#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project_env.sh
source "$SCRIPT_DIR/project_env.sh"

# =========================== 用户通常只需设置这里 ===========================
# CHECKPOINT      : 推荐直接给候选模型的绝对路径；为空时按 RUN_DIR/CHECKPOINT_STEP 查找。
# CHECKPOINT_STEP : latest 或整数，例如 75000。CHECKPOINT 非空时本项不生效。
# RUN_DIR         : 训练时间戳目录；为空时选择最新训练目录。
# OUTPUT_DIR      : 本次74-episode验证目录。继续中断任务时必须复用同一目录。
# RESUME          : 1 表示跳过已有完整 result.json 的 trial；默认1。
# DEVICE          : 默认 cuda:0。
# SMOKE           : 1 只跑规则/随机/连续/球碰撞各1例，用于检查链路，不构成准入结果。
# TRIAL_IDS       : 可选，空格分隔的精确试验ID；例如 "S_T01_small R01 C01 P_B01_1mps"。
# ALLOW_DURING_TRAINING: 默认0，避免与训练争抢GPU。
CHECKPOINT_STEP="${CHECKPOINT_STEP:-latest}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda:0}"
RESUME="${RESUME:-1}"
SMOKE="${SMOKE:-0}"
TRIAL_IDS="${TRIAL_IDS:-}"
ALLOW_DURING_TRAINING="${ALLOW_DURING_TRAINING:-0}"
TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-$FLIGHTLXX_DIR/outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0}"
MANIFEST="${MANIFEST:-$FLIGHTLXX_DIR/source/flightlxx_isaaclab/flightlxx_isaaclab/config/preflight_validation_v1.json}"

if [[ "$ALLOW_DURING_TRAINING" != "1" ]] \
    && pgrep -f 'train.py.*Isaac-FlightLxx-CTBR-Recovery-Direct-v0' >/dev/null; then
    echo "检测到训练进程。全面验证默认不与训练争抢GPU。" >&2
    echo "等待训练结束，或明确设置 ALLOW_DURING_TRAINING=1。" >&2
    exit 3
fi

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(find "$TASK_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "$CHECKPOINT" ]]; then
    if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR/checkpoints" ]]; then
        echo "未找到训练目录；请设置 CHECKPOINT 或 RUN_DIR。" >&2
        exit 2
    fi
    if [[ "$CHECKPOINT_STEP" == "latest" ]]; then
        checkpoint_file="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' \
            -printf '%f\n' | sort | tail -n 1)"
        [[ -n "$checkpoint_file" ]] && CHECKPOINT="$RUN_DIR/checkpoints/$checkpoint_file"
    elif [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
        printf -v checkpoint_file 'step_%08d.pt' "$((10#$CHECKPOINT_STEP))"
        CHECKPOINT="$RUN_DIR/checkpoints/$checkpoint_file"
    else
        echo "CHECKPOINT_STEP 只能是 latest 或非负整数。" >&2
        exit 2
    fi
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "checkpoint不存在：$CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "验证协议不存在：$MANIFEST" >&2
    exit 2
fi

size_before="$(stat -c %s "$CHECKPOINT")"
sleep 1
size_after="$(stat -c %s "$CHECKPOINT")"
if [[ "$size_before" -le 0 || "$size_before" != "$size_after" ]]; then
    echo "checkpoint仍在写入：$CHECKPOINT" >&2
    exit 4
fi

checkpoint_name="$(basename "$CHECKPOINT" .pt)"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    validation_id="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="$FLIGHTLXX_DIR/outputs/preflight/$checkpoint_name/$validation_id"
fi
mkdir -p "$OUTPUT_DIR"

runner_args=(
    --checkpoint "$CHECKPOINT"
    --manifest "$MANIFEST"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --headless
)
[[ "$RESUME" == "1" ]] && runner_args+=(--resume) || runner_args+=(--no-resume)
[[ "$SMOKE" == "1" ]] && runner_args+=(--smoke)
if [[ -n "$TRIAL_IDS" ]]; then
    read -r -a selected_ids <<< "$TRIAL_IDS"
    for trial_id in "${selected_ids[@]}"; do
        runner_args+=(--trial-id "$trial_id")
    done
fi

echo "checkpoint=$CHECKPOINT"
echo "manifest=$MANIFEST"
echo "output_dir=$OUTPUT_DIR"
echo "device=$DEVICE resume=$RESUME smoke=$SMOKE"
cd "$FASTTD3_DIR/fast_td3"
"$ISAACLAB_DIR/isaaclab.sh" -p "$FLIGHTLXX_DIR/scripts/evaluate_preflight_suite.py" "${runner_args[@]}" "$@"
