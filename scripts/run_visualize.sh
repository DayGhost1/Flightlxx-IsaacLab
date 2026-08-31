#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project_env.sh
source "$SCRIPT_DIR/project_env.sh"

# ========================= 用户通常只需覆盖以下参数 =========================
# CHECKPOINT_STEP: 默认 latest；也可写 75000、100000 等 global step。
# RUN_DIR/CHECKPOINT/OUTPUT_DIR: 与 run_evaluate.sh 相同。
# IMPACT_LEVEL    : small / medium / large，分别为基准力的 1.0 / 2.0 / 2.5 倍。
# SPEED           : 仿真回放相对实时速度，默认 1.0。
# WARMUP_UPDATES  : GUI 加载后预热的渲染帧数，默认 180。
# AUTO_START      : 默认 1，加载完成自动开始；设 0 后点击 GUI 中 Start / Replay。
# EXIT_AFTER_RUN  : 默认 0，回放结束保留 Isaac Sim；设 1 自动退出。
# OPEN_REPORT     : 默认 1，完成后自动打开曲线；设 0 不自动打开。
# ALLOW_DURING_TRAINING: 默认 0，防止 GUI 与训练争抢 GPU。
CHECKPOINT_STEP="${CHECKPOINT_STEP:-latest}"
IMPACT_LEVEL="${IMPACT_LEVEL:-small}"
ALLOW_DURING_TRAINING="${ALLOW_DURING_TRAINING:-0}"
DEVICE="${DEVICE:-cuda:0}"
SPEED="${SPEED:-1.0}"
WARMUP_UPDATES="${WARMUP_UPDATES:-180}"
AUTO_START="${AUTO_START:-1}"
EXIT_AFTER_RUN="${EXIT_AFTER_RUN:-0}"
OPEN_REPORT="${OPEN_REPORT:-1}"
TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-$FLIGHTLXX_DIR/outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"

case "$IMPACT_LEVEL" in
    small|medium|large) ;;
    *) echo "IMPACT_LEVEL must be small, medium, or large." >&2; exit 2 ;;
esac

if [[ "$ALLOW_DURING_TRAINING" != "1" ]] \
    && pgrep -f 'train.py.*Isaac-FlightLxx-CTBR-Recovery-Direct-v0' >/dev/null; then
    echo "A FlightLxx training process is active. GUI playback is blocked to avoid GPU contention." >&2
    echo "Wait for training to finish, or explicitly set ALLOW_DURING_TRAINING=1." >&2
    exit 3
fi

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(find "$TASK_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR/checkpoints" ]]; then
    echo "No valid training run was found. Set RUN_DIR to a timestamped run directory." >&2
    exit 2
fi

if [[ -z "$CHECKPOINT" ]]; then
    if [[ "$CHECKPOINT_STEP" == "latest" ]]; then
        CHECKPOINT="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name 'step_*.pt' \
            -printf '%f\n' | sort | tail -n 1)"
        [[ -n "$CHECKPOINT" ]] && CHECKPOINT="$RUN_DIR/checkpoints/$CHECKPOINT"
    elif [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
        printf -v checkpoint_file 'step_%08d.pt' "$((10#$CHECKPOINT_STEP))"
        CHECKPOINT="$RUN_DIR/checkpoints/$checkpoint_file"
    else
        echo "CHECKPOINT_STEP must be latest or a non-negative integer." >&2
        exit 2
    fi
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT:-<empty>}" >&2
    exit 2
fi

size_before="$(stat -c %s "$CHECKPOINT")"
sleep 1
size_after="$(stat -c %s "$CHECKPOINT")"
if [[ "$size_before" -le 0 || "$size_before" != "$size_after" ]]; then
    echo "Checkpoint file is still being written: $CHECKPOINT" >&2
    exit 4
fi

CHECKPOINT_NAME="$(basename "$CHECKPOINT" .pt)"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/manual_visualizations/$IMPACT_LEVEL/$CHECKPOINT_NAME}"
mkdir -p "$OUTPUT_DIR"

visual_args=(
    --checkpoint "$CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --impact_level "$IMPACT_LEVEL"
    --device "$DEVICE"
    --speed "$SPEED"
    --warmup_updates "$WARMUP_UPDATES"
)
[[ "$AUTO_START" == "1" ]] && visual_args+=(--auto_start)
[[ "$EXIT_AFTER_RUN" == "1" ]] && visual_args+=(--exit_after_run)
[[ "$OPEN_REPORT" == "0" ]] && visual_args+=(--no_open_report)

echo "checkpoint=$CHECKPOINT"
echo "output_dir=$OUTPUT_DIR"
echo "speed=$SPEED auto_start=$AUTO_START exit_after_run=$EXIT_AFTER_RUN"
echo "impact_level=$IMPACT_LEVEL"

cd "$FASTTD3_DIR/fast_td3"
"$ISAACLAB_DIR/isaaclab.sh" -p "$FLIGHTLXX_DIR/scripts/visualize_fixed_impacts.py" \
    "${visual_args[@]}" "$@"
