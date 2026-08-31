#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project_env.sh
source "$SCRIPT_DIR/project_env.sh"

REAL_PLATFORM_CONFIG="$FLIGHTLXX_DIR/source/flightlxx_isaaclab/flightlxx_isaaclab/config/snowyowl3_real_v1.json"
ALLOW_PLACEHOLDER_TARGET="${ALLOW_PLACEHOLDER_TARGET:-0}"
platform_validation_args=(--config "$REAL_PLATFORM_CONFIG")
if [[ "$ALLOW_PLACEHOLDER_TARGET" == "1" ]]; then
    platform_validation_args+=(--allow-placeholder-target)
fi
"$PYTHON_EXECUTABLE" "$SCRIPT_DIR/validate_realistic_training_config.py" "${platform_validation_args[@]}"

# ========================= 用户通常只需修改或覆盖本节参数 =========================
# 本文件给出 RTX 4090 的正式训练基线。优先在命令前临时覆盖变量，不必反复改脚本：
#   BATCH_SIZE=24576 TOTAL_TIMESTEPS=2000 ./scripts/run_train.sh
#
# TOTAL_TIMESTEPS : 一轮训练的 global steps；正式基线为 250000。
# NUM_ENVS        : 并行训练环境数；增大可提高仿真吞吐量，也会增加显存占用。
# BUFFER_SIZE     : 每个并行环境保留的 replay 长度；总 transition 数约为两者乘积。
# BATCH_SIZE      : 每次梯度更新的样本数；最直接影响网络训练显存。
# NUM_UPDATES     : 每个 global step 的梯度更新次数；增大后训练更慢且更易过拟合旧数据。
# SAVE_INTERVAL   : checkpoint 间隔；保存后训练器会自动跑固定五冲击评测。
# LOG_INTERVAL    : TensorBoard/CSV/终端日志间隔。
# LEARNING_STARTS : 开始梯度更新前先收集多少个 global steps。
# REPLAY_BAND_FRACTIONS : 每批无撞击/简单/中间/当前/探测层比例；五项与失败上下文之和为 1。
# FAILURE_CONTEXT_FRACTION: 失败发生前因果片段的独立采样比例。
# FAILURE_CONTEXT_STEPS   : 每次失败最多向前标记的控制步数（默认 50，即 1 s）。
# ACTIVE_IMPACT_FRACTION: 每个 batch 中正在受力的稀有 transition 目标比例。
# RECOVERY_PHASE_FRACTION: 每个 batch 中撞击后前 2 s 恢复 transition 的目标比例。
# OUTPUT_ROOT     : 所有时间戳 run、checkpoints、logs 和 reports 的根目录。
#
# 额外 FastTD3 参数可直接追加在脚本末尾，例如：
#   ./scripts/run_train.sh --seed 2 --target-noise-clip 0.1
# 完整操作说明见 scripts/README_CN.md。修改这里不会改变已经运行的训练进程。
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-250000}"
NUM_ENVS="${NUM_ENVS:-1024}"
BUFFER_SIZE="${BUFFER_SIZE:-3072}"
BATCH_SIZE="${BATCH_SIZE:-24576}"
NUM_UPDATES="${NUM_UPDATES:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
LEARNING_STARTS="${LEARNING_STARTS:-10}"
REPLAY_BAND_FRACTIONS="${REPLAY_BAND_FRACTIONS:-0.10 0.15 0.20 0.35 0.10}"
FAILURE_CONTEXT_FRACTION="${FAILURE_CONTEXT_FRACTION:-0.10}"
FAILURE_CONTEXT_STEPS="${FAILURE_CONTEXT_STEPS:-50}"
ACTIVE_IMPACT_FRACTION="${ACTIVE_IMPACT_FRACTION:-0.05}"
RECOVERY_PHASE_FRACTION="${RECOVERY_PHASE_FRACTION:-0.20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FLIGHTLXX_DIR/outputs/training}"

read -r -a replay_band_fractions <<< "$REPLAY_BAND_FRACTIONS"
if [[ "${#replay_band_fractions[@]}" -ne 5 ]]; then
    echo "REPLAY_BAND_FRACTIONS 必须包含五个空格分隔的数：无撞击 简单 中间 当前 探测" >&2
    exit 2
fi

LAUNCH_ID="$(date +%Y%m%d_%H%M%S)"
MONITOR_DIR="$OUTPUT_ROOT/monitoring/$LAUNCH_ID"
mkdir -p "$MONITOR_DIR"
GPU_LOG="$MONITOR_DIR/gpu_memory.csv"
PROCESS_LOG="$MONITOR_DIR/gpu_process_memory.csv"
TRAIN_LOG="$MONITOR_DIR/launcher.log"

echo "timestamp,index,utilization_gpu_percent,memory_total_mib,memory_used_mib,memory_free_mib" > "$GPU_LOG"
echo "timestamp,pid,process_name,used_memory_mib" > "$PROCESS_LOG"
PROVENANCE_ARTIFACT="$(
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/write_training_provenance.py" \
        --config "$REAL_PLATFORM_CONFIG" \
        --output-dir "$MONITOR_DIR" \
        --launch-id "$LAUNCH_ID"
)"

monitor_gpu() {
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,memory.total,memory.used,memory.free \
            --format=csv,noheader,nounits >> "$GPU_LOG" || true
        timestamp="$(date --iso-8601=seconds)"
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>/dev/null \
            | while IFS= read -r row; do printf '%s,%s\n' "$timestamp" "$row"; done \
            >> "$PROCESS_LOG" || true
        sleep 1
    done
}

monitor_gpu &
MONITOR_PID=$!
cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "FlightLxx FastTD3 launch: $LAUNCH_ID" | tee "$TRAIN_LOG"
echo "platform_provenance=$PROVENANCE_ARTIFACT" | tee -a "$TRAIN_LOG"
echo "num_envs=$NUM_ENVS buffer_size=$BUFFER_SIZE batch_size=$BATCH_SIZE num_updates=$NUM_UPDATES" | tee -a "$TRAIN_LOG"
echo "replay_bands=$REPLAY_BAND_FRACTIONS failure_context_fraction=$FAILURE_CONTEXT_FRACTION failure_context_steps=$FAILURE_CONTEXT_STEPS" | tee -a "$TRAIN_LOG"
echo "active_impact_fraction=$ACTIVE_IMPACT_FRACTION recovery_phase_fraction=$RECOVERY_PHASE_FRACTION" | tee -a "$TRAIN_LOG"
echo "monitor_dir=$MONITOR_DIR" | tee -a "$TRAIN_LOG"

cd "$FASTTD3_DIR/fast_td3"
"$ISAACLAB_DIR/isaaclab.sh" -p train.py \
    --env-name Isaac-FlightLxx-CTBR-Recovery-Direct-v0 \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --num-eval-envs "$NUM_ENVS" \
    --buffer-size "$BUFFER_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --num-updates "$NUM_UPDATES" \
    --learning-starts "$LEARNING_STARTS" \
    --band-balance-fractions "${replay_band_fractions[@]}" \
    --failure-context-fraction "$FAILURE_CONTEXT_FRACTION" \
    --failure-context-steps "$FAILURE_CONTEXT_STEPS" \
    --active-impact-fraction "$ACTIVE_IMPACT_FRACTION" \
    --recovery-phase-fraction "$RECOVERY_PHASE_FRACTION" \
    --save-interval "$SAVE_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --output-root "$OUTPUT_ROOT" \
    --no-use-wandb --no-compile "$@" 2>&1 | tee -a "$TRAIN_LOG"
