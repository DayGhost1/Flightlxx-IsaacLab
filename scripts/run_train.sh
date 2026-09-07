#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/project_env.sh"
cd "$FLIGHTLXX_DIR"
extra=()
[[ "${ALLOW_PLACEHOLDER_TARGET:-0}" == "1" ]] && extra+=(--allow-placeholder-target)
[[ -n "${RESUME:-}" ]] && extra+=(--resume "$RESUME")
exec "$ISAACLAB_DIR/isaaclab.sh" -p "$SCRIPT_DIR/train_ppo.py" \
    --headless \
    --num-envs "${NUM_ENVS:-1024}" \
    --iterations "${ITERATIONS:-2000}" \
    --rollout-steps "${ROLLOUT_STEPS:-128}" \
    --epochs "${EPOCHS:-5}" \
    --minibatches "${MINIBATCHES:-8}" \
    --save-interval "${SAVE_INTERVAL:-100}" \
    --exam-interval "${EXAM_INTERVAL:-32}" \
    --learning-rate "${LEARNING_RATE:-0.0003}" \
    --output-root "${OUTPUT_ROOT:-$FLIGHTLXX_DIR/outputs/ppo}" \
    "${extra[@]}" "$@"
