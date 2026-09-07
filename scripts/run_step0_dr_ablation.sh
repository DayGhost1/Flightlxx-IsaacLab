#!/usr/bin/env bash
# Step-0 DR ablation on a selected PPO checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project_env.sh
source "$SCRIPT_DIR/project_env.sh"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a PPO checkpoint}"
OUT_DIR="${OUT_DIR:-$FLIGHTLXX_DIR/outputs/ppo-dr-ablation}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1024}"
MODES="${MODES:-vicon vicon_realistic vicon_realistic_lpf full}"

mkdir -p "$OUT_DIR"
test -f "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }

cd "$FLIGHTLXX_DIR"
for mode in $MODES; do
  echo "===== DR mode: $mode ====="
  "$ISAACLAB_DIR/isaaclab.sh" -p "$FLIGHTLXX_DIR/scripts/dr_ablation_eval.py" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUT_DIR/${mode}.json" \
    --dr-mode "$mode" \
    --num-envs "$NUM_ENVS" \
    --device "$DEVICE" \
    --headless \
    2>&1 | tee "$OUT_DIR/${mode}.log"
done

python3 - <<PY
import json
from pathlib import Path
out = Path("$OUT_DIR")
print()
print(f"{'mode':24s} {'hover_strict':>12s} {'hover_coarse':>12s} {'v_rms':>8s} {'w_rms':>8s} {'p_rms':>8s}")
for path in sorted(out.glob("*.json")):
    d = json.loads(path.read_text())
    s = d["summary"]
    print(
        f"{d['dr_mode']:24s} "
        f"{100*s['hover_strict_success_rate']:11.1f}% "
        f"{100*s['hover_coarse_success_rate']:11.1f}% "
        f"{s['hover_linear_speed_rms']:8.3f} "
        f"{s['hover_angular_speed_rms']:8.3f} "
        f"{s['hover_position_error_rms']:8.3f}"
    )
PY
