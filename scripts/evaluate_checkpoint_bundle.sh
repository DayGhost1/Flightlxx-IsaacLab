#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project_env.sh
source "$SCRIPT_DIR/project_env.sh"

TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-$FLIGHTLXX_DIR/outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"
IMPACT_LEVEL="${IMPACT_LEVEL:-small}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR="$(find "$TASK_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR/checkpoints" ]]; then
    echo "No valid training run was found. Set RUN_DIR explicitly." >&2
    exit 2
fi
if [[ -z "$CHECKPOINT" ]]; then
    checkpoint_name="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f \
        \( -name 'step_*.pt' -o -name 'final_step_*.pt' \) -printf '%f\n' | sort | tail -n 1)"
    [[ -n "$checkpoint_name" ]] && CHECKPOINT="$RUN_DIR/checkpoints/$checkpoint_name"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT:-<empty>}" >&2
    exit 2
fi

checkpoint_size_before="$(stat -c %s "$CHECKPOINT")"
sleep 1
checkpoint_size_after="$(stat -c %s "$CHECKPOINT")"
if [[ "$checkpoint_size_before" -le 0 || "$checkpoint_size_before" != "$checkpoint_size_after" ]]; then
    echo "Checkpoint file is still being written: $CHECKPOINT" >&2
    exit 4
fi

stem="$(basename "$CHECKPOINT" .pt)"
evaluation_root="$RUN_DIR/evaluation"
impact_output="$evaluation_root/fixed_impacts/$IMPACT_LEVEL/$stem"
impact_result="$impact_output/runner_result.json"
automatic_impact_result="$evaluation_root/$stem.json"
real_state_dir="$evaluation_root/real_state"
real_state_result="$real_state_dir/$stem.json"
bundle_dir="$evaluation_root/bundles/$stem"
bundle_result="$bundle_dir/bundle_result.json"
mkdir -p "$impact_output" "$real_state_dir" "$bundle_dir"

json_stage_is_completed() {
    local result="$1"
    [[ -s "$result" ]] || return 1
    python3 - "$result" <<'PY'
import json
import sys

try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if value.get("stage") == "completed" else 1)
PY
}

if [[ -s "$automatic_impact_result" ]]; then
    impact_result="$automatic_impact_result"
elif ! json_stage_is_completed "$impact_result"; then
    RUN_DIR="$RUN_DIR" \
    CHECKPOINT="$CHECKPOINT" \
    IMPACT_LEVEL="$IMPACT_LEVEL" \
    OUTPUT_DIR="$impact_output" \
    DEVICE="$DEVICE" \
    ALLOW_DURING_TRAINING=1 \
        "$SCRIPT_DIR/run_evaluate.sh"
fi

if [[ ! -s "$real_state_result" ]]; then
    CHECKPOINT="$CHECKPOINT" \
    OUTPUT="$real_state_result" \
        "$SCRIPT_DIR/run_real_flight_replay.sh" --headless --exit_after_run --device "$DEVICE"
fi

python3 - "$CHECKPOINT" "$impact_result" "$real_state_result" "$bundle_result" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

checkpoint, impact_path, real_path, output_path = map(Path, sys.argv[1:])
impact = json.loads(impact_path.read_text(encoding="utf-8"))
real_state = json.loads(real_path.read_text(encoding="utf-8"))

scenarios = real_state.get("scenarios", [])
real_state_completed = bool(scenarios)
real_state_safe = real_state_completed and all(
    not scenario.get("crashed", False)
    and scenario.get("max_attitude_error_rad", float("inf")) < 1.0472
    and scenario.get("max_angular_speed_rad_s", float("inf")) < 10.0
    for scenario in scenarios
)
payload = {
    "stage": "completed",
    "checkpoint": str(checkpoint),
    "checkpoint_step": int(impact.get("checkpoint_step", real_state.get("checkpoint_step", 0))),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "fixed_five_impacts": impact,
    "real_state": {
        "result_path": str(real_path),
        "scenario_count": len(scenarios),
        "safe": real_state_safe,
        "scenarios": scenarios,
    },
    "passed": bool(impact.get("checkpoint_passed", impact.get("passed", False))) and real_state_safe,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True), flush=True)
PY

echo "bundle_result=$bundle_result"
