#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=project_env.sh
source "$SCRIPT_DIR/project_env.sh"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a PPO checkpoint path}"
OUTPUT="${OUTPUT:-$FLIGHTLXX_DIR/outputs/real_flight_replay/step_00050000_results.json}"
DURATION="${DURATION:-20.0}"
SPEED="${SPEED:-0.6}"
WARMUP_UPDATES="${WARMUP_UPDATES:-120}"
EXPECTED_SHA256="${EXPECTED_SHA256:-}"

test -f "$CHECKPOINT" || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }

cd "$FLIGHTLXX_DIR"
sha_args=()
if [[ -n "$EXPECTED_SHA256" ]]; then
    sha_args=(--expected_sha256 "$EXPECTED_SHA256")
fi
exec "$ISAACLAB_DIR/isaaclab.sh" -p "$FLIGHTLXX_DIR/scripts/visualize_real_flight_replay.py" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --duration "$DURATION" \
    --speed "$SPEED" \
    --warmup_updates "$WARMUP_UPDATES" \
    "${sha_args[@]}" \
    "$@"
