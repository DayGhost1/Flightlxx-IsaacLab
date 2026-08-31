#!/usr/bin/env bash
set -euo pipefail

RUN_DIR='/home/lu/FlightLxx-IsaacLab/outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0/20260827_010721_seed1'
PROJECT='/home/lu/FlightLxx-IsaacLab'
EVAL_ROOT="$RUN_DIR/manual_evaluations"
LOG_DIR="$EVAL_ROOT/six_hour_full_evaluation_logs"
LOCK="$LOG_DIR/worker.lock"
DONE="$LOG_DIR/BATCH_DONE"

mkdir -p "$LOG_DIR"
if [[ -e "$LOCK" ]]; then
  echo "Another evaluation worker holds $LOCK" >&2
  exit 3
fi
trap 'rm -f "$LOCK"' EXIT
touch "$LOCK"
rm -f "$DONE"

is_complete() {
  local result="$1"
  [[ -s "$result" ]] || return 1
  python3 - "$result" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1], encoding='utf-8'))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if value.get('stage') == 'completed'
                 and 'independent_recovered_count' in value
                 and 'sequential_recovered_count' in value else 1)
PY
}

mapfile -t checkpoints < <(find "$RUN_DIR/checkpoints" -maxdepth 1 -type f \( -name 'step_*.pt' -o -name 'final_step_*.pt' \) -printf '%f\n' | sort)
printf 'Full evaluation started at %s; checkpoints=%s\n' "$(date --iso-8601=seconds)" "${#checkpoints[@]}"

for checkpoint_name in "${checkpoints[@]}"; do
  checkpoint="$RUN_DIR/checkpoints/$checkpoint_name"
  stem="${checkpoint_name%.pt}"
  for level in small medium large; do
    output="$EVAL_ROOT/$level/$stem"
    result="$output/runner_result.json"
    logfile="$LOG_DIR/${stem}_${level}.log"
    if is_complete "$result"; then
      printf 'SKIP completed %s %s\n' "$stem" "$level"
      continue
    fi
    printf 'START %s %s %s\n' "$(date --iso-8601=seconds)" "$stem" "$level"
    (
      cd "$PROJECT"
      RUN_DIR="$RUN_DIR" \
      CHECKPOINT="$checkpoint" \
      IMPACT_LEVEL="$level" \
      ./scripts/run_evaluate.sh
    ) >"$logfile" 2>&1
    is_complete "$result"
    printf 'DONE %s %s %s\n' "$(date --iso-8601=seconds)" "$stem" "$level"
  done
done

printf 'Full evaluation completed at %s\n' "$(date --iso-8601=seconds)" | tee "$DONE"
