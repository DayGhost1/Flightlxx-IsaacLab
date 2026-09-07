#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Fresh PPO training by default. Set RESUME to a PPO checkpoint to continue.
exec "$SCRIPT_DIR/run_train.sh" "$@"
