#!/usr/bin/env bash
set -euo pipefail

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-/home/lu/isaacsim}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/lu/IsaacLab-v2.1.1}"
FLIGHTLXX_DIR="${FLIGHTLXX_DIR:-/home/lu/FlightLxx-IsaacLab}"
FASTTD3_DIR="${FASTTD3_DIR:-/home/lu/FastTD3}"

for required in \
    "$ISAAC_SIM_DIR/python.sh" \
    "$ISAACLAB_DIR/isaaclab.sh" \
    "$FLIGHTLXX_DIR/source/flightlxx_isaaclab" \
    "$FASTTD3_DIR/fast_td3/train.py"; do
    if [[ ! -e "$required" ]]; then
        echo "Required project path is missing: $required" >&2
        return 2 2>/dev/null || exit 2
    fi
done

export ISAAC_SIM_DIR ISAACLAB_DIR FLIGHTLXX_DIR FASTTD3_DIR
export PYTHONPATH="$FLIGHTLXX_DIR/source/flightlxx_isaaclab:$FASTTD3_DIR/fast_td3:$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_EXECUTABLE="$ISAAC_SIM_DIR/python.sh"
