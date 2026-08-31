#!/usr/bin/env bash
set -euo pipefail

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-$HOME/IsaacLab-v2.1.1}"
FAST_TD3_DIR="${FAST_TD3_DIR:-$HOME/FastTD3}"
MIGRATION_DIR="${MIGRATION_DIR:-$HOME/FlightLxx-IsaacLab}"
ISAAC_SIM_ZIP="${ISAAC_SIM_ZIP:-$HOME/Downloads/isaac-sim-standalone-4.5.0-linux-x86_64.zip}"
ISAAC_LAB_COMMIT="90b79bb2d44feb8d833f260f2bf37da3487180ba"
FAST_TD3_COMMIT="229ed59bbf43ea2f7a2d5d90d1076314839944d7"

require_archive_revision() {
    local source_dir="$1"
    local expected_commit="$2"
    if [[ ! -f "$source_dir/.source_revision" ]] || ! grep -qx "commit=$expected_commit" "$source_dir/.source_revision"; then
        echo "Expected source archive revision $expected_commit in $source_dir/.source_revision" >&2
        exit 3
    fi
}

if [[ "$(lsb_release -rs)" != "20.04" ]]; then
    echo "Refusing installation: expected Ubuntu 20.04.x" >&2
    exit 2
fi
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | grep -q '^NVIDIA GeForce RTX 4090, 535.230.02$'; then
    echo "Refusing installation: expected RTX 4090 with driver 535.230.02" >&2
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >&2
    exit 2
fi

if [[ ! -x "$ISAAC_SIM_DIR/python.sh" ]]; then
    if [[ -e "$ISAAC_SIM_DIR" ]]; then
        echo "Isaac Sim target exists but is incomplete; inspect it instead of overwriting: $ISAAC_SIM_DIR" >&2
        exit 3
    fi
    if [[ ! -f "$ISAAC_SIM_ZIP" ]]; then
        echo "Required NVIDIA binary is absent: $ISAAC_SIM_ZIP" >&2
        echo "Download the exact 4.5.0 Linux standalone archive, then rerun." >&2
        exit 4
    fi
    mkdir "$ISAAC_SIM_DIR"
    unzip -q "$ISAAC_SIM_ZIP" -d "$ISAAC_SIM_DIR"
    "$ISAAC_SIM_DIR/post_install.sh"
fi

PYTHON_VERSION="$($ISAAC_SIM_DIR/python.sh -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "Expected Isaac Sim Python 3.10, got $PYTHON_VERSION" >&2
    exit 5
fi

if [[ ! -d "$ISAAC_LAB_DIR/.git" ]]; then
    if [[ -e "$ISAAC_LAB_DIR" ]]; then
        require_archive_revision "$ISAAC_LAB_DIR" "$ISAAC_LAB_COMMIT"
    else
        git clone https://github.com/isaac-sim/IsaacLab.git "$ISAAC_LAB_DIR"
        git -C "$ISAAC_LAB_DIR" fetch --tags origin
        git -C "$ISAAC_LAB_DIR" checkout --detach "$ISAAC_LAB_COMMIT"
    fi
else
    if [[ -n "$(git -C "$ISAAC_LAB_DIR" status --porcelain)" ]]; then
        echo "Isaac Lab checkout has user changes; refusing to overwrite them." >&2
        exit 3
    fi
    git -C "$ISAAC_LAB_DIR" fetch --tags origin
    git -C "$ISAAC_LAB_DIR" checkout --detach "$ISAAC_LAB_COMMIT"
fi
if [[ -e "$ISAAC_LAB_DIR/_isaac_sim" && ! -L "$ISAAC_LAB_DIR/_isaac_sim" ]]; then
    echo "Refusing to replace non-symlink $ISAAC_LAB_DIR/_isaac_sim" >&2
    exit 3
fi
if [[ ! -e "$ISAAC_LAB_DIR/_isaac_sim" ]]; then
    ln -s "$ISAAC_SIM_DIR" "$ISAAC_LAB_DIR/_isaac_sim"
fi
# Install the core package without build isolation.  Isaac Sim's bundled Python
# already has the compatible build backend; build isolation misses pkg_resources
# in this pinned 4.5/v2.1.1 combination.
"$ISAAC_SIM_DIR/python.sh" -m pip install --timeout 45 --retries 3 --no-build-isolation \
    -e "$ISAAC_LAB_DIR/source/isaaclab"
"$ISAAC_LAB_DIR/isaaclab.sh" --install rsl_rl
"$ISAAC_SIM_DIR/python.sh" -c 'import isaaclab, rsl_rl; print(isaaclab.__file__); print(rsl_rl.__file__)'

if [[ ! -d "$FAST_TD3_DIR/.git" ]]; then
    if [[ -e "$FAST_TD3_DIR" ]]; then
        require_archive_revision "$FAST_TD3_DIR" "$FAST_TD3_COMMIT"
        if [[ ! -f "$FAST_TD3_DIR/.flightlxx_patch_applied" ]]; then
            (cd "$FAST_TD3_DIR" && git apply "$MIGRATION_DIR/patches/fasttd3-flightlxx.patch")
            touch "$FAST_TD3_DIR/.flightlxx_patch_applied"
        fi
    else
        git clone https://github.com/younggyoseo/FastTD3.git "$FAST_TD3_DIR"
        git -C "$FAST_TD3_DIR" checkout --detach "$FAST_TD3_COMMIT"
        git -C "$FAST_TD3_DIR" apply "$MIGRATION_DIR/patches/fasttd3-flightlxx.patch"
    fi
else
    if [[ -n "$(git -C "$FAST_TD3_DIR" status --porcelain)" ]]; then
        if ! git -C "$FAST_TD3_DIR" apply --reverse --check "$MIGRATION_DIR/patches/fasttd3-flightlxx.patch"; then
            echo "FastTD3 checkout has changes other than the expected FlightLxx patch; refusing to overwrite them." >&2
            exit 3
        fi
    else
        git -C "$FAST_TD3_DIR" fetch origin
        git -C "$FAST_TD3_DIR" checkout --detach "$FAST_TD3_COMMIT"
        git -C "$FAST_TD3_DIR" apply "$MIGRATION_DIR/patches/fasttd3-flightlxx.patch"
    fi
fi
if [[ -d "$FAST_TD3_DIR/.git" ]] && [[ "$(git -C "$FAST_TD3_DIR" rev-parse HEAD)" != "$FAST_TD3_COMMIT" ]]; then
    echo "FastTD3 base commit is not the required commit." >&2
    exit 3
fi

ISAAC_PYTHON="$ISAAC_SIM_DIR/python.sh"
"$ISAAC_PYTHON" -m pip install -r "$MIGRATION_DIR/requirements/fasttd3_isaaclab45.txt"
"$ISAAC_PYTHON" -m pip install --no-deps -e "$FAST_TD3_DIR"
"$ISAAC_PYTHON" -m pip install --no-deps -e "$MIGRATION_DIR/source/flightlxx_isaaclab"

echo "Installed fixed Isaac Sim, Isaac Lab, FastTD3, and FlightLxx migration."
