#!/usr/bin/env bash
set -euo pipefail

echo "=== operating system ==="
lsb_release -a
uname -a
echo "=== gpu ==="
nvidia-smi
echo "=== cpu and memory ==="
lscpu
free -h
echo "=== disks ==="
df -hT "$HOME"
echo "=== existing installations ==="
for path in "$HOME/isaacsim" "$HOME/IsaacLab-v2.1.1" "$HOME/FastTD3" "$HOME/FlightLxx-IsaacLab"; do
    if [[ -e "$path" ]]; then
        printf 'EXISTS %s -> %s\n' "$path" "$(realpath "$path")"
    else
        printf 'MISSING %s\n' "$path"
    fi
done
find "$HOME/Downloads" -maxdepth 1 -type f -name 'isaac-sim-standalone-4.5.0-linux-x86_64.zip' -print 2>/dev/null || true

