# Installation

## Version lock

- Isaac Sim standalone binary 4.5.0, Python 3.10
- Isaac Lab v2.1.1 (`90b79bb2d44feb8d833f260f2bf37da3487180ba`)
- FastTD3 (`229ed59bbf43ea2f7a2d5d90d1076314839944d7`)

Ubuntu 20.04 must use Isaac Sim's binary installation route. Extract `isaac-sim-standalone-4.5.0-linux-x86_64.zip` to `~/isaacsim`, run `~/isaacsim/post_install.sh`, and make Isaac Lab's `_isaac_sim` symlink point at `~/isaacsim`.

For this installation, the pinned GitHub source archives are retained under `~/Downloads` and their exact commit plus SHA-256 are recorded in `~/IsaacLab-v2.1.1/.source_revision` and `~/FastTD3/.source_revision`. This avoids an unreliable host-side GitHub clone while preserving the requested sources.

```bash
ln -s ~/isaacsim ~/IsaacLab-v2.1.1/_isaac_sim
cd ~/IsaacLab-v2.1.1
~/isaacsim/python.sh -m pip install --timeout 45 --retries 3 --no-build-isolation \
  -e ~/IsaacLab-v2.1.1/source/isaaclab
./isaaclab.sh --install rsl_rl
./isaaclab.sh -p -m pip install --no-deps -e ~/FlightLxx-IsaacLab/source/flightlxx_isaaclab
~/isaacsim/python.sh -m pip install -r ~/FlightLxx-IsaacLab/requirements/fasttd3_isaaclab45.txt
~/isaacsim/python.sh -m pip install --no-deps -e ~/FastTD3
```

Do not install FastTD3's generic `requirements/requirements.txt`: it pins PyTorch 2.6/CUDA 12.4 and would replace Isaac Lab v2.1.1's PyTorch 2.7/CUDA 12.8. The dedicated requirements file intentionally excludes PyTorch packages.

For a guarded installation after copying this project to the host, run `scripts/remote_inventory.sh`, inspect its output, then run `scripts/remote_install.sh`. It refuses unexpected OS/GPU/driver combinations, incomplete target directories, mismatched source revisions, and user-modified FastTD3 checkouts.

## Smoke commands

```bash
~/isaacsim/python.sh -c 'import sys, torch; print(sys.version); print(torch.cuda.get_device_name())'
cd ~/IsaacLab-v2.1.1
./isaaclab.sh -p ~/FlightLxx-IsaacLab/scripts/isaac_sim_app_smoke.py --headless
./isaaclab.sh -p ~/FlightLxx-IsaacLab/scripts/official_quadcopter_smoke.py \
  --headless --device cuda:0 --num_envs 8 --steps 32
./isaaclab.sh -p ~/FlightLxx-IsaacLab/scripts/smoke_env.py --headless --device cuda:0 --num_envs 8
```

The final runtime verification commands and their outputs are recorded in `TEST_RESULTS.md` after the headless checks complete.
