# Training and evaluation

Install this project and apply `patches/fasttd3-flightlxx.patch` to the pinned FastTD3 checkout. Start with a 4090-safe smoke configuration:

```bash
cd ~/FastTD3/fast_td3
~/IsaacLab-v2.1.1/isaaclab.sh -p train.py \
  --env-name Isaac-FlightLxx-CTBR-Recovery-Direct-v0 \
  --num-envs 64 --buffer-size 512 --batch-size 16 \
  --learning-starts 8 --total-timesteps 80 \
  --save-interval 40 --eval-interval 40 \
  --no-use-wandb --no-compile
```

For a normal headless run, use the project launcher. It loads the fixed Isaac Sim/Isaac Lab/FastTD3 paths, starts one-second GPU monitoring, and then runs the measured RTX 4090 defaults: 1024 environments, replay 3072, batch 24576, AMP BF16, no `torch.compile`, a 100-step logging interval, a **250,000-global-step full round**, and a 25,000-step checkpoint interval:

```bash
cd ~/FlightLxx-IsaacLab
./scripts/run_train.sh
```

The task-specific default is the first full RTX 4090 baseline: 1024 environments, a 3072-step-per-environment replay, global batch 24576, 251 distributional atoms, and Q support `[-10, 10]`. A 200-step capacity run measured a 19,994 MiB whole-GPU peak, 16.86 GiB peak PyTorch reserve, 86% peak utilization, and about 12.8k samples/s. Batch 32768 increased whole-GPU usage by only 520 MiB while reducing throughput by about 23%, so it is not the default. The unified normalized reciprocal reward has a maximum non-terminal rate of 2.9 before time scaling and retains the `-2` terminal failure penalty; monitor support-boundary probability before changing the distributional support. If memory is exhausted, reduce replay length, batch size, and environment count in that order. Do not replace FastTD3 with PPO.

The reward has one continuous objective rather than separate wide and precision terms. Its physical scales are 0.30 m, 0.30 m/s, 10 degrees, and 0.50 rad/s. Reward shaping, curriculum promotion and precision-hover reporting are deliberately independent: curriculum recovery uses 0.15 m / 0.15 m/s / 5 degrees / 0.25 rad/s with a 0.5 s dwell, while the strict precision metric uses 0.05 m / 0.05 m/s / 2 degrees / 0.05 rad/s with a 2 s dwell. The fixed evaluation protocol remains independent at 0.10 m / 0.10 m/s / 3 degrees / 0.10 rad/s with a 1 s dwell.

Each launch writes `gpu_memory.csv`, `gpu_process_memory.csv`, and `launcher.log` under `outputs/training/monitoring/<timestamp>/`. FastTD3 also logs current and peak PyTorch allocated/reserved memory in `progress.csv`. For a bounded capacity probe, override one parameter at a time, for example:

```bash
cd ~/FlightLxx-IsaacLab
TOTAL_TIMESTEPS=2000 SAVE_INTERVAL=0 BATCH_SIZE=24576 ./scripts/run_train.sh
```

Optimize GPU utilization and samples/s rather than memory occupancy alone. A headless RTX 4090 run should retain roughly 2--4 GiB for Isaac/PhysX and transient allocations; do not target an exact 24 GiB allocation.

Every launch creates an isolated timestamped directory under `outputs/<environment>/<timestamp>_seed<seed>/`. Terminal metrics are also written to `logs/progress.csv` and `tensorboard/`; model files go only to `checkpoints/`. Open TensorBoard with:

```bash
~/isaacsim/kit/python/bin/tensorboard \
  --logdir ~/FastTD3/fast_td3/outputs --port 6006
```

The default checkpoint interval is 25,000 global steps. This matches the old FlightLxx sample cadence: one legacy PPO iteration collected 500 steps/environment across 100 environments, so 500 iterations contained 25,000,000 transitions. With 1024 Isaac Lab environments, 25,000 global steps contain 25.6 million transitions. A full round is 250,000 global steps, therefore it contains ten evaluation nodes (25k through 225k plus the final 250k checkpoint).

Resume using `--checkpoint-path <checkpoint>`. A normal checkpoint includes Actor, twin Critic, target Critic, optimizers, LR schedulers, normalizers, and curriculum state. Add `--checkpoint-replay` only for sparse long-milestone snapshots because a 1024×2048 raw-history replay is large. Deterministic evaluation uses the Actor's `explore(..., deterministic=True)` path.

## Fixed five-impact checkpoint evaluation

Every FlightLxx checkpoint automatically runs the immutable protocol in
`source/flightlxx_isaaclab/flightlxx_isaaclab/config/fixed_five_impacts.json`.
It starts at nominal hover with no observation noise, no domain randomization,
no exploration noise, and no curriculum/normalizer/optimizer/replay mutation.
The five fixed body-frame, off-centre impacts occur at 3, 7, 11, 15, and 19 s
of a 23 s rollout. Each requires position `<0.10 m`, speed `<0.10 m/s`,
attitude `<0.0523598776 rad`, angular speed `<0.10 rad/s` for 1.0 s; any arena,
ground, ceiling, NaN, or numerical termination fails the checkpoint.

For one timestamped run, outputs are separated as follows:

- `checkpoints/`: model and complete training state;
- `evaluation/step_<step>.json`: protocol, five per-impact outcomes and scalar result;
- `evaluation/timeseries/step_<step>.csv`: environment-0 time series;
- `evaluation/summary.csv`: all checkpoint summaries;
- `reports/round_report.md`, `training_reward.png`, `five_impact_recovery.png`: round overview.

The bounded reproducer is `scripts/remote_fasttd3_smoke.sh`; it verifies two
checkpoint evaluations at steps 20 and 40, artifact creation, deterministic
actions, and checkpoint continuation.

No manual curriculum switch is needed. Online episodes use five persistent bands: 15% no-impact, 15% easy, 25% middle, 35% current and 10% probe. Three consecutive mastery windows promote by 0.05; three consecutive failure windows demote by 0.025, with a floor of 0.10 and a 5,000-step cooldown. Replay uses six mutually exclusive quotas: 10/15/20/35/10% for those five bands plus 10% failure context. It additionally targets at least 5% active-force transitions and 20% early-recovery transitions, falling back explicitly when a class is temporarily empty.

Logged diagnostics include Q1/Q2 gap, distributional support-boundary probability, target range, Actor/Critic gradient norms, measured Replay occupancy and sampled class/phase fractions, curriculum promotions/demotions, broad recovery success, strict precision-hover success and post-impact crash rate. The launcher exposes `REPLAY_BAND_FRACTIONS`, `FAILURE_CONTEXT_FRACTION`, `FAILURE_CONTEXT_STEPS`, `ACTIVE_IMPACT_FRACTION` and `RECOVERY_PHASE_FRACTION`; defaults are `0.10 0.15 0.20 0.35 0.10`, `0.10`, `50`, `0.05` and `0.20`.

## 上实机前全面撞击验证

固定五冲击筛出候选 checkpoint 后，使用独立的验证专用复合四旋翼碰撞体运行版本化准入套件。该套件不修改训练任务、奖励、课程、Replay或网络：

```bash
cd ~/FlightLxx-IsaacLab
CHECKPOINT=/绝对路径/step_XXXXXXXX.pt ./scripts/run_preflight_validation.sh
```

完整套件固定为74个episode和92次撞击，包含规则外力、固定种子随机外力、连续撞击和0.6 kg刚体球的定向物理碰撞。全部参数、准入判据、断点续跑和输出说明见`scripts/README_CN.md`第7节；正式结果必须保留`resolved_manifest.json`、checkpoint SHA-256和逐trial时间序列。
