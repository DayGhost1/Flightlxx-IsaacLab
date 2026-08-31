# FlightLxx-IsaacLab 训练与评测操作手册

本文只说明日常操作。所有命令都在 4090 主机的项目根目录执行：

```bash
cd /home/lu/FlightLxx-IsaacLab
```

环境路径统一由 `scripts/project_env.sh` 设置，正常情况下不需要每次手动激活 Conda，也不要使用系统 Python 3.8。训练和评测均通过 Isaac Lab 调用 Isaac Sim 自带的 Python 3.10。

## 1. 一键开始训练

正式 250k 基线：

```bash
./scripts/run_train.sh
```

临时改参数时，把变量写在命令前，不需要编辑脚本：

```bash
TOTAL_TIMESTEPS=2000 NUM_ENVS=256 BUFFER_SIZE=1024 BATCH_SIZE=4096 \
  SAVE_INTERVAL=1000 ./scripts/run_train.sh
```

可调参数及当前默认值：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `TOTAL_TIMESTEPS` | 250000 | 一整轮 global steps |
| `NUM_ENVS` | 1024 | GPU 并行 Isaac Lab 环境数 |
| `BUFFER_SIZE` | 3072 | 每环境 replay 长度，总容量约为二者乘积 |
| `BATCH_SIZE` | 24576 | 每次梯度更新的 transition 数 |
| `NUM_UPDATES` | 2 | 每个 global step 的梯度更新次数 |
| `SAVE_INTERVAL` | 25000 | checkpoint 保存及自动五冲击评测间隔 |
| `LOG_INTERVAL` | 100 | 终端、CSV、TensorBoard 记录间隔 |
| `LEARNING_STARTS` | 10 | 开始更新网络前的采样步数 |
| `REPLAY_BAND_FRACTIONS` | `0.10 0.15 0.20 0.35 0.10` | Replay 每批无撞击/简单/中间/当前/探测层的目标比例 |
| `FAILURE_CONTEXT_FRACTION` | 0.10 | 失败前后文的独立目标比例 |
| `FAILURE_CONTEXT_STEPS` | 50 | 失败前最多回溯步数（20 ms 控制周期下为 1 s） |
| `ACTIVE_IMPACT_FRACTION` | 0.05 | 每批中正在受力的稀有 transition 目标比例 |
| `RECOVERY_PHASE_FRACTION` | 0.20 | 每批中撞击后前 2 s 恢复 transition 的目标比例 |
| `OUTPUT_ROOT` | `outputs/training` | 时间戳训练目录根路径 |

额外 FastTD3 命令行参数可以直接附加：

```bash
./scripts/run_train.sh --seed 2 --target-noise-clip 0.1
```

参数只在进程启动时读取；修改脚本或环境变量不会改变已经运行的训练。

每个 run 使用独立时间戳目录，基本结构为：

```text
outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0/<时间_seed>/
├── checkpoints/        # step_00025000.pt 等模型
├── logs/               # progress.csv
├── tensorboard/        # TensorBoard event 文件
├── evaluation/         # 训练器保存后自动生成的固定五冲击 JSON/CSV
└── reports/            # 本轮曲线与汇总
```

`run_train.sh` 还会在 `outputs/training/monitoring/<启动时间>/` 保存 GPU 显存和启动日志。

## 2. 查看训练指标

另开终端启动 TensorBoard：

```bash
cd /home/lu/FlightLxx-IsaacLab
source ./scripts/project_env.sh
$PYTHON_EXECUTABLE -m tensorboard --logdir \
  /home/lu/FlightLxx-IsaacLab/outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0 \
  --host 0.0.0.0 --port 6006
```

主机浏览器打开 `http://127.0.0.1:6006`。常用指标包括 episode return、恢复成功率、精细悬停成功率、撞地率、稳态误差、课程难度、Q loss、Q1/Q2 gap、support 边界概率、Actor/Critic 梯度范数以及 `Replay/*` 分层比例。某些 episode 指标只在环境完成或 reset 后才更新；短时间为 0 不一定表示日志失效，应结合环境终止次数和原始 `logs/progress.csv` 判断。

自动课程不再使用精细悬停标准直接晋级。它根据连续统计窗自动升降级，不需要手动切换：

- **课程恢复标准**：位置 `< 0.15 m`、线速度 `< 0.15 m/s`、姿态 `< 5°`、角速度 `< 0.25 rad/s`，连续 0.5 s；只用于判断是否可以提高难度。
- **精细悬停标准**：位置 `< 0.05 m`、线速度 `< 0.05 m/s`、姿态 `< 2°`、角速度 `< 0.05 rad/s`，连续 2.0 s；只作为最终稳定性的严格训练指标。
- **在线任务组成**：15% 无撞击、15% 简单档 `U(0, 0.3d)`、25% 中间档 `U(0.3d, 0.7d)`、35% 当前档 `U(0.7d, d)`、10% 下一档探测；`d` 是当前课程难度。
- **晋级**：无撞击成功率、当前档成功率/撞击率和探测档成功率连续 3 个统计窗满足门槛，`d += 0.05`。
- **降级**：当前档成功率过低或撞击率过高连续 3 个统计窗成立，`d -= 0.025`；最低为 0.10，改变难度后冷却 5000 global steps。
- **Replay 六类互斥采样**：10% 无撞击、15% 简单、20% 中间、35% 当前、10% 探测、10% 失败前后文；此外保证约 5% 正在施力、20% 撞击后前 2 s 恢复片段。某类不足时回退采样并记录 `Replay/fallback_fraction`。

`Curriculum/difficulty` 允许小步升降；`Curriculum/mastered_difficulty` 只记录曾经满足晋级条件的最高难度。结合 `promotion_count`、`demotion_count`、`last_action`、三个档位成功率和撞击率判断课程是否正常。`Replay/occupancy_*` 是缓冲区真实构成，`Replay/sample_*` 是最近一次 batch 的真实构成，两者不是同一个概念。

终端实时看 GPU：

```bash
watch -n 1 nvidia-smi
```

只看关键数字：

```bash
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader'
```

## 3. 固定五冲击自动评测

训练器每保存一次 checkpoint，就会自动用确定性 Actor 连续完成两种固定五冲击评测：

1. **independent**：每次撞击后 reset，五次撞击互不影响，用于测单次恢复能力；
2. **sequential**：不 reset，按原协议连续承受五次撞击，用于测累积扰动和长期稳定性。

两种模式属于同一次评测调用，共用同一个 `IMPACT_LEVEL`，不提供也不需要 mode 参数。因此训练期间通常不需要另开评测脚本，也不要为看一次效果而停止训练。

协议源文件为：

```text
source/flightlxx_isaaclab/flightlxx_isaaclab/config/fixed_five_impacts.json
```

当前五次冲击如下。作用点和力均在机体系表达；冲量为 `|F| × 持续时间`。按标称质量 0.78 kg 估算的 `Δv` 只表示忽略控制与重力时的瞬时速度变化。

| impact_id | 作用点 (m) | 力向量 (N) | `|F|` (N) | 时长 (s) | 冲量 (N·s) | 理想 `Δv` (m/s) |
|---|---|---|---:|---:|---:|---:|
| `lateral_center` | `[0, 0, 0]` | `[5, 0, 0]` | 5.000 | 0.12 | 0.600 | 0.769 |
| `yaw_offset` | `[0.025, 0, 0]` | `[0, -6, 0]` | 6.000 | 0.10 | 0.600 | 0.769 |
| `diagonal_offset` | `[0, 0.03, 0.015]` | `[-4, 4, 0]` | 5.657 | 0.14 | 0.792 | 1.015 |
| `downward_roll` | `[0, -0.025, 0]` | `[2, 0, -5]` | 5.385 | 0.10 | 0.539 | 0.690 |
| `oblique_final` | `[0.02, 0.02, -0.015]` | `[-7, -3, 1]` | 7.681 | 0.12 | 0.922 | 1.182 |

如何直观理解 5 N：它约等于 0.51 kg 物体静止时的重量。施加在 0.78 kg 无人机上时，单由这股力产生的加速度约为 6.41 m/s²（0.65 g）；持续 0.12 s 的矩形脉冲产生 0.600 N·s 冲量，对应约 0.769 m/s 的理想速度突变。真实球体碰撞通常只有几十毫秒且峰值力远高于平均力，因此论文比较应优先报告冲量、接触时长和状态变化，不能只比较“多少 N”。

“最大能承受多大撞击”不能由一个峰值力直接回答。当前固定回归协议中最大的已测试冲击为 `oblique_final`：7.681 N、0.12 s、0.922 N·s；75,000 checkpoint 已在标称模型上通过全部五次。训练采样器在课程难度 1.0 时的配置上限为 `Δv=3.0 m/s`，对应标称质量下 2.34 N·s；若同时取 0.04 s 最短脉冲，则等效矩形力为 58.5 N。这个 58.5 N 只是扰动生成器的数学上限，不代表当前策略已经通过，更不代表机架、电机或桨叶的实物结构上限。

固定评测要求每次扰动后连续 1.0 s 同时满足：位置误差 `< 0.10 m`、线速度 `< 0.10 m/s`、姿态误差 `< 3°`、角速度 `< 0.10 rad/s`，且没有撞地或越界。这里的“通过”是工程回归测试，不等于论文最终实验结论。

原先笼统写成“训练恢复标准”的内容现在必须拆成三个概念：

- **课程恢复标准**：0.15 m / 0.15 m/s / 5° / 0.25 rad/s、连续 0.5 s；给自动课程提供密度足够的晋级信号。
- **精细悬停标准**：0.05 m / 0.05 m/s / 2° / 0.05 rad/s、连续 2.0 s；它约束策略真正稳定悬停，但不阻塞课程晋级。
- **固定五冲击评测标准**：上面的 0.10 m / 0.10 m/s / 3° / 0.10 rad/s、连续 1.0 s，用来在所有 checkpoint 间做一致比较。

固定五冲击现在提供三个等级。三者严格复用相同的五个方向、作用点、触发时刻、持续时间和通过阈值，只缩放力，因此可以直接比较 checkpoint 的强度裕量：

| `IMPACT_LEVEL` | 力倍率 | 最大力幅值 | 最大单次冲量 | 用途 |
|---|---:|---:|---:|---|
| `small` | 1.0 | 7.681 N | 0.922 N·s | 原始回归基线，内容保持不变 |
| `medium` | 2.0 | 15.362 N | 1.844 N·s | 中等强度泛化测试 |
| `large` | 2.5 | 19.203 N | 2.304 N·s | 大强度上限测试 |

自动 checkpoint 评测仍固定使用 `small`，避免训练期间的历史结果定义发生变化；`medium` 和 `large` 用于手动评测与可视化。

## 4. 手动运行无界面评测

训练完成后，评测最新 run 的最新 checkpoint：

```bash
./scripts/run_evaluate.sh
```

指定 75,000 checkpoint：

```bash
CHECKPOINT_STEP=75000 ./scripts/run_evaluate.sh
```

指定冲击等级（默认 `small`）：

```bash
CHECKPOINT_STEP=75000 IMPACT_LEVEL=medium ./scripts/run_evaluate.sh
CHECKPOINT_STEP=75000 IMPACT_LEVEL=large  ./scripts/run_evaluate.sh
```

指定某轮训练：

```bash
RUN_DIR=/home/lu/FlightLxx-IsaacLab/outputs/training/Isaac-FlightLxx-CTBR-Recovery-Direct-v0/<时间_seed> \
CHECKPOINT_STEP=75000 ./scripts/run_evaluate.sh
```

也可以用 `CHECKPOINT=/绝对路径/step_00075000.pt`。一次命令会自动完成 independent 和 sequential 两种模式；结果默认写到该 run 的 `manual_evaluations/<small|medium|large>/step_00075000/`，包含统一 JSON、带 mode/trial 标记的时间序列 CSV、两组曲线和简短报告。不同强度不会相互覆盖。

脚本默认检查训练进程并拒绝抢占 GPU。确有需要时可显式写 `ALLOW_DURING_TRAINING=1`，但这可能降低训练吞吐、增加 OOM 风险。

## 5. 在 Isaac Sim 中可视化

训练完成后回放最新 checkpoint：

```bash
./scripts/run_visualize.sh
```

回放 75,000 checkpoint，并在完成后自动退出：

```bash
CHECKPOINT_STEP=75000 EXIT_AFTER_RUN=1 ./scripts/run_visualize.sh
```

回放中等或大外力只需加等级变量：

```bash
CHECKPOINT_STEP=75000 IMPACT_LEVEL=medium ./scripts/run_visualize.sh
CHECKPOINT_STEP=75000 IMPACT_LEVEL=large  ./scripts/run_visualize.sh
```

常用变量：

- `SPEED=0.5`：半速回放，便于观察撞击瞬间；默认 `1.0`。
- `WARMUP_UPDATES=240`：慢机器可增加 GUI 预热帧数；默认 `180`。
- `AUTO_START=0`：加载后手动点 `Start / Replay`；默认自动开始。
- `OPEN_REPORT=0`：结束后不自动打开曲线。
- `CHECKPOINT_STEP`、`RUN_DIR`、`CHECKPOINT`：选择模型的方法同无界面评测。
- `IMPACT_LEVEL=small|medium|large`：选择外力等级；默认 `small`。

这是“启动 Isaac Sim 并载入任务”的一键入口。已经打开的普通 Isaac Sim 进程不能可靠地热注入此 Python 任务；保留窗口的推荐方式是使用默认 `EXIT_AFTER_RUN=0`，然后在同一窗口中点击 `Start / Replay` 重播。

## 6. 常见问题

- 找不到 checkpoint：先确认 `RUN_DIR/checkpoints/step_XXXXXXXX.pt` 存在，或显式设置 `RUN_DIR`。
- 提示 checkpoint 正在写入：等待片刻重试；脚本会比较两次文件大小，避免加载半写文件。
- 提示训练正在运行：训练器已自动评测保存点；如非必要不要设置 `ALLOW_DURING_TRAINING=1`。
- GUI 黑屏：先等状态变成 ready，第一次加载扩展和 RTX 渲染通常较慢。
- OOM：下次启动训练时依次降低 `BATCH_SIZE`、`BUFFER_SIZE`、`NUM_ENVS`；不要修改正在运行的进程。
- 评测结果位置：自动结果看 run 下的 `evaluation/` 和 `reports/`；手动结果看 `manual_evaluations/` 或 `manual_visualizations/`。

## 7. 上实机前全面撞击准入验证

固定五冲击用于快速筛选 checkpoint；候选模型准备上实机前，再运行本节的版本化全面验证：

```bash
CHECKPOINT=/绝对路径/step_00100000.pt ./scripts/run_preflight_validation.sh
```

默认执行74个episode、92次撞击：36次规则单撞击、20次固定种子随机单撞击、6组四连撞、12次刚体球碰撞。验证任务使用0.78 kg单刚体复合碰撞体（机身、四机臂、四电机），不会修改训练任务和训练参数。通过标准固定为位置0.10 m、线速度0.10 m/s、姿态3°、角速度0.10 rad/s连续保持1 s；为排除短暂进入恢复区后再次漂移，末段2 s的四项悬停RMS也必须分别小于相同阈值。

先检查完整链路可使用四例烟雾测试：

```bash
CHECKPOINT=/绝对路径/step_00100000.pt SMOKE=1 ./scripts/run_preflight_validation.sh
```

烟雾测试会因为缺少其余70例而输出`NOT_QUALIFIED`，这是正常现象。完整验证中断后，复用原来的输出目录继续：

```bash
CHECKPOINT=/绝对路径/step_00100000.pt \
OUTPUT_DIR=/home/lu/FlightLxx-IsaacLab/outputs/preflight/step_00100000/<时间戳> \
RESUME=1 ./scripts/run_preflight_validation.sh
```

也可只复查指定案例：

```bash
CHECKPOINT=/绝对路径/step_00100000.pt \
TRIAL_IDS="S_T01_large R01 C01 P_B01_3mps" \
./scripts/run_preflight_validation.sh
```

结果目录包含：

```text
resolved_manifest.json        # 这次实际执行的显式固定清单
run_metadata.json             # checkpoint哈希、代码版本、设备
progress.json                 # 已完成/待完成trial，供断点续跑
trials/<group>/<trial_id>/
  ├── result.json             # 原子完成标志、判定和峰值
  └── telemetry.csv           # 20 ms时间序列；球实验含球位置/速度/命中证据
qualification_summary.json    # 机器可读总结果
trial_summary.csv             # 全部trial标量汇总
qualification_report.md       # 中文准入结论和最坏案例
figures/*.pdf                 # 论文可用矢量图
figures/*.png                 # 300 dpi预览图
runner_result.json            # 脚本最终状态
```

只有规则36/36、随机至少19/20且无硬失败、分离四连撞全部恢复、压力四连撞至少2/3最终恢复、球碰撞12/12，并且74项证据完整时才会输出`QUALIFIED`。撞地、撞墙、撞顶、越界、NaN和提前终止均为硬失败。不同 checkpoint 必须使用同一个`preflight_validation_v1.json`，不能为让某个模型通过而临时改清单。
