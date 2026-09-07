# CTBR PPO（Isaac Lab / PyTorch）

`CTBR-PPO` 分支默认使用 RSL-RL 2.3.3 的 PPO。环境仍是
`Isaac-FlightLxx-CTBR-Recovery-Direct-v0`；本次不改变 CTBR 动作、奖励、动力学、
平台参数或域随机化，也不改成直接电机转速动作。

## 实现与参考

参考 [FlightLxx 原仓库](https://github.com/lixiaoxin97/FlightLxx/tree/53eee1327d67cd410b9db7d13b52f6f99ed9c837)，
尤其 `Simulation_Experiments/train.py` 和 `rl/lxx_baselines/ppo/ppo2.py`。
PPO 更新和 GAE 使用上游 RSL-RL，不逐行移植 TensorFlow 实现。

沿用原 PPO 的 gamma=0.99、lambda=0.95、学习率 3e-4、clip=0.2、
entropy coefficient=0.0001、value coefficient=0.5、max gradient norm=0.5，
以及独立的 64×64 tanh actor/critic 主干。保留当前工程的因果 TCN 历史编码：
actor 输入 625 维，critic 输入 645 维；部署只使用 actor。

面向 GPU 并行环境，默认 rollout=128、minibatches=8、epochs=5、num_envs=1024，
区别于旧仓库的 500/10/10。它们是起始参数，不代表已经验证最优。
初始高斯标准差为 0.3，使用 log 标准差参数化。采样动作在环境接口裁剪到 [-1,1]，
PPO 存储原始采样值及其 log probability；确定性评估/导出采用裁剪后的均值。
不再使用 TD3 replay buffer、双 Q、目标网络、EMA actor 或 AR(1) 外加探索噪声。

## 安装

先完成 Isaac Lab 2.1.1 / Isaac Sim 4.5 安装，再在相同 Python 环境安装：

```bash
/home/lu/isaacsim/python.sh -m pip install -r requirements-ppo.txt
```

`scripts/project_env.sh` 可通过环境变量覆盖安装位置，PPO 主流程不需要 FastTD3 目录。
原 FastTD3 专用导出/安装脚本属于历史工具；旧 checkpoint 请在 `isaaclab-fasttd3` 分支使用。

## 训练

```bash
cd /home/lu/FlightLxx-IsaacLab
bash scripts/run_train.sh
```

正式训练保留原平台参数检查。如果目标坐标仍为占位值，只有显式设置
`ALLOW_PLACEHOLDER_TARGET=1` 才允许运行，适用于仿真管线短测。

```bash
ALLOW_PLACEHOLDER_TARGET=1 NUM_ENVS=16 ITERATIONS=2 ROLLOUT_STEPS=16 \
EPOCHS=1 MINIBATCHES=2 EXAM_INTERVAL=1 bash scripts/run_train.sh
```

`ITERATIONS` 是 PPO 更新轮数，不是旧 FastTD3 的 global steps。
每轮环境步数为 `ROLLOUT_STEPS`，transition 数为 `NUM_ENVS * ROLLOUT_STEPS`。
默认每 32 轮（4096 环境步）进行一次原有标准的课程考试，考试数据不加入 PPO rollout，
也不更新观测归一化统计。考试后重置环境，再开始新 rollout。
固定五冲击评估通过下面的独立命令运行，不宣称每个 checkpoint 都自动通过该评测。

输出目录为 `outputs/ppo/<时间戳>_seed<seed>/`：

- TensorBoard 事件、PPO 配置、启动参数和平台配置快照；
- RSL-RL 定期保存的 `model_<iteration>.pt`；
- 每次课程考试/训练结束后的 `checkpoints/step_<环境步数>.pt`；
- `curriculum.jsonl` 记录课程考试指标。

恢复训练：

```bash
RESUME=/absolute/path/to/ppo/checkpoint.pt ITERATIONS=200 bash scripts/run_train.sh
```

恢复 actor、critic、优化器、学习率、归一化、课程状态和计数；使用 checkpoint 中的 PPO
超参数。`ITERATIONS` 表示追加轮数。物理场景从新 episode 开始，不声称逐帧精确续接。
FastTD3 checkpoint 不兼容 PPO 续训。

## 评估、回放与导出

```bash
CHECKPOINT=/absolute/path/to/ppo/checkpoint.pt bash scripts/run_evaluate.sh
CHECKPOINT=/absolute/path/to/ppo/checkpoint.pt bash scripts/run_visualize.sh
CHECKPOINT=/absolute/path/to/ppo/checkpoint.pt bash scripts/run_real_flight_replay.sh

source scripts/project_env.sh
"$PYTHON_EXECUTABLE" scripts/export_ppo_deployment.py \
  --checkpoint /absolute/path/to/ppo/checkpoint.pt --output-dir outputs/ppo-export
```

导出 ONNX、golden vectors、metadata；归一化和动作裁剪嵌入 ONNX。
脚本使用 ONNX Runtime 检查与 PyTorch 的输出误差。接口标识为 `snowyowl3_ppo_v1`。
Jetson 的旧 FastTD3 bundle 加载器可能需要适配该标识和文件名；本次不改写或启用实机控制。

## 回归检查

```bash
PYTHONPATH=source/flightlxx_isaaclab /home/lu/isaacsim/python.sh \
  -m pytest tests/test_ppo_integration.py -q
```

测试包含真实 RSL-RL PPO 更新、历史编码梯度、超时 bootstrap、checkpoint 续训、
续训首帧归一化，以及 ONNX 输出一致性。短测只验证接口和数值链路，不代表策略已收敛。

2026-09-07 主机验证：6 项 PPO 回归测试通过；16 环境、2 轮训练及两次课程考试完成，
恢复后追加 1 轮训练完成；固定冲击评估生成结果；1024 环境、128 步 rollout、5 epochs、
8 minibatches 的默认配置完成 1 轮更新（131072 transitions）。测试 checkpoint 的 ONNX
与 PyTorch 最大绝对误差为 5.59e-9。短测策略固定冲击恢复为 0/5，尚未进行收敛训练。
