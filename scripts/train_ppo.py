#!/usr/bin/env python3
"""Train the existing CTBR task using the Isaac Lab RSL-RL PPO stack."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs', type=int, default=1024)
parser.add_argument('--iterations', type=int, default=2000, help='Additional PPO iterations, also on resume')
parser.add_argument('--rollout-steps', type=int, default=128)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--minibatches', type=int, default=8)
parser.add_argument('--learning-rate', type=float, default=3e-4)
parser.add_argument('--save-interval', type=int, default=100)
parser.add_argument('--exam-interval', type=int, default=32, help='PPO iterations between curriculum exams; 0 disables')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--resume', type=Path)
parser.add_argument('--output-root', type=Path, default=Path('outputs/ppo'))
parser.add_argument('--allow-placeholder-target', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.iterations, args.rollout_steps, args.epochs, args.minibatches, args.save_interval) <= 0:
    parser.error('Environment, rollout, update and save counts must be positive')
if args.exam_interval < 0 or args.learning_rate <= 0:
    parser.error('Invalid exam interval or learning rate')
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import torch
    import flightlxx_isaaclab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from flightlxx_isaaclab.ppo import PPORunner, PPOObservationCache, default_ppo_config, read_ppo_checkpoint
    from flightlxx_isaaclab.ppo_evaluation import run_curriculum_exam
    from flightlxx_isaaclab.core.platform import SnowyOwl3PlatformCfg

    task = 'Isaac-FlightLxx-CTBR-Recovery-Direct-v0'
    platform_path = Path(__file__).resolve().parents[1] / 'source/flightlxx_isaaclab/flightlxx_isaaclab/config/snowyowl3_real_v1.json'
    SnowyOwl3PlatformCfg.from_json(platform_path).validate_for_training(
        allow_placeholder_target=args.allow_placeholder_target)
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    train_cfg = default_ppo_config()
    train_cfg.update(num_steps_per_env=args.rollout_steps, save_interval=args.save_interval)
    train_cfg['algorithm'].update(num_learning_epochs=args.epochs,
                                  num_mini_batches=args.minibatches, learning_rate=args.learning_rate)
    if args.resume:
        train_cfg = read_ppo_checkpoint(args.resume)['train_cfg']
        print('Resume uses saved PPO hyperparameters, including rollout length and save interval.', flush=True)
    if args.num_envs * train_cfg['num_steps_per_env'] % train_cfg['algorithm']['num_mini_batches']:
        raise ValueError('num_envs * rollout_steps must be divisible by minibatches')
    run_dir = args.output_root.resolve() / f'{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'config.json').write_text(json.dumps(train_cfg, indent=2) + '\n')
    (run_dir / 'platform.json').write_text(platform_path.read_text())
    (run_dir / 'launch.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    env = gym.make(task, cfg=cfg)
    class CTBRPPOWrapper(PPOObservationCache, RslRlVecEnvWrapper):
        pass
    wrapped = CTBRPPOWrapper(env, clip_actions=1.0)
    try:
        env.unwrapped.set_training_discount(train_cfg['algorithm']['gamma'])
        runner = PPORunner(wrapped, train_cfg, str(run_dir), args.device)
        runner.add_git_repo_to_log(__file__)
        if args.resume:
            runner.load(str(args.resume))
            wrapped.reset()  # Start a new rollout using the restored curriculum.
        remaining = args.iterations
        print(f'PPO_RUN_DIR={run_dir}', flush=True)
        while remaining:
            count = min(remaining, args.exam_interval) if args.exam_interval else remaining
            runner.learn(count)
            remaining -= count
            if args.exam_interval:
                runner.eval_mode()
                diagnostics = run_curriculum_exam(env, runner)
                with (run_dir / 'curriculum.jsonl').open('a') as f:
                    f.write(json.dumps({'iteration': runner.current_learning_iteration, **diagnostics}) + '\n')
                for name, value in diagnostics.items():
                    runner.writer.add_scalar('Exam/' + name, value, runner.current_learning_iteration)
                with torch.inference_mode():
                    wrapped.reset()
            runner.save(str(run_dir / 'checkpoints' / f'step_{runner.env_steps:08d}.pt'))
        runner.writer.flush()
        runner.writer.close()
    finally:
        wrapped.close()


if __name__ == '__main__':
    import sys
    import traceback
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        status = 1
    finally:
        sys.stdout.flush()
        try:
            app.close()
        finally:
            if status:
                raise SystemExit(status)
