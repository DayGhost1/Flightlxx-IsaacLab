"""Algorithm-independent evaluation adapter and PPO curriculum exam."""
from __future__ import annotations
import math
import torch
from .core.curriculum import assess_curriculum_exam


class DirectGymPolicyAdapter:
    def __init__(self, env):
        self.envs = env

    def reset(self, random_start_init=False):
        observations, _ = self.envs.reset()
        return observations['policy']

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.envs.step(actions)
        return obs['policy'], reward, terminated | truncated, info


@torch.inference_mode()
def run_curriculum_exam(env, runner):
    raw = env.unwrapped
    seed = raw.begin_curriculum_evaluation()
    limits = {'position_error': 0.15, 'linear_speed': 0.15,
              'attitude_error_rad': math.radians(5.0), 'angular_speed': 0.25}
    device = torch.device(raw.device)
    devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            obs, _ = env.reset()
            groups = {k: v.clone() for k, v in raw.curriculum_evaluation_groups().items()}
            crashed = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
            tail = {key: [] for key in limits}
            steps = raw.max_episode_length
            tail_steps = max(1, round(1.0 / raw.step_dt))
            with torch.inference_mode():
                for i in range(steps):
                    actions = runner.alg.policy.act_inference(runner.obs_normalizer(obs['policy'])).clamp(-1, 1)
                    obs, _, terminated, _, _ = env.step(actions)
                    metrics = raw.evaluation_step_metrics()
                    crashed |= terminated.bool() | metrics['failure'].bool()
                    if i >= steps - tail_steps:
                        for key in tail:
                            tail[key].append(metrics[key].clone())
            assessment = assess_curriculum_exam({k: torch.stack(v) for k, v in tail.items()},
                                                 crashed=crashed, groups=groups, limits=limits)
            return {**raw.complete_curriculum_evaluation(assessment), 'exam_seed': float(seed)}
    finally:
        raw.end_curriculum_evaluation()
