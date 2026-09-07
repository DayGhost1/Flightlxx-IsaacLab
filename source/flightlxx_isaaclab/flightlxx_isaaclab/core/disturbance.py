from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImpactSamplingCfg:
    start_s: tuple[float, float] = (0.5, 1.5)
    duration_s: tuple[float, float] = (0.04, 0.25)
    application_xy_m: float = 0.125
    application_z_m: float = 0.03
    min_fraction_of_max: float = 0.25


@dataclass
class ImpactSample:
    start_s: torch.Tensor
    duration_s: torch.Tensor
    application_point_b: torch.Tensor
    delta_velocity_b: torch.Tensor
    force_b: torch.Tensor
    torque_b: torch.Tensor


def classify_impact_phase(
    enabled: torch.Tensor,
    episode_step: torch.Tensor,
    start_step: torch.Tensor,
    end_step: torch.Tensor,
    recovery_window_steps: int | None = None,
) -> torch.Tensor:
    """Classify clean, pre-impact, active, early-recovery, and late-recovery steps."""

    if recovery_window_steps is not None and recovery_window_steps <= 0:
        raise ValueError("recovery_window_steps must be positive")

    enabled = enabled.bool()
    phase = torch.zeros_like(episode_step, dtype=torch.uint8)
    phase[enabled & (episode_step < start_step)] = 1
    phase[enabled & (episode_step >= start_step) & (episode_step < end_step)] = 2
    phase[enabled & (episode_step >= end_step)] = 3
    if recovery_window_steps is not None:
        phase[enabled & (episode_step >= end_step + recovery_window_steps)] = 4
    return phase


def _unit_vectors(num, device, generator):
    vectors = torch.randn(num, 3, device=device, generator=generator)
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def sample_impact_wrench(mass, inertia, difficulty, cfg=None, seed=None) -> ImpactSample:
    cfg = cfg or ImpactSamplingCfg()
    device, num_envs = mass.device, mass.shape[0]
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    def uniform(shape, low, high):
        return torch.empty(shape, device=device).uniform_(low, high, generator=generator)

    start_s = uniform((num_envs,), *cfg.start_s)
    duration_s = uniform((num_envs,), *cfg.duration_s)
    bounded_difficulty = difficulty.clamp(0.0, 1.0)
    point = torch.empty(num_envs, 3, device=device)
    point[:, :2] = uniform((num_envs, 2), -cfg.application_xy_m, cfg.application_xy_m)
    point[:, 2] = uniform((num_envs,), -cfg.application_z_m, cfg.application_z_m)
    point *= bounded_difficulty.unsqueeze(-1)
    max_delta_v = 0.3 + 2.7 * bounded_difficulty
    delta_velocity = _unit_vectors(num_envs, device, generator) * (
        max_delta_v * uniform((num_envs,), cfg.min_fraction_of_max, 1.0)
    ).unsqueeze(-1)
    force = mass.unsqueeze(-1) * delta_velocity / duration_s.unsqueeze(-1)
    torque = torch.linalg.cross(point, force, dim=-1)
    return ImpactSample(start_s, duration_s, point, delta_velocity, force, torque)
