"""Full-state handoff distributions for realistic policy takeover training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class HandoffState:
    position_error_m: torch.Tensor
    linear_velocity_mps: torch.Tensor
    orientation_wxyz: torch.Tensor
    angular_velocity_radps: torch.Tensor


def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    result = torch.Generator(device=device)
    result.manual_seed(seed)
    return result


def _uniform_quaternion(count: int, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    """Shoemake Haar-uniform SO(3) sampler, returned in wxyz convention."""

    u = torch.rand(count, 3, device=device, generator=generator)
    root_a = torch.sqrt(1.0 - u[:, 0])
    root_b = torch.sqrt(u[:, 0])
    phase_a = 2.0 * math.pi * u[:, 1]
    phase_b = 2.0 * math.pi * u[:, 2]
    return torch.stack(
        (
            root_b * torch.cos(phase_b),
            root_a * torch.sin(phase_a),
            root_a * torch.cos(phase_a),
            root_b * torch.sin(phase_b),
        ),
        dim=-1,
    )


def sample_handoff_state(
    count: int,
    device: torch.device | str,
    *,
    difficulty: float | torch.Tensor,
    seed: int | None = None,
) -> HandoffState:
    """Sample valid takeovers, reaching the full requested range at difficulty 1."""

    if count <= 0:
        raise ValueError("count must be positive")
    device = torch.device(device)
    generator = _generator(device, seed)
    scale = torch.as_tensor(difficulty, device=device, dtype=torch.float32).expand(count)
    if not bool(torch.all((scale >= 0.0) & (scale <= 1.0))):
        raise ValueError("difficulty must lie in [0, 1]")
    position = (2.0 * torch.rand(count, 3, device=device, generator=generator) - 1.0) * (2.5 * scale[:, None])
    velocity = (2.0 * torch.rand(count, 3, device=device, generator=generator) - 1.0) * (5.0 * scale[:, None])
    angular_velocity = (2.0 * torch.rand(count, 3, device=device, generator=generator) - 1.0) * (3.0 * math.pi * scale[:, None])
    axis = torch.randn(count, 3, device=device, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    angle = torch.rand(count, device=device, generator=generator) * (math.pi * scale)
    orientation = torch.cat((torch.cos(angle[:, None] / 2.0), axis * torch.sin(angle[:, None] / 2.0)), dim=-1)
    full = scale == 1.0
    if full.any():
        orientation[full] = _uniform_quaternion(int(full.sum()), device, generator)
    return HandoffState(position, velocity, orientation, angular_velocity)
def fixed_target_hover_state(
    default_root_state: torch.Tensor,
    target_position: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a zero-velocity reset state at the configured world target."""
    count = default_root_state.shape[0]
    if default_root_state.ndim != 2 or default_root_state.shape[1] < 13:
        raise ValueError("default_root_state must have shape [num_envs, >=13]")
    if target_position.shape != (count, 3) or target_quat.shape != (count, 4):
        raise ValueError("target_position and target_quat must match the environment batch")
    pose = default_root_state[:, :7].clone()
    pose[:, :3] = target_position
    pose[:, 3:7] = target_quat
    velocity = torch.zeros(count, 6, device=default_root_state.device, dtype=default_root_state.dtype)
    return pose, velocity
