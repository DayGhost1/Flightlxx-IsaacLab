from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecoveryCriteria:
    """State bounds and continuous dwell required for a recovery event."""

    position_m: float
    linear_speed_mps: float
    attitude_rad: float
    angular_speed_rps: float
    dwell_s: float


def update_recovery_dwell(
    dwell: torch.Tensor,
    position_error: torch.Tensor,
    linear_speed: torch.Tensor,
    attitude_error: torch.Tensor,
    angular_speed: torch.Tensor,
    eligible: torch.Tensor,
    step_dt: float,
    criteria: RecoveryCriteria,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance dwell inside a recovery set and reset it after any violation."""

    inside = (
        (position_error < criteria.position_m)
        & (linear_speed < criteria.linear_speed_mps)
        & (attitude_error < criteria.attitude_rad)
        & (angular_speed < criteria.angular_speed_rps)
        & eligible.bool()
    )
    next_dwell = torch.where(inside, dwell + step_dt, torch.zeros_like(dwell))
    return next_dwell, next_dwell >= criteria.dwell_s
