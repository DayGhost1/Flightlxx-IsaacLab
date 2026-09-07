from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RecoveryCriteria:
    """State bounds and continuous dwell required for a recovery event."""

    position_m: float | torch.Tensor
    linear_speed_mps: float | torch.Tensor
    attitude_rad: float | torch.Tensor
    angular_speed_rps: float | torch.Tensor
    dwell_s: float | torch.Tensor


_CURRICULUM_RECOVERY_ROWS = (
    (0.80, 1.00, 35.0, 2.50, 0.20),
    (0.70, 0.90, 30.0, 2.00, 0.25),
    (0.60, 0.80, 25.0, 1.60, 0.30),
    (0.50, 0.65, 20.0, 1.25, 0.35),
    (0.42, 0.55, 16.0, 1.00, 0.40),
    (0.35, 0.45, 13.0, 0.80, 0.50),
    (0.29, 0.36, 10.0, 0.60, 0.60),
    (0.23, 0.28, 8.0, 0.45, 0.70),
    (0.19, 0.21, 6.0, 0.33, 0.80),
    (0.15, 0.15, 5.0, 0.25, 1.00),
)


def curriculum_episode_level(
    a_level: torch.Tensor,
    b_level: torch.Tensor,
    c_level: torch.Tensor,
) -> torch.Tensor:
    """Return the strictest active A/B/C level for each episode."""

    if a_level.shape != b_level.shape or a_level.shape != c_level.shape:
        raise ValueError("A/B/C curriculum level tensors must have matching shapes")
    return torch.maximum(torch.maximum(a_level, b_level), c_level).to(torch.long).clamp_min(1)


def curriculum_recovery_criteria(
    level: torch.Tensor,
    band: torch.Tensor,
    *,
    mastered_band: int = 0,
) -> RecoveryCriteria:
    """Select the progressive recovery set for each curriculum episode.

    A mastered-band episode confirms a level with the next stricter set.  L10
    remains capped at its own set; the separate precision metric retains the
    final 5 cm / 2 degree research target.
    """

    if level.shape != band.shape:
        raise ValueError("curriculum level and band tensors must have matching shapes")
    effective_level = level.to(torch.long).clamp(1, len(_CURRICULUM_RECOVERY_ROWS))
    effective_level = torch.where(
        band.to(torch.long) == mastered_band,
        (effective_level + 1).clamp_max(len(_CURRICULUM_RECOVERY_ROWS)),
        effective_level,
    )
    table = torch.tensor(
        _CURRICULUM_RECOVERY_ROWS,
        device=level.device,
        dtype=torch.float32,
    )
    selected = table[effective_level - 1]
    return RecoveryCriteria(
        position_m=selected[..., 0],
        linear_speed_mps=selected[..., 1],
        attitude_rad=selected[..., 2] * (math.pi / 180.0),
        angular_speed_rps=selected[..., 3],
        dwell_s=selected[..., 4],
    )


def recovery_reached(
    dwell: torch.Tensor,
    criteria: RecoveryCriteria,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare dwell with scalar or per-environment thresholds on a subset."""

    selected_dwell = dwell if indices is None else dwell[indices]
    threshold = criteria.dwell_s
    if isinstance(threshold, torch.Tensor) and indices is not None and threshold.ndim > 0:
        threshold = threshold[indices]
    return selected_dwell >= threshold


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
