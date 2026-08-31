from __future__ import annotations

import torch


def arena_failure_mask(
    relative_position: torch.Tensor,
    state_is_finite: torch.Tensor,
    *,
    half_extent_xy: float = 12.5,
    height: float = 25.0,
    body_margin: float = 0.125,
    ground_clearance: float = 0.08,
) -> torch.Tensor:
    horizontal_limit = half_extent_xy - body_margin
    ceiling_limit = height - body_margin
    outside = (
        (relative_position[:, 0].abs() >= horizontal_limit)
        | (relative_position[:, 1].abs() >= horizontal_limit)
        | (relative_position[:, 2] <= ground_clearance)
        | (relative_position[:, 2] >= ceiling_limit)
    )
    return outside | ~state_is_finite
