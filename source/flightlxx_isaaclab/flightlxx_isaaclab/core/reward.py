"""Minimal continuous recovery objective shared by every curriculum lesson."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecoveryRewardCfg:
    """Physical state cost and one policy smoothness regularizer.

    The terms intentionally use measured units instead of staged success gates:
    a policy is always rewarded for reducing position, attitude, velocity and
    body-rate error, whether it is close to hover or recovering from a large
    takeover error.  The action term is only a temporal-difference cost; it
    never prefers a low-amplitude command and therefore does not compete with
    the collective thrust required to hover.
    """

    position_weight: float = 1.0
    attitude_weight: float = 1.0
    linear_speed_weight: float = 0.10
    angular_speed_weight: float = 0.10
    action_delta_weight: float = 0.05

    def __post_init__(self) -> None:
        if any(
            value < 0.0
            for value in (
                self.position_weight,
                self.attitude_weight,
                self.linear_speed_weight,
                self.angular_speed_weight,
                self.action_delta_weight,
            )
        ):
            raise ValueError("recovery reward weights must be non-negative")


def _same_shape(*values: torch.Tensor) -> None:
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("all state-error tensors must have matching shapes")


def recovery_state_cost(
    *,
    position_error: torch.Tensor,
    linear_speed: torch.Tensor,
    attitude_error_rad: torch.Tensor,
    angular_speed: torch.Tensor,
    cfg: RecoveryRewardCfg | None = None,
) -> torch.Tensor:
    """Return ``E(s)`` in physical units for the recovery objective."""

    _same_shape(position_error, linear_speed, attitude_error_rad, angular_speed)
    cfg = cfg or RecoveryRewardCfg()
    return (
        cfg.position_weight * position_error.clamp_min(0.0)
        + cfg.attitude_weight * attitude_error_rad.clamp_min(0.0)
        + cfg.linear_speed_weight * linear_speed.clamp_min(0.0)
        + cfg.angular_speed_weight * angular_speed.clamp_min(0.0)
    )


def recovery_reward(
    *,
    position_error: torch.Tensor,
    linear_speed: torch.Tensor,
    attitude_error_rad: torch.Tensor,
    angular_speed: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    failure: torch.Tensor,
    step_dt: float,
    failure_penalty: float,
    cfg: RecoveryRewardCfg | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``-dt * (E(s) + 0.05 ||a_t-a_(t-1)||²)`` plus physical failure.

    Time limits are deliberately excluded from ``failure`` by the environment,
    so their value bootstraps in FastTD3.  Only a physical terminal transition
    receives the one-off failure cost.
    """

    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    if actions.ndim != 2 or actions.shape[-1] != 4 or actions.shape != previous_actions.shape:
        raise ValueError("actions and previous_actions must have matching [N, 4] shapes")
    if failure.shape != position_error.shape:
        raise ValueError("failure must have one value per environment")
    cfg = cfg or RecoveryRewardCfg()
    state_cost = recovery_state_cost(
        position_error=position_error,
        linear_speed=linear_speed,
        attitude_error_rad=attitude_error_rad,
        angular_speed=angular_speed,
        cfg=cfg,
    )
    action_delta_cost = cfg.action_delta_weight * (actions - previous_actions).square().sum(dim=-1)
    components = {
        "state_cost": -step_dt * state_cost,
        "action_delta": -step_dt * action_delta_cost,
        "failure": failure.to(dtype=position_error.dtype) * failure_penalty,
    }
    return torch.stack(tuple(components.values())).sum(dim=0), components
