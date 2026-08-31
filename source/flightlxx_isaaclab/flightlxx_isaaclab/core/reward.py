from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HoverRewardCfg:
    """Weights and physical scales for the unified hover reward."""

    position_weight: float = 1.0
    linear_speed_weight: float = 0.6
    attitude_weight: float = 1.0
    angular_speed_weight: float = 1.0
    position_scale_m: float = 0.15
    linear_speed_scale_mps: float = 0.15
    attitude_scale_rad: float = 0.0872664626  # 5 deg
    angular_speed_scale_rps: float = 0.25
    collective_action_magnitude_weight: float = 0.002
    body_rate_action_magnitude_weight: float = 0.015
    collective_action_rate_weight: float = 0.01
    action_rate_far_weight: float = 0.01
    action_rate_near_weight: float = 0.08
    collective_action_delta_reference: float = 0.20
    body_rate_action_delta_reference: float = 0.10
    motor_saturation_weight: float = 0.10
    loose_recovery_weight: float = 0.5
    precision_recovery_weight: float = 1.0
    recovery_completion_bonus: float = 1.0
    timeout_without_recovery_penalty: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "position_scale_m",
            "linear_speed_scale_mps",
            "attitude_scale_rad",
            "angular_speed_scale_rps",
            "collective_action_delta_reference",
            "body_rate_action_delta_reference",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.action_rate_far_weight < 0.0:
            raise ValueError("action_rate_far_weight must be non-negative")
        if self.action_rate_near_weight < self.action_rate_far_weight:
            raise ValueError("action_rate_near_weight must not be smaller than the far weight")


def unified_hover_reward(
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
    cfg: HoverRewardCfg | None = None,
    motor_saturation_fraction: torch.Tensor | None = None,
    loose_inside: torch.Tensor | None = None,
    precision_inside: torch.Tensor | None = None,
    recovery_completed: torch.Tensor | None = None,
    timed_out_without_recovery: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a single-scale dense hover reward and its additive components.

    The normalized reciprocal terms retain useful signal at large errors while
    keeping the unique maximum at the exact hover equilibrium.  The CTBR
    action mapping is centered at nominal hover, so ``actions == 0`` is the
    action regularization target.
    """

    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    if actions.shape != previous_actions.shape:
        raise ValueError("actions and previous_actions must have matching shapes")
    cfg = cfg or HoverRewardCfg()
    if motor_saturation_fraction is None:
        motor_saturation_fraction = torch.zeros_like(position_error)
    if motor_saturation_fraction.shape != position_error.shape:
        raise ValueError("motor_saturation_fraction must have shape [num_envs]")
    zero_flags = torch.zeros_like(position_error, dtype=torch.bool)
    loose_inside = zero_flags if loose_inside is None else loose_inside.bool()
    precision_inside = zero_flags if precision_inside is None else precision_inside.bool()
    recovery_completed = zero_flags if recovery_completed is None else recovery_completed.bool()
    timed_out_without_recovery = (
        zero_flags if timed_out_without_recovery is None else timed_out_without_recovery.bool()
    )
    for name, flag in (
        ("loose_inside", loose_inside),
        ("precision_inside", precision_inside),
        ("recovery_completed", recovery_completed),
        ("timed_out_without_recovery", timed_out_without_recovery),
    ):
        if flag.shape != position_error.shape:
            raise ValueError(f"{name} must have shape [num_envs]")
    position_error = position_error.clamp_min(0.0)
    linear_speed = linear_speed.clamp_min(0.0)
    attitude_error_rad = attitude_error_rad.clamp_min(0.0)
    angular_speed = angular_speed.clamp_min(0.0)

    position_score = 1.0 / (1.0 + position_error / cfg.position_scale_m)
    linear_speed_score = 1.0 / (1.0 + linear_speed / cfg.linear_speed_scale_mps)
    attitude_score = 1.0 / (1.0 + attitude_error_rad / cfg.attitude_scale_rad)
    angular_speed_score = 1.0 / (1.0 + angular_speed / cfg.angular_speed_scale_rps)
    hover_gate = position_score * linear_speed_score * attitude_score * angular_speed_score
    action_rate_weight = cfg.action_rate_far_weight + hover_gate * (
        cfg.action_rate_near_weight - cfg.action_rate_far_weight
    )
    action_delta = actions - previous_actions
    collective_action_rate = (
        action_delta[:, 0] / cfg.collective_action_delta_reference
    ).square()
    body_rate_action_rate = (
        action_delta[:, 1:] / cfg.body_rate_action_delta_reference
    ).square().sum(dim=-1)

    components = {
        "position": step_dt * cfg.position_weight * position_score,
        "linear_velocity": step_dt * cfg.linear_speed_weight * linear_speed_score,
        "attitude": step_dt * cfg.attitude_weight * attitude_score,
        "angular_velocity": step_dt * cfg.angular_speed_weight * angular_speed_score,
        "action_magnitude": -step_dt
        * (
            cfg.collective_action_magnitude_weight * actions[:, 0].square()
            + cfg.body_rate_action_magnitude_weight * actions[:, 1:].square().sum(dim=-1)
        ),
        "action_rate": -step_dt
        * (
            cfg.collective_action_rate_weight * collective_action_rate
            + action_rate_weight * body_rate_action_rate
        ),
        "motor_saturation": -step_dt
        * cfg.motor_saturation_weight
        * motor_saturation_fraction.clamp(0.0, 1.0),
        "failure": failure_penalty * failure.to(dtype=position_error.dtype),
        "loose_recovery": step_dt
        * cfg.loose_recovery_weight
        * loose_inside.to(dtype=position_error.dtype),
        "precision_recovery": step_dt
        * cfg.precision_recovery_weight
        * precision_inside.to(dtype=position_error.dtype),
        "recovery_completion": cfg.recovery_completion_bonus
        * recovery_completed.to(dtype=position_error.dtype),
        "timeout_without_recovery": -cfg.timeout_without_recovery_penalty
        * timed_out_without_recovery.to(dtype=position_error.dtype),
    }
    reward = torch.stack(tuple(components.values())).sum(dim=0)
    return reward, components
