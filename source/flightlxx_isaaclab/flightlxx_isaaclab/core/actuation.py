"""Vectorized 4S motor, propeller and strategy-authority model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def physx_angular_velocity_limit_deg_s(
    max_body_rate_rps: tuple[float, float, float], *, headroom: float = 2.0
) -> float:
    """Convert CTBR rad/s limits to the deg/s limit expected by PhysX."""
    if headroom < 1.0:
        raise ValueError("headroom must be at least 1.0")
    return math.degrees(max(abs(float(value)) for value in max_body_rate_rps)) * headroom


class ActionDelayBuffer:
    """Per-environment integer control-step delay for the executed CTBR command."""

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        max_delay_steps: int,
        device: torch.device | str,
    ) -> None:
        if num_envs <= 0 or action_dim <= 0 or max_delay_steps < 0:
            raise ValueError("num_envs/action_dim must be positive and max_delay_steps non-negative")
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.max_delay_steps = max_delay_steps
        self.device = torch.device(device)
        self.delay_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._buffer = torch.zeros(
            num_envs, max_delay_steps + 1, action_dim, device=self.device
        )

    def set_delay_steps(self, env_ids: torch.Tensor, delay_steps: torch.Tensor) -> None:
        if delay_steps.shape != (len(env_ids),):
            raise ValueError("delay_steps must match the selected environments")
        self.delay_steps[env_ids] = delay_steps.to(self.device).long().clamp(
            0, self.max_delay_steps
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._buffer.zero_()
        else:
            self._buffer[env_ids] = 0.0

    def step(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != (self.num_envs, self.action_dim):
            raise ValueError("action must match [num_envs, action_dim]")
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
        self._buffer[:, -1] = action
        indices = self.max_delay_steps - self.delay_steps
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self._buffer[env_ids, indices].clone()


@dataclass(frozen=True)
class MotorActuatorCfg:
    dt: float = 0.02
    mass: float = 0.78
    gravity: float = 9.81
    thrust_coefficient_n_per_rpm2: float = 1.78036e-8
    motor_kv_rpm_per_v: float = 2550.0
    battery_voltage_v: float = 16.8
    battery_internal_resistance_ohm: float = 0.0
    max_current_per_motor_a: float = 40.0
    rpm_official_max: float = 30000.0
    policy_rpm_fraction: float = 0.8
    rpm_tau_up_s: float = 0.0625
    rpm_tau_down_s: float = 0.0625
    arm_length: float = 0.125
    yaw_moment_coefficient: float = 0.0125
    max_body_rate: tuple[float, float, float] = (6.2831852, 6.2831852, 3.1415926)
    rate_kp: tuple[float, float, float] = (0.03794, 0.03794, 0.01778)
    torque_limit: tuple[float, float, float] = (0.45, 0.45, 0.18)


class MotorActuator:
    """Map CTBR through an authority governor into four 4S motor RPM states.

    The policy receives only 80 percent of the official maximum RPM for its
    collective authority.  The allocator itself still has the physical ceiling
    available for rate-loop corrections, matching the intended real-flight
    separation between policy authority and motor capability.
    """

    def __init__(self, num_envs: int, device: torch.device | str, cfg: MotorActuatorCfg):
        if cfg.dt <= 0.0 or cfg.mass <= 0.0 or cfg.thrust_coefficient_n_per_rpm2 <= 0.0:
            raise ValueError("dt, mass and thrust coefficient must be positive")
        if cfg.motor_kv_rpm_per_v <= 0.0 or cfg.rpm_official_max <= 0.0:
            raise ValueError("motor KV and official RPM maximum must be positive")
        if not 0.0 < cfg.policy_rpm_fraction <= 1.0:
            raise ValueError("policy_rpm_fraction must lie in (0, 1]")
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.mass = torch.full((num_envs,), cfg.mass, device=self.device)
        self.thrust_coefficient = torch.full(
            (num_envs, 4), cfg.thrust_coefficient_n_per_rpm2, device=self.device
        )
        self.rpm_tau_up_s = torch.full((num_envs,), cfg.rpm_tau_up_s, device=self.device)
        self.rpm_tau_down_s = torch.full((num_envs,), cfg.rpm_tau_down_s, device=self.device)
        self.battery_voltage_v = torch.full((num_envs,), cfg.battery_voltage_v, device=self.device)
        self.battery_internal_resistance_ohm = torch.full(
            (num_envs,), cfg.battery_internal_resistance_ohm, device=self.device
        )
        self.rate_limit = torch.tensor(cfg.max_body_rate, device=self.device)
        self.kp = torch.tensor(cfg.rate_kp, device=self.device)
        self.torque_limit = torch.tensor(cfg.torque_limit, device=self.device)
        self._rpm_official_max = torch.tensor(cfg.rpm_official_max, device=self.device)
        self._policy_rpm_ceiling = torch.tensor(
            cfg.rpm_official_max * cfg.policy_rpm_fraction,
            device=self.device,
        )
        lever = cfg.arm_length / math.sqrt(2.0)
        yaw = cfg.yaw_moment_coefficient
        self.allocation = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [lever, -lever, -lever, lever], [-lever, -lever, lever, lever], [yaw, -yaw, yaw, -yaw]],
            device=self.device,
        )
        self.allocation_inverse = torch.linalg.inv(self.allocation)
        self._pid_limit = torch.tensor((500.0, 500.0, 400.0), device=self.device)
        self._betaflight_mixer = torch.tensor(
            (
                (1.0, -1.0, 1.0),
                (-1.0, -1.0, -1.0),
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
            ),
            device=self.device,
        )
        self.motor_rpm = torch.zeros(num_envs, 4, device=self.device)
        self._last_hardware_rpm_limit = torch.zeros(num_envs, device=self.device)
        self._last_policy_rpm_limit = torch.zeros(num_envs, device=self.device)
        self._last_motor_thrust = torch.zeros(num_envs, 4, device=self.device)
        self._last_governor_scale = torch.ones(num_envs, device=self.device)

    def set_battery_parameters(
        self,
        env_ids: torch.Tensor,
        *,
        voltage_v: torch.Tensor,
        internal_resistance_ohm: torch.Tensor,
    ) -> None:
        self.battery_voltage_v[env_ids] = voltage_v.clamp_min(0.0)
        self.battery_internal_resistance_ohm[env_ids] = internal_resistance_ohm.clamp_min(0.0)

    def set_domain_parameters(
        self,
        env_ids: torch.Tensor,
        *,
        mass: torch.Tensor,
        thrust_scale: torch.Tensor,
        motor_efficiency: torch.Tensor,
        rpm_tau_s: torch.Tensor,
    ) -> None:
        """Apply the per-environment randomization used by the rigid body.

        ``thrust_scale`` represents measured propeller-model uncertainty;
        ``motor_efficiency`` retains individual motor mismatch.
        """
        count = len(env_ids)
        if (
            mass.shape != (count,)
            or thrust_scale.shape != (count,)
            or motor_efficiency.shape != (count, 4)
            or rpm_tau_s.shape != (count,)
        ):
            raise ValueError("domain parameter shapes must match selected environments")
        self.mass[env_ids] = mass.clamp_min(1.0e-6)
        self.thrust_coefficient[env_ids] = (
            self.cfg.thrust_coefficient_n_per_rpm2
            * thrust_scale.clamp_min(1.0e-6)[:, None]
            * motor_efficiency.clamp_min(1.0e-6)
        )
        self.rpm_tau_up_s[env_ids] = rpm_tau_s.clamp_min(0.0)
        self.rpm_tau_down_s[env_ids] = rpm_tau_s.clamp_min(0.0)

    def _limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_fraction = (self.motor_rpm / self.cfg.rpm_official_max).square().clamp(0.0, 1.0)
        total_current = current_fraction.sum(dim=-1) * self.cfg.max_current_per_motor_a
        loaded_voltage = (self.battery_voltage_v - total_current * self.battery_internal_resistance_ohm).clamp_min(0.0)
        electrical_rpm = loaded_voltage * self.cfg.motor_kv_rpm_per_v
        hardware = torch.minimum(electrical_rpm, self._rpm_official_max)
        policy = torch.minimum(hardware, self._policy_rpm_ceiling)
        return hardware, policy

    def reset(self, env_ids: torch.Tensor) -> None:
        hardware, _ = self._limits()
        hover_per_motor = self.mass[env_ids] * self.cfg.gravity / 4.0
        hover_rpm = torch.sqrt(hover_per_motor[:, None] / self.thrust_coefficient[env_ids])
        self.motor_rpm[env_ids] = hover_rpm.minimum(hardware[env_ids, None])
        self._last_hardware_rpm_limit[env_ids] = hardware[env_ids]
        self._last_policy_rpm_limit[env_ids] = torch.minimum(
            hardware[env_ids],
            self._policy_rpm_ceiling,
        )
        self._last_motor_thrust[env_ids] = self.thrust_coefficient[env_ids] * self.motor_rpm[env_ids].square()
        self._last_governor_scale[env_ids] = 1.0

    def step(self, action: torch.Tensor, body_rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action.shape != (self.num_envs, 4) or body_rate.shape != (self.num_envs, 3):
            raise ValueError("action must be [num_envs, 4] and body_rate must be [num_envs, 3]")
        hardware_rpm, policy_rpm = self._limits()
        action = action.clamp(-1.0, 1.0)
        hover_total = self.mass * self.cfg.gravity
        policy_total = (self.thrust_coefficient * policy_rpm[:, None].square()).sum(dim=-1)
        collective = action[:, 0]
        desired_total = torch.where(
            collective <= 0.0,
            hover_total * (collective + 1.0),
            hover_total + collective * (policy_total - hover_total).clamp_min(0.0),
        ).clamp_min(0.0)
        rate_sp = action[:, 1:] * self.rate_limit
        torque = self.kp * (rate_sp - body_rate)
        torque = torch.maximum(torch.minimum(torque, self.torque_limit), -self.torque_limit)
        desired_wrench = torch.cat((desired_total[:, None], torque), dim=-1)
        desired_thrust = desired_wrench @ self.allocation_inverse.T
        max_motor_thrust = self.thrust_coefficient * hardware_rpm[:, None].square()
        desired_thrust = desired_thrust.clamp_min(0.0).minimum(max_motor_thrust)
        desired_rpm = torch.sqrt(desired_thrust / self.cfg.thrust_coefficient_n_per_rpm2)
        spinning_up = desired_rpm >= self.motor_rpm
        tau = torch.where(spinning_up, self.rpm_tau_up_s[:, None], self.rpm_tau_down_s[:, None])
        alpha = torch.where(tau <= 0.0, torch.ones_like(tau), 1.0 - torch.exp(-self.cfg.dt / tau.clamp_min(1.0e-9)))
        self.motor_rpm += alpha * (desired_rpm - self.motor_rpm)
        self.motor_rpm = self.motor_rpm.clamp_min(0.0).minimum(hardware_rpm[:, None])
        motor_thrust = self.thrust_coefficient * self.motor_rpm.square()
        wrench = motor_thrust @ self.allocation.T
        self._last_hardware_rpm_limit.copy_(hardware_rpm)
        self._last_policy_rpm_limit.copy_(policy_rpm)
        self._last_motor_thrust.copy_(motor_thrust)
        self._last_governor_scale.copy_(torch.where(policy_total > 0.0, desired_total / policy_total, torch.ones_like(policy_total)).clamp(max=1.0))
        return wrench[:, :1], wrench[:, 1:]

    def step_betaflight(self, action: torch.Tensor, pid_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply firmware PID sums through the Betaflight quad-X motor mixer.

        ``action[:, 0]`` is first constrained to the policy's 80-percent RPM
        authority.  The mixer then adds its P/I/D correction in full hardware
        output space, so the unavailable policy headroom remains usable by the
        flight-controller inner loop.
        """
        if action.shape != (self.num_envs, 4) or pid_sum.shape != (self.num_envs, 3):
            raise ValueError("action must be [num_envs, 4] and pid_sum must be [num_envs, 3]")
        hardware_rpm, policy_rpm = self._limits()
        collective = action[:, 0].clamp(-1.0, 1.0)
        hover_total = self.mass * self.cfg.gravity
        policy_total = (self.thrust_coefficient * policy_rpm[:, None].square()).sum(dim=-1)
        desired_total = torch.where(
            collective <= 0.0,
            hover_total * (collective + 1.0),
            hover_total + collective * (policy_total - hover_total).clamp_min(0.0),
        ).clamp_min(0.0)
        baseline_output = torch.sqrt(
            desired_total[:, None] / (4.0 * self.thrust_coefficient)
        ) / hardware_rpm[:, None]
        scaled_pid = pid_sum.clamp(-self._pid_limit, self._pid_limit) / 1000.0
        desired_output = (
            baseline_output + scaled_pid @ self._betaflight_mixer.T
        ).clamp(0.0, 1.0)
        desired_rpm = desired_output * hardware_rpm[:, None]
        spinning_up = desired_rpm >= self.motor_rpm
        tau = torch.where(
            spinning_up,
            self.rpm_tau_up_s[:, None],
            self.rpm_tau_down_s[:, None],
        )
        alpha = torch.where(tau <= 0.0, torch.ones_like(tau), 1.0 - torch.exp(-self.cfg.dt / tau.clamp_min(1.0e-9)))
        self.motor_rpm += alpha * (desired_rpm - self.motor_rpm)
        self.motor_rpm = self.motor_rpm.clamp_min(0.0).minimum(hardware_rpm[:, None])
        motor_thrust = self.thrust_coefficient * self.motor_rpm.square()
        wrench = motor_thrust @ self.allocation.T
        self._last_hardware_rpm_limit.copy_(hardware_rpm)
        self._last_policy_rpm_limit.copy_(policy_rpm)
        self._last_motor_thrust.copy_(motor_thrust)
        self._last_governor_scale.copy_(torch.where(policy_total > 0.0, desired_total / policy_total, torch.ones_like(policy_total)).clamp(max=1.0))
        return wrench[:, :1], wrench[:, 1:]

    def diagnostics(self, *, copy: bool = True) -> dict[str, torch.Tensor]:
        values = {
            "hardware_rpm_limit": self._last_hardware_rpm_limit,
            "policy_rpm_limit": self._last_policy_rpm_limit,
            "motor_rpm": self.motor_rpm,
            "motor_thrust_n": self._last_motor_thrust,
            "policy_governor_scale": self._last_governor_scale,
        }
        if copy:
            return {name: value.clone() for name, value in values.items()}
        return values
