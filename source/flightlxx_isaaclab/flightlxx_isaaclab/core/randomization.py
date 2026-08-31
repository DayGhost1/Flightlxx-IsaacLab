from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DomainRandomizationCfg:
    """Selective quadrotor sim-to-real randomization around measured parameters."""

    mass_scale: tuple[float, float] = (0.95, 1.05)
    inertia_scale: tuple[float, float] = (0.90, 1.10)
    com_xy_m: float = 0.005
    com_z_m: float = 0.003
    thrust_scale: tuple[float, float] = (0.90, 1.10)
    motor_scale: tuple[float, float] = (0.97, 1.03)
    actuator_tau_s: tuple[float, float] = (0.04, 0.09)
    action_delay_steps: tuple[int, int] = (0, 2)
    # Calibrated from the 2026-08-29 SnowyOwl3 4S blackbox logs: fitted
    # open-circuit voltages 14.8--16.4 V and load-sag resistance 0.02--0.06 ohm.
    battery_voltage_v: tuple[float, float] = (14.8, 16.4)
    battery_internal_resistance_ohm: tuple[float, float] = (0.02, 0.06)
    position_noise_std_m: tuple[float, float] = (0.003, 0.015)
    velocity_noise_std_mps: tuple[float, float] = (0.01, 0.05)
    attitude_noise_std_rad: tuple[float, float] = (0.00175, 0.00873)
    gyro_noise_std_radps: tuple[float, float] = (0.002, 0.01)
    gyro_bias_radps: tuple[float, float] = (-0.01, 0.01)
    vicon_dropout_probability: tuple[float, float] = (0.0, 0.01)


@dataclass
class DomainParameters:
    mass: torch.Tensor
    inertia: torch.Tensor
    com: torch.Tensor
    thrust_scale: torch.Tensor
    motor_scale: torch.Tensor
    actuator_tau: torch.Tensor
    delay_steps: torch.Tensor
    battery_voltage_v: torch.Tensor
    battery_internal_resistance_ohm: torch.Tensor
    position_noise_std: torch.Tensor
    velocity_noise_std: torch.Tensor
    attitude_noise_std: torch.Tensor
    gyro_noise_std: torch.Tensor
    gyro_bias: torch.Tensor
    vicon_dropout_probability: torch.Tensor


def write_com_offsets(
    com_tensor: torch.Tensor,
    env_ids: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Write xyz COM offsets for either RigidObject or Articulation views.

    Isaac Lab 2.1 returns ``(num_envs, 7)`` from a rigid-object PhysX view and
    ``(num_envs, num_bodies, 7)`` from an articulation view.  Keeping this
    compatibility at the boundary avoids coupling domain randomization to one
    asset class.
    """

    if com_tensor.ndim == 2:
        com_tensor[env_ids, :3] = offsets
    elif com_tensor.ndim == 3:
        com_tensor[env_ids, :, :3] = offsets.unsqueeze(1)
    else:
        raise ValueError(
            f"Expected a 2D or 3D PhysX COM tensor, got shape {tuple(com_tensor.shape)}"
        )
    return com_tensor


def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _uniform(shape, low, high, device, generator):
    return torch.empty(shape, device=device).uniform_(low, high, generator=generator)


def sample_domain_parameters(
    num_envs: int,
    device: torch.device | str,
    nominal_mass: float,
    nominal_inertia: tuple[float, float, float],
    cfg: DomainRandomizationCfg | None = None,
    seed: int | None = None,
) -> DomainParameters:
    cfg = cfg or DomainRandomizationCfg()
    device = torch.device(device)
    generator = _generator(device, seed)
    mass = nominal_mass * _uniform((num_envs,), *cfg.mass_scale, device, generator)
    inertia = torch.tensor(nominal_inertia, device=device).unsqueeze(0)
    inertia = inertia * _uniform((num_envs, 3), *cfg.inertia_scale, device, generator)
    com = torch.empty(num_envs, 3, device=device)
    com[:, :2] = _uniform((num_envs, 2), -cfg.com_xy_m, cfg.com_xy_m, device, generator)
    com[:, 2] = _uniform((num_envs,), -cfg.com_z_m, cfg.com_z_m, device, generator)
    thrust_scale = _uniform((num_envs,), *cfg.thrust_scale, device, generator)
    motor_scale = _uniform((num_envs, 4), *cfg.motor_scale, device, generator)
    actuator_tau = _uniform((num_envs,), *cfg.actuator_tau_s, device, generator)
    delay_steps = torch.randint(
        cfg.action_delay_steps[0], cfg.action_delay_steps[1] + 1, (num_envs,), device=device, generator=generator
    )
    battery_voltage_v = _uniform((num_envs,), *cfg.battery_voltage_v, device, generator)
    battery_internal_resistance_ohm = _uniform(
        (num_envs,), *cfg.battery_internal_resistance_ohm, device, generator
    )
    position_noise_std = _uniform((num_envs,), *cfg.position_noise_std_m, device, generator)
    velocity_noise_std = _uniform((num_envs,), *cfg.velocity_noise_std_mps, device, generator)
    attitude_noise_std = _uniform((num_envs,), *cfg.attitude_noise_std_rad, device, generator)
    gyro_noise_std = _uniform((num_envs,), *cfg.gyro_noise_std_radps, device, generator)
    gyro_bias = _uniform((num_envs, 3), *cfg.gyro_bias_radps, device, generator)
    vicon_dropout_probability = _uniform(
        (num_envs,), *cfg.vicon_dropout_probability, device, generator
    )
    return DomainParameters(
        mass, inertia, com, thrust_scale, motor_scale, actuator_tau, delay_steps,
        battery_voltage_v, battery_internal_resistance_ohm,
        position_noise_std, velocity_noise_std, attitude_noise_std, gyro_noise_std, gyro_bias,
        vicon_dropout_probability,
    )
