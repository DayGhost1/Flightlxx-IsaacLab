"""Small Torch implementation of the Vicon pose-to-state bridge."""

from __future__ import annotations

import math
import random

import torch


class ViconSampleClock:
    """Monotonic 100 Hz Vicon sampling clock with bounded arrival jitter."""

    def __init__(
        self,
        *,
        nominal_period_s: float = 0.01,
        jitter_s: float = 0.0,
        seed: int = 0,
    ) -> None:
        if nominal_period_s <= 0.0 or jitter_s < 0.0 or jitter_s >= nominal_period_s:
            raise ValueError("sampling period must be positive and jitter must be smaller than it")
        self.nominal_period_s = nominal_period_s
        self.jitter_s = jitter_s
        self._random = random.Random(seed)
        self.next_sample_time_s = 0.0

    def consume_if_due(self, timestamp_s: float) -> bool:
        """Advance once when a Vicon sample is due at ``timestamp_s``."""
        if timestamp_s + 1.0e-12 < self.next_sample_time_s:
            return False
        jitter = self._random.uniform(-self.jitter_s, self.jitter_s)
        self.next_sample_time_s += self.nominal_period_s + jitter
        return True


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[:, 1:] *= -1.0
    return result


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry,
         lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw),
        dim=-1,
    )


def _angular_velocity(previous: torch.Tensor, current: torch.Tensor, dt_s: float) -> torch.Tensor:
    if dt_s <= 0.0:
        return torch.zeros_like(current[:, 1:])
    delta = _quat_mul(_quat_conjugate(previous), current)
    delta = torch.where(delta[:, :1] < 0.0, -delta, delta)
    vector = delta[:, 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, delta[:, :1].clamp_min(1.0e-12))
    axis = vector / vector_norm.clamp_min(1.0e-12)
    return axis * (angle / dt_s)


class VirtualViconBridge:
    """Sample-and-hold Vicon bridge with timestamp-based 60 ms derivatives."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        output_hz: float = 50.0,
        angular_window_s: float = 0.06,
        measurement_delay_s: float = 0.0,
    ):
        if output_hz <= 0.0 or angular_window_s <= 0.0 or measurement_delay_s < 0.0:
            raise ValueError("output_hz and angular_window_s must be positive and measurement_delay_s non-negative")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.output_period_s = 1.0 / output_hz
        self.angular_window_s = angular_window_s
        self.measurement_delay_s = measurement_delay_s
        self.position_noise_std_m = torch.zeros(num_envs, device=self.device)
        self.attitude_noise_std_rad = torch.zeros(num_envs, device=self.device)
        self.linear_velocity_noise_std_mps = torch.zeros(num_envs, device=self.device)
        self.angular_velocity_noise_std_radps = torch.zeros(num_envs, device=self.device)
        self.angular_velocity_bias_radps = torch.zeros(num_envs, 3, device=self.device)
        self.dropout_probability = torch.zeros(num_envs, device=self.device)
        self._samples: list[tuple[float, torch.Tensor, torch.Tensor]] = []
        self._valid_after_time = torch.full((num_envs,), float("-inf"), device=self.device)
        self._reset_pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._last_output_time = torch.full((num_envs,), float("-inf"), device=self.device)
        self._last_output: torch.Tensor | None = None

    def set_measurement_noise(
        self,
        position_noise_std_m: torch.Tensor,
        attitude_noise_std_rad: torch.Tensor,
    ) -> None:
        """Set independent per-environment Vicon position and attitude noise."""
        expected = (self.num_envs,)
        if position_noise_std_m.shape != expected or attitude_noise_std_rad.shape != expected:
            raise ValueError("measurement noise must have shape [num_envs]")
        self.position_noise_std_m.copy_(position_noise_std_m.to(self.device).clamp_min(0.0))
        self.attitude_noise_std_rad.copy_(attitude_noise_std_rad.to(self.device).clamp_min(0.0))

    def set_dropout_probability(self, probability: torch.Tensor) -> None:
        if probability.shape != (self.num_envs,):
            raise ValueError("dropout probability must have shape [num_envs]")
        self.dropout_probability.copy_(probability.to(self.device).clamp(0.0, 1.0))

    def set_derived_state_noise(
        self,
        *,
        linear_velocity_noise_std_mps: torch.Tensor,
        angular_velocity_noise_std_radps: torch.Tensor,
        angular_velocity_bias_radps: torch.Tensor,
    ) -> None:
        expected = (self.num_envs,)
        if (
            linear_velocity_noise_std_mps.shape != expected
            or angular_velocity_noise_std_radps.shape != expected
            or angular_velocity_bias_radps.shape != (self.num_envs, 3)
        ):
            raise ValueError("derived-state noise must match the environment batch")
        self.linear_velocity_noise_std_mps.copy_(
            linear_velocity_noise_std_mps.to(self.device).clamp_min(0.0)
        )
        self.angular_velocity_noise_std_radps.copy_(
            angular_velocity_noise_std_radps.to(self.device).clamp_min(0.0)
        )
        self.angular_velocity_bias_radps.copy_(angular_velocity_bias_radps.to(self.device))

    def push(self, position_w: torch.Tensor, orientation_wxyz: torch.Tensor, *, timestamp_s: float) -> None:
        if position_w.shape != (self.num_envs, 3) or orientation_wxyz.shape != (self.num_envs, 4):
            raise ValueError("position must be [num_envs, 3] and orientation must be [num_envs, 4]")
        if self._samples and timestamp_s < self._samples[-1][0]:
            raise ValueError("Vicon timestamps must be monotonic")
        self._valid_after_time[self._reset_pending] = float(timestamp_s)
        self._reset_pending.zero_()
        position = position_w.to(self.device).clone()
        position += torch.randn_like(position) * self.position_noise_std_m[:, None]
        quaternion = orientation_wxyz.to(self.device).clone()
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        noise_vector = torch.randn(self.num_envs, 3, device=self.device) * self.attitude_noise_std_rad[:, None]
        noise_angle = noise_vector.norm(dim=-1, keepdim=True)
        noise_quaternion = torch.cat(
            (torch.cos(noise_angle / 2.0), noise_vector * torch.sin(noise_angle / 2.0) / noise_angle.clamp_min(1.0e-12)),
            dim=-1,
        )
        quaternion = _quat_mul(noise_quaternion, quaternion)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        if self._samples:
            dropped = torch.rand(self.num_envs, device=self.device) < self.dropout_probability
            _, previous_position, previous_quaternion = self._samples[-1]
            position[dropped] = previous_position[dropped]
            quaternion[dropped] = previous_quaternion[dropped]
        self._samples.append((float(timestamp_s), position, quaternion))
        oldest_time = float(timestamp_s) - self.measurement_delay_s - 4.0 * self.angular_window_s
        while len(self._samples) > 2 and self._samples[0][0] < oldest_time:
            self._samples.pop(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart only selected environments' derivative windows."""
        if env_ids is None:
            self._samples.clear()
            self._valid_after_time.fill_(float("-inf"))
            self._reset_pending.zero_()
            self._last_output_time.fill_(float("-inf"))
            self._last_output = None
            return
        self._valid_after_time[env_ids] = float("inf")
        self._reset_pending[env_ids] = True
        self._last_output_time[env_ids] = float("-inf")
        if self._last_output is not None:
            self._last_output[env_ids] = 0.0

    def observe(self, *, now_s: float) -> torch.Tensor | None:
        if not self._samples:
            return None
        cutoff_time = now_s - self.measurement_delay_s + 1.0e-9
        times = torch.tensor([sample[0] for sample in self._samples], device=self.device)
        positions = torch.stack([sample[1] for sample in self._samples])
        quaternions = torch.stack([sample[2] for sample in self._samples])
        frame_indices = torch.arange(len(self._samples), device=self.device)[:, None]
        valid = (times[:, None] <= cutoff_time) & (times[:, None] >= self._valid_after_time[None, :])
        latest_indices = torch.where(valid, frame_indices, -torch.ones_like(frame_indices)).max(dim=0).values
        has_measurement = latest_indices >= 0
        selected_indices = latest_indices.clamp_min(0)
        env_indices = torch.arange(self.num_envs, device=self.device)
        latest_position = positions[selected_indices, env_indices]
        latest_quaternion = quaternions[selected_indices, env_indices]
        latest_time = times[selected_indices]
        target_time = latest_time - self.angular_window_s
        earlier_valid = valid & (times[:, None] <= target_time[None, :])
        earlier_indices = torch.where(earlier_valid, frame_indices, -torch.ones_like(frame_indices)).max(dim=0).values
        earlier_indices = torch.where(earlier_indices >= 0, earlier_indices, selected_indices)
        earlier_position = positions[earlier_indices, env_indices]
        earlier_quaternion = quaternions[earlier_indices, env_indices]
        dt_s = (latest_time - times[earlier_indices]).clamp_min(0.0)
        linear_velocity = torch.where(
            (dt_s > 0.0)[:, None],
            (latest_position - earlier_position) / dt_s[:, None].clamp_min(1.0e-12),
            torch.zeros_like(latest_position),
        )
        angular_velocity = _angular_velocity(earlier_quaternion, latest_quaternion, 1.0)
        angular_velocity = angular_velocity / dt_s[:, None].clamp_min(1.0e-12)
        angular_velocity = torch.where((dt_s > 0.0)[:, None], angular_velocity, torch.zeros_like(angular_velocity))
        linear_velocity = linear_velocity + (
            torch.randn_like(linear_velocity) * self.linear_velocity_noise_std_mps[:, None]
        )
        angular_velocity = (
            angular_velocity
            + self.angular_velocity_bias_radps
            + torch.randn_like(angular_velocity) * self.angular_velocity_noise_std_radps[:, None]
        )
        output = torch.cat((latest_position, linear_velocity, latest_quaternion, angular_velocity), dim=-1)
        if self._last_output is None:
            self._last_output = torch.zeros_like(output)
        due = now_s - self._last_output_time >= self.output_period_s
        update = has_measurement & due
        self._last_output[update] = output[update]
        self._last_output_time[update] = float(now_s)
        return self._last_output.clone()
