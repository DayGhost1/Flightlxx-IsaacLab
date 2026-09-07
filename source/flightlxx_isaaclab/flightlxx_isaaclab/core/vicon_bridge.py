"""Small Torch implementation of the deployed Vicon pose/twist bridge."""

from __future__ import annotations

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
    result[..., 1:] *= -1.0
    return result


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry,
         lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw),
        dim=-1,
    )


def _rotation_vector_world(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """World-frame rotation vector from body-to-world quaternions (WXYZ).

    Matches the deployed Jetson helper: ``q_delta = q_curr * q_prev^{-1}``.
    """
    dots = (previous * current).sum(dim=-1, keepdim=True)
    current = torch.where(dots < 0.0, -current, current)
    delta = _quat_mul(current, _quat_conjugate(previous))
    delta = torch.where(delta[..., :1] < 0.0, -delta, delta)
    vector = delta[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, delta[..., :1].clamp_min(1.0e-12))
    small = vector_norm < 1.0e-12
    return torch.where(
        small,
        2.0 * vector,
        vector * (angle / vector_norm.clamp_min(1.0e-12)),
    )


class VirtualViconBridge:
    """Sample-and-hold Vicon bridge matching the Jetson observation contract.

    Linear velocity is the VRPN world-frame twist, not a pose finite difference.
    Angular velocity is a 60 ms world-frame least-squares fit on quaternions.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        output_hz: float = 50.0,
        angular_window_s: float = 0.06,
        max_angular_dt_s: float = 0.05,
        measurement_delay_s: float = 0.0,
    ):
        if output_hz <= 0.0 or angular_window_s <= 0.0 or max_angular_dt_s <= 0.0:
            raise ValueError("output_hz, angular_window_s and max_angular_dt_s must be positive")
        if measurement_delay_s < 0.0:
            raise ValueError("measurement_delay_s must be non-negative")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.output_period_s = 1.0 / output_hz
        self.angular_window_s = angular_window_s
        self.max_angular_dt_s = max_angular_dt_s
        self.measurement_delay_s = measurement_delay_s
        self.measurement_age_s = torch.full(
            (num_envs,), float(measurement_delay_s), device=self.device
        )
        self.position_noise_std_m = torch.zeros(num_envs, device=self.device)
        self.attitude_noise_std_rad = torch.zeros(num_envs, device=self.device)
        self.linear_velocity_noise_std_mps = torch.zeros(num_envs, device=self.device)
        self.angular_velocity_noise_std_radps = torch.zeros(num_envs, device=self.device)
        self.angular_velocity_bias_radps = torch.zeros(num_envs, 3, device=self.device)
        self.dropout_probability = torch.zeros(num_envs, device=self.device)
        self._samples: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._valid_after_time = torch.full((num_envs,), float("-inf"), device=self.device)
        self._reset_pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._last_output_time = torch.full((num_envs,), float("-inf"), device=self.device)
        self._last_output: torch.Tensor | None = None

    def set_measurement_age(self, measurement_age_s: torch.Tensor) -> None:
        """Set the timestamp-selection age independently for each environment."""
        if measurement_age_s.shape != (self.num_envs,):
            raise ValueError("measurement age must have shape [num_envs]")
        if torch.any(measurement_age_s < 0.0):
            raise ValueError("measurement age must be non-negative")
        self.measurement_age_s.copy_(measurement_age_s.to(self.device))

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

    def push(
        self,
        position_w: torch.Tensor,
        orientation_wxyz: torch.Tensor,
        *,
        linear_velocity_w: torch.Tensor,
        timestamp_s: float,
    ) -> None:
        if (
            position_w.shape != (self.num_envs, 3)
            or orientation_wxyz.shape != (self.num_envs, 4)
            or linear_velocity_w.shape != (self.num_envs, 3)
        ):
            raise ValueError(
                "position and linear_velocity_w must be [num_envs, 3] and orientation [num_envs, 4]"
            )
        if self._samples and timestamp_s < self._samples[-1][0]:
            raise ValueError("Vicon timestamps must be monotonic")
        self._valid_after_time[self._reset_pending] = float(timestamp_s)
        self._reset_pending.zero_()
        position = position_w.to(self.device).clone()
        position += torch.randn_like(position) * self.position_noise_std_m[:, None]
        linear_velocity = linear_velocity_w.to(self.device).clone()
        linear_velocity += torch.randn_like(linear_velocity) * self.linear_velocity_noise_std_mps[:, None]
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
            _, previous_position, previous_quaternion, previous_linear_velocity = self._samples[-1]
            position[dropped] = previous_position[dropped]
            quaternion[dropped] = previous_quaternion[dropped]
            linear_velocity[dropped] = previous_linear_velocity[dropped]
        self._samples.append((float(timestamp_s), position, quaternion, linear_velocity))
        oldest_time = (
            float(timestamp_s)
            - float(self.measurement_age_s.max().item())
            - 4.0 * self.angular_window_s
        )
        while len(self._samples) > 2 and self._samples[0][0] < oldest_time:
            self._samples.pop(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart only selected environments' angular-velocity windows."""
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
        cutoff_time = now_s - self.measurement_age_s + 1.0e-9
        times = torch.tensor([sample[0] for sample in self._samples], device=self.device)
        positions = torch.stack([sample[1] for sample in self._samples])
        quaternions = torch.stack([sample[2] for sample in self._samples])
        linear_velocities = torch.stack([sample[3] for sample in self._samples])
        frame_indices = torch.arange(len(self._samples), device=self.device)[:, None]
        valid = (times[:, None] <= cutoff_time[None, :]) & (
            times[:, None] >= self._valid_after_time[None, :]
        )
        latest_indices = torch.where(valid, frame_indices, -torch.ones_like(frame_indices)).max(dim=0).values
        has_measurement = latest_indices >= 0
        selected_indices = latest_indices.clamp_min(0)
        env_indices = torch.arange(self.num_envs, device=self.device)
        latest_position = positions[selected_indices, env_indices]
        latest_quaternion = quaternions[selected_indices, env_indices]
        latest_linear_velocity = linear_velocities[selected_indices, env_indices]
        latest_time = times[selected_indices]
        angular_velocity = self._world_angular_velocity_least_squares(
            times=times,
            quaternions=quaternions,
            valid=valid,
            latest_indices=selected_indices,
            latest_time=latest_time,
            window_ready_gate=has_measurement,
        )
        angular_velocity = (
            angular_velocity
            + self.angular_velocity_bias_radps
            + torch.randn_like(angular_velocity) * self.angular_velocity_noise_std_radps[:, None]
        )
        output = torch.cat(
            (latest_position, latest_linear_velocity, latest_quaternion, angular_velocity),
            dim=-1,
        )
        if self._last_output is None:
            self._last_output = torch.zeros_like(output)
        due = now_s - self._last_output_time >= self.output_period_s
        update = has_measurement & due
        self._last_output[update] = output[update]
        self._last_output_time[update] = float(now_s)
        return self._last_output.clone()

    def _world_angular_velocity_least_squares(
        self,
        *,
        times: torch.Tensor,
        quaternions: torch.Tensor,
        valid: torch.Tensor,
        latest_indices: torch.Tensor,
        latest_time: torch.Tensor,
        window_ready_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized copy of deployed ``WorldAngularVelocityEstimator.update``."""
        num_samples = times.shape[0]
        env_indices = torch.arange(self.num_envs, device=self.device)
        target_time = latest_time - self.angular_window_s
        earlier_valid = valid & (times[:, None] <= target_time[None, :] + 1.0e-9)
        frame_indices = torch.arange(num_samples, device=self.device)[:, None]
        earlier_indices = torch.where(
            earlier_valid, frame_indices, -torch.ones_like(frame_indices)
        ).max(dim=0).values
        window_ready = window_ready_gate & (earlier_indices >= 0)
        earlier_indices = earlier_indices.clamp_min(0)
        first_time = times[earlier_indices]
        in_window = (
            valid
            & (times[:, None] + 1.0e-9 >= first_time[None, :])
            & (times[:, None] <= latest_time[None, :] + 1.0e-9)
        )
        if num_samples >= 2:
            adjacent_dt = times[1:] - times[:-1]
            consecutive = in_window[1:] & in_window[:-1]
            gap_too_large = consecutive & (adjacent_dt[:, None] > self.max_angular_dt_s)
            window_ready = window_ready & ~gap_too_large.any(dim=0)
        baseline = quaternions[earlier_indices, env_indices]
        baseline = baseline.unsqueeze(0).expand(num_samples, -1, -1)
        rotation_vectors = _rotation_vector_world(baseline, quaternions)
        mask = in_window & window_ready
        count = mask.sum(dim=0).clamp_min(1).to(dtype=rotation_vectors.dtype)
        times_rel = times[:, None] - first_time[None, :]
        mean_time = (times_rel * mask).sum(dim=0) / count
        mean_vector = (rotation_vectors * mask[:, :, None]).sum(dim=0) / count[:, None]
        centered_time = times_rel - mean_time
        centered_vector = rotation_vectors - mean_vector
        denominator = (centered_time.square() * mask).sum(dim=0)
        numerator = (centered_time[:, :, None] * centered_vector * mask[:, :, None]).sum(dim=0)
        angular_velocity = numerator / denominator.clamp_min(1.0e-18)[:, None]
        ready = window_ready & (denominator > 1.0e-18)
        return torch.where(ready[:, None], angular_velocity, torch.zeros_like(angular_velocity))
