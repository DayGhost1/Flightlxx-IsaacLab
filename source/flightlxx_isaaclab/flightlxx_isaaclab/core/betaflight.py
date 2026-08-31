"""Raw Betaflight profile representation for the real-airframe model.

The Betaflight GUI PID numbers are retained as firmware-native values.  This
module deliberately does not reinterpret them as SI torque gains.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .platform import BetaflightAxisPidCfg, BetaflightProfileCfg


@dataclass(frozen=True)
class BetaflightProfile:
    firmware_version: str
    pid_loop_hz: int | None
    rate_profile_source: str
    filter_profile_source: str
    rc_rate: tuple[int, int, int]
    super_rate: tuple[int, int, int]
    roll: BetaflightAxisPidCfg
    pitch: BetaflightAxisPidCfg
    yaw: BetaflightAxisPidCfg
    d_max_gain: int
    d_max_advance: int
    dterm_lpf1_hz: int
    dterm_lpf2_hz: int
    iterm_relax_cutoff_hz: int
    tpa_rate_percent: int
    tpa_breakpoint_us: int
    feedforward_smooth_factor: int
    feedforward_jitter_factor: int
    feedforward_boost: int
    feedforward_max_rate_limit: int
    rc_smoothing_enabled: bool
    rc_smoothing_auto_factor_rpy: int
    rc_smoothing_feedforward_cutoff_hz: int
    rc_smoothing_setpoint_cutoff_hz: int
    control_link_hz: float
    actor_rate_max_dps: tuple[float, float, float]
    bridge_rate_max_dps: tuple[float, float, float]
    bridge_yaw_reversed: bool

    @classmethod
    def from_platform(cls, cfg: BetaflightProfileCfg) -> "BetaflightProfile":
        return cls(
            firmware_version=cfg.firmware_version,
            pid_loop_hz=cfg.pid_loop_hz,
            rate_profile_source=cfg.rate_profile_source,
            filter_profile_source=cfg.filter_profile_source,
            rc_rate=cfg.rc_rate,
            super_rate=cfg.super_rate,
            roll=cfg.roll,
            pitch=cfg.pitch,
            yaw=cfg.yaw,
            d_max_gain=cfg.d_max_gain,
            d_max_advance=cfg.d_max_advance,
            dterm_lpf1_hz=cfg.dterm_lpf1_hz,
            dterm_lpf2_hz=cfg.dterm_lpf2_hz,
            iterm_relax_cutoff_hz=cfg.iterm_relax_cutoff_hz,
            tpa_rate_percent=cfg.tpa_rate_percent,
            tpa_breakpoint_us=cfg.tpa_breakpoint_us,
            feedforward_smooth_factor=cfg.feedforward_smooth_factor,
            feedforward_jitter_factor=cfg.feedforward_jitter_factor,
            feedforward_boost=cfg.feedforward_boost,
            feedforward_max_rate_limit=cfg.feedforward_max_rate_limit,
            rc_smoothing_enabled=cfg.rc_smoothing_enabled,
            rc_smoothing_auto_factor_rpy=cfg.rc_smoothing_auto_factor_rpy,
            rc_smoothing_feedforward_cutoff_hz=cfg.rc_smoothing_feedforward_cutoff_hz,
            rc_smoothing_setpoint_cutoff_hz=cfg.rc_smoothing_setpoint_cutoff_hz,
            control_link_hz=cfg.control_link_hz,
            actor_rate_max_dps=cfg.actor_rate_max_dps,
            bridge_rate_max_dps=cfg.bridge_rate_max_dps,
            bridge_yaw_reversed=cfg.bridge_yaw_reversed,
        )

    def build_rate_loop(self, num_envs: int = 1, device: torch.device | str = "cpu") -> "BetaflightRateLoop":
        if self.pid_loop_hz is None or self.pid_loop_hz <= 0:
            raise ValueError("pid_loop_hz must be measured before building a Betaflight rate loop")
        return BetaflightRateLoop(profile=self, num_envs=num_envs, device=device)

    def action_to_rate_setpoint_dps(self, normalized_body_rate: torch.Tensor) -> torch.Tensor:
        """Map the three normalized CTBR rate commands via Rateprofile 0.

        Betaflight's native rate type uses ``200 * rc_rate * stick`` and
        multiplies it by the super-rate factor ``1/(1-|stick|*super_rate)``.
        CLI values are stored as integer percentages.
        """
        if normalized_body_rate.ndim != 2 or normalized_body_rate.shape[-1] != 3:
            raise ValueError("normalized_body_rate must have shape [num_envs, 3]")
        command = normalized_body_rate.clamp(-1.0, 1.0)
        rc_rate = torch.tensor(self.rc_rate, device=command.device, dtype=command.dtype) / 100.0
        super_rate = torch.tensor(self.super_rate, device=command.device, dtype=command.dtype) / 100.0
        return 200.0 * rc_rate * command / (1.0 - command.abs() * super_rate).clamp_min(1.0e-6)

    def ctbr_action_to_rc_deflection(self, normalized_body_rate: torch.Tensor) -> torch.Tensor:
        """Reproduce the deployed actor -> CTBR -> SBus bridge mapping."""
        if normalized_body_rate.ndim != 2 or normalized_body_rate.shape[-1] != 3:
            raise ValueError("normalized_body_rate must have shape [num_envs, 3]")
        actor_max = torch.tensor(
            self.actor_rate_max_dps, device=normalized_body_rate.device, dtype=normalized_body_rate.dtype
        )
        bridge_max = torch.tensor(
            self.bridge_rate_max_dps, device=normalized_body_rate.device, dtype=normalized_body_rate.dtype
        )
        deflection = normalized_body_rate.clamp(-1.0, 1.0) * actor_max / bridge_max
        if self.bridge_yaw_reversed:
            deflection = deflection.clone()
            deflection[:, 2].neg_()
        return deflection.clamp(-1.0, 1.0)

    def ctbr_action_to_rate_setpoint_dps(self, normalized_body_rate: torch.Tensor) -> torch.Tensor:
        return self.action_to_rate_setpoint_dps(self.ctbr_action_to_rc_deflection(normalized_body_rate))


@dataclass(frozen=True)
class RateLoopAdvance:
    pid_sum: torch.Tensor
    inner_ticks: int


class BetaflightRateLoop:
    """Betaflight 4.3 P/I/D rate loop evaluated on the measured firmware clock."""

    _PTERM_SCALE = 0.032029
    _ITERM_SCALE = 0.244381
    _DTERM_SCALE = 0.000529
    _FEEDFORWARD_SCALE = 0.013754

    def __init__(self, profile: BetaflightProfile, num_envs: int, device: torch.device | str):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if profile.pid_loop_hz is None or profile.pid_loop_hz <= 0:
            raise ValueError("pid_loop_hz must be measured before building a Betaflight rate loop")
        self.profile = profile
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.dt_s = 1.0 / profile.pid_loop_hz
        self.dterm_lpf1_hz = profile.dterm_lpf1_hz
        self.dterm_lpf2_hz = profile.dterm_lpf2_hz
        self.iterm_relax_cutoff_hz = profile.iterm_relax_cutoff_hz
        self.tpa_rate = profile.tpa_rate_percent / 100.0
        self.tpa_breakpoint = (profile.tpa_breakpoint_us - 1000) / 1000.0
        axes = (profile.roll, profile.pitch, profile.yaw)
        self._kp = torch.tensor([axis.p * self._PTERM_SCALE for axis in axes], device=self.device)
        self._ki = torch.tensor([axis.i * self._ITERM_SCALE for axis in axes], device=self.device)
        self._kd = torch.tensor([axis.d * self._DTERM_SCALE for axis in axes], device=self.device)
        self._kf = torch.tensor(
            [axis.f * self._FEEDFORWARD_SCALE / 100.0 for axis in axes], device=self.device
        )
        self._dmin_percent = torch.tensor(
            [axis.d_min / axis.d if 0 < axis.d_min < axis.d else 0.0 for axis in axes],
            device=self.device,
        )
        self._dmin_enabled = self._dmin_percent > 0.0
        self._dmin_gyro_gain = profile.d_max_gain * 0.00008 / 35.0
        self._dmin_setpoint_gain = (
            profile.d_max_gain * 0.00008 * profile.d_max_advance * profile.pid_loop_hz / (100.0 * 35.0)
        )
        self._iterm = torch.zeros(num_envs, 3, device=self.device)
        self._previous_gyro_dps = torch.zeros(num_envs, 3, device=self.device)
        self._dterm_lpf1 = torch.zeros(num_envs, 3, device=self.device)
        self._dterm_lpf2 = torch.zeros(num_envs, 3, device=self.device)
        cutoff_correction = 1.0 / math.sqrt(math.sqrt(2.0) - 1.0)
        self._dmin_range_gain = self.dt_s / (
            self.dt_s + 1.0 / (2.0 * cutoff_correction * math.pi * 85.0)
        )
        self._dmin_lowpass_gain = self.dt_s / (
            self.dt_s + 1.0 / (2.0 * cutoff_correction * math.pi * 35.0)
        )
        self._dmin_range_state1 = torch.zeros(num_envs, 3, device=self.device)
        self._dmin_range_state = torch.zeros(num_envs, 3, device=self.device)
        self._dmin_lowpass_state1 = torch.zeros(num_envs, 3, device=self.device)
        self._dmin_lowpass_state = torch.zeros(num_envs, 3, device=self.device)
        self._previous_setpoint_dps = torch.zeros(num_envs, 3, device=self.device)
        self._setpoint_lpf = torch.zeros(num_envs, 3, device=self.device)
        self._ff_previous_setpoint = torch.zeros(num_envs, 3, device=self.device)
        self._ff_previous_speed = torch.zeros(num_envs, 3, device=self.device)
        self._ff_previous_acceleration = torch.zeros(num_envs, 3, device=self.device)
        self._ff_delta = torch.zeros(num_envs, 3, device=self.device)
        self._ff_previous_rc_command = torch.zeros(num_envs, 3, device=self.device)
        self._ff_duplicate_count = torch.zeros(num_envs, 3, device=self.device, dtype=torch.int64)
        self._rc_setpoint_state1 = torch.zeros(num_envs, 3, device=self.device)
        self._rc_setpoint_state2 = torch.zeros(num_envs, 3, device=self.device)
        self._rc_setpoint_state = torch.zeros(num_envs, 3, device=self.device)
        self._ff_lpf_state1 = torch.zeros(num_envs, 3, device=self.device)
        self._ff_lpf_state2 = torch.zeros(num_envs, 3, device=self.device)
        self._ff_lpf_state = torch.zeros(num_envs, 3, device=self.device)
        self._rc_smoothing_setpoint_gain = self._rc_smoothing_gain(profile.rc_smoothing_setpoint_cutoff_hz)
        self._rc_smoothing_feedforward_gain = self._rc_smoothing_gain(
            profile.rc_smoothing_feedforward_cutoff_hz
        )
        self._feedforward_max_rate = self._max_rate_dps()
        self._tick_fraction = 0.0
        self._last_rate_setpoint_dps = torch.zeros(num_envs, 3, device=self.device)
        self._last_gyro_rate_dps = torch.zeros(num_envs, 3, device=self.device)
        self._last_rate_error_dps = torch.zeros(num_envs, 3, device=self.device)
        self._last_pid_sum = torch.zeros(num_envs, 3, device=self.device)
        self._last_inner_ticks = 0

    def _auto_rc_smoothing_cutoff_hz(self) -> float:
        return float(round(self.profile.control_link_hz * 1.5 / (1.0 + self.profile.rc_smoothing_auto_factor_rpy / 10.0)))

    def _rc_smoothing_gain(self, configured_cutoff_hz: int) -> float:
        if not self.profile.rc_smoothing_enabled:
            return 1.0
        cutoff_hz = float(configured_cutoff_hz) if configured_cutoff_hz > 0 else self._auto_rc_smoothing_cutoff_hz()
        correction = 1.0 / math.sqrt(2.0 ** (1.0 / 3.0) - 1.0)
        return self.dt_s / (self.dt_s + 1.0 / (2.0 * correction * math.pi * cutoff_hz))

    def _max_rate_dps(self) -> torch.Tensor:
        command = torch.ones(1, 3, device=self.device)
        return self.profile.action_to_rate_setpoint_dps(command)[0]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._iterm.zero_()
            self._previous_gyro_dps.zero_()
            self._dterm_lpf1.zero_()
            self._dterm_lpf2.zero_()
            self._dmin_range_state1.zero_()
            self._dmin_range_state.zero_()
            self._dmin_lowpass_state1.zero_()
            self._dmin_lowpass_state.zero_()
            self._previous_setpoint_dps.zero_()
            self._setpoint_lpf.zero_()
            self._ff_previous_setpoint.zero_()
            self._ff_previous_speed.zero_()
            self._ff_previous_acceleration.zero_()
            self._ff_delta.zero_()
            self._ff_previous_rc_command.zero_()
            self._ff_duplicate_count.zero_()
            self._rc_setpoint_state1.zero_()
            self._rc_setpoint_state2.zero_()
            self._rc_setpoint_state.zero_()
            self._ff_lpf_state1.zero_()
            self._ff_lpf_state2.zero_()
            self._ff_lpf_state.zero_()
            self._tick_fraction = 0.0
            self._last_rate_setpoint_dps.zero_()
            self._last_gyro_rate_dps.zero_()
            self._last_rate_error_dps.zero_()
            self._last_pid_sum.zero_()
            self._last_inner_ticks = 0
            return
        self._iterm[env_ids] = 0.0
        self._previous_gyro_dps[env_ids] = 0.0
        self._dterm_lpf1[env_ids] = 0.0
        self._dterm_lpf2[env_ids] = 0.0
        self._dmin_range_state1[env_ids] = 0.0
        self._dmin_range_state[env_ids] = 0.0
        self._dmin_lowpass_state1[env_ids] = 0.0
        self._dmin_lowpass_state[env_ids] = 0.0
        self._previous_setpoint_dps[env_ids] = 0.0
        self._setpoint_lpf[env_ids] = 0.0
        self._ff_previous_setpoint[env_ids] = 0.0
        self._ff_previous_speed[env_ids] = 0.0
        self._ff_previous_acceleration[env_ids] = 0.0
        self._ff_delta[env_ids] = 0.0
        self._ff_previous_rc_command[env_ids] = 0.0
        self._ff_duplicate_count[env_ids] = 0
        self._rc_setpoint_state1[env_ids] = 0.0
        self._rc_setpoint_state2[env_ids] = 0.0
        self._rc_setpoint_state[env_ids] = 0.0
        self._ff_lpf_state1[env_ids] = 0.0
        self._ff_lpf_state2[env_ids] = 0.0
        self._ff_lpf_state[env_ids] = 0.0
        self._last_rate_setpoint_dps[env_ids] = 0.0
        self._last_gyro_rate_dps[env_ids] = 0.0
        self._last_rate_error_dps[env_ids] = 0.0
        self._last_pid_sum[env_ids] = 0.0

    def _infer_rc_deflection(self, rate_setpoint_dps: torch.Tensor) -> torch.Tensor:
        """Inverse of the native Betaflight rate curve for direct unit callers."""
        rc_rate = torch.tensor(self.profile.rc_rate, device=self.device, dtype=rate_setpoint_dps.dtype) / 100.0
        super_rate = torch.tensor(self.profile.super_rate, device=self.device, dtype=rate_setpoint_dps.dtype) / 100.0
        magnitude = rate_setpoint_dps.abs()
        return (rate_setpoint_dps.sign() * magnitude / (200.0 * rc_rate + super_rate * magnitude)).clamp(-1.0, 1.0)

    def _apply_pt3(
        self,
        input_value: torch.Tensor,
        state1: torch.Tensor,
        state2: torch.Tensor,
        state: torch.Tensor,
        gain: float,
    ) -> torch.Tensor:
        state1 += gain * (input_value - state1)
        state2 += gain * (state1 - state2)
        state += gain * (state2 - state)
        return state

    def _update_feedforward(self, raw_setpoint_dps: torch.Tensor, rc_deflection: torch.Tensor) -> None:
        rx_rate = self.profile.control_link_hz
        smooth_factor = 1.0 - self.profile.feedforward_smooth_factor / 100.0
        boost_factor = self.profile.feedforward_boost / 10.0
        rc_command = rc_deflection * 500.0
        rc_delta = rc_command - self._ff_previous_rc_command
        abs_rc_delta = rc_delta.abs()
        speed = (raw_setpoint_dps - self._ff_previous_setpoint) * rx_rate
        previous_speed = self._ff_previous_speed.clone()
        previous_acceleration = self._ff_previous_acceleration.clone()
        moving = abs_rc_delta > 0.0
        duplicate = self._ff_duplicate_count > 0
        speed = torch.where(duplicate & moving, speed / (self._ff_duplicate_count + 1), speed)
        smoothed_speed = previous_speed + smooth_factor * (speed - previous_speed)
        acceleration = (smoothed_speed - previous_speed) * rx_rate * 0.01
        smoothed_acceleration = previous_acceleration + smooth_factor * (acceleration - previous_acceleration)
        jitter = torch.ones_like(abs_rc_delta)
        jitter_mask = (abs_rc_delta > 0.0) & (abs_rc_delta < self.profile.feedforward_jitter_factor)
        if self.profile.feedforward_jitter_factor > 0:
            residual = 1.0 - abs_rc_delta / self.profile.feedforward_jitter_factor
            jitter = torch.where(jitter_mask, 1.0 - residual.square(), jitter)
        near_endpoint = (raw_setpoint_dps.abs() / self._feedforward_max_rate).gt(0.95)
        prevent_kick = near_endpoint & (speed.abs() < 3.0 * previous_speed.abs())
        smoothed_speed = torch.where(prevent_kick, torch.zeros_like(smoothed_speed), smoothed_speed)
        smoothed_acceleration = torch.where(
            prevent_kick, torch.zeros_like(smoothed_acceleration), smoothed_acceleration
        )
        candidate_delta = (smoothed_speed + boost_factor * smoothed_acceleration) * self.dt_s * jitter
        self._ff_delta = torch.where(moving, candidate_delta, self._ff_delta)
        first_duplicate = ~moving & ~duplicate
        repeated_duplicate = ~moving & duplicate
        self._ff_duplicate_count = torch.where(
            moving,
            torch.zeros_like(self._ff_duplicate_count),
            torch.where(
                first_duplicate,
                torch.ones_like(self._ff_duplicate_count),
                torch.minimum(self._ff_duplicate_count + 1, torch.full_like(self._ff_duplicate_count, 2)),
            ),
        )
        self._ff_delta = torch.where(repeated_duplicate, torch.zeros_like(self._ff_delta), self._ff_delta)
        self._ff_previous_speed = torch.where(repeated_duplicate, torch.zeros_like(smoothed_speed), smoothed_speed)
        self._ff_previous_acceleration = torch.where(
            repeated_duplicate, torch.zeros_like(smoothed_acceleration), smoothed_acceleration
        )
        self._ff_previous_setpoint.copy_(raw_setpoint_dps)
        self._ff_previous_rc_command.copy_(rc_command)

    def _feedforward_term(self, current_setpoint_dps: torch.Tensor) -> torch.Tensor:
        f_term = self._kf * self._ff_delta / self.dt_s
        limit = self._feedforward_max_rate * (self.profile.feedforward_max_rate_limit / 100.0)
        same_direction = f_term * current_setpoint_dps > 0.0
        below_limit = current_setpoint_dps.abs() <= limit
        lower = (-limit - current_setpoint_dps) * self._kp
        upper = (limit - current_setpoint_dps) * self._kp
        clamped = torch.maximum(torch.minimum(f_term, upper), lower)
        apply_limit = same_direction & below_limit
        f_term = torch.where(apply_limit, clamped, f_term)
        f_term[:, 2] = self._kf[2] * self._ff_delta[:, 2] / self.dt_s
        return self._apply_pt3(
            f_term,
            self._ff_lpf_state1,
            self._ff_lpf_state2,
            self._ff_lpf_state,
            self._rc_smoothing_feedforward_gain,
        )

    def step(
        self,
        rate_setpoint_dps: torch.Tensor,
        gyro_rate_dps: torch.Tensor,
        *,
        throttle: torch.Tensor | float = 0.0,
        rc_deflection: torch.Tensor | None = None,
        new_rc_frame: bool = True,
    ) -> torch.Tensor:
        """Execute one firmware-timebase PID tick and return raw Betaflight PID sums."""
        expected = (self.num_envs, 3)
        if rate_setpoint_dps.shape != expected or gyro_rate_dps.shape != expected:
            raise ValueError("rate_setpoint_dps and gyro_rate_dps must be [num_envs, 3]")
        if rc_deflection is None:
            rc_deflection = self._infer_rc_deflection(rate_setpoint_dps)
        if rc_deflection.shape != expected:
            raise ValueError("rc_deflection must be [num_envs, 3]")
        if new_rc_frame:
            self._update_feedforward(rate_setpoint_dps, rc_deflection)
        current_setpoint_dps = self._apply_pt3(
            rate_setpoint_dps,
            self._rc_setpoint_state1,
            self._rc_setpoint_state2,
            self._rc_setpoint_state,
            self._rc_smoothing_setpoint_gain,
        )
        error = current_setpoint_dps - gyro_rate_dps
        throttle_tensor = torch.as_tensor(throttle, device=self.device, dtype=rate_setpoint_dps.dtype).reshape(-1)
        if throttle_tensor.numel() == 1:
            throttle_tensor = throttle_tensor.expand(self.num_envs)
        if throttle_tensor.shape != (self.num_envs,):
            raise ValueError("throttle must be scalar or [num_envs]")
        tpa_factor = 1.0 - self.tpa_rate * (
            (throttle_tensor - self.tpa_breakpoint) / (1.0 - self.tpa_breakpoint)
        ).clamp(0.0, 1.0)
        p_term = self._kp * error * tpa_factor[:, None]
        relax_alpha = self.dt_s / (self.dt_s + 1.0 / (2.0 * torch.pi * self.iterm_relax_cutoff_hz))
        self._setpoint_lpf += relax_alpha * (current_setpoint_dps - self._setpoint_lpf)
        iterm_error = error.clone()
        iterm_relax_factor = (1.0 - (current_setpoint_dps - self._setpoint_lpf).abs() / 40.0).clamp_min(0.0)
        iterm_error[:, :2] *= iterm_relax_factor[:, :2]
        self._iterm = (self._iterm + self._ki * self.dt_s * iterm_error).clamp(-400.0, 400.0)
        lpf1_alpha = self.dt_s / (self.dt_s + 1.0 / (2.0 * torch.pi * self.dterm_lpf1_hz))
        lpf2_alpha = self.dt_s / (self.dt_s + 1.0 / (2.0 * torch.pi * self.dterm_lpf2_hz))
        self._dterm_lpf1 += lpf1_alpha * (gyro_rate_dps - self._dterm_lpf1)
        self._dterm_lpf2 += lpf2_alpha * (self._dterm_lpf1 - self._dterm_lpf2)
        derivative = -(self._dterm_lpf2 - self._previous_gyro_dps) / self.dt_s
        setpoint_delta = current_setpoint_dps - self._previous_setpoint_dps
        self._dmin_range_state1 += self._dmin_range_gain * (derivative - self._dmin_range_state1)
        self._dmin_range_state += self._dmin_range_gain * (
            self._dmin_range_state1 - self._dmin_range_state
        )
        dynamic_d = torch.maximum(
            self._dmin_range_state.abs() * self._dmin_gyro_gain,
            setpoint_delta.abs() * self._dmin_setpoint_gain,
        )
        dmin_target = self._dmin_percent + (1.0 - self._dmin_percent) * dynamic_d
        self._dmin_lowpass_state1 += self._dmin_lowpass_gain * (
            dmin_target - self._dmin_lowpass_state1
        )
        self._dmin_lowpass_state += self._dmin_lowpass_gain * (
            self._dmin_lowpass_state1 - self._dmin_lowpass_state
        )
        dmin_factor = torch.where(
            self._dmin_enabled,
            self._dmin_lowpass_state.clamp(max=1.0),
            torch.ones_like(self._dmin_lowpass_state),
        )
        d_term = self._kd * derivative * dmin_factor * tpa_factor[:, None]
        f_term = self._feedforward_term(current_setpoint_dps)
        self._previous_gyro_dps.copy_(self._dterm_lpf2)
        self._previous_setpoint_dps.copy_(current_setpoint_dps)
        pid_sum = p_term + self._iterm + d_term + f_term
        self._last_rate_setpoint_dps.copy_(current_setpoint_dps)
        self._last_gyro_rate_dps.copy_(gyro_rate_dps)
        self._last_rate_error_dps.copy_(error)
        self._last_pid_sum.copy_(pid_sum)
        self._last_inner_ticks = 1
        return pid_sum

    def advance_sample_hold(
        self,
        rate_setpoint_dps: torch.Tensor,
        gyro_rate_dps: torch.Tensor,
        *,
        hold_s: float,
        throttle: torch.Tensor | float = 0.0,
        rc_deflection: torch.Tensor | None = None,
    ) -> RateLoopAdvance:
        if hold_s <= 0.0:
            raise ValueError("hold_s must be positive")
        tick_budget = hold_s / self.dt_s + self._tick_fraction
        inner_ticks = int(tick_budget)
        self._tick_fraction = tick_budget - inner_ticks
        if inner_ticks <= 0:
            raise RuntimeError("policy hold interval is shorter than one PID tick")
        pid_sum = torch.zeros(self.num_envs, 3, device=self.device)
        for tick in range(inner_ticks):
            pid_sum += self.step(
                rate_setpoint_dps,
                gyro_rate_dps,
                throttle=throttle,
                rc_deflection=rc_deflection,
                new_rc_frame=tick == 0,
            )
        average_pid_sum = pid_sum / inner_ticks
        self._last_pid_sum.copy_(average_pid_sum)
        self._last_inner_ticks = inner_ticks
        return RateLoopAdvance(pid_sum=average_pid_sum, inner_ticks=inner_ticks)

    def diagnostics(self) -> dict[str, torch.Tensor | int]:
        return {
            "rate_setpoint_dps": self._last_rate_setpoint_dps.clone(),
            "gyro_rate_dps": self._last_gyro_rate_dps.clone(),
            "rate_error_dps": self._last_rate_error_dps.clone(),
            "pid_sum": self._last_pid_sum.clone(),
            "inner_ticks": self._last_inner_ticks,
        }
