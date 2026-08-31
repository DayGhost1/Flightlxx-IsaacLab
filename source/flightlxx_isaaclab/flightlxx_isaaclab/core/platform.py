"""Traceable real-airframe configuration used by the realistic training path."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BetaflightAxisPidCfg:
    """Raw Betaflight PID profile values, deliberately not SI torque gains."""

    p: int
    i: int
    d: int
    d_min: int
    f: int


@dataclass(frozen=True)
class BetaflightProfileCfg:
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
    dterm_lpf1_hz: int = 75
    dterm_lpf2_hz: int = 150
    iterm_relax_cutoff_hz: int = 15
    tpa_rate_percent: int = 65
    tpa_breakpoint_us: int = 1350
    feedforward_smooth_factor: int = 25
    feedforward_jitter_factor: int = 7
    feedforward_boost: int = 15
    feedforward_max_rate_limit: int = 90
    rc_smoothing_enabled: bool = True
    rc_smoothing_auto_factor_rpy: int = 30
    rc_smoothing_feedforward_cutoff_hz: int = 0
    rc_smoothing_setpoint_cutoff_hz: int = 0
    control_link_hz: float = 50.0
    actor_rate_max_dps: tuple[float, float, float] = (360.0, 360.0, 180.0)
    bridge_rate_max_dps: tuple[float, float, float] = (720.0, 720.0, 360.0)
    bridge_yaw_reversed: bool = True


@dataclass(frozen=True)
class ViconBridgeCfg:
    """Measured timing settings for the Vicon-to-policy state path."""

    sample_hz: float
    output_hz: float
    angular_window_s: float
    measurement_age_s: float
    sampling_jitter_s: float
    source: str


@dataclass(frozen=True)
class SnowyOwl3PlatformCfg:
    """Platform facts that must be traceable before a formal run can start."""

    platform_id: str
    battery_cells: int
    target_position_vicon_m: tuple[float, float, float]
    target_rpy_deg: tuple[float, float, float]
    target_is_placeholder: bool
    target_source: str
    propeller_dataset: str
    propeller_thrust_coefficient: float
    motor_model: str
    rpm_official_max: float | None
    rpm_official_max_source: str
    policy_rpm_fraction: float
    vicon: ViconBridgeCfg
    betaflight: BetaflightProfileCfg

    @classmethod
    def from_json(cls, path: str | Path) -> "SnowyOwl3PlatformCfg":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            motor = payload["motor"]
            propeller = payload["propeller"]
            target = payload["target"]
            betaflight = payload["betaflight"]
            raw_axes = betaflight["pid_profile"]
            raw_rates = betaflight["rate_profile"]
        except KeyError as error:
            raise ValueError(f"Platform configuration is missing {error.args[0]!r}") from error

        def axis(name: str) -> BetaflightAxisPidCfg:
            values: dict[str, Any] = raw_axes[name]
            required = ("p", "i", "d", "d_min", "f")
            missing = [key for key in required if key not in values]
            if missing:
                raise ValueError(f"Betaflight PID axis {name!r} is missing {missing!r}")
            return BetaflightAxisPidCfg(**{key: int(values[key]) for key in required})

        def rates(name: str) -> tuple[int, int, int]:
            values = raw_rates[name]
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"Betaflight rate profile {name!r} must have roll/pitch/yaw values")
            return tuple(int(value) for value in values)

        def float_rates(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
            values = betaflight.get(name, default)
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError(f"Betaflight {name!r} must have roll/pitch/yaw values")
            return tuple(float(value) for value in values)

        def vector3(values: Any, name: str) -> tuple[float, float, float]:
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError(f"{name} must contain exactly three values")
            return tuple(float(value) for value in values)

        rpm_value = motor.get("rpm_official_max")
        raw_vicon = payload.get("vicon", {})
        return cls(
            platform_id=str(payload.get("platform_id", "")),
            battery_cells=int(payload.get("battery_cells", 0)),
            target_position_vicon_m=vector3(target.get("position_vicon_m"), "target.position_vicon_m"),
            target_rpy_deg=vector3(target.get("rpy_deg"), "target.rpy_deg"),
            target_is_placeholder=bool(target.get("placeholder", False)),
            target_source=str(target.get("source", "")),
            propeller_dataset=str(propeller.get("dataset", "")),
            propeller_thrust_coefficient=float(propeller.get("thrust_coefficient_n_per_rpm2", 0.0)),
            motor_model=str(motor.get("model", "")),
            rpm_official_max=None if rpm_value is None else float(rpm_value),
            rpm_official_max_source=str(motor.get("rpm_official_max_source", "")),
            policy_rpm_fraction=float(motor.get("policy_rpm_fraction", 0.0)),
            vicon=ViconBridgeCfg(
                sample_hz=float(raw_vicon.get("sample_hz", 0.0)),
                output_hz=float(raw_vicon.get("output_hz", 0.0)),
                angular_window_s=float(raw_vicon.get("angular_window_s", 0.0)),
                measurement_age_s=float(raw_vicon.get("measurement_age_s", -1.0)),
                sampling_jitter_s=float(raw_vicon.get("sampling_jitter_s", -1.0)),
                source=str(raw_vicon.get("source", "")),
            ),
            betaflight=BetaflightProfileCfg(
                firmware_version=str(betaflight.get("firmware_version", "")),
                pid_loop_hz=None if betaflight.get("pid_loop_hz") is None else int(betaflight["pid_loop_hz"]),
                rate_profile_source=str(betaflight.get("rate_profile_source", "")),
                filter_profile_source=str(betaflight.get("filter_profile_source", "")),
                rc_rate=rates("rc_rate"),
                super_rate=rates("super_rate"),
                roll=axis("roll"),
                pitch=axis("pitch"),
                yaw=axis("yaw"),
                d_max_gain=int(betaflight.get("d_max_gain", 0)),
                d_max_advance=int(betaflight.get("d_max_advance", 0)),
                dterm_lpf1_hz=int(betaflight.get("dterm_lpf1_hz", 75)),
                dterm_lpf2_hz=int(betaflight.get("dterm_lpf2_hz", 150)),
                iterm_relax_cutoff_hz=int(betaflight.get("iterm_relax_cutoff_hz", 15)),
                tpa_rate_percent=int(betaflight.get("tpa_rate_percent", 65)),
                tpa_breakpoint_us=int(betaflight.get("tpa_breakpoint_us", 1350)),
                feedforward_smooth_factor=int(betaflight.get("feedforward_smooth_factor", 25)),
                feedforward_jitter_factor=int(betaflight.get("feedforward_jitter_factor", 7)),
                feedforward_boost=int(betaflight.get("feedforward_boost", 15)),
                feedforward_max_rate_limit=int(betaflight.get("feedforward_max_rate_limit", 90)),
                rc_smoothing_enabled=bool(betaflight.get("rc_smoothing_enabled", True)),
                rc_smoothing_auto_factor_rpy=int(betaflight.get("rc_smoothing_auto_factor_rpy", 30)),
                rc_smoothing_feedforward_cutoff_hz=int(
                    betaflight.get("rc_smoothing_feedforward_cutoff_hz", 0)
                ),
                rc_smoothing_setpoint_cutoff_hz=int(
                    betaflight.get("rc_smoothing_setpoint_cutoff_hz", 0)
                ),
                control_link_hz=float(betaflight.get("control_link_hz", 50.0)),
                actor_rate_max_dps=float_rates(
                    "actor_rate_max_dps", (360.0, 360.0, 180.0)
                ),
                bridge_rate_max_dps=float_rates(
                    "bridge_rate_max_dps", (720.0, 720.0, 360.0)
                ),
                bridge_yaw_reversed=bool(betaflight.get("bridge_yaw_reversed", True)),
            ),
        )

    @property
    def hardware_rpm_max(self) -> float:
        if self.rpm_official_max is None:
            raise ValueError("rpm_official_max is required before querying the hardware ceiling")
        return self.rpm_official_max

    @property
    def policy_rpm_max(self) -> float:
        return self.hardware_rpm_max * self.policy_rpm_fraction

    def validate_for_training(self, *, allow_placeholder_target: bool = False) -> None:
        if self.platform_id != "snowyowl3_real_v1":
            raise ValueError("platform_id must be 'snowyowl3_real_v1'")
        if self.battery_cells != 4:
            raise ValueError("battery_cells must be 4 for the real-flight model")
        if self.target_rpy_deg != (0.0, 0.0, 0.0):
            raise ValueError("target Roll, Pitch, and Yaw must all be zero")
        if self.target_is_placeholder and not allow_placeholder_target:
            raise ValueError(
                "measured Vicon target XYZ must replace the placeholder before formal training or flight"
            )
        if self.propeller_thrust_coefficient <= 0.0 or not self.propeller_dataset:
            raise ValueError("measured propeller dataset and positive thrust coefficient are required")
        if not self.motor_model:
            raise ValueError("motor model is required")
        if self.rpm_official_max is None or self.rpm_official_max <= 0.0:
            raise ValueError("rpm_official_max must come from a traceable official source")
        if not self.rpm_official_max_source:
            raise ValueError("rpm_official_max_source is required")
        if not 0.0 < self.policy_rpm_fraction <= 1.0:
            raise ValueError("policy_rpm_fraction must lie in (0, 1]")
        if self.vicon.sample_hz <= 0.0 or self.vicon.output_hz <= 0.0:
            raise ValueError("Vicon sample_hz and output_hz must be positive")
        if self.vicon.angular_window_s <= 0.0 or self.vicon.measurement_age_s < 0.0:
            raise ValueError("Vicon angular_window_s must be positive and measurement_age_s non-negative")
        if self.vicon.sampling_jitter_s < 0.0 or self.vicon.sampling_jitter_s >= 1.0 / self.vicon.sample_hz:
            raise ValueError("Vicon sampling_jitter_s must be non-negative and smaller than the sample period")
        if not self.vicon.source:
            raise ValueError("Vicon source is required")
        if not self.betaflight.firmware_version:
            raise ValueError("Betaflight firmware_version is required")
        if self.betaflight.pid_loop_hz is None or self.betaflight.pid_loop_hz <= 0:
            raise ValueError("Betaflight pid_loop_hz is required")
        if not self.betaflight.rate_profile_source:
            raise ValueError("Betaflight rate_profile_source is required")
        if any(value <= 0 or value >= 100 for value in self.betaflight.rc_rate + self.betaflight.super_rate):
            raise ValueError("Betaflight rc_rate and super_rate must be percentages in (0, 100)")
        if not self.betaflight.filter_profile_source:
            raise ValueError("Betaflight filter_profile_source is required")
        if self.betaflight.control_link_hz <= 0.0:
            raise ValueError("Betaflight control_link_hz must be positive")
        if any(value <= 0.0 for value in self.betaflight.actor_rate_max_dps + self.betaflight.bridge_rate_max_dps):
            raise ValueError("Betaflight actor and bridge rate maxima must be positive")
