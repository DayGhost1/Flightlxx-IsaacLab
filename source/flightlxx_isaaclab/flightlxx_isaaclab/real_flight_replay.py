"""Pure helpers and measured baselines for the 2026-08-29 real-flight replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

import torch


DEPLOYED_50K_SHA256 = "1bde8be47ca1cc36faa2f353f4cf23e3f5a9b733d16c970fec46f2003113918b"
DEFAULT_REPLAY_DURATION_S = 20.0
TARGET_POSITION_VICON_M = (-0.2, -0.75, 1.50)
TARGET_EULER_XYZ_DEG = (0.0, 0.0, 0.0)

_ERROR_SCALES = {
    "position_error": 0.15,
    "linear_speed": 0.15,
    "attitude_error_rad": math.radians(5.0),
    "angular_speed": 0.25,
}
_ORDINARY_RECOVERY = ({
    "position_error": 0.15,
    "linear_speed": 0.15,
    "attitude_error_rad": math.radians(5.0),
    "angular_speed": 0.25,
}, 0.5)
_STRICT_RECOVERY = ({
    "position_error": 0.05,
    "linear_speed": 0.05,
    "attitude_error_rad": math.radians(2.0),
    "angular_speed": 0.05,
}, 2.0)
_SUMMARY_NAMES = {
    "position_error": "position_error_m",
    "linear_speed": "linear_speed_m_s",
    "attitude_error_rad": "attitude_error_rad",
    "angular_speed": "angular_speed_rad_s",
}

REAL_FLIGHT_BASELINE = {
    "duration_s": 0.7705247402,
    "first_normalized_action": (0.2453044951, -0.9047320485, 0.8726417422, -0.1165461689),
    "first_physical_command": (19.4140483745, -5.6845991141, 5.4829697732, -0.3661405881),
    "any_rate_saturation_fraction": 0.70,
    "max_position_error_m": 0.3081997489,
    "max_tilt_deg": 49.4785546767,
    "max_abs_reported_body_rate_rad_s": (10.9696110651, 10.4890743429, 3.7372169643),
}


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    duration_s: float
    observation_delay_steps: int
    initial_position_vicon_m: tuple[float, float, float]
    target_position_vicon_m: tuple[float, float, float]
    target_euler_xyz_deg: tuple[float, float, float]
    initial_linear_velocity_w: tuple[float, float, float]
    initial_body_rate: tuple[float, float, float]
    initial_euler_xyz_deg: tuple[float, float, float]


def delay_steps(delay_s: float, step_dt: float) -> int:
    if delay_s < 0.0:
        raise ValueError("delay_s must be non-negative")
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    return int(round(delay_s / step_dt))


def replay_scenarios(
    step_dt: float,
    duration_s: float = DEFAULT_REPLAY_DURATION_S,
) -> tuple[ReplayScenario, ReplayScenario]:
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    return (
        ReplayScenario(
            name="training_nominal",
            duration_s=duration_s,
            observation_delay_steps=0,
            initial_position_vicon_m=TARGET_POSITION_VICON_M,
            target_position_vicon_m=TARGET_POSITION_VICON_M,
            target_euler_xyz_deg=TARGET_EULER_XYZ_DEG,
            initial_linear_velocity_w=(0.0, 0.0, 0.0),
            initial_body_rate=(0.0, 0.0, 0.0),
            initial_euler_xyz_deg=(0.0, 0.0, 0.0),
        ),
        ReplayScenario(
            name="real_flight_replay",
            duration_s=duration_s,
            observation_delay_steps=0,
            initial_position_vicon_m=(-0.7957784121, -0.6366853643, 0.1050470831),
            target_position_vicon_m=TARGET_POSITION_VICON_M,
            target_euler_xyz_deg=TARGET_EULER_XYZ_DEG,
            initial_linear_velocity_w=(-0.4072348931, 0.2350416887, -0.2090959158),
            initial_body_rate=(0.0144270683, 0.0131436142, -0.0351822994),
            initial_euler_xyz_deg=(0.6415964570, -0.1303676168, -86.9650172753),
        ),
    )


class ObservationDelay:
    """Fixed control-sample delay with initial-state history padding."""

    def __init__(self, delay_steps: int):
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.delay_steps = int(delay_steps)
        self._queue: deque[torch.Tensor] = deque()

    def reset(self, observation: torch.Tensor) -> torch.Tensor:
        self._queue.clear()
        for _ in range(self.delay_steps):
            self._queue.append(observation.detach().clone())
        return observation

    def push(self, observation: torch.Tensor) -> torch.Tensor:
        if self.delay_steps == 0:
            return observation
        self._queue.append(observation.detach().clone())
        return self._queue.popleft()


class ReplayMetrics:
    def __init__(self, *, step_dt: float, tail_window_s: float = 2.0):
        if step_dt <= 0.0 or tail_window_s <= 0.0:
            raise ValueError("step_dt and tail_window_s must be positive")
        self.step_dt = float(step_dt)
        self.tail_window_s = float(tail_window_s)
        self._tail_steps = max(1, math.ceil(tail_window_s / step_dt - 1.0e-12))
        self._actions: list[list[float]] = []
        self._states: list[dict[str, float]] = []
        self._maxima = {
            "position_error": 0.0,
            "attitude_error_rad": 0.0,
            "angular_speed": 0.0,
            "linear_speed": 0.0,
        }
        self._saturated = 0
        self._normalized_error_auc = 0.0
        self._crash = False
        self._elapsed_s = 0.0
        self._ordinary_dwell_s = 0.0
        self._strict_dwell_s = 0.0
        self._ordinary_recovery_time_s: float | None = None
        self._strict_recovery_time_s: float | None = None

    @staticmethod
    def _inside(state: Mapping[str, float], limits: Mapping[str, float]) -> bool:
        return all(float(state[key]) < limit for key, limit in limits.items())

    def _advance_recovery(
        self,
        state: Mapping[str, float],
        *,
        crash: bool,
        limits: Mapping[str, float],
        required_dwell_s: float,
        dwell_attribute: str,
        recovery_attribute: str,
    ) -> None:
        dwell = getattr(self, dwell_attribute)
        dwell = dwell + self.step_dt if not crash and self._inside(state, limits) else 0.0
        setattr(self, dwell_attribute, dwell)
        if getattr(self, recovery_attribute) is None and dwell + 1.0e-12 >= required_dwell_s:
            setattr(self, recovery_attribute, self._elapsed_s)

    def record(self, action: torch.Tensor, state: Mapping[str, float]) -> None:
        values = [float(value) for value in action[0].detach().cpu().tolist()]
        self._actions.append(values)
        sample = {key: float(state[key]) for key in _ERROR_SCALES}
        self._states.append(sample)
        self._elapsed_s += self.step_dt
        crash = bool(state.get("failure", False))
        self._crash = self._crash or crash
        if any(abs(value) >= 0.95 for value in values[1:]):
            self._saturated += 1
        for key in self._maxima:
            self._maxima[key] = max(self._maxima[key], sample[key])
        self._normalized_error_auc += self.step_dt * sum(
            min(sample[key] / scale, 10.0) for key, scale in _ERROR_SCALES.items()
        ) / len(_ERROR_SCALES)
        ordinary_limits, ordinary_dwell_s = _ORDINARY_RECOVERY
        strict_limits, strict_dwell_s = _STRICT_RECOVERY
        self._advance_recovery(
            sample,
            crash=crash,
            limits=ordinary_limits,
            required_dwell_s=ordinary_dwell_s,
            dwell_attribute="_ordinary_dwell_s",
            recovery_attribute="_ordinary_recovery_time_s",
        )
        self._advance_recovery(
            sample,
            crash=crash,
            limits=strict_limits,
            required_dwell_s=strict_dwell_s,
            dwell_attribute="_strict_dwell_s",
            recovery_attribute="_strict_recovery_time_s",
        )

    def summary(self) -> dict[str, object]:
        first = self._actions[0] if self._actions else [float("nan")] * 4
        real_first = REAL_FLIGHT_BASELINE["first_normalized_action"]
        final = self._states[-1] if self._states else {key: float("nan") for key in _ERROR_SCALES}
        tail = self._states[-self._tail_steps :]
        tail_rms = {
            _SUMMARY_NAMES[key]: (
                math.sqrt(sum(sample[key] ** 2 for sample in tail) / len(tail))
                if tail
                else float("nan")
            )
            for key in _ERROR_SCALES
        }
        return {
            "sample_count": len(self._actions),
            "first_normalized_action": first,
            "first_action_minus_real": [value - float(real) for value, real in zip(first, real_first)],
            "any_rate_saturation_fraction": self._saturated / max(1, len(self._actions)),
            "ordinary_recovery_time_s": self._ordinary_recovery_time_s,
            "strict_recovery_time_s": self._strict_recovery_time_s,
            "final_errors": {_SUMMARY_NAMES[key]: final[key] for key in _ERROR_SCALES},
            "tail_2s_rms": tail_rms,
            "normalized_error_auc": self._normalized_error_auc,
            "max_position_error_m": self._maxima["position_error"],
            "max_attitude_error_rad": self._maxima["attitude_error_rad"],
            "max_linear_speed_m_s": self._maxima["linear_speed"],
            "max_angular_speed_rad_s": self._maxima["angular_speed"],
            "crash": self._crash,
        }


def verify_checkpoint(path: str | Path, expected_sha256: str | None = None) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return digest


def reset_replay_control_state(raw_env, env_ids: torch.Tensor) -> None:
    """Reset the control states used by the current Betaflight execution path."""
    raw_env._betaflight.reset(env_ids)
    raw_env._actuator.reset(env_ids)
