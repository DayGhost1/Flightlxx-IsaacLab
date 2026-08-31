"""Deterministic, simulator-independent fixed-impact evaluation protocol."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


IMPACT_LEVEL_SCALES: Mapping[str, float] = {
    "small": 1.0,
    "medium": 2.0,
    "large": 2.5,
}


def evaluation_horizon_steps(
    total_duration_s: float,
    step_dt: float,
    *,
    margin_steps: int = 2,
) -> int:
    """Return an environment timeout safely beyond the sampled rollout."""

    if total_duration_s <= 0.0:
        raise ValueError("total_duration_s must be positive")
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    if margin_steps < 1:
        raise ValueError("margin_steps must be at least one")
    return math.ceil(total_duration_s / step_dt) + margin_steps


@dataclass(frozen=True)
class FixedImpact:
    """One body-frame force applied at an offset from the vehicle CoM."""

    impact_id: str
    trigger_time_s: float
    duration_s: float
    application_point_b: torch.Tensor
    force_b: torch.Tensor

    @property
    def end_time_s(self) -> float:
        return self.trigger_time_s + self.duration_s

    @property
    def equivalent_torque_b(self) -> torch.Tensor:
        return torch.linalg.cross(self.application_point_b, self.force_b)


@dataclass(frozen=True)
class FixedImpactProtocol:
    protocol_id: str
    total_duration_s: float
    recovery_dwell_s: float
    thresholds: Mapping[str, float]
    impacts: tuple[FixedImpact, ...]
    impact_level: str = "small"
    force_scale: float = 1.0


@dataclass
class EvaluationModeState:
    """Explicitly separates deterministic evaluation from training state."""

    active: bool = False
    noisy_observations: bool = True
    domain_randomization: bool = True
    curriculum_updates: bool = True

    def begin_fixed_evaluation(self, protocol: FixedImpactProtocol) -> None:
        del protocol  # The environment owns scheduling; this object owns isolation flags.
        self.active = True
        self.noisy_observations = False
        self.domain_randomization = False
        self.curriculum_updates = False

    def end_fixed_evaluation(self) -> None:
        self.active = False
        self.noisy_observations = True
        self.domain_randomization = True
        self.curriculum_updates = True


def _vector(value: Any, name: str) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=torch.float32)
    if tensor.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values, got {tensor.shape}")
    return tensor


def load_fixed_protocol(path: str | Path) -> FixedImpactProtocol:
    """Load and validate the versioned fixed-impact protocol JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    impacts = tuple(
        FixedImpact(
            impact_id=str(item["impact_id"]),
            trigger_time_s=float(item["trigger_time_s"]),
            duration_s=float(item["duration_s"]),
            application_point_b=_vector(item["application_point_b"], "application_point_b"),
            force_b=_vector(item["force_b"], "force_b"),
        )
        for item in raw["impacts"]
    )
    if len(impacts) != 5:
        raise ValueError(f"fixed evaluation requires exactly five impacts, got {len(impacts)}")
    if any(impact.duration_s <= 0.0 for impact in impacts):
        raise ValueError("all impacts must have positive duration")
    if tuple(sorted(impact.trigger_time_s for impact in impacts)) != tuple(
        impact.trigger_time_s for impact in impacts
    ):
        raise ValueError("impacts must be ordered by trigger_time_s")
    if impacts[-1].end_time_s > float(raw["total_duration_s"]):
        raise ValueError("total_duration_s ends before the final impact")
    thresholds = {key: float(value) for key, value in raw["thresholds"].items()}
    required = {"position_error", "linear_speed", "attitude_error_rad", "angular_speed"}
    if set(thresholds) != required:
        raise ValueError(f"thresholds must be {sorted(required)}")
    return FixedImpactProtocol(
        protocol_id=str(raw["protocol_id"]),
        total_duration_s=float(raw["total_duration_s"]),
        recovery_dwell_s=float(raw["recovery_dwell_s"]),
        thresholds=thresholds,
        impacts=impacts,
    )


def protocol_for_impact_level(
    base_protocol: FixedImpactProtocol,
    impact_level: str,
) -> FixedImpactProtocol:
    """Create one named force tier without changing geometry, timing, or pass limits."""

    if impact_level not in IMPACT_LEVEL_SCALES:
        choices = ", ".join(IMPACT_LEVEL_SCALES)
        raise ValueError(f"unknown impact level {impact_level!r}; choose one of: {choices}")
    force_scale = IMPACT_LEVEL_SCALES[impact_level]
    protocol_id = (
        base_protocol.protocol_id
        if impact_level == "small"
        else f"{base_protocol.protocol_id}_{impact_level}"
    )
    impacts = tuple(
        FixedImpact(
            impact_id=impact.impact_id,
            trigger_time_s=impact.trigger_time_s,
            duration_s=impact.duration_s,
            application_point_b=impact.application_point_b.clone(),
            force_b=impact.force_b * force_scale,
        )
        for impact in base_protocol.impacts
    )
    return FixedImpactProtocol(
        protocol_id=protocol_id,
        total_duration_s=base_protocol.total_duration_s,
        recovery_dwell_s=base_protocol.recovery_dwell_s,
        thresholds=dict(base_protocol.thresholds),
        impacts=impacts,
        impact_level=impact_level,
        force_scale=force_scale,
    )


def protocol_for_single_impact(
    base_protocol: FixedImpactProtocol,
    impact_index: int,
    *,
    settle_s: float = 1.0,
    recovery_s: float = 4.0,
) -> FixedImpactProtocol:
    """Build one reset-isolated trial from a fixed five-impact protocol."""

    if impact_index < 0 or impact_index >= len(base_protocol.impacts):
        raise IndexError(
            f"impact_index must be in [0, {len(base_protocol.impacts) - 1}], got {impact_index}"
        )
    if settle_s < 0.0:
        raise ValueError("settle_s must be non-negative")
    if recovery_s <= 0.0:
        raise ValueError("recovery_s must be positive")

    source = base_protocol.impacts[impact_index]
    impact = FixedImpact(
        impact_id=source.impact_id,
        trigger_time_s=float(settle_s),
        duration_s=source.duration_s,
        application_point_b=source.application_point_b.clone(),
        force_b=source.force_b.clone(),
    )
    return FixedImpactProtocol(
        protocol_id=(
            f"{base_protocol.protocol_id}_independent_{source.impact_id}"
        ),
        total_duration_s=float(settle_s + source.duration_s + recovery_s),
        recovery_dwell_s=base_protocol.recovery_dwell_s,
        thresholds=dict(base_protocol.thresholds),
        impacts=(impact,),
        impact_level=base_protocol.impact_level,
        force_scale=base_protocol.force_scale,
    )


def protocol_to_dict(protocol: FixedImpactProtocol) -> dict[str, Any]:
    """Return JSON-safe protocol data stored alongside every evaluation result."""

    return {
        "protocol_id": protocol.protocol_id,
        "impact_level": protocol.impact_level,
        "force_scale": protocol.force_scale,
        "total_duration_s": protocol.total_duration_s,
        "recovery_dwell_s": protocol.recovery_dwell_s,
        "thresholds": dict(protocol.thresholds),
        "impacts": [
            {
                "impact_id": impact.impact_id,
                "trigger_time_s": impact.trigger_time_s,
                "duration_s": impact.duration_s,
                "application_point_b": impact.application_point_b.tolist(),
                "force_b": impact.force_b.tolist(),
                "equivalent_torque_b": impact.equivalent_torque_b.tolist(),
            }
            for impact in protocol.impacts
        ],
    }


class ImpactRecoveryTracker:
    """Finalize recovery only when stability persists until the next impact."""

    _TIME_TOLERANCE_S = 1.0e-9

    def __init__(self, protocol: FixedImpactProtocol):
        self.protocol = protocol
        self._next_index = 0
        self._safe_from_s: float | None = None
        self.records: list[dict[str, Any]] = []

    @property
    def complete(self) -> bool:
        return len(self.records) == len(self.protocol.impacts)

    def _safe(self, metrics: Mapping[str, float]) -> bool:
        return all(float(metrics[name]) < limit for name, limit in self.protocol.thresholds.items())

    def _record(
        self,
        impact: FixedImpact,
        *,
        recovered: bool,
        failure: bool,
        time_s: float,
        recovery_time_s: float | None = None,
    ) -> dict[str, Any]:
        record = {
            "impact_id": impact.impact_id,
            "trigger_time_s": impact.trigger_time_s,
            "impact_end_time_s": impact.end_time_s,
            "recovered": recovered,
            "failure": failure,
            "recovery_time_s": None if not recovered else recovery_time_s,
        }
        self.records.append(record)
        self._next_index += 1
        self._safe_from_s = None
        return record

    def step(self, time_s: float, metrics: Mapping[str, float], failure: bool) -> list[dict[str, Any]]:
        """Advance the state machine and return any records completed at this step."""

        if self.complete:
            return []
        impact = self.protocol.impacts[self._next_index]
        if failure:
            return [self._record(impact, recovered=False, failure=True, time_s=time_s)]
        boundary_s = (
            self.protocol.impacts[self._next_index + 1].trigger_time_s
            if self._next_index + 1 < len(self.protocol.impacts)
            else self.protocol.total_duration_s
        )
        if time_s + self._TIME_TOLERANCE_S >= boundary_s:
            recovered = (
                self._safe_from_s is not None
                and boundary_s - self._safe_from_s + self._TIME_TOLERANCE_S
                >= self.protocol.recovery_dwell_s
            )
            recovery_time_s = None
            if recovered:
                recovery_time_s = max(
                    0.0,
                    self._safe_from_s + self.protocol.recovery_dwell_s - impact.end_time_s,
                )
            return [
                self._record(
                    impact,
                    recovered=recovered,
                    failure=False,
                    time_s=boundary_s,
                    recovery_time_s=recovery_time_s,
                )
            ]
        if time_s + self._TIME_TOLERANCE_S < impact.end_time_s:
            return []
        if not self._safe(metrics):
            self._safe_from_s = None
            return []
        if self._safe_from_s is None:
            self._safe_from_s = time_s
        return []
