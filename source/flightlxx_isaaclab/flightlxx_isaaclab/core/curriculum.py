"""Current-difficulty curriculum for hover, handoff and impact recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


HANDOFF_SCENARIO = 0
IMPACT_SCENARIO = 1
HANDOFF_IMPACT_SCENARIO = 2
# Kept as an alias for callers using the previous name.
SINGLE_IMPACT_SCENARIO = IMPACT_SCENARIO
RECOVERY_SCENARIO_NAMES = ("handoff", "impact", "handoff_impact")

MIX_HOVER = 0
MIX_CURRENT = 1
MIX_REHEARSAL = 2
MIX_PROBE = 3


def assess_curriculum_exam(
    tail_metrics: Mapping[str, torch.Tensor],
    *,
    crashed: torch.Tensor,
    groups: Mapping[str, torch.Tensor],
    limits: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Summarize one exam from per-step physical errors in its final window.

    Pass/fail uses per-environment p95 only.  RMS is still logged so brief
    spikes remain visible, but they no longer decide promotion or retreat.
    """

    strict_success = ~crashed.bool()
    coarse_success = ~crashed.bool()
    summaries: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, limit in limits.items():
        values = tail_metrics[name]
        rms = values.square().mean(dim=0).sqrt()
        p95 = torch.quantile(values, 0.95, dim=0)
        strict_failure = p95 > limit
        coarse_failure = p95 > 2.0 * limit
        summaries[name] = (rms, p95, strict_failure, coarse_failure)
        strict_success &= p95 <= limit
        coarse_success &= p95 <= 2.0 * limit

    assessment: dict[str, dict[str, float]] = {}
    for group_name, ids in groups.items():
        metrics = {
            "strict_success_rate": strict_success[ids].float().mean().item(),
            "coarse_success_rate": coarse_success[ids].float().mean().item(),
            "crash_rate": crashed[ids].float().mean().item(),
        }
        for metric_name, (rms, p95, strict_failure, coarse_failure) in summaries.items():
            metrics[f"{metric_name}_rms"] = rms[ids].mean().item()
            metrics[f"{metric_name}_p95"] = p95[ids].mean().item()
            metrics[f"{metric_name}_strict_failure_rate"] = (
                strict_failure[ids].float().mean().item()
            )
            metrics[f"{metric_name}_coarse_failure_rate"] = (
                coarse_failure[ids].float().mean().item()
            )
        assessment[group_name] = metrics
    return assessment


def _allocate_counts(total: int, weights: torch.Tensor) -> torch.Tensor:
    """Round weights to integer counts that sum to ``total``."""

    if total <= 0:
        return torch.zeros(weights.numel(), dtype=torch.long)
    normalized = weights / weights.clamp_min(1.0e-8).sum()
    raw = normalized * float(total)
    counts = raw.floor().to(torch.long)
    remainder = int(total - int(counts.sum().item()))
    if remainder > 0:
        frac = raw - raw.floor()
        _, extra = torch.topk(frac, remainder)
        counts[extra] += 1
    return counts


@dataclass(frozen=True)
class ContinuousCurriculumSample:
    difficulty: torch.Tensor
    scenario_type: torch.Tensor
    hover_anchor: torch.Tensor
    handoff_difficulty: torch.Tensor
    impact_difficulty: torch.Tensor
    impact_enabled: torch.Tensor
    mix_kind: torch.Tensor


class ContinuousRecoveryCurriculum:
    """Adapt one shared difficulty from deterministic current-level exams."""

    STATE_VERSION = 7
    advance_delta = 0.05
    retreat_delta = 0.025
    required_streak = 2
    hover_fraction = 0.05
    current_fraction = 0.65
    rehearsal_fraction = 0.20
    probe_fraction = 0.10
    min_recovery_scenario_fraction = 0.10
    success_thresholds = {
        "hover": 0.80,
        "handoff": 0.80,
        "impact": 0.70,
        "handoff_impact": 0.65,
    }
    final_success_thresholds = {
        "hover": 0.95,
        "handoff": 0.80,
        "impact": 0.75,
        "handoff_impact": 0.60,
    }
    maximum_crash_rate = 0.05
    severe_success_rate = 0.35
    severe_crash_rate = 0.15

    def __init__(
        self,
        *,
        initial_difficulty: float = 0.0,
        retention_floor: float = 0.0,
        seed: int = 0,
    ):
        self.difficulty = float(initial_difficulty)
        self.initial_difficulty = float(initial_difficulty)
        self.retention_floor = float(retention_floor)
        self.highest_mastered_difficulty = 0.0
        self.advance_streak = 0
        self.retreat_streak = 0
        self.evaluations = 0
        self.exams_at_difficulty = 0
        self.last_exam_passed = False
        self.last_exam_severe = False
        self.last_exam_metrics: dict[str, dict[str, float]] = {}
        self.last_scenario_weights = torch.ones(3, dtype=torch.float32) / 3.0
        self.seed = int(seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

    def sample(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        seed: int | None = None,
        fixed_difficulty: float | None = None,
    ) -> ContinuousCurriculumSample:
        device = torch.device(device)
        generator = self._generator(seed)

        if fixed_difficulty is not None:
            difficulty = torch.full((num_envs,), float(fixed_difficulty), dtype=torch.float32)
            scenario_type = torch.arange(num_envs, dtype=torch.long) % 3
            mix_kind = torch.full((num_envs,), MIX_CURRENT, dtype=torch.long)
            return self.parameters(
                difficulty.to(device),
                scenario_type.to(device),
                mix_kind=mix_kind.to(device),
            )

        hover_count = int(round(num_envs * self.hover_fraction))
        recovery_count = num_envs - hover_count
        current_count = min(recovery_count, int(round(num_envs * self.current_fraction)))
        rehearsal_count = min(
            recovery_count - current_count,
            int(round(num_envs * self.rehearsal_fraction)),
        )
        probe_count = recovery_count - current_count - rehearsal_count

        difficulty = torch.zeros(num_envs, dtype=torch.float32)
        hover_anchor = torch.zeros(num_envs, dtype=torch.bool)
        mix_kind = torch.zeros(num_envs, dtype=torch.long)
        hover_anchor[:hover_count] = True
        mix_kind[:hover_count] = MIX_HOVER
        recovery_ids = torch.arange(hover_count, num_envs)
        current_ids = recovery_ids[:current_count]
        rehearsal_ids = recovery_ids[current_count : current_count + rehearsal_count]
        probe_ids = recovery_ids[current_count + rehearsal_count :]
        difficulty[current_ids] = self.difficulty
        mix_kind[current_ids] = MIX_CURRENT
        if rehearsal_count:
            difficulty[rehearsal_ids] = (
                torch.rand(rehearsal_count, generator=generator) * self.difficulty
            )
            mix_kind[rehearsal_ids] = MIX_REHEARSAL
        if probe_count:
            difficulty[probe_ids] = min(1.0, self.difficulty + self.advance_delta)
            mix_kind[probe_ids] = MIX_PROBE

        scenario_weights = self.recovery_scenario_weights()
        self.last_scenario_weights = scenario_weights.clone()
        scenario_counts = _allocate_counts(recovery_count, scenario_weights)
        scenario_type = torch.full((num_envs,), HANDOFF_SCENARIO, dtype=torch.long)
        cursor = 0
        for scenario_id, count in enumerate(scenario_counts.tolist()):
            if count:
                scenario_type[recovery_ids[cursor : cursor + count]] = scenario_id
                cursor += count
        recovery_order = recovery_ids[torch.randperm(recovery_count, generator=generator)]
        scenario_type[recovery_ids] = scenario_type[recovery_order]
        order = torch.randperm(num_envs, generator=generator)
        return self.parameters(
            difficulty[order].to(device),
            scenario_type[order].to(device),
            hover_anchor=hover_anchor[order].to(device),
            mix_kind=mix_kind[order].to(device),
        )

    def sample_exam(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        seed: int | None = None,
    ) -> tuple[ContinuousCurriculumSample, dict[str, torch.Tensor]]:
        """Build a balanced exam containing only the current difficulty."""

        generator = self._generator(seed)
        order = torch.randperm(num_envs, generator=generator)
        chunks = torch.tensor_split(order, 4)
        names = ("hover", "handoff", "impact", "handoff_impact")
        exam_group = {name: chunk.to(device) for name, chunk in zip(names, chunks)}
        difficulty = torch.full((num_envs,), self.difficulty, dtype=torch.float32)
        hover_anchor = torch.zeros(num_envs, dtype=torch.bool)
        scenario_type = torch.full((num_envs,), HANDOFF_SCENARIO, dtype=torch.long)
        mix_kind = torch.full((num_envs,), MIX_CURRENT, dtype=torch.long)
        hover_anchor[chunks[0]] = True
        mix_kind[chunks[0]] = MIX_HOVER
        difficulty[chunks[0]] = 0.0
        scenario_type[chunks[2]] = IMPACT_SCENARIO
        scenario_type[chunks[3]] = HANDOFF_IMPACT_SCENARIO
        sample = self.parameters(
            difficulty.to(device),
            scenario_type.to(device),
            hover_anchor=hover_anchor.to(device),
            mix_kind=mix_kind.to(device),
        )
        return sample, exam_group

    def next_exam_seed(self) -> int:
        """Alternate forever between two fixed current-level exam papers."""

        return self.seed + self.evaluations % 2

    @staticmethod
    def parameters(
        difficulty: torch.Tensor,
        scenario_type: torch.Tensor,
        *,
        hover_anchor: torch.Tensor | None = None,
        mix_kind: torch.Tensor | None = None,
    ) -> ContinuousCurriculumSample:
        if hover_anchor is None:
            hover_anchor = torch.zeros_like(scenario_type, dtype=torch.bool)
        if mix_kind is None:
            mix_kind = torch.where(
                hover_anchor,
                torch.full_like(scenario_type, MIX_HOVER),
                torch.full_like(scenario_type, MIX_CURRENT),
            )
        impact_enabled = (scenario_type != HANDOFF_SCENARIO) & ~hover_anchor
        handoff_enabled = (scenario_type != IMPACT_SCENARIO) & ~hover_anchor
        return ContinuousCurriculumSample(
            difficulty=difficulty,
            scenario_type=scenario_type,
            hover_anchor=hover_anchor,
            handoff_difficulty=torch.where(handoff_enabled, difficulty, 0.0),
            impact_difficulty=torch.where(impact_enabled, difficulty, 0.0),
            impact_enabled=impact_enabled,
            mix_kind=mix_kind,
        )

    def recovery_scenario_weights(self) -> torch.Tensor:
        """Weight handoff / impact / handoff+impact by the latest exam deficit."""

        equal = torch.ones(3, dtype=torch.float32) / 3.0
        floor = self.min_recovery_scenario_fraction
        metrics = self.last_exam_metrics
        if not metrics or any(name not in metrics for name in RECOVERY_SCENARIO_NAMES):
            return equal
        passed = all(
            metrics[name]["strict_success_rate"] >= self.success_thresholds[name]
            and metrics[name]["crash_rate"] <= self.maximum_crash_rate
            for name in RECOVERY_SCENARIO_NAMES
        )
        if passed:
            return equal
        gaps = torch.tensor(
            [
                max(
                    0.0,
                    self.success_thresholds[name]
                    - metrics[name]["strict_success_rate"],
                )
                for name in RECOVERY_SCENARIO_NAMES
            ],
            dtype=torch.float32,
        )
        if float(gaps.sum()) <= 0.0:
            return equal
        gaps = gaps / gaps.sum()
        remaining = 1.0 - 3.0 * floor
        return remaining * gaps + floor

    def update_exam(self, assessment: Mapping[str, Mapping[str, float]]) -> float:
        """Update from one deterministic exam at the current difficulty."""

        required = tuple(self.success_thresholds)
        metrics = {
            name: {str(key): float(value) for key, value in assessment[name].items()}
            for name in self.success_thresholds
        }
        passed = all(
            metrics[name]["strict_success_rate"] >= self.success_thresholds[name]
            and metrics[name]["crash_rate"] <= self.maximum_crash_rate
            for name in required
        )
        severe = any(
            metrics[name]["coarse_success_rate"] < self.severe_success_rate
            or metrics[name]["crash_rate"] > self.severe_crash_rate
            for name in required
        )
        self.evaluations += 1
        self.exams_at_difficulty += 1
        self.last_exam_metrics = metrics
        self.last_exam_passed = passed
        self.last_exam_severe = severe

        if passed:
            self.advance_streak += 1
            self.retreat_streak = 0
            if self.advance_streak >= self.required_streak:
                mastered = self.difficulty
                self.difficulty = min(1.0, self.difficulty + self.advance_delta)
                self.highest_mastered_difficulty = max(
                    self.highest_mastered_difficulty, mastered
                )
                self.exams_at_difficulty = 0
                self._clear_streaks()
        elif severe:
            self.retreat_streak += 1
            self.advance_streak = 0
            if self.retreat_streak >= self.required_streak:
                self.difficulty = max(0.0, self.difficulty - self.retreat_delta)
                self.exams_at_difficulty = 0
                self._clear_streaks()
        else:
            self._clear_streaks()
        return self.difficulty

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self.exams_at_difficulty = 0
        self._clear_streaks()

    @property
    def retention_ceiling(self) -> float:
        return max(self.retention_floor, self.highest_mastered_difficulty)

    def status_diagnostics(self) -> dict[str, float]:
        """Live curriculum fields that should update between exams."""

        weights = self.last_scenario_weights
        return {
            "difficulty": self.difficulty,
            "initial_difficulty": self.initial_difficulty,
            "highest_mastered_difficulty": self.highest_mastered_difficulty,
            "advance_streak": float(self.advance_streak),
            "retreat_streak": float(self.retreat_streak),
            "evaluations": float(self.evaluations),
            "exams_at_difficulty": float(self.exams_at_difficulty),
            "exam_passed": float(self.last_exam_passed),
            "exam_severe": float(self.last_exam_severe),
            "scenario_weight_handoff": float(weights[0]),
            "scenario_weight_impact": float(weights[1]),
            "scenario_weight_handoff_impact": float(weights[2]),
        }

    def diagnostics(self) -> dict[str, float]:
        values = self.status_diagnostics()
        for name, metrics in self.last_exam_metrics.items():
            for metric_name, value in metrics.items():
                values[f"{name}_{metric_name}"] = value
        return values

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "difficulty": self.difficulty,
            "initial_difficulty": self.initial_difficulty,
            "retention_floor": self.retention_floor,
            "highest_mastered_difficulty": self.highest_mastered_difficulty,
            "advance_streak": self.advance_streak,
            "retreat_streak": self.retreat_streak,
            "evaluations": self.evaluations,
            "exams_at_difficulty": self.exams_at_difficulty,
            "last_exam_passed": self.last_exam_passed,
            "last_exam_severe": self.last_exam_severe,
            "last_exam_metrics": self.last_exam_metrics,
            "seed": self.seed,
            "rng_state": self._rng.get_state(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.difficulty = float(state["difficulty"])
        self.initial_difficulty = float(
            state.get("initial_difficulty", self.difficulty)
        )
        self.retention_floor = float(
            state.get("retention_floor", state.get("initial_difficulty", 0.0))
        )
        self.highest_mastered_difficulty = float(
            state.get("highest_mastered_difficulty", 0.0)
        )
        self.advance_streak = int(state.get("advance_streak", 0))
        self.retreat_streak = int(state.get("retreat_streak", 0))
        self.evaluations = int(state.get("evaluations", 0))
        self.exams_at_difficulty = int(state.get("exams_at_difficulty", 0))
        self.last_exam_passed = bool(state.get("last_exam_passed", False))
        self.last_exam_severe = bool(state.get("last_exam_severe", False))
        self.last_exam_metrics = {
            str(name): {str(key): float(value) for key, value in metrics.items()}
            for name, metrics in state.get("last_exam_metrics", {}).items()
        }
        self.seed = int(state.get("seed", 0))
        self._rng = torch.Generator(device="cpu")
        if "rng_state" in state:
            self._rng.set_state(state["rng_state"].cpu())
        else:
            self._rng.manual_seed(self.seed)

    def _generator(self, seed: int | None) -> torch.Generator:
        if seed is None:
            return self._rng
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return generator

    def _clear_streaks(self) -> None:
        self.advance_streak = 0
        self.retreat_streak = 0
