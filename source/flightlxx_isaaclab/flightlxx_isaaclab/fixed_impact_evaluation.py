"""Deterministic checkpoint evaluation for FlightLxx's five fixed impacts."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from flightlxx_isaaclab.evaluation import (
    FixedImpactProtocol,
    ImpactRecoveryTracker,
    load_fixed_protocol,
    protocol_for_single_impact,
    protocol_to_dict,
)


@dataclass(frozen=True)
class EvaluationArtifactPaths:
    json_path: Path
    timeseries_path: Path
    summary_csv: Path


_SUMMARY_FIELDS = (
    "checkpoint_step",
    "passed",
    "recovered_count",
    "crashed",
    "max_position_error",
    "max_attitude_error_rad",
    "max_linear_speed",
    "max_angular_speed",
    "final_steady_position_error",
)


def write_evaluation_artifacts(output_dir: str | Path, step: int, result: Mapping[str, Any]) -> EvaluationArtifactPaths:
    """Persist one immutable checkpoint result plus a run-level CSV index."""

    evaluation_dir = Path(output_dir) / "evaluation"
    timeseries_dir = evaluation_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    json_path = evaluation_dir / f"step_{step:08d}.json"
    timeseries_path = timeseries_dir / f"step_{step:08d}.csv"
    summary_csv = evaluation_dir / "summary.csv"
    payload = {"checkpoint_step": int(step), **dict(result)}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = list(result.get("time_series", []))
    fields = sorted({key for row in rows for key in row}) or ["time_s"]
    with timeseries_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_row = {name: payload.get(name) for name in _SUMMARY_FIELDS}
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    return EvaluationArtifactPaths(json_path, timeseries_path, summary_csv)


def _scalar(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics[key]
    if isinstance(value, torch.Tensor):
        return float(value[0].detach().item())
    return float(value)


def _steady_rms(
    time_series: list[dict[str, float | bool | str]],
    window_s: float,
) -> dict[str, float | None]:
    """Compute terminal-window RMS from pre-reset telemetry only."""

    metric_keys = (
        "position_error",
        "linear_speed",
        "attitude_error_rad",
        "angular_speed",
    )
    if not time_series:
        return {key: None for key in metric_keys}
    end_time = float(time_series[-1]["time_s"])
    start_time = end_time - window_s
    rows = [row for row in time_series if float(row["time_s"]) > start_time + 1.0e-9]
    return {
        key: math.sqrt(sum(float(row[key]) ** 2 for row in rows) / len(rows))
        for key in metric_keys
    }


def _run_protocol(
    env: Any,
    raw_env: Any,
    actor: Any,
    normalize_obs: Callable[..., torch.Tensor],
    protocol: FixedImpactProtocol,
    *,
    mode: str,
    trial_impact_id: str | None,
    on_step: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    raw_env.begin_fixed_evaluation(protocol)
    tracker = ImpactRecoveryTracker(protocol)
    time_series: list[dict[str, float | bool | str]] = []
    crashed = False
    invalid_early_truncation = False
    maxima = {
        "position_error": 0.0,
        "attitude_error_rad": 0.0,
        "linear_speed": 0.0,
        "angular_speed": 0.0,
    }
    try:
        observations = env.reset(random_start_init=False)
        total_steps = math.ceil(protocol.total_duration_s / raw_env.step_dt)
        for step_index in range(total_steps):
            with torch.no_grad():
                normalized = normalize_obs(observations, update=False)
                if hasattr(actor, "explore"):
                    actions = actor.explore(normalized, deterministic=True)
                else:
                    actions = actor(normalized)
            observations, _, dones, _ = env.step(actions.float())
            metrics_tensor = raw_env.evaluation_step_metrics()
            if not metrics_tensor:
                raise RuntimeError("evaluation environment did not publish step metrics")
            scalar_metrics = {
                key: _scalar(metrics_tensor, key)
                for key in metrics_tensor
                if key != "failure"
            }
            metrics = {key: scalar_metrics[key] for key in protocol.thresholds}
            failure = bool(metrics_tensor["failure"][0].item())
            time_s = (step_index + 1) * raw_env.step_dt
            completed_impacts = tracker.step(time_s, metrics, failure)
            row: dict[str, float | bool | str] = {
                "mode": mode,
                "trial_impact_id": trial_impact_id or "",
                "time_s": time_s,
                **scalar_metrics,
                "failure": failure,
            }
            time_series.append(row)
            if on_step is not None:
                on_step(
                    {
                        "mode": mode,
                        "trial_impact_id": trial_impact_id,
                        "time_s": time_s,
                        "metrics": scalar_metrics,
                        "actions": actions.detach(),
                        "completed_impacts": completed_impacts,
                        "failure": failure,
                    }
                )
            for key in maxima:
                maxima[key] = max(maxima[key], metrics[key])

            done = bool(dones[0].item()) if isinstance(dones, torch.Tensor) else bool(dones)
            if done and step_index + 1 < total_steps:
                invalid_early_truncation = True
                break
            if failure:
                crashed = True
                break

        steady = _steady_rms(time_series, protocol.recovery_dwell_s)
        steady_safe = all(
            steady[key] is not None and float(steady[key]) < limit
            for key, limit in protocol.thresholds.items()
        )
        records = [dict(record) for record in tracker.records]
        if len(records) == 1:
            records[0].update(
                {
                    "max_position_error": maxima["position_error"],
                    "max_attitude_error_rad": maxima["attitude_error_rad"],
                    "max_linear_speed": maxima["linear_speed"],
                    "max_angular_speed": maxima["angular_speed"],
                    "steady_position_error_rms": steady["position_error"],
                }
            )
        return {
            "protocol": protocol_to_dict(protocol),
            "passed": (
                tracker.complete
                and all(record["recovered"] for record in records)
                and not crashed
                and not invalid_early_truncation
                and steady_safe
            ),
            "recovered_count": sum(bool(record["recovered"]) for record in records),
            "crashed": crashed,
            "invalid_early_truncation": invalid_early_truncation,
            "max_position_error": maxima["position_error"],
            "max_attitude_error_rad": maxima["attitude_error_rad"],
            "max_linear_speed": maxima["linear_speed"],
            "max_angular_speed": maxima["angular_speed"],
            "steady_position_error_rms": steady["position_error"],
            "steady_linear_speed_rms": steady["linear_speed"],
            "steady_attitude_error_rad_rms": steady["attitude_error_rad"],
            "steady_angular_speed_rms": steady["angular_speed"],
            "final_steady_position_error": steady["position_error"],
            "impact_records": records,
            "time_series": time_series,
        }
    finally:
        raw_env.end_fixed_evaluation()


def _aggregate_independent(trials: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for trial in trials for record in trial["impact_records"]]
    time_series = [row for trial in trials for row in trial["time_series"]]

    def maximum(name: str) -> float:
        return max((float(trial[name]) for trial in trials), default=0.0)

    steady_values = [
        float(trial["steady_position_error_rms"])
        for trial in trials
        if trial["steady_position_error_rms"] is not None
    ]
    return {
        "passed": len(trials) == 5 and all(bool(trial["passed"]) for trial in trials),
        "recovered_count": sum(bool(record["recovered"]) for record in records),
        "crashed": any(bool(trial["crashed"]) for trial in trials),
        "invalid_early_truncation": any(
            bool(trial["invalid_early_truncation"]) for trial in trials
        ),
        "max_position_error": maximum("max_position_error"),
        "max_attitude_error_rad": maximum("max_attitude_error_rad"),
        "max_linear_speed": maximum("max_linear_speed"),
        "max_angular_speed": maximum("max_angular_speed"),
        "steady_position_error_rms": max(steady_values, default=None),
        "final_steady_position_error": max(steady_values, default=None),
        "impact_records": records,
        "time_series": time_series,
        "trials": trials,
    }


def evaluate_fixed_five_impacts(
    env: Any,
    actor: Any,
    normalize_obs: Callable[..., torch.Tensor],
    checkpoint_step: int,
    output_dir: str | Path,
    on_step: Callable[[Mapping[str, Any]], None] | None = None,
    protocol: FixedImpactProtocol | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic actor actions without touching training state.

    ``env`` is the FastTD3 IsaacLab wrapper.  No replay buffer, optimizer,
    normalizer update, curriculum update, or exploration noise is invoked.
    """

    if protocol is None:
        package_root = Path(__import__("flightlxx_isaaclab").__file__).parent
        protocol = load_fixed_protocol(package_root / "config" / "fixed_five_impacts.json")
    raw_env = env.envs.unwrapped
    try:
        independent_trials = [
            _run_protocol(
                env,
                raw_env,
                actor,
                normalize_obs,
                protocol_for_single_impact(protocol, index),
                mode="independent",
                trial_impact_id=impact.impact_id,
                on_step=on_step,
            )
            for index, impact in enumerate(protocol.impacts)
        ]
        result: dict[str, Any] = {
            "protocol": protocol_to_dict(protocol),
            **_aggregate_independent(independent_trials),
        }
        paths = write_evaluation_artifacts(output_dir, checkpoint_step, result)
        result.update({"json_path": str(paths.json_path), "timeseries_path": str(paths.timeseries_path)})
        return result
    finally:
        # This reset has a guard in the task so evaluation cannot update the
        # training curriculum or episode metrics.
        env.reset(random_start_init=False)
