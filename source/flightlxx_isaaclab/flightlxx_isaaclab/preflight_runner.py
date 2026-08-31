"""Simulator-independent resume bookkeeping for the preflight runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .preflight_protocol import PreflightManifest
from .preflight_report import TrialOutcome


_RUN_IDENTITY_KEYS = (
    "protocol_id",
    "checkpoint_sha256",
    "resolved_manifest_sha256",
    "evaluator_sha256",
    "task",
)


def terminal_rms_is_safe(
    steady_rms: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> bool:
    return all(
        key in steady_rms
        and math.isfinite(float(steady_rms[key]))
        and float(steady_rms[key]) < limit
        for key, limit in thresholds.items()
    )


def first_stable_window_end(
    rows: Sequence[Mapping[str, Any]],
    *,
    dwell_s: float,
    step_dt: float,
    thresholds: Mapping[str, float],
) -> float | None:
    """Return the first timestamp ending a continuous stable dwell window."""

    required = max(1, math.ceil(dwell_s / step_dt - 1.0e-9))
    stable_steps = 0
    for row in rows:
        stable_steps = stable_steps + 1 if all(
            key in row
            and math.isfinite(float(row[key]))
            and float(row[key]) < limit
            for key, limit in thresholds.items()
        ) else 0
        if stable_steps >= required:
            return float(row["time_s"])
    return None


def preimpact_window_is_stable(
    rows: Sequence[Mapping[str, Any]],
    *,
    trigger_time_s: float,
    dwell_s: float,
    step_dt: float,
    thresholds: Mapping[str, float],
) -> bool:
    """Check only samples strictly before the trigger-stamped post-impact frame."""

    selected = [
        row
        for row in rows
        if float(row["time_s"]) + 1.0e-9 >= trigger_time_s - dwell_s
        and float(row["time_s"]) < trigger_time_s - 1.0e-9
    ]
    required = max(1, math.ceil(dwell_s / step_dt - 1.0e-9))
    return len(selected) >= required and all(
        all(
            key in row and math.isfinite(float(row[key])) and float(row[key]) < limit
            for key, limit in thresholds.items()
        )
        for row in selected
    )


def expected_trial_ids(manifest: PreflightManifest) -> tuple[str, ...]:
    return (
        tuple(case.case_id for case in manifest.structured)
        + tuple(case.case_id for case in manifest.randomized)
        + tuple(episode.episode_id for episode in manifest.continuous)
        + tuple(case.case_id for case in manifest.balls)
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def write_trial_result(trial_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish the one completion marker used by resume logic."""

    if payload.get("status") != "completed":
        raise ValueError("only completed trial payloads may be published as result.json")
    outcome = payload.get("outcome")
    if not isinstance(outcome, Mapping) or not outcome.get("trial_id"):
        raise ValueError("completed trial payload requires outcome.trial_id")
    return _atomic_json(Path(trial_dir) / "result.json", payload)


def ensure_run_identity(output_dir: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Prevent resume from combining results produced by different evidence sources."""

    missing = [key for key in _RUN_IDENTITY_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"run metadata is missing identity keys: {', '.join(missing)}")
    path = Path(output_dir) / "run_metadata.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"existing run metadata is unreadable: {path}") from exc
        differences = [
            key for key in _RUN_IDENTITY_KEYS if existing.get(key) != metadata.get(key)
        ]
        if differences:
            raise RuntimeError(
                "output directory belongs to a different run identity; mismatched: "
                + ", ".join(differences)
            )
        return path
    return _atomic_json(path, metadata)


def _completed_payloads(output_dir: str | Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted((Path(output_dir) / "trials").glob("*/*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        outcome = payload.get("outcome")
        if payload.get("status") == "completed" and isinstance(outcome, dict) and outcome.get("trial_id"):
            payloads.append(payload)
    return payloads


def completed_trial_ids(output_dir: str | Path) -> set[str]:
    return {str(payload["outcome"]["trial_id"]) for payload in _completed_payloads(output_dir)}


def load_completed_outcomes(output_dir: str | Path) -> list[TrialOutcome]:
    outcomes: list[TrialOutcome] = []
    for payload in _completed_payloads(output_dir):
        raw = dict(payload["outcome"])
        raw["impact_recoveries"] = tuple(bool(value) for value in raw.get("impact_recoveries", ()))
        outcomes.append(TrialOutcome(**raw))
    return outcomes


def write_progress(
    output_dir: str | Path,
    manifest: PreflightManifest,
    completed_ids: set[str],
) -> Path:
    expected = expected_trial_ids(manifest)
    completed_ordered = [trial_id for trial_id in expected if trial_id in completed_ids]
    remaining = [trial_id for trial_id in expected if trial_id not in completed_ids]
    return _atomic_json(
        Path(output_dir) / "progress.json",
        {
            "protocol_id": manifest.protocol_id,
            "total": len(expected),
            "completed": len(completed_ordered),
            "remaining": len(remaining),
            "completed_trial_ids": completed_ordered,
            "remaining_trial_ids": remaining,
        },
    )
