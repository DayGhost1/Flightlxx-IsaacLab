"""Qualification aggregation and durable reports for preflight validation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .preflight_protocol import PreflightManifest


@dataclass(frozen=True)
class TrialOutcome:
    trial_id: str
    group: str
    recovered: bool
    hard_failure: bool
    failure_reason: str
    recovery_time_s: float | None
    max_position_error_m: float
    max_linear_speed_mps: float
    max_attitude_error_deg: float
    max_angular_speed_radps: float
    steady_rms: Mapping[str, float]
    impact_recoveries: tuple[bool, ...] = ()


@dataclass(frozen=True)
class GroupResult:
    name: str
    passed: int
    total: int
    hard_failures: int
    qualified: bool
    detail: str


@dataclass(frozen=True)
class QualificationSummary:
    status: str
    qualified: bool
    group_results: Mapping[str, GroupResult]
    outcomes: tuple[TrialOutcome, ...]
    missing_trial_ids: tuple[str, ...]
    duplicate_trial_ids: tuple[str, ...]
    unexpected_trial_ids: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    worst_cases: tuple[TrialOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expected_ids(manifest: PreflightManifest) -> dict[str, str]:
    expected = {case.case_id: "structured" for case in manifest.structured}
    expected.update({case.case_id: "randomized" for case in manifest.randomized})
    expected.update({episode.episode_id: "continuous" for episode in manifest.continuous})
    expected.update({case.case_id: "balls" for case in manifest.balls})
    return expected


def _worst_key(outcome: TrialOutcome) -> tuple[float, ...]:
    return (
        1.0 if outcome.hard_failure else 0.0,
        0.0 if outcome.recovered else 1.0,
        outcome.recovery_time_s if outcome.recovery_time_s is not None else 1.0e9,
        outcome.max_position_error_m,
        outcome.max_attitude_error_deg,
        outcome.max_angular_speed_radps,
    )


def qualify_preflight(
    manifest: PreflightManifest,
    outcomes: Sequence[TrialOutcome],
) -> QualificationSummary:
    """Apply the versioned group requirements; missing evidence always fails closed."""

    expected = _expected_ids(manifest)
    counts = Counter(outcome.trial_id for outcome in outcomes)
    duplicates = tuple(sorted(trial_id for trial_id, count in counts.items() if count > 1))
    missing = tuple(sorted(set(expected) - set(counts)))
    unexpected = tuple(sorted(set(counts) - set(expected)))
    unique = {outcome.trial_id: outcome for outcome in outcomes if counts[outcome.trial_id] == 1}

    group_outcomes: dict[str, list[TrialOutcome]] = {
        name: [
            unique[trial_id]
            for trial_id, group in expected.items()
            if group == name and trial_id in unique and unique[trial_id].group == name
        ]
        for name in ("structured", "randomized", "continuous", "balls")
    }
    failures: list[str] = []
    if missing:
        failures.append(f"missing results: {', '.join(missing)}")
    if duplicates:
        failures.append(f"duplicate results: {', '.join(duplicates)}")
    if unexpected:
        failures.append(f"unexpected results: {', '.join(unexpected)}")
    wrong_group = sorted(
        outcome.trial_id
        for outcome in outcomes
        if outcome.trial_id in expected and outcome.group != expected[outcome.trial_id]
    )
    if wrong_group:
        failures.append(f"wrong group labels: {', '.join(wrong_group)}")

    group_results: dict[str, GroupResult] = {}
    structured = group_outcomes["structured"]
    structured_hard = sum(outcome.hard_failure for outcome in structured)
    structured_passed = sum(outcome.recovered and not outcome.hard_failure for outcome in structured)
    structured_ok = len(structured) == 36 and structured_passed == 36 and structured_hard == 0
    group_results["structured"] = GroupResult(
        "structured", structured_passed, 36, structured_hard, structured_ok, "规则单撞击要求36/36"
    )

    randomized = group_outcomes["randomized"]
    random_hard = sum(outcome.hard_failure for outcome in randomized)
    random_passed = sum(outcome.recovered and not outcome.hard_failure for outcome in randomized)
    random_ok = len(randomized) == 20 and random_passed >= 19 and random_hard == 0
    group_results["randomized"] = GroupResult(
        "randomized", random_passed, 20, random_hard, random_ok, "随机单撞击要求至少19/20且无硬失败"
    )

    continuous = group_outcomes["continuous"]
    continuous_hard = sum(outcome.hard_failure for outcome in continuous)
    separated = {outcome.trial_id: outcome for outcome in continuous if outcome.trial_id in {"C01", "C02", "C03"}}
    stress = {outcome.trial_id: outcome for outcome in continuous if outcome.trial_id in {"C04", "C05", "C06"}}
    separated_ok = len(separated) == 3 and all(
        len(outcome.impact_recoveries) == 4 and all(outcome.impact_recoveries)
        for outcome in separated.values()
    )
    stress_passed = sum(outcome.recovered for outcome in stress.values())
    continuous_ok = (
        len(continuous) == 6
        and separated_ok
        and len(stress) == 3
        and stress_passed >= 2
        and continuous_hard == 0
    )
    continuous_passed = sum(
        (
            len(outcome.impact_recoveries) == 4 and all(outcome.impact_recoveries)
            if outcome.trial_id in separated
            else outcome.recovered
        )
        and not outcome.hard_failure
        for outcome in continuous
    )
    group_results["continuous"] = GroupResult(
        "continuous",
        continuous_passed,
        6,
        continuous_hard,
        continuous_ok,
        "C01-C03每次恢复；C04-C06至少2/3最终恢复；全部无硬失败",
    )

    balls = group_outcomes["balls"]
    ball_hard = sum(outcome.hard_failure for outcome in balls)
    ball_passed = sum(outcome.recovered and not outcome.hard_failure for outcome in balls)
    ball_ok = len(balls) == 12 and ball_passed == 12 and ball_hard == 0
    group_results["balls"] = GroupResult(
        "balls", ball_passed, 12, ball_hard, ball_ok, "刚体球碰撞要求12/12"
    )

    for name, result in group_results.items():
        if not result.qualified:
            failures.append(
                f"{name} failed: {result.passed}/{result.total}, {result.hard_failures} hard failures"
            )
    if any(outcome.hard_failure for outcome in outcomes):
        failures.append("at least one trial has a hard failure")

    qualified = not failures and all(result.qualified for result in group_results.values())
    worst = tuple(sorted(outcomes, key=_worst_key, reverse=True)[:5])
    return QualificationSummary(
        status="QUALIFIED" if qualified else "NOT_QUALIFIED",
        qualified=qualified,
        group_results=group_results,
        outcomes=tuple(outcomes),
        missing_trial_ids=missing,
        duplicate_trial_ids=duplicates,
        unexpected_trial_ids=unexpected,
        failure_reasons=tuple(failures),
        worst_cases=worst,
    )


def write_preflight_report(
    summary: QualificationSummary,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write machine-readable and concise Chinese qualification reports."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "qualification_summary.json"
    csv_path = root / "trial_summary.csv"
    markdown_path = root / "qualification_report.md"
    json_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial_id",
                "group",
                "recovered",
                "hard_failure",
                "failure_reason",
                "recovery_time_s",
                "max_position_error_m",
                "max_linear_speed_mps",
                "max_attitude_error_deg",
                "max_angular_speed_radps",
                "steady_rms_json",
                "impact_recoveries_json",
            ],
        )
        writer.writeheader()
        for outcome in summary.outcomes:
            row = asdict(outcome)
            row["steady_rms_json"] = json.dumps(row.pop("steady_rms"), ensure_ascii=False)
            row["impact_recoveries_json"] = json.dumps(row.pop("impact_recoveries"))
            writer.writerow(row)

    lines = [
        "# 四旋翼全面撞击验证报告",
        "",
        f"## 上实机准入结论：{summary.status}",
        "",
        "| 分组 | 通过 | 硬失败 | 要求 | 结论 |",
        "|---|---:|---:|---|---|",
    ]
    for result in summary.group_results.values():
        lines.append(
            f"| {result.name} | {result.passed}/{result.total} | {result.hard_failures} | "
            f"{result.detail} | {'通过' if result.qualified else '失败'} |"
        )
    lines.extend(["", "## 最坏5个案例", "", "| ID | 分组 | 恢复时间(s) | 位置峰值(m) | 姿态峰值(deg) | 末段 RMS |", "|---|---|---:|---:|---:|---|"])
    for outcome in summary.worst_cases:
        recovery = "未恢复" if outcome.recovery_time_s is None else f"{outcome.recovery_time_s:.3f}"
        rms = ", ".join(f"{key}={value:.4g}" for key, value in outcome.steady_rms.items())
        lines.append(
            f"| {outcome.trial_id} | {outcome.group} | {recovery} | "
            f"{outcome.max_position_error_m:.4f} | {outcome.max_attitude_error_deg:.3f} | {rms} |"
        )
    if summary.failure_reasons:
        lines.extend(["", "## 未通过原因", ""])
        lines.extend(f"- {reason}" for reason in summary.failure_reasons)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def write_preflight_figures(
    summary: QualificationSummary,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export reproducible paper-ready overview figures as PDF and 300-dpi PNG."""

    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(output_dir) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    paths: dict[str, Path] = {}

    def save(fig, stem: str) -> None:
        pdf = root / f"{stem}.pdf"
        png = root / f"{stem}.png"
        fig.savefig(pdf)
        fig.savefig(png, dpi=300)
        plt.close(fig)
        paths[f"{stem.removeprefix('preflight_')}_pdf"] = pdf
        paths[f"{stem.removeprefix('preflight_')}_png"] = png

    names = ("structured", "randomized", "continuous", "balls")
    results = [summary.group_results[name] for name in names]
    pass_rates = [100.0 * result.passed / result.total for result in results]
    fig, ax = plt.subplots(figsize=(6.75, 2.8))
    x = np.arange(len(names))
    bars = ax.bar(x, pass_rates, color=colors, width=0.62, edgecolor="white", linewidth=0.6)
    ax.axhline(100.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, ["Structured", "Random", "Sequential", "Rigid-ball"])
    ax.set_ylim(0.0, 108.0)
    ax.set_ylabel("Passed trials (%)")
    ax.set_title(f"Preflight qualification: {summary.status}")
    for bar, result, rate in zip(bars, results, pass_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            rate + 1.5,
            f"{result.passed}/{result.total}\nHF={result.hard_failures}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save(fig, "preflight_group_summary")

    level_names = ("small", "medium", "large")
    matrix = np.full((12, 3), np.nan)
    recovered = np.zeros((12, 3), dtype=bool)
    for outcome in summary.outcomes:
        if outcome.group != "structured":
            continue
        _, template_id, level = outcome.trial_id.split("_", 2)
        row = int(template_id[1:]) - 1
        column = level_names.index(level)
        matrix[row, column] = outcome.recovery_time_s if outcome.recovery_time_s is not None else np.nan
        recovered[row, column] = outcome.recovered and not outcome.hard_failure
    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0)
    ax.grid(False)
    ax.set_xticks(np.arange(3), ["0.6", "1.2", "1.5"])
    ax.set_yticks(np.arange(12), [f"T{index:02d}" for index in range(1, 13)])
    ax.set_xlabel("Impulse (N s)")
    ax.set_ylabel("Direction / application template")
    ax.set_title("Structured recovery time (s)")
    for row in range(12):
        for column in range(3):
            label = f"{matrix[row, column]:.2f}" if recovered[row, column] else "FAIL"
            ax.text(column, row, label, ha="center", va="center", fontsize=7, color="#222222")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Recovery time (s)")
    save(fig, "preflight_structured_matrix")

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.0), constrained_layout=True)
    metrics = (
        ("Recovery time (s)", lambda outcome: outcome.recovery_time_s),
        ("Peak position error (m)", lambda outcome: outcome.max_position_error_m),
        ("Peak attitude error (deg)", lambda outcome: outcome.max_attitude_error_deg),
        ("Terminal position RMS (m)", lambda outcome: outcome.steady_rms.get("position_error")),
    )
    for ax, (label, accessor) in zip(axes.flat, metrics):
        for group_index, name in enumerate(names):
            values = [
                float(value)
                for outcome in summary.outcomes
                if outcome.group == name
                for value in [accessor(outcome)]
                if value is not None and math.isfinite(float(value))
            ]
            if not values:
                continue
            offsets = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                group_index + offsets,
                values,
                s=18,
                alpha=0.72,
                color=colors[group_index],
                edgecolors="white",
                linewidths=0.35,
            )
            ax.plot(
                [group_index - 0.22, group_index + 0.22],
                [np.median(values), np.median(values)],
                color="#222222",
                linewidth=1.2,
            )
        ax.set_xticks(range(4), ["Struct.", "Random", "Seq.", "Ball"])
        ax.set_ylabel(label)
    fig.suptitle("Preflight response metrics (points: trials; bars: median)", fontsize=11, fontweight="bold")
    save(fig, "preflight_response_metrics")
    return paths
