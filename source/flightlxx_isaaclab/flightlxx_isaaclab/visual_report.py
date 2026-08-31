"""Static analysis artifacts for the fixed five-impact GUI playback."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flightlxx_isaaclab.evaluation import FixedImpactProtocol


@dataclass(frozen=True)
class VisualReportPaths:
    response_figure: Path
    summary_figure: Path
    independent_response_figure: Path
    independent_summary_figure: Path
    aligned_csv: Path
    markdown: Path


def align_impact_windows(
    time_series: Sequence[Mapping[str, Any]],
    protocol: FixedImpactProtocol,
    *,
    pre_s: float = 0.5,
    post_s: float = 3.5,
) -> list[dict[str, Any]]:
    """Copy telemetry into five impact-relative windows."""

    if pre_s < 0.0 or post_s <= 0.0:
        raise ValueError("pre_s must be non-negative and post_s must be positive")
    aligned: list[dict[str, Any]] = []
    for impact_index, impact in enumerate(protocol.impacts, start=1):
        force_norm = float(impact.force_b.norm())
        for source in time_series:
            relative_time = float(source["time_s"]) - impact.trigger_time_s
            if -pre_s - 1.0e-9 <= relative_time <= post_s + 1.0e-9:
                aligned.append(
                    {
                        "impact_index": impact_index,
                        "impact_id": impact.impact_id,
                        "relative_time_s": relative_time,
                        "force_norm_n": force_norm,
                        "duration_s": impact.duration_s,
                        **dict(source),
                    }
                )
    return aligned


def align_independent_windows(
    time_series: Sequence[Mapping[str, Any]],
    protocol: FixedImpactProtocol,
    *,
    trigger_time_s: float = 1.0,
    pre_s: float = 0.5,
    post_s: float = 3.5,
) -> list[dict[str, Any]]:
    """Align five reset-isolated trials at their shared local trigger."""

    if pre_s < 0.0 or post_s <= 0.0:
        raise ValueError("pre_s must be non-negative and post_s must be positive")
    by_id = {impact.impact_id: (index, impact) for index, impact in enumerate(protocol.impacts, 1)}
    aligned: list[dict[str, Any]] = []
    for source in time_series:
        trial_id = str(source.get("trial_impact_id", ""))
        if trial_id not in by_id:
            continue
        impact_index, impact = by_id[trial_id]
        relative_time = float(source["time_s"]) - trigger_time_s
        if -pre_s - 1.0e-9 <= relative_time <= post_s + 1.0e-9:
            aligned.append(
                {
                    "mode": "independent",
                    "impact_index": impact_index,
                    "impact_id": trial_id,
                    "relative_time_s": relative_time,
                    "force_norm_n": float(impact.force_b.norm()),
                    "duration_s": impact.duration_s,
                    **dict(source),
                }
            )
    return aligned


_COLORS = ("#3B73B9", "#E0782F", "#7A8B2E", "#CC79A7", "#D8A600")
_LINESTYLES = ("-", "--", "-.", ":", "-")
_MARKERS = (None, None, None, None, "o")


def _rows_for_impact(aligned: Iterable[Mapping[str, Any]], impact_id: str) -> list[Mapping[str, Any]]:
    return [row for row in aligned if row["impact_id"] == impact_id]


def _write_aligned_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    primary_fields = (
        "mode",
        "trial_impact_id",
        "impact_index",
        "impact_id",
        "relative_time_s",
        "force_norm_n",
        "duration_s",
        "time_s",
        "position_error",
        "linear_speed",
        "attitude_error_rad",
        "angular_speed",
        "failure",
    )
    additional_fields = sorted(
        {key for row in rows for key in row}.difference(primary_fields)
    )
    fields = (*primary_fields, *additional_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field) for field in fields}
            payload["relative_time_s"] = f"{float(row['relative_time_s']):.6f}"
            payload["force_norm_n"] = f"{float(row['force_norm_n']):.6f}"
            writer.writerow(payload)


def _response_figure(
    path: Path,
    aligned: Sequence[Mapping[str, Any]],
    protocol: FixedImpactProtocol,
    checkpoint_step: int,
    passed: bool,
    mode_title: str = "Fixed five-impact recovery",
) -> None:
    metric_specs = (
        ("position_error", "Position error (m)", protocol.thresholds["position_error"], 1.0),
        ("linear_speed", "Linear speed (m/s)", protocol.thresholds["linear_speed"], 1.0),
        (
            "attitude_error_rad",
            "Attitude error (deg)",
            math.degrees(protocol.thresholds["attitude_error_rad"]),
            180.0 / math.pi,
        ),
        ("angular_speed", "Angular speed (rad/s)", protocol.thresholds["angular_speed"], 1.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.5), sharex=True, constrained_layout=True)
    max_duration = max(impact.duration_s for impact in protocol.impacts)
    for axis, (key, ylabel, threshold, scale) in zip(axes.flat, metric_specs):
        axis.axvspan(0.0, max_duration, color="#D62728", alpha=0.08, label="force active")
        axis.axhline(threshold, color="#333333", linestyle=(0, (4, 3)), linewidth=1.1, label="recovery limit")
        for index, impact in enumerate(protocol.impacts):
            rows = _rows_for_impact(aligned, impact.impact_id)
            axis.plot(
                [float(row["relative_time_s"]) for row in rows],
                [float(row[key]) * scale for row in rows],
                color=_COLORS[index],
                linestyle=_LINESTYLES[index],
                marker=_MARKERS[index],
                markevery=20 if _MARKERS[index] else None,
                linewidth=1.8,
                markersize=3.0,
                label=f"{index + 1}. {impact.impact_id}",
            )
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time from impact trigger (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    outcome = "PASS" if passed else "FAIL"
    fig.suptitle(
        f"{mode_title} — checkpoint {checkpoint_step:,} ({outcome})\n"
        "Curves are aligned at impact trigger; shaded interval spans the longest force pulse",
        fontsize=14,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _summary_figure(
    path: Path,
    aligned: Sequence[Mapping[str, Any]],
    protocol: FixedImpactProtocol,
    records: Sequence[Mapping[str, Any]],
    mode_title: str = "Fixed protocol impact summary",
) -> None:
    names = [f"{index + 1}" for index in range(len(protocol.impacts))]
    recovery_by_id = {str(record["impact_id"]): record.get("recovery_time_s") for record in records}
    recovery_times = [recovery_by_id.get(impact.impact_id) for impact in protocol.impacts]
    recovery_values = [0.0 if value is None else float(value) for value in recovery_times]

    metrics = ("position_error", "linear_speed", "attitude_error_rad", "angular_speed")
    metric_labels = ("position", "speed", "attitude", "angular speed")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axes[0].bar(names, recovery_values, color=_COLORS, edgecolor="#333333", linewidth=0.6)
    axes[0].set_title("Recovery time by impact")
    axes[0].set_xlabel("Impact number")
    axes[0].set_ylabel("Recovery time after force ends (s)")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.7)

    width = 0.17
    centres = list(range(len(protocol.impacts)))
    for metric_index, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ratios = []
        for impact in protocol.impacts:
            post_rows = [
                row
                for row in _rows_for_impact(aligned, impact.impact_id)
                if float(row["relative_time_s"]) >= 0.0
            ]
            peak = max((float(row[metric]) for row in post_rows), default=0.0)
            ratios.append(peak / protocol.thresholds[metric])
        offsets = [centre + (metric_index - 1.5) * width for centre in centres]
        axes[1].bar(offsets, ratios, width=width, label=label)
    axes[1].axhline(1.0, color="#333333", linestyle=(0, (4, 3)), linewidth=1.1)
    axes[1].set_xticks(centres, names)
    axes[1].set_xlabel("Impact number")
    axes[1].set_ylabel("Peak response / recovery limit")
    axes[1].set_title("Post-impact peak severity")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.7)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle(mode_title, fontsize=14)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _mode_markdown(
    title: str,
    mode_result: Mapping[str, Any],
    protocol: FixedImpactProtocol,
    response_figure: Path,
    summary_figure: Path,
) -> list[str]:
    records = {
        str(record["impact_id"]): record
        for record in mode_result.get("impact_records", [])
    }
    lines = [
        f"## {title}",
        "",
        f"- Result: {'PASS' if mode_result.get('passed') else 'FAIL'}",
        f"- Recovered: {int(mode_result.get('recovered_count', 0))}/5",
        f"- Crashed: {bool(mode_result.get('crashed', False))}",
        f"- Invalid early truncation: {bool(mode_result.get('invalid_early_truncation', False))}",
        "",
        f"![Impact-aligned response]({response_figure.name})",
        "",
        f"![Impact summary]({summary_figure.name})",
        "",
        "| # | Impact | Force (N) | Duration (s) | Recovered | Recovery time (s) |",
        "|---:|---|---:|---:|:---:|---:|",
    ]
    for index, impact in enumerate(protocol.impacts, start=1):
        record = records.get(impact.impact_id, {})
        recovery_time = record.get("recovery_time_s")
        recovery_text = "—" if recovery_time is None else f"{float(recovery_time):.3f}"
        lines.append(
            f"| {index} | {impact.impact_id} | {float(impact.force_b.norm()):.3f} | "
            f"{impact.duration_s:.3f} | {'yes' if record.get('recovered') else 'no'} | {recovery_text} |"
        )
    return lines


def _write_markdown(
    path: Path,
    result: Mapping[str, Any],
    protocol: FixedImpactProtocol,
    paths: VisualReportPaths,
) -> None:
    modes = result.get("modes") or {"sequential": result}
    lines = [
        "# Fixed five-impact playback report",
        "",
        f"- Checkpoint step: {int(result.get('checkpoint_step', 0)):,}",
        f"- Combined result: {'PASS' if result.get('passed') else 'FAIL'}",
        "",
    ]
    if "independent" in modes:
        lines.extend(
            _mode_markdown(
                "Independent reset trials",
                modes["independent"],
                protocol,
                paths.independent_response_figure,
                paths.independent_summary_figure,
            )
        )
        lines.append("")
    lines.extend(
        _mode_markdown(
            "Sequential stress trial",
            modes["sequential"],
            protocol,
            paths.response_figure,
            paths.summary_figure,
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_visual_evaluation_report(
    result: Mapping[str, Any],
    protocol: FixedImpactProtocol,
    output_dir: str | Path,
) -> VisualReportPaths:
    """Write impact-aligned figures and durable tabular/Markdown artifacts."""

    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    paths = VisualReportPaths(
        response_figure=analysis_dir / "five_impact_response.png",
        summary_figure=analysis_dir / "impact_summary.png",
        independent_response_figure=analysis_dir / "independent_five_impact_response.png",
        independent_summary_figure=analysis_dir / "independent_impact_summary.png",
        aligned_csv=analysis_dir / "impact_aligned.csv",
        markdown=analysis_dir / "report.md",
    )
    modes = result.get("modes") or {"sequential": result}
    sequential_result = modes["sequential"]
    sequential_aligned = align_impact_windows(
        sequential_result.get("time_series", []), protocol
    )
    independent_result = modes.get("independent")
    independent_aligned = (
        align_independent_windows(independent_result.get("time_series", []), protocol)
        if independent_result is not None
        else []
    )
    if not sequential_aligned:
        raise ValueError("evaluation result does not contain telemetry in the five impact windows")
    combined_aligned = [
        {"mode": "independent", **row} for row in independent_aligned
    ] + [{"mode": "sequential", **row} for row in sequential_aligned]
    _write_aligned_csv(paths.aligned_csv, combined_aligned)
    checkpoint_step = int(result.get("checkpoint_step", 0))
    _response_figure(
        paths.response_figure,
        sequential_aligned,
        protocol,
        checkpoint_step,
        bool(sequential_result.get("passed", False)),
        "Sequential five-impact recovery",
    )
    _summary_figure(
        paths.summary_figure,
        sequential_aligned,
        protocol,
        sequential_result.get("impact_records", []),
        "Sequential stress-trial summary",
    )
    if independent_result is not None and independent_aligned:
        _response_figure(
            paths.independent_response_figure,
            independent_aligned,
            protocol,
            checkpoint_step,
            bool(independent_result.get("passed", False)),
            "Independent reset-trial recovery",
        )
        _summary_figure(
            paths.independent_summary_figure,
            independent_aligned,
            protocol,
            independent_result.get("impact_records", []),
            "Independent reset-trial summary",
        )
    _write_markdown(paths.markdown, result, protocol, paths)
    return paths
