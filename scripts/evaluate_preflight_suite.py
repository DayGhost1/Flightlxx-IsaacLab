"""Run the checkpoint-independent comprehensive preflight impact suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--smoke", action="store_true", help="Run one trial from each group, then report missing evidence.")
parser.add_argument("--trial-id", action="append", default=[], help="Run only selected trial IDs; repeat as needed.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app


TASK_NAME = "Isaac-FlightLxx-CTBR-Preflight-Direct-v0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _safe(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> bool:
    return all(math.isfinite(float(metrics[key])) and float(metrics[key]) < limit for key, limit in thresholds.items())


def _steady_rms(
    rows: Sequence[Mapping[str, Any]],
    window_s: float,
) -> dict[str, float]:
    keys = ("position_error", "linear_speed", "attitude_error_rad", "angular_speed")
    if not rows:
        return {key: math.inf for key in keys}
    start = float(rows[-1]["time_s"]) - window_s
    selected = [row for row in rows if float(row["time_s"]) > start + 1.0e-9]
    if not selected:
        return {key: math.inf for key in keys}
    return {
        key: math.sqrt(sum(float(row[key]) ** 2 for row in selected) / len(selected))
        for key in keys
    }


def _recovery_for_window(
    rows: Sequence[Mapping[str, Any]],
    *,
    impact_end_s: float,
    deadline_s: float,
    dwell_s: float,
    thresholds: Mapping[str, float],
    step_dt: float,
) -> tuple[bool, float | None]:
    safe_steps = 0
    required_steps = max(1, math.ceil(dwell_s / step_dt - 1.0e-9))
    for row in rows:
        time_s = float(row["time_s"])
        if time_s + 1.0e-9 < impact_end_s:
            continue
        if time_s > deadline_s + 1.0e-9:
            break
        metrics = {key: float(row[key]) for key in thresholds}
        safe_steps = safe_steps + 1 if _safe(metrics, thresholds) else 0
        if safe_steps >= required_steps:
            recovery_time = max(0.0, time_s - (required_steps - 1) * step_dt - impact_end_s + dwell_s)
            return True, recovery_time
    return False, None


def _trial_protocol(manifest, trial) -> tuple[Any, list[dict[str, Any]], float]:
    import torch

    from flightlxx_isaaclab.evaluation import FixedImpact, FixedImpactProtocol

    if hasattr(trial, "impacts"):
        # The first impact is not allowed until the policy has had the full
        # pre-impact acquisition window.  This avoids conflating initial
        # transient settling with disturbance recovery.
        triggers = [manifest.preimpact_timeout_s]
        for interval in trial.intervals_s:
            triggers.append(triggers[-1] + interval)
        cases = trial.impacts
        policy = trial.recovery_policy
    else:
        triggers = [manifest.preimpact_timeout_s]
        cases = (trial,)
        policy = "single"
    impacts = []
    events = []
    for trigger, case in zip(triggers, cases):
        impact = FixedImpact(
            impact_id=case.case_id,
            trigger_time_s=float(trigger),
            duration_s=float(case.duration_s),
            application_point_b=torch.tensor(case.application_point_b, dtype=torch.float32),
            force_b=torch.tensor(case.force_b, dtype=torch.float32),
        )
        impacts.append(impact)
        events.append({"impact_id": case.case_id, "trigger_time_s": trigger, "end_time_s": impact.end_time_s})
    total = impacts[-1].end_time_s + manifest.recovery_window_s + manifest.final_rms_window_s
    return (
        FixedImpactProtocol(
            protocol_id=f"{manifest.protocol_id}_{getattr(trial, 'episode_id', getattr(trial, 'case_id', 'trial'))}",
            total_duration_s=total,
            recovery_dwell_s=manifest.recovery_dwell_s,
            thresholds=dict(manifest.thresholds),
            impacts=tuple(impacts),
        ),
        events,
        policy,
    )


def _ball_protocol(manifest, ball_case, step_dt: float) -> tuple[Any, list[dict[str, Any]]]:
    import torch

    from flightlxx_isaaclab.evaluation import FixedImpact, FixedImpactProtocol

    # Keep a small contact margin after the pre-hover qualification window:
    # PhysX collision may be resolved one simulation step before the nominal
    # contact timestamp.
    trigger_time_s = manifest.preimpact_timeout_s + 5.0 * step_dt
    dummy = FixedImpact(
        impact_id=f"{ball_case.case_id}_contact_window",
        trigger_time_s=trigger_time_s,
        duration_s=step_dt,
        application_point_b=torch.zeros(3),
        force_b=torch.zeros(3),
    )
    total = trigger_time_s + manifest.recovery_window_s + manifest.final_rms_window_s
    protocol = FixedImpactProtocol(
        protocol_id=f"{manifest.protocol_id}_{ball_case.case_id}",
        total_duration_s=total,
        recovery_dwell_s=manifest.recovery_dwell_s,
        thresholds=dict(manifest.thresholds),
        impacts=(dummy,),
    )
    return protocol, [{"impact_id": ball_case.case_id, "trigger_time_s": trigger_time_s, "end_time_s": trigger_time_s}]


def _write_telemetry(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["time_s"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _run_rollout(
    gym_env,
    actor,
    normalizer,
    protocol,
    events,
    manifest,
    *,
    trial_id: str,
    group: str,
    recovery_policy: str,
    ball_case=None,
):
    import torch

    from flightlxx_isaaclab.preflight_report import TrialOutcome
    from flightlxx_isaaclab.preflight_runner import (
        preimpact_window_is_stable,
        terminal_rms_is_safe,
    )

    raw_env = gym_env.unwrapped
    raw_env.begin_fixed_evaluation(protocol)
    rows: list[dict[str, Any]] = []
    hard_failure = False
    failure_reason = ""
    ball_launched = False
    ball_contact = ball_case is None
    launch_time_s = events[0]["trigger_time_s"] - 0.5 if ball_case is not None else None
    launch_state: dict[str, torch.Tensor] | None = None
    env_ids = torch.tensor([0], device=raw_env.device, dtype=torch.long)
    try:
        observations, _ = gym_env.reset()
        total_steps = math.ceil(protocol.total_duration_s / raw_env.step_dt)
        for step_index in range(total_steps):
            time_before_s = step_index * raw_env.step_dt
            if ball_case is not None and not ball_launched and time_before_s + 1.0e-9 >= launch_time_s:
                launch_state = raw_env.launch_targeted_ball(
                    env_ids,
                    torch.tensor([ball_case.target_point_b], device=raw_env.device, dtype=torch.float32),
                    torch.tensor([ball_case.approach_direction_b], device=raw_env.device, dtype=torch.float32),
                    ball_case.impact_speed_mps,
                    flight_time_s=0.5,
                    contact_clearance_m=ball_case.contact_clearance_m,
                )
                ball_launched = True
            with torch.no_grad():
                normalized = normalizer(observations["policy"], update=False)
                actions = actor.explore(normalized, deterministic=True)
            observations, _, terminated, truncated, _ = gym_env.step(actions.float())
            metrics_tensor = raw_env.evaluation_step_metrics()
            if not metrics_tensor:
                raise RuntimeError("evaluation environment did not publish step metrics")
            metrics = {
                key: float(value[0].detach().item())
                for key, value in metrics_tensor.items()
                if key != "failure"
            }
            failure = bool(metrics_tensor["failure"][0].item())
            time_s = (step_index + 1) * raw_env.step_dt
            row: dict[str, Any] = {"time_s": time_s, **metrics, "failure": failure}
            for action_index, name in enumerate(("collective", "body_rate_x", "body_rate_y", "body_rate_z")):
                row[f"policy_action_{name}"] = float(actions[0, action_index].detach().item())

            if ball_launched and launch_state is not None:
                ball_position = raw_env._ball.data.root_pos_w[0]
                ball_velocity = raw_env._ball.data.root_lin_vel_w[0]
                robot_position = raw_env._robot.data.root_pos_w[0]
                elapsed = max(0.0, time_s - float(launch_time_s))
                gravity = torch.tensor(raw_env.cfg.sim.gravity, device=raw_env.device)
                expected_velocity = launch_state["initial_velocity_w"][0] + gravity * elapsed
                velocity_deviation = torch.linalg.vector_norm(ball_velocity - expected_velocity)
                center_distance = torch.linalg.vector_norm(ball_position - robot_position)
                if (
                    time_s <= float(events[0]["trigger_time_s"]) + 0.6
                    and float(center_distance.item()) <= 0.32
                    and float(velocity_deviation.item()) >= 0.10
                ):
                    ball_contact = True
                for axis, suffix in enumerate(("x", "y", "z")):
                    row[f"ball_position_{suffix}"] = float(ball_position[axis].item())
                    row[f"ball_velocity_{suffix}"] = float(ball_velocity[axis].item())
                row["ball_robot_center_distance"] = float(center_distance.item())
                row["ball_freeflight_velocity_deviation"] = float(velocity_deviation.item())
                row["ball_contact_detected"] = ball_contact
            rows.append(row)

            finite = all(math.isfinite(float(metrics[key])) for key in manifest.thresholds)
            if not finite:
                hard_failure = True
                failure_reason = "numerical_failure"
                break
            if failure:
                hard_failure = True
                failure_reason = "ground_or_arena_collision"
                break
            done = bool((terminated | truncated)[0].item())
            if done and step_index + 1 < total_steps:
                hard_failure = True
                failure_reason = "early_termination"
                break

        impact_recoveries: list[bool] = []
        recovery_times: list[float | None] = []
        for index, event in enumerate(events):
            if index + 1 < len(events):
                next_trigger = float(events[index + 1]["trigger_time_s"])
                deadline = min(float(event["end_time_s"]) + manifest.recovery_window_s, next_trigger)
            else:
                deadline = float(event["end_time_s"]) + manifest.recovery_window_s
            recovered, recovery_time = _recovery_for_window(
                rows,
                impact_end_s=float(event["end_time_s"]),
                deadline_s=deadline,
                dwell_s=manifest.recovery_dwell_s,
                thresholds=manifest.thresholds,
                step_dt=raw_env.step_dt,
            )
            impact_recoveries.append(recovered)
            recovery_times.append(recovery_time)

        if recovery_policy == "final":
            recovered = bool(impact_recoveries and impact_recoveries[-1])
            recovery_time_s = recovery_times[-1] if recovery_times else None
        else:
            recovered = bool(impact_recoveries and all(impact_recoveries))
            recovery_time_s = max((value for value in recovery_times if value is not None), default=None)
        if ball_case is not None and not ball_contact:
            recovered = False
            failure_reason = failure_reason or "ball_target_missed"

        # A rigid ball is launched 0.5 s before its planned contact.  Its
        # approach phase must not be included in the vehicle-only hover gate.
        preimpact_trigger_s = launch_time_s if ball_case is not None else float(events[0]["trigger_time_s"])
        preimpact_stable = preimpact_window_is_stable(
            rows,
            trigger_time_s=float(preimpact_trigger_s),
            dwell_s=manifest.recovery_dwell_s,
            step_dt=raw_env.step_dt,
            thresholds=manifest.thresholds,
        )
        if not preimpact_stable:
            recovered = False
            failure_reason = failure_reason or "preimpact_hover_not_stable"
        steady = _steady_rms(rows, manifest.final_rms_window_s)
        terminal_stable = terminal_rms_is_safe(steady, manifest.thresholds)
        if not terminal_stable:
            recovered = False
            failure_reason = failure_reason or "terminal_hover_not_stable"
        maxima = {
            "position_error": max((float(row["position_error"]) for row in rows), default=math.inf),
            "linear_speed": max((float(row["linear_speed"]) for row in rows), default=math.inf),
            "attitude_error_rad": max((float(row["attitude_error_rad"]) for row in rows), default=math.inf),
            "angular_speed": max((float(row["angular_speed"]) for row in rows), default=math.inf),
        }
        return TrialOutcome(
            trial_id=trial_id,
            group=group,
            recovered=recovered and not hard_failure,
            hard_failure=hard_failure,
            failure_reason=failure_reason,
            recovery_time_s=recovery_time_s,
            max_position_error_m=maxima["position_error"],
            max_linear_speed_mps=maxima["linear_speed"],
            max_attitude_error_deg=math.degrees(maxima["attitude_error_rad"]),
            max_angular_speed_radps=maxima["angular_speed"],
            steady_rms=steady,
            impact_recoveries=tuple(impact_recoveries),
        ), rows, {
            "ball_contact_detected": ball_contact,
            "preimpact_stable": preimpact_stable,
            "terminal_hover_stable": terminal_stable,
        }
    finally:
        raw_env.end_fixed_evaluation()
        gym_env.reset()


def main() -> None:
    import gymnasium as gym
    import torch

    import flightlxx_isaaclab
    import flightlxx_isaaclab.tasks  # noqa: F401
    from fast_td3_utils import EmpiricalNormalization
    from flightlxx_isaaclab.fast_td3_models import HistoryActor, POLICY_RAW_DIM
    from flightlxx_isaaclab.preflight_protocol import load_preflight_manifest
    from flightlxx_isaaclab.preflight_report import (
        qualify_preflight,
        write_preflight_figures,
        write_preflight_report,
    )
    from flightlxx_isaaclab.preflight_runner import (
        completed_trial_ids,
        ensure_run_identity,
        expected_trial_ids,
        load_completed_outcomes,
        write_progress,
        write_trial_result,
    )
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_preflight_manifest(manifest_path)
    resolved_manifest_path = output_dir / "resolved_manifest.json"
    resolved_json = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if resolved_manifest_path.exists() and resolved_manifest_path.read_text(encoding="utf-8") != resolved_json:
        raise RuntimeError("existing output directory contains a different resolved manifest")
    resolved_manifest_path.write_text(resolved_json, encoding="utf-8")

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    saved_args = checkpoint["args"]
    actor = HistoryActor(
        n_obs=POLICY_RAW_DIM,
        n_act=4,
        num_envs=int(saved_args["num_envs"]),
        device=args.device,
        init_scale=float(saved_args.get("init_scale", 0.01)),
        hidden_dim=int(saved_args.get("actor_hidden_dim", 512)),
        std_min=float(saved_args.get("std_min", 0.05)),
        std_max=float(saved_args.get("std_max", 0.20)),
        sim_type=str(saved_args.get("sim_type", "")),
        sim_dimension=int(saved_args.get("sim_dimension", 64)),
        seq_len=int(saved_args.get("actor_seq_len", 8)),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    normalizer = EmpiricalNormalization(shape=POLICY_RAW_DIM, device=args.device)
    if checkpoint.get("obs_normalizer_state"):
        normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    normalizer.eval()

    cfg = parse_env_cfg(TASK_NAME, device=args.device, num_envs=1)
    cfg.seed = 1
    gym_env = gym.make(TASK_NAME, cfg=cfg)
    metadata = {
        "protocol_id": manifest.protocol_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("global_step", 0)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "resolved_manifest_sha256": hashlib.sha256(manifest.to_json().encode("utf-8")).hexdigest(),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "manifest_source": str(manifest_path),
        "code_commit": _git_commit(Path(flightlxx_isaaclab.__file__).parents[3]),
        "device": args.device,
        "task": TASK_NAME,
    }
    ensure_run_identity(output_dir, metadata)

    lookup: dict[str, tuple[str, Any]] = {}
    lookup.update({case.case_id: ("structured", case) for case in manifest.structured})
    lookup.update({case.case_id: ("randomized", case) for case in manifest.randomized})
    lookup.update({episode.episode_id: ("continuous", episode) for episode in manifest.continuous})
    lookup.update({case.case_id: ("balls", case) for case in manifest.balls})
    selected = list(expected_trial_ids(manifest))
    if args.smoke:
        selected = [manifest.structured[0].case_id, manifest.randomized[0].case_id, manifest.continuous[0].episode_id, manifest.balls[0].case_id]
    if args.trial_id:
        unknown = sorted(set(args.trial_id) - set(lookup))
        if unknown:
            raise ValueError(f"unknown trial IDs: {', '.join(unknown)}")
        selected = list(dict.fromkeys(args.trial_id))
    if not args.resume:
        overlap = completed_trial_ids(output_dir) & set(selected)
        if overlap:
            raise RuntimeError("completed results already exist and --no-resume was requested")

    completed = completed_trial_ids(output_dir)
    write_progress(output_dir, manifest, completed)
    try:
        for trial_id in selected:
            if args.resume and trial_id in completed:
                continue
            group, trial = lookup[trial_id]
            trial_dir = output_dir / "trials" / group / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            if group == "balls":
                protocol, events = _ball_protocol(manifest, trial, gym_env.unwrapped.step_dt)
                recovery_policy = "single"
                ball_case = trial
            else:
                protocol, events, recovery_policy = _trial_protocol(manifest, trial)
                ball_case = None
            outcome, rows, diagnostics = _run_rollout(
                gym_env,
                actor,
                normalizer,
                protocol,
                events,
                manifest,
                trial_id=trial_id,
                group=group,
                recovery_policy=recovery_policy,
                ball_case=ball_case,
            )
            _write_telemetry(trial_dir / "telemetry.csv", rows)
            write_trial_result(
                trial_dir,
                {
                    "schema_version": 1,
                    "status": "completed",
                    "case": asdict(trial),
                    "events": events,
                    "diagnostics": diagnostics,
                    "outcome": asdict(outcome),
                },
            )
            completed.add(trial_id)
            write_progress(output_dir, manifest, completed)
            print(
                f"[{len(completed):02d}/{manifest.episode_count}] {trial_id}: "
                f"{'PASS' if outcome.recovered and not outcome.hard_failure else 'FAIL'}",
                flush=True,
            )
    finally:
        gym_env.close()

    outcomes = load_completed_outcomes(output_dir)
    summary = qualify_preflight(manifest, outcomes)
    report_paths = write_preflight_report(summary, output_dir)
    figure_paths = write_preflight_figures(summary, output_dir)
    runner_result = {
        "stage": "completed",
        "status": summary.status,
        "qualified": summary.qualified,
        "completed_trials": len(outcomes),
        "total_trials": manifest.episode_count,
        "full_suite": len(outcomes) == manifest.episode_count,
        "report_paths": {
            **{name: str(path) for name, path in report_paths.items()},
            **{name: str(path) for name, path in figure_paths.items()},
        },
    }
    (output_dir / "runner_result.json").write_text(
        json.dumps(runner_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(runner_result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "runner_result.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    finally:
        simulation_app.close()
