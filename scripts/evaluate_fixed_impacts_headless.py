"""Evaluate one FastTD3 checkpoint with the deterministic five-impact protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--impact_level", choices=("small", "medium", "large"), default="small")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

TASK_NAME = "Isaac-FlightLxx-CTBR-Recovery-Direct-v0"


def write_runner_result(payload: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runner_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def safe_print(payload: dict) -> None:
    """Do not turn a completed evaluation into a failure if its SSH pipe closes."""

    try:
        print(payload, flush=True)
    except BrokenPipeError:
        pass


def main() -> None:
    import gymnasium as gym
    import torch

    import flightlxx_isaaclab
    import flightlxx_isaaclab.tasks  # noqa: F401
    from fast_td3_utils import EmpiricalNormalization
    from fixed_impact_evaluation import evaluate_fixed_five_impacts
    from flightlxx_isaaclab.evaluation import load_fixed_protocol, protocol_for_impact_level
    from flightlxx_isaaclab.fast_td3_models import HistoryActor, POLICY_RAW_DIM
    from flightlxx_isaaclab.visual_report import write_visual_evaluation_report
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    class DirectGymPolicyAdapter:
        def __init__(self, env):
            self.envs = env

        def reset(self, random_start_init: bool = False):
            del random_start_init
            observations, _ = self.envs.reset()
            return observations["policy"]

        def step(self, actions):
            observations, rewards, terminated, truncated, info = self.envs.step(actions)
            return observations["policy"], rewards, terminated | truncated, info

    checkpoint_path = args.checkpoint.expanduser().resolve()
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
    env = DirectGymPolicyAdapter(gym_env)
    checkpoint_step = int(checkpoint.get("global_step", 0))
    base_protocol = load_fixed_protocol(
        Path(flightlxx_isaaclab.__file__).parent / "config" / "fixed_five_impacts.json"
    )
    protocol = protocol_for_impact_level(base_protocol, args.impact_level)
    result = evaluate_fixed_five_impacts(
        env,
        actor,
        normalizer,
        checkpoint_step,
        args.output_dir,
        protocol=protocol,
    )
    result["checkpoint_step"] = checkpoint_step
    independent = result["modes"]["independent"]
    sequential = result["modes"]["sequential"]
    report_paths = None
    report_warning = None
    try:
        report_paths = write_visual_evaluation_report(result, protocol, args.output_dir)
    except ValueError as exc:
        if "does not contain telemetry in the five impact windows" not in str(exc):
            raise
        # A policy can fail before the first scheduled impact.  The JSON and
        # full time-series are still valid artifacts; there is simply no
        # impact-aligned window from which to build the usual figures.
        report_warning = str(exc)
    summary = {
        "stage": "completed",
        "pass": True,
        "checkpoint_passed": bool(result["passed"]),
        "checkpoint_step": checkpoint_step,
        "impact_level": args.impact_level,
        "force_scale": protocol.force_scale,
        "recovered_count": int(result["recovered_count"]),
        "independent_passed": bool(independent["passed"]),
        "independent_recovered_count": int(independent["recovered_count"]),
        "independent_crashed": bool(independent["crashed"]),
        "sequential_passed": bool(sequential["passed"]),
        "sequential_recovered_count": int(sequential["recovered_count"]),
        "sequential_crashed": bool(sequential["crashed"]),
        "crashed": bool(result["crashed"]),
        "max_position_error": float(result["max_position_error"]),
        "max_attitude_error_rad": float(result["max_attitude_error_rad"]),
        "max_linear_speed": float(result["max_linear_speed"]),
        "max_angular_speed": float(result["max_angular_speed"]),
        "evaluation_json": str(result["json_path"]),
        "timeseries_csv": str(result["timeseries_path"]),
        "response_figure": None if report_paths is None else str(report_paths.response_figure),
        "report": None if report_paths is None else str(report_paths.markdown),
        "report_warning": report_warning,
    }
    write_runner_result(summary)
    safe_print(summary)
    gym_env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        failure = {
            "stage": "failed",
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_runner_result(failure)
        safe_print(failure)
        raise
    finally:
        app.close()
