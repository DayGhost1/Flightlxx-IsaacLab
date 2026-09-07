"""Evaluate one PPO checkpoint with the deterministic five-impact protocol."""

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
    from flightlxx_isaaclab.fixed_impact_evaluation import evaluate_fixed_five_impacts
    from flightlxx_isaaclab.evaluation import load_fixed_protocol, protocol_for_impact_level
    from flightlxx_isaaclab.ppo import load_ppo_policy
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
    actor, normalizer, checkpoint_step = load_ppo_policy(checkpoint_path, args.device)

    cfg = parse_env_cfg(TASK_NAME, device=args.device, num_envs=1)
    cfg.seed = 1
    gym_env = gym.make(TASK_NAME, cfg=cfg)
    env = DirectGymPolicyAdapter(gym_env)
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
    summary = {
        "stage": "completed",
        "pass": True,
        "checkpoint_passed": bool(result["passed"]),
        "checkpoint_step": checkpoint_step,
        "impact_level": args.impact_level,
        "force_scale": protocol.force_scale,
        "recovered_count": int(result["recovered_count"]),
        "crashed": bool(result["crashed"]),
        "max_position_error": float(result["max_position_error"]),
        "max_attitude_error_rad": float(result["max_attitude_error_rad"]),
        "max_linear_speed": float(result["max_linear_speed"]),
        "max_angular_speed": float(result["max_angular_speed"]),
        "evaluation_json": str(result["json_path"]),
        "timeseries_csv": str(result["timeseries_path"]),
        "response_figure": None,
        "report": None,
        "report_warning": None,
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
