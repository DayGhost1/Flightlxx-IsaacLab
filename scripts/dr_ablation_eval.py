"""Deterministic A/B curriculum-exam DR ablation for one checkpoint.

Step-0 diagnostic modes:
  - vicon_realistic: keep age/dropout/jitter, shrink pose noise to
    0.3--1 mm / 0.05--0.15 deg.
  - vicon_realistic_lpf: same plus a first-order low-pass on the
    bridge's published linear/angular velocity (tau=70 ms).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--dr-mode",
    choices=(
        "full",
        "nominal",
        "actuation",
        "vicon",
        "physical",
        "vicon_realistic",
        "vicon_realistic_lpf",
        "vicon_realistic_pose_only",
    ),
    required=True,
)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--velocity-lpf-tau-s", type=float, default=0.07)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

TASK_NAME = "Isaac-FlightLxx-CTBR-Recovery-Direct-v0"
LIMITS = {
    "position_error": 0.15,
    "linear_speed": 0.15,
    "attitude_error_rad": math.radians(5.0),
    "angular_speed": 0.25,
}
REALISTIC_POSITION_NOISE_M = (0.0003, 0.001)
REALISTIC_ATTITUDE_NOISE_RAD = (math.radians(0.05), math.radians(0.15))


def domain_config(raw_env, mode: str):
    from flightlxx_isaaclab.core.randomization import DomainRandomizationCfg

    full = DomainRandomizationCfg()
    nominal = DomainRandomizationCfg(
        mass_scale=(1.0, 1.0),
        inertia_scale=(1.0, 1.0),
        com_xy_m=0.0,
        com_z_m=0.0,
        thrust_scale=(1.0, 1.0),
        motor_scale=(1.0, 1.0),
        actuator_tau_s=(raw_env.cfg.nominal_actuator_tau_s,) * 2,
        action_delay_steps=(0, 0),
        battery_voltage_v=(16.0, 16.0),
        battery_internal_resistance_ohm=(0.035, 0.035),
        position_noise_std_m=(0.0, 0.0),
        velocity_noise_std_mps=(0.0, 0.0),
        attitude_noise_std_rad=(0.0, 0.0),
        gyro_noise_std_radps=(0.0, 0.0),
        gyro_bias_radps=(0.0, 0.0),
        vicon_measurement_age_s=(raw_env._platform.vicon.measurement_age_s,) * 2,
        vicon_dropout_probability=(0.0, 0.0),
    )
    if mode == "full":
        return full
    if mode == "nominal":
        return nominal
    if mode == "actuation":
        return replace(
            nominal,
            actuator_tau_s=full.actuator_tau_s,
            action_delay_steps=full.action_delay_steps,
        )
    if mode == "vicon":
        return replace(
            nominal,
            position_noise_std_m=full.position_noise_std_m,
            velocity_noise_std_mps=full.velocity_noise_std_mps,
            attitude_noise_std_rad=full.attitude_noise_std_rad,
            gyro_noise_std_radps=full.gyro_noise_std_radps,
            gyro_bias_radps=full.gyro_bias_radps,
            vicon_measurement_age_s=full.vicon_measurement_age_s,
            vicon_dropout_probability=full.vicon_dropout_probability,
        )
    if mode == "vicon_realistic_pose_only":
        # Only shrink pose noise; zero additive derived-state noise and fix age/dropout.
        return replace(
            nominal,
            position_noise_std_m=REALISTIC_POSITION_NOISE_M,
            attitude_noise_std_rad=REALISTIC_ATTITUDE_NOISE_RAD,
            vicon_measurement_age_s=full.vicon_measurement_age_s,  # keep age range
            vicon_dropout_probability=full.vicon_dropout_probability,
        )
    if mode in ("vicon_realistic", "vicon_realistic_lpf"):
        # Keep age/dropout and additive derived-state noise as-is so the
        # only changed variable is pose-noise magnitude.
        return replace(
            nominal,
            position_noise_std_m=REALISTIC_POSITION_NOISE_M,
            velocity_noise_std_mps=full.velocity_noise_std_mps,
            attitude_noise_std_rad=REALISTIC_ATTITUDE_NOISE_RAD,
            gyro_noise_std_radps=full.gyro_noise_std_radps,
            gyro_bias_radps=full.gyro_bias_radps,
            vicon_measurement_age_s=full.vicon_measurement_age_s,
            vicon_dropout_probability=full.vicon_dropout_probability,
        )
    return replace(
        nominal,
        mass_scale=full.mass_scale,
        inertia_scale=full.inertia_scale,
        com_xy_m=full.com_xy_m,
        com_z_m=full.com_z_m,
        thrust_scale=full.thrust_scale,
        motor_scale=full.motor_scale,
        battery_voltage_v=full.battery_voltage_v,
        battery_internal_resistance_ohm=full.battery_internal_resistance_ohm,
    )


def enable_velocity_low_pass(bridge, *, tau_s: float) -> None:
    """Patch VirtualViconBridge so derived velocities are low-pass filtered."""

    import torch

    if tau_s <= 0.0:
        raise ValueError("velocity LPF tau must be positive")
    bridge._velocity_lpf_tau_s = float(tau_s)
    bridge._filt_lin_vel = None
    bridge._filt_ang_vel = None
    original_observe = bridge.observe
    original_reset = bridge.reset

    def observe_filtered(*, now_s: float):
        output = original_observe(now_s=now_s)
        if output is None:
            return None
        # Environments whose hold was refreshed on this call.
        updated = torch.isclose(
            bridge._last_output_time,
            torch.full_like(bridge._last_output_time, float(now_s)),
            atol=1.0e-9,
            rtol=0.0,
        )
        lin = output[:, 3:6]
        ang = output[:, 10:13]
        if bridge._filt_lin_vel is None:
            bridge._filt_lin_vel = lin.clone()
            bridge._filt_ang_vel = ang.clone()
        else:
            alpha = 1.0 - math.exp(-bridge.output_period_s / bridge._velocity_lpf_tau_s)
            bridge._filt_lin_vel = torch.where(
                updated[:, None],
                (1.0 - alpha) * bridge._filt_lin_vel + alpha * lin,
                bridge._filt_lin_vel,
            )
            bridge._filt_ang_vel = torch.where(
                updated[:, None],
                (1.0 - alpha) * bridge._filt_ang_vel + alpha * ang,
                bridge._filt_ang_vel,
            )
        filtered = output.clone()
        filtered[:, 3:6] = bridge._filt_lin_vel
        filtered[:, 10:13] = bridge._filt_ang_vel
        # Keep the sample-and-hold buffer consistent with the filtered view.
        if bridge._last_output is not None:
            bridge._last_output.copy_(filtered)
        return filtered

    def reset_filtered(env_ids=None):
        original_reset(env_ids)
        if env_ids is None or bridge._filt_lin_vel is None:
            bridge._filt_lin_vel = None
            bridge._filt_ang_vel = None
            return
        bridge._filt_lin_vel[env_ids] = 0.0
        bridge._filt_ang_vel[env_ids] = 0.0

    bridge.observe = observe_filtered
    bridge.reset = reset_filtered


def mean_group_metric(papers: dict, group: str, key: str) -> float:
    values = [float(papers[paper][group][key]) for paper in ("A", "B")]
    return sum(values) / len(values)


def main():
    import gymnasium as gym
    import torch

    import flightlxx_isaaclab.tasks  # noqa: F401
    from fast_td3_utils import EmpiricalNormalization
    from flightlxx_isaaclab.core import assess_curriculum_exam
    from flightlxx_isaaclab.fast_td3_models import HistoryActor, POLICY_RAW_DIM
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    saved = checkpoint["args"]
    actor = HistoryActor(
        n_obs=POLICY_RAW_DIM,
        n_act=4,
        num_envs=args.num_envs,
        device=args.device,
        init_scale=float(saved.get("init_scale", 0.01)),
        hidden_dim=int(saved.get("actor_hidden_dim", 512)),
        std_min=float(saved.get("std_min", 0.0)),
        std_max=float(saved.get("std_max", 0.2)),
        sim_type=str(saved.get("sim_type", "")),
        sim_dimension=int(saved.get("sim_dimension", 64)),
        seq_len=int(saved.get("actor_seq_len", 8)),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    normalizer = EmpiricalNormalization(POLICY_RAW_DIM, args.device)
    normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    normalizer.eval()

    cfg = parse_env_cfg(TASK_NAME, device=args.device, num_envs=args.num_envs)
    cfg.seed = 1
    gym_env = gym.make(TASK_NAME, cfg=cfg)
    raw = gym_env.unwrapped
    raw.initialize_curriculum(0.0, retention_floor=0.0)
    raw._domain_cfg = domain_config(raw, args.dr_mode)
    if args.dr_mode == "vicon_realistic_lpf":
        enable_velocity_low_pass(raw._vicon, tau_s=args.velocity_lpf_tau_s)

    papers = {}
    for paper_index, paper_name in enumerate(("A", "B")):
        exam_seed = raw.begin_curriculum_evaluation(paper_index=paper_index)
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(exam_seed)
                observations, _ = gym_env.reset()
                groups = raw.curriculum_evaluation_groups()
                tail_steps = max(1, int(round(1.0 / raw.step_dt)))
                tail = {name: [] for name in LIMITS}
                crashed = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                for step_index in range(raw.max_episode_length):
                    with torch.no_grad():
                        actions = actor.explore(
                            normalizer(observations["policy"], update=False),
                            deterministic=True,
                        )
                    observations, _, _, _, _ = gym_env.step(actions.float())
                    metrics = raw.evaluation_step_metrics()
                    if step_index >= raw.max_episode_length - tail_steps:
                        for name in tail:
                            tail[name].append(metrics[name])
                    crashed |= metrics["failure"].bool()
                papers[paper_name] = assess_curriculum_exam(
                    {name: torch.stack(values) for name, values in tail.items()},
                    crashed=crashed,
                    groups=groups,
                    limits=LIMITS,
                )
                papers[paper_name]["exam_seed"] = exam_seed
        finally:
            raw.end_curriculum_evaluation()

    summary = {
        "hover_strict_success_rate": mean_group_metric(papers, "hover", "strict_success_rate"),
        "hover_coarse_success_rate": mean_group_metric(papers, "hover", "coarse_success_rate"),
        "hover_linear_speed_rms": mean_group_metric(papers, "hover", "linear_speed_rms"),
        "hover_angular_speed_rms": mean_group_metric(papers, "hover", "angular_speed_rms"),
        "hover_position_error_rms": mean_group_metric(papers, "hover", "position_error_rms"),
        "hover_attitude_error_rad_rms": mean_group_metric(
            papers, "hover", "attitude_error_rad_rms"
        ),
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("global_step", 0)),
        "dr_mode": args.dr_mode,
        "velocity_lpf_tau_s": (
            float(args.velocity_lpf_tau_s) if args.dr_mode == "vicon_realistic_lpf" else None
        ),
        "realistic_position_noise_std_m": REALISTIC_POSITION_NOISE_M,
        "realistic_attitude_noise_std_rad": REALISTIC_ATTITUDE_NOISE_RAD,
        "num_envs": args.num_envs,
        "summary": summary,
        "papers": papers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"dr_mode": args.dr_mode, **summary}, sort_keys=True), flush=True)
    gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
