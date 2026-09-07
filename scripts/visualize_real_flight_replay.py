"""Evaluate a policy in nominal and measured real-flight handoff conditions."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--speed", type=float, default=0.6)
parser.add_argument("--warmup_updates", type=int, default=120)
parser.add_argument("--expected_sha256", default=None)
parser.add_argument("--exit_after_run", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration <= 0.0 or args.speed <= 0.0:
    parser.error("--duration and --speed must be positive")

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import omni.timeline  # noqa: E402
if args.headless:
    ui = None
else:
    import omni.ui as ui  # noqa: E402

import flightlxx_isaaclab.tasks  # noqa: E402,F401
from flightlxx_isaaclab.evaluation import FixedImpact, FixedImpactProtocol  # noqa: E402
from flightlxx_isaaclab.ppo import load_ppo_policy
from flightlxx_isaaclab.real_flight_replay import (  # noqa: E402
    DEPLOYED_50K_SHA256,
    ObservationDelay,
    REAL_FLIGHT_BASELINE,
    ReplayMetrics,
    replay_scenarios,
    reset_replay_control_state,
    verify_checkpoint,
)
from flightlxx_isaaclab.core.math import quat_error, quat_rotate_inverse  # noqa: E402
from flightlxx_isaaclab.visual_playback import configure_visual_scene, transform_body_points  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz  # noqa: E402


TASK_NAME = "Isaac-FlightLxx-CTBR-Recovery-Direct-v0"


def material(color, opacity=1.0):
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=tuple(0.65 * value for value in color),
        opacity=opacity,
        roughness=0.8,
    )


class ReplayVisuals:
    def __init__(self, raw_env):
        self.raw_env = raw_env
        airframe_cfg = VisualizationMarkersCfg(
            prim_path="/World/RealFlightReplay/Airframe",
            markers={
                "arm_x": sim_utils.CuboidCfg(size=(0.86, 0.06, 0.035), visual_material=material((0.05, 0.85, 1.0))),
                "arm_y": sim_utils.CuboidCfg(size=(0.06, 0.86, 0.035), visual_material=material((1.0, 0.38, 0.05))),
                "nose": sim_utils.CuboidCfg(size=(0.15, 0.14, 0.08), visual_material=material((0.20, 1.0, 0.28))),
            },
        )
        trail_cfg = VisualizationMarkersCfg(
            prim_path="/World/RealFlightReplay/Trail",
            markers={"point": sim_utils.SphereCfg(radius=0.045, visual_material=material((1.0, 0.82, 0.05), 0.82))},
        )
        goal_cfg = VisualizationMarkersCfg(
            prim_path="/World/RealFlightReplay/Goal",
            markers={"goal": sim_utils.SphereCfg(radius=0.09, visual_material=material((0.20, 1.0, 0.25), 0.65))},
        )
        self.airframe = VisualizationMarkers(airframe_cfg)
        self.trail = VisualizationMarkers(trail_cfg)
        self.goal = VisualizationMarkers(goal_cfg)
        self.offsets = torch.tensor(
            ((0.0, 0.0, 0.06), (0.0, 0.0, 0.08), (0.35, 0.0, 0.11)),
            device=raw_env.device,
        )
        self.points: list[torch.Tensor] = []

    def reset(self):
        self.points.clear()
        self.trail.set_visibility(False)
        self.goal.visualize(translations=self.raw_env._target_position[0:1])
        self.update(0)

    def update(self, step: int):
        position = self.raw_env._robot.data.root_pos_w[0:1]
        quaternion = self.raw_env._robot.data.root_quat_w[0:1]
        points = transform_body_points(position, quaternion, self.offsets)[0]
        self.airframe.visualize(
            translations=points,
            orientations=quaternion.expand(3, -1),
            marker_indices=[0, 1, 2],
        )
        if step % 5 == 0:
            self.points.append(position[0].detach().cpu().clone())
            self.trail.set_visibility(True)
            self.trail.visualize(translations=torch.stack(self.points).to(self.raw_env.device))


class StatusWindow:
    def __init__(self):
        self.window = ui.Window("50k Real-Flight Replay", width=430, height=250, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=10, style={"margin": 12}):
                ui.Label("FastTD3 50k — no-impact handoff test", height=28)
                self.scenario = ui.Label("Loading...", height=42, word_wrap=True)
                self.details = ui.Label("", height=120, word_wrap=True)

    def set(self, scenario: str, details: str):
        self.scenario.text = scenario
        self.details.text = details

    def close(self):
        self.window.visible = False
        self.window.destroy()


class DirectAdapter:
    def __init__(self, env):
        self.envs = env

    def reset(self):
        observations, _ = self.envs.reset()
        return observations["policy"]

    def step(self, actions):
        observations, rewards, terminated, truncated, info = self.envs.step(actions)
        return observations["policy"], rewards, terminated | truncated, info


def load_actor(checkpoint_path: Path, device: str):
    return load_ppo_policy(checkpoint_path, device)


def no_impact_protocol(duration_s: float) -> FixedImpactProtocol:
    return FixedImpactProtocol(
        protocol_id="real_flight_no_impact_v1",
        total_duration_s=duration_s,
        recovery_dwell_s=0.5,
        thresholds={
            "position_error": 0.10,
            "linear_speed": 0.10,
            "attitude_error_rad": math.radians(3.0),
            "angular_speed": 0.10,
        },
        impacts=(
            FixedImpact(
                impact_id="disabled",
                trigger_time_s=duration_s + 100.0,
                duration_s=0.01,
                application_point_b=torch.zeros(3),
                force_b=torch.zeros(3),
            ),
        ),
    )


def scalar_metrics(raw_env):
    metrics = raw_env.evaluation_step_metrics()
    result = {
        key: float(metrics[key][0].detach().item())
        for key in ("position_error", "attitude_error_rad", "angular_speed", "linear_speed")
    }
    result["failure"] = bool(metrics["failure"][0].detach().item())
    return result


def set_initial_state(raw_env, scenario):
    device = raw_env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    angles = torch.deg2rad(torch.tensor([scenario.initial_euler_xyz_deg], device=device))
    quaternion = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    target_angles = torch.deg2rad(
        torch.tensor([scenario.target_euler_xyz_deg], device=device)
    )
    target_quaternion = quat_from_euler_xyz(
        target_angles[:, 0], target_angles[:, 1], target_angles[:, 2]
    )
    target_position = raw_env.scene.env_origins[env_ids] + torch.tensor(
        [scenario.target_position_vicon_m], device=device
    )
    initial_position = raw_env.scene.env_origins[env_ids] + torch.tensor(
        [scenario.initial_position_vicon_m], device=device
    )
    raw_env._target_position[env_ids] = target_position
    raw_env._target_quat[env_ids] = target_quaternion
    pose = torch.cat((initial_position, quaternion), dim=-1)
    body_rate = torch.tensor([scenario.initial_body_rate], device=device)
    angular_velocity_w = quat_apply(quaternion, body_rate)
    linear_velocity_w = torch.tensor([scenario.initial_linear_velocity_w], device=device)
    velocity = torch.cat((linear_velocity_w, angular_velocity_w), dim=-1)
    raw_env._robot.write_root_pose_to_sim(pose, env_ids)
    raw_env._robot.write_root_velocity_to_sim(velocity, env_ids)
    raw_env._actions[env_ids] = 0.0
    raw_env._previous_actions[env_ids] = 0.0
    reset_replay_control_state(raw_env, env_ids)
    # The first inference must see the measured handoff velocity/rate instead
    # of the hover sample that DirectRLEnv wrote during reset.  Subsequent
    # observations again come from the normal virtual-Vicon path.
    raw_env._vicon.reset()
    initial_state = torch.cat(
        (
            quat_rotate_inverse(quaternion, initial_position - target_position),
            quat_rotate_inverse(quaternion, linear_velocity_w),
            quat_error(target_quaternion, quaternion),
            body_rate,
        ),
        dim=-1,
    )
    initial_feature = torch.cat((initial_state, raw_env._actions[env_ids]), dim=-1)
    raw_env._history.reset(env_ids, initial_feature)
    fast = raw_env._history.latest(raw_env.cfg.fast_history).flatten(1)
    slow = raw_env._history.buffer.flatten(1)
    return torch.cat((initial_state, fast, slow), dim=-1)


def main():
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha = verify_checkpoint(checkpoint, expected_sha256=args.expected_sha256)
    cfg = parse_env_cfg(TASK_NAME, device=args.device, num_envs=1)
    cfg.seed = 1
    configure_visual_scene(cfg)
    gym_env = gym.make(TASK_NAME, cfg=cfg, render_mode=None if args.headless else "human")
    env = DirectAdapter(gym_env)
    raw_env = gym_env.unwrapped
    actor, normalizer, checkpoint_step = load_actor(checkpoint, args.device)
    protocol = no_impact_protocol(args.duration)
    timeline = omni.timeline.get_timeline_interface()
    status = None if args.headless else StatusWindow()
    visuals = None if args.headless else ReplayVisuals(raw_env)

    if not args.headless:
        timeline.pause()
        for _ in range(args.warmup_updates):
            if not simulation_app.is_running():
                return
            simulation_app.update()
            time.sleep(0.01)

    results = []
    try:
        for scenario_index, scenario in enumerate(replay_scenarios(raw_env.step_dt, args.duration), start=1):
            raw_env.begin_fixed_evaluation(protocol)
            observations = env.reset()
            observations = set_initial_state(raw_env, scenario)
            delay = ObservationDelay(scenario.observation_delay_steps)
            delayed_observations = delay.reset(observations)
            metrics = ReplayMetrics(step_dt=raw_env.step_dt)
            if visuals is not None:
                timeline.play()
                visuals.reset()
                status.set(
                    f"Scene {scenario_index}/2: {scenario.name}",
                    f"feedback delay = {scenario.observation_delay_steps * raw_env.step_dt * 1000:.0f} ms\n"
                    "green sphere = hover target; yellow dots = trajectory",
                )
            wall_start = time.perf_counter()
            total_steps = math.ceil(scenario.duration_s / raw_env.step_dt)
            for step in range(total_steps):
                with torch.no_grad():
                    action = actor.explore(normalizer(delayed_observations, update=False), deterministic=True)
                observations, _, dones, _ = env.step(action.float())
                delayed_observations = delay.push(observations)
                state = scalar_metrics(raw_env)
                metrics.record(action, state)
                if visuals is not None:
                    target_wall = wall_start + (step + 1) * raw_env.step_dt / args.speed
                    wait = target_wall - time.perf_counter()
                    if wait > 0.0:
                        time.sleep(wait)
                    visuals.update(step)
                if bool(dones[0].item()):
                    break
            summary = {
                "scenario": scenario.name,
                "observation_delay_ms": scenario.observation_delay_steps * raw_env.step_dt * 1000.0,
                **metrics.summary(),
            }
            results.append(summary)
            print({"scene_complete": summary}, flush=True)
            raw_env.end_fixed_evaluation()
            if status is not None:
                status.set(
                    f"Completed: {scenario.name}",
                    f"first action={summary['first_normalized_action']}\n"
                    f"rate saturation={100.0 * summary['any_rate_saturation_fraction']:.1f}%\n"
                    f"max attitude={math.degrees(summary['max_attitude_error_rad']):.1f} deg, "
                    f"max rate={summary['max_angular_speed_rad_s']:.2f} rad/s",
                )
                pause_until = time.perf_counter() + 1.5
                while simulation_app.is_running() and time.perf_counter() < pause_until:
                    simulation_app.update()
                    time.sleep(0.01)

        payload = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_step,
            "checkpoint_sha256": checkpoint_sha,
            "expected_checkpoint_sha256": args.expected_sha256,
            "real_flight_baseline": REAL_FLIGHT_BASELINE,
            "scenarios": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print({"replay_complete": str(args.output), "results": results}, flush=True)
        if status is not None:
            status.set("Both scenes complete", f"Results saved to:\n{args.output}\nWindow remains open for inspection.")
        if not args.exit_after_run and not args.headless:
            timeline.pause()
            while simulation_app.is_running():
                simulation_app.update()
                time.sleep(0.02)
    finally:
        if status is not None:
            status.close()
        gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
