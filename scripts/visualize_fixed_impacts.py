"""Replay the deterministic five-impact evaluation in the Isaac Sim GUI."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, default=None)
parser.add_argument("--impact_level", choices=("small", "medium", "large"), default="small")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed relative to real time")
parser.add_argument("--warmup_updates", type=int, default=180, help="Renderer updates before enabling Start")
parser.add_argument("--auto_start", action="store_true", help="Start once after warmup instead of waiting for the GUI button")
parser.add_argument("--exit_after_run", action="store_true", help="Exit cleanly after one completed replay")
parser.add_argument("--no_open_report", action="store_true", help="Do not automatically open the response figure")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.speed <= 0.0:
    parser.error("--speed must be positive")
if args.warmup_updates < 1:
    parser.error("--warmup_updates must be positive")
if args.headless:
    parser.error("This visualizer requires GUI mode; omit --headless")

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import omni.timeline  # noqa: E402
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import flightlxx_isaaclab  # noqa: E402
import flightlxx_isaaclab.tasks  # noqa: E402,F401
from fast_td3_utils import EmpiricalNormalization  # noqa: E402
from fixed_impact_evaluation import evaluate_fixed_five_impacts  # noqa: E402
from flightlxx_isaaclab.evaluation import load_fixed_protocol, protocol_for_impact_level  # noqa: E402
from flightlxx_isaaclab.fast_td3_models import HistoryActor, POLICY_RAW_DIM  # noqa: E402
from flightlxx_isaaclab.visual_playback import (  # noqa: E402
    PlaybackState,
    configure_visual_scene,
    quaternion_from_x_axis,
    transform_body_points,
)
from flightlxx_isaaclab.visual_report import write_visual_evaluation_report  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402


TASK_NAME = "Isaac-FlightLxx-CTBR-Recovery-Direct-v0"


def vivid_material(color, *, opacity=1.0):
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=tuple(0.75 * component for component in color),
        roughness=0.8,
        opacity=opacity,
    )


class DirectGymPolicyAdapter:
    """Expose a one-environment DirectRLEnv through the FastTD3 eval API."""

    def __init__(self, env):
        self.envs = env

    def reset(self, random_start_init: bool = False):
        del random_start_init
        observations, _ = self.envs.reset()
        return observations["policy"]

    def step(self, actions):
        observations, rewards, terminated, truncated, info = self.envs.step(actions)
        return observations["policy"], rewards, terminated | truncated, info


class ReplayControlWindow:
    """Small in-simulator control panel for starting and replaying the rollout."""

    def __init__(self, state: PlaybackState, on_start, on_open_report):
        self.state = state
        self.window = ui.Window("FlightLxx Five-Impact Replay", width=400, height=365, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=10, style={"margin": 12}):
                ui.Label("FastTD3 — fixed five-impact recovery", height=28)
                self.status_label = ui.Label(state.status, word_wrap=True, height=62)
                self.detail_label = ui.Label(
                    "Wide fixed world view. The high-visibility shell is enlarged; red arrow is force and yellow dots are trajectory.",
                    word_wrap=True,
                    height=72,
                )
                self.start_button = ui.Button("Start / Replay", height=42, clicked_fn=on_start)
                self.start_button.enabled = False
                self.report_button = ui.Button("Open response report", height=36, clicked_fn=on_open_report)
                self.report_button.enabled = False

    def dock_to_property_panel(self) -> None:
        property_window = ui.Workspace.get_window("Property")
        if property_window is not None:
            self.window.dock_in(property_window, ui.DockPosition.SAME, 1.0)
            self.window.focus()

    def sync(self) -> None:
        self.status_label.text = self.state.status
        self.start_button.enabled = self.state.phase in {"ready", "complete"}

    def show_detail(self, text: str) -> None:
        self.detail_label.text = text

    def enable_report(self) -> None:
        self.report_button.enabled = True

    def destroy(self) -> None:
        self.window.visible = False
        self.window.destroy()


def add_scene_references() -> None:
    """Add static, non-physical hover references and a brighter key light."""

    reference_material = vivid_material((0.20, 1.0, 0.28), opacity=0.60)
    hover_axis_cfg = sim_utils.CylinderCfg(
        radius=0.018,
        height=4.70,
        visual_material=reference_material,
    )
    hover_axis_cfg.func(
        "/World/ReplayMarkers/HoverAxis",
        hover_axis_cfg,
        translation=(0.0, 0.0, 2.35),
    )
    ground_target_cfg = sim_utils.CylinderCfg(
        radius=0.30,
        height=0.015,
        visual_material=reference_material,
    )
    ground_target_cfg.func(
        "/World/ReplayMarkers/GroundTarget",
        ground_target_cfg,
        translation=(0.0, 0.0, 0.012),
    )
    key_light_cfg = sim_utils.DistantLightCfg(
        color=(1.0, 0.97, 0.90),
        intensity=3500.0,
        angle=0.6,
    )
    key_light_cfg.func("/World/ReplayKeyLight", key_light_cfg)


class PlaybackVisuals:
    """Visualization-only markers driven by the true evaluation state."""

    def __init__(self, raw_env):
        self.raw_env = raw_env
        airframe_cfg = VisualizationMarkersCfg(
            prim_path="/World/ReplayMarkers/Airframe",
            markers={
                "arm_x": sim_utils.CuboidCfg(
                    size=(0.86, 0.060, 0.032),
                    visual_material=vivid_material((0.05, 0.85, 1.0)),
                ),
                "arm_y": sim_utils.CuboidCfg(
                    size=(0.060, 0.86, 0.032),
                    visual_material=vivid_material((1.0, 0.38, 0.05)),
                ),
                "nose": sim_utils.CuboidCfg(
                    size=(0.130, 0.140, 0.075),
                    visual_material=vivid_material((0.20, 1.0, 0.28)),
                ),
                "rotor_front": sim_utils.CylinderCfg(
                    radius=0.120,
                    height=0.014,
                    visual_material=vivid_material((0.20, 1.0, 0.28), opacity=0.90),
                ),
                "rotor_rear": sim_utils.CylinderCfg(
                    radius=0.120,
                    height=0.014,
                    visual_material=vivid_material((1.0, 0.20, 0.12), opacity=0.90),
                ),
                "rotor_left": sim_utils.CylinderCfg(
                    radius=0.120,
                    height=0.014,
                    visual_material=vivid_material((0.05, 0.85, 1.0), opacity=0.90),
                ),
                "rotor_right": sim_utils.CylinderCfg(
                    radius=0.120,
                    height=0.014,
                    visual_material=vivid_material((1.0, 0.62, 0.05), opacity=0.90),
                ),
            },
        )
        goal_cfg = VisualizationMarkersCfg(
            prim_path="/World/ReplayMarkers/Goal",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.25, 1.0, 0.25),
                        emissive_color=(0.18, 0.75, 0.18),
                        opacity=0.65,
                    ),
                )
            },
        )
        trail_cfg = VisualizationMarkersCfg(
            prim_path="/World/ReplayMarkers/Trajectory",
            markers={
                "point": sim_utils.SphereCfg(
                    radius=0.050,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.82, 0.05),
                        emissive_color=(0.80, 0.60, 0.02),
                        opacity=0.82,
                    ),
                )
            },
        )
        force_cfg = RED_ARROW_X_MARKER_CFG.copy()
        force_cfg.prim_path = "/World/ReplayMarkers/AppliedForce"
        force_cfg.markers["arrow"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        force_cfg.markers["arrow"].visual_material.emissive_color = (1.0, 0.05, 0.05)
        self.airframe = VisualizationMarkers(airframe_cfg)
        self.goal = VisualizationMarkers(goal_cfg)
        self.trail = VisualizationMarkers(trail_cfg)
        self.force = VisualizationMarkers(force_cfg)
        self._airframe_offsets_b = torch.tensor(
            (
                (0.0, 0.0, 0.055),
                (0.0, 0.0, 0.075),
                (0.350, 0.0, 0.110),
                (0.420, 0.0, 0.080),
                (-0.420, 0.0, 0.080),
                (0.0, 0.420, 0.080),
                (0.0, -0.420, 0.080),
            ),
            device=raw_env.device,
        )
        self._airframe_indices = list(range(len(self._airframe_offsets_b)))
        self.goal.visualize(translations=raw_env._target_position[0:1])
        self._trail_positions: list[torch.Tensor] = []
        self._next_trail_time_s = 0.0
        self.trail.set_visibility(False)
        self.force.set_visibility(False)

    def reset(self) -> None:
        self._trail_positions.clear()
        self._next_trail_time_s = 0.0
        self.trail.set_visibility(False)
        self.force.set_visibility(False)
        self._update_airframe()

    def _update_airframe(self) -> None:
        robot_position = self.raw_env._robot.data.root_pos_w[0:1]
        robot_quaternion = self.raw_env._robot.data.root_quat_w[0:1]
        positions_w = transform_body_points(
            robot_position,
            robot_quaternion,
            self._airframe_offsets_b,
        )[0]
        orientations_w = robot_quaternion.expand(len(self._airframe_offsets_b), -1)
        self.airframe.visualize(
            translations=positions_w,
            orientations=orientations_w,
            marker_indices=self._airframe_indices,
        )

    def update(self, time_s: float, protocol) -> None:
        self._update_airframe()
        robot_position = self.raw_env._robot.data.root_pos_w[0:1]
        if time_s + 1.0e-9 >= self._next_trail_time_s:
            self._trail_positions.append(robot_position[0].detach().cpu().clone())
            self._next_trail_time_s += 0.10
            self.trail.set_visibility(True)
            self.trail.visualize(translations=torch.stack(self._trail_positions))

        active_impact = next(
            (
                impact
                for impact in protocol.impacts
                if impact.trigger_time_s <= time_s < impact.end_time_s
            ),
            None,
        )
        if active_impact is None:
            self.force.set_visibility(False)
            return

        quaternion_w = self.raw_env._robot.data.root_quat_w[0:1]
        force_b = active_impact.force_b.to(self.raw_env.device).unsqueeze(0)
        point_b = active_impact.application_point_b.to(self.raw_env.device).unsqueeze(0)
        force_w = quat_apply(quaternion_w, force_b)
        application_point_w = robot_position + quat_apply(quaternion_w, point_b)
        orientation_w = quaternion_from_x_axis(force_w)
        arrow_length = 0.17 * torch.linalg.vector_norm(force_w, dim=-1, keepdim=True)
        arrow_scale = torch.cat(
            (arrow_length.clamp(min=0.65, max=1.35), torch.full_like(arrow_length, 1.20), torch.full_like(arrow_length, 1.20)),
            dim=-1,
        )
        self.force.set_visibility(True)
        self.force.visualize(
            translations=application_point_w,
            orientations=orientation_w,
            scales=arrow_scale,
        )


def open_report(path: Path | None) -> None:
    if path is None or not path.is_file():
        return
    subprocess.Popen(
        ["xdg-open", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def load_actor_and_normalizer(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint["args"]
    actor = HistoryActor(
        n_obs=POLICY_RAW_DIM,
        n_act=4,
        num_envs=int(saved_args["num_envs"]),
        device=device,
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

    normalizer = EmpiricalNormalization(shape=POLICY_RAW_DIM, device=device)
    normalizer_state = checkpoint.get("obs_normalizer_state")
    if normalizer_state:
        normalizer.load_state_dict(normalizer_state)
    normalizer.eval()
    return actor, normalizer, int(checkpoint.get("global_step", 0))


def main() -> None:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = parse_env_cfg(TASK_NAME, device=args.device, num_envs=1)
    cfg.seed = 1
    cfg.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.70, 0.72, 0.76),
        emissive_color=(0.10, 0.10, 0.10),
        roughness=0.9,
    )
    configure_visual_scene(cfg)
    gym_env = gym.make(TASK_NAME, cfg=cfg, render_mode="human")
    env = DirectGymPolicyAdapter(gym_env)
    actor, normalizer, checkpoint_step = load_actor_and_normalizer(checkpoint_path, args.device)

    package_root = Path(flightlxx_isaaclab.__file__).parent
    base_protocol = load_fixed_protocol(package_root / "config" / "fixed_five_impacts.json")
    protocol = protocol_for_impact_level(base_protocol, args.impact_level)
    output_dir = args.output_dir or checkpoint_path.parent.parent / "visual_replay"
    raw_env = gym_env.unwrapped
    timeline = omni.timeline.get_timeline_interface()
    state = PlaybackState()
    latest_report: Path | None = None

    def handle_open_report() -> None:
        open_report(latest_report)

    controls = ReplayControlWindow(state, state.request_start, handle_open_report)

    def prime_hover_scene() -> None:
        timeline.play()
        raw_env.begin_fixed_evaluation(protocol)
        env.reset(random_start_init=False)
        timeline.pause()
        raw_env.viewport_camera_controller.update_view_to_world()

    timeline.pause()
    add_scene_references()
    visuals = PlaybackVisuals(raw_env)
    prime_hover_scene()
    visuals.reset()

    print("Loading task, checkpoint, camera, and renderer...", flush=True)
    for _ in range(args.warmup_updates):
        if not simulation_app.is_running():
            return
        simulation_app.update()
        time.sleep(0.01)
    controls.dock_to_property_panel()
    simulation_settings = ui.Workspace.get_window("Simulation Settings")
    if simulation_settings is not None:
        simulation_settings.visible = False
    for _ in range(10):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(cfg.viewer.cam_prim_path)
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    camera_translation = None
    camera_forward = None
    robot_bounds = None
    if camera_prim.IsValid():
        camera_transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_translation = tuple(camera_transform.ExtractTranslation())
        camera_forward = tuple(camera_transform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized())
    if robot_prim.IsValid():
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        aligned_range = cache.ComputeWorldBound(robot_prim).ComputeAlignedRange()
        robot_bounds = {
            "min": tuple(aligned_range.GetMin()),
            "max": tuple(aligned_range.GetMax()),
        }
    print(
        {
            "robot_root_position": raw_env._robot.data.root_pos_w[0].detach().cpu().tolist(),
            "viewer_origin_type": raw_env.viewport_camera_controller.cfg.origin_type,
            "camera_eye": raw_env.viewport_camera_controller.default_cam_eye.tolist(),
            "camera_lookat": raw_env.viewport_camera_controller.default_cam_lookat.tolist(),
            "camera_translation": camera_translation,
            "camera_forward": camera_forward,
            "robot_prim_valid": robot_prim.IsValid(),
            "robot_visibility": UsdGeom.Imageable(robot_prim).ComputeVisibility() if robot_prim.IsValid() else None,
            "robot_world_bounds": robot_bounds,
            "goal_marker_valid": stage.GetPrimAtPath("/World/ReplayMarkers/Goal").IsValid(),
            "fixed_protocol": protocol.protocol_id,
            "impact_level": protocol.impact_level,
            "force_scale": protocol.force_scale,
        },
        flush=True,
    )
    state.mark_ready()
    controls.sync()
    if args.auto_start:
        state.request_start()

    try:
        print(
            f"Scene ready: checkpoint step {checkpoint_step}. Click Start / Replay in Isaac Sim. "
            f"Each rollout is {protocol.total_duration_s:.1f}s at {args.speed:.2f}x speed.",
            flush=True,
        )
        while simulation_app.is_running():
            if not state.consume_start_request():
                simulation_app.update()
                time.sleep(0.01)
                continue

            state.mark_running()
            controls.sync()
            timeline.play()
            visuals.reset()
            wall_start = time.perf_counter()
            announced_impacts: set[str] = set()

            def show_frame(frame):
                if not simulation_app.is_running():
                    raise KeyboardInterrupt
                time_s = float(frame["time_s"])
                target_wall_time = wall_start + time_s / args.speed
                delay = target_wall_time - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                visuals.update(time_s, protocol)

                for impact_index, impact in enumerate(protocol.impacts, start=1):
                    if time_s >= impact.trigger_time_s and impact.impact_id not in announced_impacts:
                        announced_impacts.add(impact.impact_id)
                        detail = (
                            f"Impact {impact_index}/5: {impact.impact_id}\n"
                            f"force_b={impact.force_b.tolist()} N, point_b={impact.application_point_b.tolist()} m"
                        )
                        controls.show_detail(detail)
                        print(f"[impact] t={time_s:5.2f}s  {detail.replace(chr(10), '  ')}", flush=True)
                for record in frame["completed_impacts"]:
                    recovery_time = record["recovery_time_s"]
                    recovery_text = "not recovered" if recovery_time is None else f"recovered in {recovery_time:.2f} s"
                    controls.show_detail(
                        f"{record['impact_id']}: {recovery_text}"
                    )
                    print(
                        f"[recovery] {record['impact_id']}  recovered={record['recovered']}  "
                        f"recovery_time={record['recovery_time_s']}",
                        flush=True,
                    )

            result = evaluate_fixed_five_impacts(
                env,
                actor,
                lambda observations, update=False: normalizer(observations, update=update),
                checkpoint_step,
                output_dir,
                on_step=show_frame,
                protocol=protocol,
            )
            result["checkpoint_step"] = checkpoint_step
            independent = result["modes"]["independent"]
            sequential = result["modes"]["sequential"]
            report_paths = write_visual_evaluation_report(result, protocol, output_dir)
            latest_report = report_paths.response_figure
            state.mark_complete(
                passed=result["passed"],
                recovered_count=min(
                    int(independent["recovered_count"]),
                    int(sequential["recovered_count"]),
                ),
            )
            controls.show_detail(
                f"independent={independent['recovered_count']}/5, "
                f"sequential={sequential['recovered_count']}/5\n"
                f"crashed={result['crashed']}, max position error={result['max_position_error']:.3f} m\n"
                f"report={report_paths.response_figure}"
            )
            controls.enable_report()
            print(
                f"[result] passed={result['passed']} "
                f"independent={independent['recovered_count']}/5 "
                f"sequential={sequential['recovered_count']}/5 "
                f"crashed={result['crashed']} max_position_error={result['max_position_error']:.3f} m",
                flush=True,
            )
            print(f"[report] {report_paths.markdown}", flush=True)
            if not args.no_open_report:
                open_report(latest_report)
            if args.exit_after_run:
                break
            prime_hover_scene()
            visuals.reset()
            controls.sync()
            for _ in range(10):
                simulation_app.update()
    finally:
        controls.destroy()
        gym_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
