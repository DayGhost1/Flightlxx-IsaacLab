from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from flightlxx_isaaclab.core import (
    MotorActuator,
    MotorActuatorCfg,
    ActionDelayBuffer,
    BetaflightProfile,
    SnowyOwl3PlatformCfg,
    DomainRandomizationCfg,
    ImpactCurriculum,
    ImpactCurriculumCfg,
    TwoStageCurriculum,
    ImpactSamplingCfg,
    HoverRewardCfg,
    RecoveryCriteria,
    VectorizedHistory,
    fixed_target_hover_state,
    arena_failure_mask,
    classify_impact_phase,
    physical_impact_metadata,
    quat_error,
    quat_mul,
    physx_angular_velocity_limit_deg_s,
    sample_domain_parameters,
    sample_handoff_state,
    VirtualViconBridge,
    sample_impact_wrench,
    unified_hover_reward,
    update_recovery_dwell,
    write_com_offsets,
)
from flightlxx_isaaclab.core.math import quat_rotate_inverse
from flightlxx_isaaclab.core.vicon_bridge import ViconSampleClock
from flightlxx_isaaclab.evaluation import (
    EvaluationModeState,
    FixedImpactProtocol,
    evaluation_horizon_steps,
)


STATE_DIM = 13
ACTION_DIM = 4
FEATURE_DIM = STATE_DIM + ACTION_DIM
# mass, inertia, CoM, current impact wrench, thrust/tau/delay, difficulty/time/active
PRIVILEGED_DIM = 19


@configclass
class CTBRRecoveryEnvCfg(DirectRLEnvCfg):
    episode_length_s = 8.0
    decimation = 10
    action_space = ACTION_DIM
    fast_history = 4
    slow_history = 32
    observation_space = STATE_DIM + FEATURE_DIM * (fast_history + slow_history)
    state_space = observation_space + PRIVILEGED_DIM
    sim = SimulationCfg(dt=0.002, render_interval=decimation)
    scene = InteractiveSceneCfg(num_envs=512, env_spacing=27.0, replicate_physics=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        debug_vis=False,
    )
    robot = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.055),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=30.0,
                # PhysX expects deg/s here. Keep two-times headroom above the
                # largest CTBR command (2*pi rad/s -> 720 deg/s) so physics
                # does not silently become the body-rate limiter.
                max_angular_velocity=physx_angular_velocity_limit_deg_s(
                    MotorActuatorCfg().max_body_rate,
                    headroom=2.0,
                ),
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.78),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.12, 0.18)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_linear_velocity=30.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )

    mass = 0.78
    arm_length = 0.125
    inertia = (0.00228515625, 0.00228515625, 0.0035546875)
    nominal_actuator_tau_s = 0.0625
    arena_size_m = 25.0
    arena_wall_thickness_m = 0.10
    body_boundary_margin_m = 0.125
    failure_penalty = -5.0
    curriculum_recovery_position_error_m = 0.15
    curriculum_recovery_linear_speed_mps = 0.15
    curriculum_recovery_attitude_error_rad = 0.0872664626
    curriculum_recovery_angular_speed_rps = 0.25
    curriculum_recovery_dwell_s = 0.5
    precision_recovery_position_error_m = 0.05
    precision_recovery_linear_speed_mps = 0.05
    precision_recovery_attitude_error_rad = 0.0349065850
    precision_recovery_angular_speed_rps = 0.05
    precision_recovery_dwell_s = 2.0
    enable_domain_randomization = True
    enable_impacts = True
    curriculum_initial_difficulty = 0.05


PREFLIGHT_COLLISION_ASSET = (
    Path(__file__).resolve().parents[3] / "assets" / "flightlxx_quadrotor_collision.usda"
)


@configclass
class CTBRPreflightEnvCfg(CTBRRecoveryEnvCfg):
    """Validation-only compound collider; the training task remains unchanged."""

    scene = InteractiveSceneCfg(num_envs=1, env_spacing=27.0, replicate_physics=True)
    enable_domain_randomization = False
    enable_impacts = False
    robot = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PREFLIGHT_COLLISION_ASSET),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=30.0,
                max_angular_velocity=physx_angular_velocity_limit_deg_s(
                    MotorActuatorCfg().max_body_rate,
                    headroom=2.0,
                ),
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )


class CTBRRecoveryEnv(DirectRLEnv):
    cfg: CTBRRecoveryEnvCfg

    def __init__(self, cfg: CTBRRecoveryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._executed_actions = torch.zeros_like(self._actions)
        self._domain_cfg = DomainRandomizationCfg()
        self._action_delay = ActionDelayBuffer(
            self.num_envs,
            ACTION_DIM,
            max(self._domain_cfg.action_delay_steps),
            self.device,
        )
        platform_path = Path(__file__).resolve().parents[3] / "config" / "snowyowl3_real_v1.json"
        self._platform = SnowyOwl3PlatformCfg.from_json(platform_path)
        # The launch script rejects a placeholder for formal training.  The
        # environment itself permits it so short smoke tests can validate the
        # implementation before the final Vicon XYZ is measured.
        self._platform.validate_for_training(allow_placeholder_target=True)
        self._actuator = MotorActuator(
            self.num_envs,
            self.device,
            MotorActuatorCfg(
                dt=cfg.sim.dt,
                mass=cfg.mass,
                thrust_coefficient_n_per_rpm2=self._platform.propeller_thrust_coefficient,
                motor_kv_rpm_per_v=2550.0,
                rpm_official_max=self._platform.hardware_rpm_max,
                policy_rpm_fraction=self._platform.policy_rpm_fraction,
                arm_length=cfg.arm_length,
            ),
        )
        self._betaflight = BetaflightProfile.from_platform(self._platform.betaflight).build_rate_loop(
            self.num_envs, self.device
        )
        self._history = VectorizedHistory(self.num_envs, cfg.slow_history, FEATURE_DIM, self.device)
        # The 120 ms observed in flight is command-to-rigid-body response;
        # motor, Betaflight and plant dynamics model it separately.  The bags
        # show roughly a 9 ms state-to-policy scheduler phase, so model a
        # 10 ms Vicon sample age rather than double-counting plant response.
        self._vicon = VirtualViconBridge(
            self.num_envs,
            self.device,
            output_hz=self._platform.vicon.output_hz,
            angular_window_s=self._platform.vicon.angular_window_s,
            measurement_delay_s=self._platform.vicon.measurement_age_s,
        )
        self._vicon_time_s = 0.0
        # The rosbag shows a nominal 100 Hz Vicon stream with a narrow,
        # arrival-time jitter.  Sampling is global because it is one physical
        # Vicon stream shared by all vectorized training environments.
        self._vicon_sample_clock = ViconSampleClock(
            nominal_period_s=1.0 / self._platform.vicon.sample_hz,
            jitter_s=self._platform.vicon.sampling_jitter_s,
            seed=0,
        )
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._control_torque = torch.zeros_like(self._thrust)
        self._rc_deflection = torch.zeros(self.num_envs, 3, device=self.device)
        self._rate_setpoint_dps = torch.zeros_like(self._rc_deflection)
        self._throttle = torch.zeros(self.num_envs, device=self.device)
        self._new_rc_frame_pending = False
        self._disturbance_force = torch.zeros_like(self._thrust)
        self._disturbance_torque = torch.zeros_like(self._thrust)
        self._scheduled_force = torch.zeros_like(self._thrust)
        self._scheduled_torque = torch.zeros_like(self._thrust)
        self._application_point = torch.zeros(self.num_envs, 3, device=self.device)
        self._disturbance_start = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._disturbance_end = torch.zeros_like(self._disturbance_start)
        self._disturbance_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._impact_enabled = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._impact_happened = torch.zeros_like(self._impact_enabled)
        self._impact_band = torch.zeros(self.num_envs, device=self.device, dtype=torch.uint8)
        self._episode_difficulty = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_recovery_dwell = torch.zeros(self.num_envs, device=self.device)
        self._precision_recovery_dwell = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_recovery_criteria = RecoveryCriteria(
            cfg.curriculum_recovery_position_error_m,
            cfg.curriculum_recovery_linear_speed_mps,
            cfg.curriculum_recovery_attitude_error_rad,
            cfg.curriculum_recovery_angular_speed_rps,
            cfg.curriculum_recovery_dwell_s,
        )
        self._precision_recovery_criteria = RecoveryCriteria(
            cfg.precision_recovery_position_error_m,
            cfg.precision_recovery_linear_speed_mps,
            cfg.precision_recovery_attitude_error_rad,
            cfg.precision_recovery_angular_speed_rps,
            cfg.precision_recovery_dwell_s,
        )
        self._steady_position_sum = torch.zeros(self.num_envs, device=self.device)
        self._steady_position_count = torch.zeros(self.num_envs, device=self.device)
        self._hover_reward_cfg = HoverRewardCfg()
        self._reward_component_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "position",
                "linear_velocity",
                "attitude",
                "angular_velocity",
                "action_magnitude",
                "action_rate",
                "motor_saturation",
                "failure",
                "loose_recovery",
                "precision_recovery",
                "recovery_completion",
                "timeout_without_recovery",
            )
        }

        target_offset = torch.tensor(
            self._platform.target_position_vicon_m,
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        self._target_position = self.scene.env_origins + target_offset
        self._target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat[:, 0] = 1.0
        self._domain = sample_domain_parameters(self.num_envs, self.device, cfg.mass, cfg.inertia, self._domain_cfg)
        self._curriculum = ImpactCurriculum(
            ImpactCurriculumCfg(), initial_difficulty=cfg.curriculum_initial_difficulty
        )
        self._stage_curriculum = TwoStageCurriculum(initial_difficulty=cfg.curriculum_initial_difficulty)
        self._impact_cfg = ImpactSamplingCfg()
        self._evaluation_state = EvaluationModeState()
        self._evaluation_protocol: FixedImpactProtocol | None = None
        self._evaluation_last_metrics: dict[str, torch.Tensor] = {}
        self._discard_evaluation_episode = False
        self._episode_metrics = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "max_position_error",
                "max_attitude_error_rad",
                "max_angular_velocity",
                "recovery_time",
                "success_recovery",
                "precision_recovery_time",
                "precision_recovery_success",
            )
        }
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._apply_domain_randomization(all_ids)

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot)
        self._ball = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["robot"] = self._robot
        self.scene.rigid_objects["ball"] = self._ball
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self._spawn_arena_boundaries()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0)
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_arena_boundaries(self):
        half = self.cfg.arena_size_m / 2.0
        height = self.cfg.arena_size_m
        thickness = self.cfg.arena_wall_thickness_m
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.18, 0.28), opacity=0.12)
        boundaries = (
            ("WallXPos", (thickness, self.cfg.arena_size_m, height), (half, 0.0, height / 2.0)),
            ("WallXNeg", (thickness, self.cfg.arena_size_m, height), (-half, 0.0, height / 2.0)),
            ("WallYPos", (self.cfg.arena_size_m, thickness, height), (0.0, half, height / 2.0)),
            ("WallYNeg", (self.cfg.arena_size_m, thickness, height), (0.0, -half, height / 2.0)),
            ("Ceiling", (self.cfg.arena_size_m, self.cfg.arena_size_m, thickness), (0.0, 0.0, height)),
        )
        for name, size, translation in boundaries:
            spawn_cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=material,
            )
            # The regex spawner clones the leaf below each matched, existing
            # environment prim.  A nested ``Arena`` parent does not exist and
            # therefore cannot be used as the regex source path.
            spawn_cfg.func(f"/World/envs/env_.*/Arena_{name}", spawn_cfg, translation=translation)

    def _apply_domain_randomization(self, env_ids: torch.Tensor, *, nominal: bool = False):
        count = len(env_ids)
        if self.cfg.enable_domain_randomization and not nominal:
            sampled = sample_domain_parameters(count, self.device, self.cfg.mass, self.cfg.inertia, self._domain_cfg)
        else:
            nominal_cfg = DomainRandomizationCfg(
                mass_scale=(1.0, 1.0),
                inertia_scale=(1.0, 1.0),
                com_xy_m=0.0,
                com_z_m=0.0,
                thrust_scale=(1.0, 1.0),
                motor_scale=(1.0, 1.0),
                actuator_tau_s=(
                    self.cfg.nominal_actuator_tau_s,
                    self.cfg.nominal_actuator_tau_s,
                ),
                action_delay_steps=(0, 0),
                battery_voltage_v=(16.0, 16.0),
                battery_internal_resistance_ohm=(0.035, 0.035),
                vicon_dropout_probability=(0.0, 0.0),
            )
            sampled = sample_domain_parameters(count, self.device, self.cfg.mass, self.cfg.inertia, nominal_cfg)
        for field in fields(self._domain):
            getattr(self._domain, field.name)[env_ids] = getattr(sampled, field.name)

        view_ids = env_ids.to(device="cpu", dtype=torch.long)
        masses = self._robot.root_physx_view.get_masses()
        masses[view_ids, 0] = self._domain.mass[env_ids].cpu()
        self._robot.root_physx_view.set_masses(masses, view_ids)
        inertias = self._robot.root_physx_view.get_inertias()
        inertias[view_ids] = 0.0
        randomized_inertia = self._domain.inertia[env_ids].cpu()
        inertias[view_ids, 0] = randomized_inertia[:, 0]
        inertias[view_ids, 4] = randomized_inertia[:, 1]
        inertias[view_ids, 8] = randomized_inertia[:, 2]
        self._robot.root_physx_view.set_inertias(inertias, view_ids)
        coms = self._robot.root_physx_view.get_coms()
        write_com_offsets(coms, view_ids, self._domain.com[env_ids].cpu())
        self._robot.root_physx_view.set_coms(coms, view_ids)
        self._actuator.set_domain_parameters(
            env_ids,
            mass=self._domain.mass[env_ids],
            thrust_scale=self._domain.thrust_scale[env_ids],
            motor_efficiency=self._domain.motor_scale[env_ids],
            rpm_tau_s=self._domain.actuator_tau[env_ids],
        )
        self._actuator.set_battery_parameters(
            env_ids,
            voltage_v=self._domain.battery_voltage_v[env_ids],
            internal_resistance_ohm=self._domain.battery_internal_resistance_ohm[env_ids],
        )
        self._action_delay.set_delay_steps(env_ids, self._domain.delay_steps[env_ids])
        self._vicon.set_measurement_noise(
            self._domain.position_noise_std,
            self._domain.attitude_noise_std,
        )
        self._vicon.set_derived_state_noise(
            linear_velocity_noise_std_mps=self._domain.velocity_noise_std,
            angular_velocity_noise_std_radps=self._domain.gyro_noise_std,
            angular_velocity_bias_radps=self._domain.gyro_bias,
        )
        self._vicon.set_dropout_probability(self._domain.vicon_dropout_probability)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        self._executed_actions.copy_(self._action_delay.step(self._actions))
        self._rc_deflection.copy_(
            self._betaflight.profile.ctbr_action_to_rc_deflection(
                self._executed_actions[:, 1:]
            )
        )
        self._rate_setpoint_dps.copy_(
            self._betaflight.profile.ctbr_action_to_rate_setpoint_dps(
                self._executed_actions[:, 1:]
            )
        )
        self._throttle.copy_((self._executed_actions[:, 0] + 1.0) * 0.5)
        self._new_rc_frame_pending = True
        self._update_disturbances()

    def _apply_action(self):
        gyro_rate_dps = self._robot.data.root_ang_vel_b * (180.0 / math.pi)
        pid = self._betaflight.advance_physics_tick(
            self._rate_setpoint_dps,
            gyro_rate_dps,
            physics_dt_s=self.cfg.sim.dt,
            throttle=self._throttle,
            rc_deflection=self._rc_deflection,
            new_rc_frame=self._new_rc_frame_pending,
        )
        self._new_rc_frame_pending = False
        thrust, torque = self._actuator.step_betaflight(
            self._executed_actions, pid.pid_sum
        )
        self._thrust.zero_()
        self._thrust[:, 0, 2] = thrust[:, 0]
        self._control_torque[:, 0] = torque
        self._vicon_time_s += self.cfg.sim.dt
        if self._vicon_sample_clock.consume_if_due(self._vicon_time_s):
            self._vicon.push(
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                timestamp_s=self._vicon_time_s,
            )
        self._robot.set_external_force_and_torque(
            self._thrust + self._disturbance_force,
            self._control_torque + self._disturbance_torque,
            is_global=False,
        )

    def _current_state(self, noisy: bool = True) -> torch.Tensor:
        measured = self._vicon.observe(now_s=self._vicon_time_s)
        if measured is not None:
            position_error_b = quat_rotate_inverse(
                measured[:, 6:10], measured[:, :3] - self._target_position
            )
            linear_velocity_b = quat_rotate_inverse(measured[:, 6:10], measured[:, 3:6])
            error_quat = quat_error(self._target_quat, measured[:, 6:10])
            return torch.cat((position_error_b, linear_velocity_b, error_quat, measured[:, 10:13]), dim=-1)
        position_error_b = quat_rotate_inverse(
            self._robot.data.root_quat_w, self._robot.data.root_pos_w - self._target_position
        )
        linear_velocity_b = self._robot.data.root_lin_vel_b
        error_quat = quat_error(self._target_quat, self._robot.data.root_quat_w)
        angular_velocity_b = self._robot.data.root_ang_vel_b
        if noisy:
            position_error_b = position_error_b + torch.randn_like(position_error_b) * self._domain.position_noise_std[:, None]
            linear_velocity_b = linear_velocity_b + torch.randn_like(linear_velocity_b) * self._domain.velocity_noise_std[:, None]
            attitude_noise = torch.randn(self.num_envs, 3, device=self.device) * self._domain.attitude_noise_std[:, None]
            noise_quat = quat_from_euler_xyz(attitude_noise[:, 0], attitude_noise[:, 1], attitude_noise[:, 2])
            error_quat = quat_mul(noise_quat, error_quat)
            error_quat = error_quat / error_quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            error_quat = torch.where(error_quat[:, :1] < 0.0, -error_quat, error_quat)
            angular_velocity_b = angular_velocity_b + self._domain.gyro_bias
            angular_velocity_b = angular_velocity_b + torch.randn_like(angular_velocity_b) * self._domain.gyro_noise_std[:, None]
        return torch.cat((position_error_b, linear_velocity_b, error_quat, angular_velocity_b), dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self._current_state(noisy=self._evaluation_state.noisy_observations)
        feature = torch.cat((state, self._actions), dim=-1)
        self._history.append(feature)
        fast = self._history.latest(self.cfg.fast_history).flatten(1)
        slow = self._history.buffer.flatten(1)
        policy = torch.cat((state, fast, slow), dim=-1)
        active = torch.linalg.vector_norm(self._disturbance_force[:, 0], dim=-1) > 0.0
        privileged = torch.cat(
            (
                self._domain.mass[:, None],
                self._domain.inertia,
                self._domain.com,
                self._disturbance_force[:, 0],
                self._disturbance_torque[:, 0],
                self._domain.thrust_scale[:, None],
                self._domain.actuator_tau[:, None],
                self._domain.delay_steps[:, None].float(),
                self._episode_difficulty[:, None],
                self._disturbance_elapsed[:, None],
                active[:, None].float(),
            ),
            dim=-1,
        )
        event_type, effective_band, impact_phase = physical_impact_metadata(
            impact_enabled=self._impact_enabled,
            impact_happened=self._impact_happened,
            curriculum_band=self._impact_band,
            episode_step=self.episode_length_buf,
            disturbance_start=self._disturbance_start,
            disturbance_end=self._disturbance_end,
            recovery_window_steps=max(1, int(round(2.0 / self.step_dt))),
        )
        self.extras["event_type"] = event_type
        self.extras["curriculum_band"] = effective_band
        self.extras["impact_phase"] = impact_phase
        self.extras["episode_difficulty"] = self._episode_difficulty.clone()
        self.extras["initial_difficulty"] = torch.full_like(
            self._episode_difficulty, self._stage_curriculum.initial_difficulty
        )
        self.extras["disturbance_difficulty"] = torch.full_like(
            self._episode_difficulty, self._stage_curriculum.disturbance_difficulty
        )
        actuator_diagnostics = self._actuator.diagnostics(copy=False)
        self.extras["motor_saturation_fraction"] = (
            actuator_diagnostics["motor_rpm"]
            >= 0.99 * actuator_diagnostics["hardware_rpm_limit"][:, None]
        ).float().mean(dim=-1)
        self.extras["domain_parameters"] = torch.stack(
            (
                self._actuator.battery_voltage_v,
                self._actuator.battery_internal_resistance_ohm,
                self._domain.thrust_scale,
                self._domain.motor_scale.mean(dim=-1),
                self._domain.actuator_tau,
                self._domain.position_noise_std,
                self._domain.attitude_noise_std,
                self._domain.vicon_dropout_probability,
            ),
            dim=-1,
        )
        return {"policy": policy, "critic": torch.cat((policy, privileged), dim=-1)}

    def _failure_mask(self) -> torch.Tensor:
        relative = self._robot.data.root_pos_w - self.scene.env_origins
        finite = torch.isfinite(self._robot.data.root_state_w).all(dim=-1)
        return arena_failure_mask(
            relative,
            finite,
            half_extent_xy=self.cfg.arena_size_m / 2.0,
            height=self.cfg.arena_size_m,
            body_margin=self.cfg.body_boundary_margin_m,
        )

    def _get_rewards(self) -> torch.Tensor:
        position_error = torch.linalg.vector_norm(self._robot.data.root_pos_w - self._target_position, dim=-1)
        linear_speed = torch.linalg.vector_norm(self._robot.data.root_lin_vel_w, dim=-1)
        attitude_angle = 2.0 * torch.acos(
            torch.sum(self._target_quat * self._robot.data.root_quat_w, dim=-1).abs().clamp(max=1.0)
        )
        angular_speed = torch.linalg.vector_norm(self._robot.data.root_ang_vel_b, dim=-1)
        disturbance_finished = ~self._impact_enabled | (self.episode_length_buf >= self._disturbance_end)
        # Clean-hover episodes must be able to satisfy the no-impact mastery
        # gate.  Impact episodes only become eligible after their pulse.
        recovery_eligible = disturbance_finished & (~self._impact_enabled | self._impact_happened)
        was_recovered = self._curriculum_recovery_dwell >= self._curriculum_recovery_criteria.dwell_s
        self._curriculum_recovery_dwell, recovered = update_recovery_dwell(
            self._curriculum_recovery_dwell,
            position_error,
            linear_speed,
            attitude_angle,
            angular_speed,
            recovery_eligible,
            self.step_dt,
            self._curriculum_recovery_criteria,
        )
        self._precision_recovery_dwell, precision_recovered = update_recovery_dwell(
            self._precision_recovery_dwell,
            position_error,
            linear_speed,
            attitude_angle,
            angular_speed,
            recovery_eligible,
            self.step_dt,
            self._precision_recovery_criteria,
        )
        failure = self._failure_mask()
        actuator_diagnostics = self._actuator.diagnostics(copy=False)
        motor_saturation_fraction = (
            actuator_diagnostics["motor_rpm"]
            >= 0.99 * actuator_diagnostics["hardware_rpm_limit"][:, None]
        ).float().mean(dim=-1)
        loose_inside = self._curriculum_recovery_dwell > 0.0
        precision_inside = self._precision_recovery_dwell > 0.0
        recovery_completed = recovered & ~was_recovered
        timed_out_without_recovery = (
            (self.episode_length_buf >= self.max_episode_length - 1)
            & recovery_eligible
            & ~recovered
            & (self._episode_metrics["success_recovery"] == 0.0)
        )
        reward, reward_components = unified_hover_reward(
            position_error=position_error,
            linear_speed=linear_speed,
            attitude_error_rad=attitude_angle,
            angular_speed=angular_speed,
            actions=self._actions,
            previous_actions=self._previous_actions,
            failure=failure,
            step_dt=self.step_dt,
            failure_penalty=self.cfg.failure_penalty,
            cfg=self._hover_reward_cfg,
            motor_saturation_fraction=motor_saturation_fraction,
            loose_inside=loose_inside,
            precision_inside=precision_inside,
            recovery_completed=recovery_completed,
            timed_out_without_recovery=timed_out_without_recovery,
        )
        for name, values in reward_components.items():
            self._reward_component_sums[name] += values

        self._evaluation_last_metrics = {
            "position_error": position_error.detach().clone(),
            "linear_speed": linear_speed.detach().clone(),
            "attitude_error_rad": attitude_angle.detach().clone(),
            "angular_speed": angular_speed.detach().clone(),
            "failure": failure.detach().clone(),
        }
        if self._evaluation_state.active:
            position_error_b = quat_rotate_inverse(
                self._robot.data.root_quat_w,
                self._robot.data.root_pos_w - self._target_position,
            )
            attitude_error_quat = quat_error(self._target_quat, self._robot.data.root_quat_w)
            controller = self._betaflight.diagnostics()
            actuator = self._actuator.diagnostics(copy=False)
            diagnostic_vectors = {
                "position_error_b": position_error_b,
                "linear_velocity_b": self._robot.data.root_lin_vel_b,
                "attitude_error_q": attitude_error_quat,
                "body_rate": self._robot.data.root_ang_vel_b,
                "body_rate_setpoint": controller["rate_setpoint_dps"] * (math.pi / 180.0),
                "body_rate_error": controller["rate_error_dps"] * (math.pi / 180.0),
                "betaflight_gyro_dps": controller["gyro_rate_dps"],
                "betaflight_pid_sum": controller["pid_sum"],
                "control_torque": self._control_torque[:, 0],
                "disturbance_force": self._disturbance_force[:, 0, :],
                "disturbance_torque": self._disturbance_torque[:, 0, :],
            }
            for prefix, values in diagnostic_vectors.items():
                suffixes = ("w", "x", "y", "z") if values.shape[-1] == 4 else ("x", "y", "z")
                for axis, suffix in enumerate(suffixes):
                    self._evaluation_last_metrics[f"{prefix}_{suffix}"] = values[:, axis].detach().clone()
            delayed_action = self._executed_actions
            for axis, name in enumerate(("collective", "body_rate_x", "body_rate_y", "body_rate_z")):
                self._evaluation_last_metrics[f"action_{name}"] = self._actions[:, axis].detach().clone()
                self._evaluation_last_metrics[f"delayed_action_{name}"] = delayed_action[:, axis].detach().clone()
            self._evaluation_last_metrics["collective_thrust_n"] = self._thrust[
                :, 0, 2
            ].detach().clone()
            self._evaluation_last_metrics["betaflight_inner_ticks"] = torch.full(
                (self.num_envs,),
                float(controller["inner_ticks"]),
                device=self.device,
            )
            for motor in range(4):
                self._evaluation_last_metrics[f"motor_rpm_{motor + 1}"] = actuator[
                    "motor_rpm"
                ][:, motor].detach().clone()
            return reward

        self._episode_metrics["max_position_error"] = torch.maximum(
            self._episode_metrics["max_position_error"], position_error
        )
        self._episode_metrics["max_attitude_error_rad"] = torch.maximum(
            self._episode_metrics["max_attitude_error_rad"], attitude_angle
        )
        self._episode_metrics["max_angular_velocity"] = torch.maximum(
            self._episode_metrics["max_angular_velocity"], angular_speed
        )
        not_recorded = self._episode_metrics["recovery_time"] == 0.0
        self._episode_metrics["recovery_time"] = torch.where(
            recovered & not_recorded & self._impact_happened,
            self._disturbance_elapsed,
            self._episode_metrics["recovery_time"],
        )
        self._episode_metrics["success_recovery"] = torch.maximum(
            self._episode_metrics["success_recovery"], recovered.float()
        )
        precision_not_recorded = self._episode_metrics["precision_recovery_time"] == 0.0
        self._episode_metrics["precision_recovery_time"] = torch.where(
            precision_recovered & precision_not_recorded & self._impact_happened,
            self._disturbance_elapsed,
            self._episode_metrics["precision_recovery_time"],
        )
        self._episode_metrics["precision_recovery_success"] = torch.maximum(
            self._episode_metrics["precision_recovery_success"], precision_recovered.float()
        )
        self._steady_position_sum += torch.where(precision_recovered, position_error, 0.0)
        self._steady_position_count += precision_recovered.float()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_steps = self.max_episode_length
        if self._evaluation_protocol is not None:
            max_steps = evaluation_horizon_steps(
                self._evaluation_protocol.total_duration_s,
                self.step_dt,
            )
        return self._failure_mask(), self.episode_length_buf >= max_steps - 1

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if hasattr(self, "_episode_metrics") and not self._evaluation_state.active and not self._discard_evaluation_episode:
            crashed = self.reset_terminated[env_ids].bool()
            counted_impact = self._impact_enabled[env_ids] & self._impact_happened[env_ids]
            curriculum_band = torch.where(
                counted_impact,
                self._impact_band[env_ids],
                torch.zeros_like(self._impact_band[env_ids]),
            )
            stage_before = self._stage_curriculum.sample()
            stage_after = self._stage_curriculum.record_window(
                success_rate=self._episode_metrics["success_recovery"][env_ids].float().mean().item(),
                crash_rate=crashed.float().mean().item(),
                global_step=int(getattr(self, "common_step_counter", 0)),
            )
            stage_updated = stage_after != stage_before
            curriculum_updated = self._curriculum.record_batch(
                self._episode_metrics["success_recovery"][env_ids].bool(),
                crashed,
                curriculum_band,
                global_step=int(getattr(self, "common_step_counter", 0)),
            )
            self.extras["log"] = {
                f"Metrics/{name}": values[env_ids].mean().item() for name, values in self._episode_metrics.items()
            }
            steady_error = self._steady_position_sum[env_ids] / self._steady_position_count[env_ids].clamp_min(1.0)
            self.extras["log"]["Metrics/steady_position_error"] = steady_error.mean().item()
            impacted_count = counted_impact.float().sum().clamp_min(1.0)
            self.extras["log"]["Metrics/success_recovery"] = (
                self._episode_metrics["success_recovery"][env_ids] * counted_impact
            ).sum().div(impacted_count).item()
            self.extras["log"]["Metrics/precision_recovery_success"] = (
                self._episode_metrics["precision_recovery_success"][env_ids] * counted_impact
            ).sum().div(impacted_count).item()
            self.extras["log"]["Metrics/post_impact_crash_rate"] = (
                crashed & counted_impact
            ).float().sum().div(impacted_count).item()
            self.extras["log"]["Curriculum/difficulty"] = self._curriculum.difficulty
            self.extras["log"]["Curriculum/mastered_difficulty"] = self._curriculum.mastered_difficulty
            self.extras["log"]["Curriculum/promotion_count"] = self._curriculum.promotion_count
            self.extras["log"]["Curriculum/demotion_count"] = self._curriculum.demotion_count
            self.extras["log"]["Curriculum/promotion_streak"] = self._curriculum.promotion_streak
            self.extras["log"]["Curriculum/demotion_streak"] = self._curriculum.demotion_streak
            self.extras["log"]["Curriculum/no_impact_success_rate"] = (
                self._curriculum.last_no_impact_success_rate
            )
            self.extras["log"]["Curriculum/no_impact_crash_rate"] = (
                self._curriculum.last_no_impact_crash_rate
            )
            self.extras["log"]["Curriculum/current_success_rate"] = self._curriculum.last_current_success_rate
            self.extras["log"]["Curriculum/current_crash_rate"] = self._curriculum.last_current_crash_rate
            self.extras["log"]["Curriculum/probe_success_rate"] = self._curriculum.last_probe_success_rate
            self.extras["log"]["Curriculum/probe_crash_rate"] = self._curriculum.last_probe_crash_rate
            self.extras["log"]["Curriculum/last_action"] = float(
                {"demote": -1, "none": 0, "legacy_reset": 0, "promote": 1}.get(
                    self._curriculum.last_action,
                    0,
                )
            )
            self.extras["log"]["Curriculum/updated"] = float(curriculum_updated)
            self.extras["log"]["Curriculum/stage_updated"] = float(stage_updated)
            self.extras["log"]["Curriculum/initial_difficulty"] = self._stage_curriculum.initial_difficulty
            self.extras["log"]["Curriculum/disturbance_difficulty"] = self._stage_curriculum.disturbance_difficulty
            self.extras["log"]["Curriculum/impact_probability"] = self._stage_curriculum.impact_probability
            episode_steps = self.episode_length_buf[env_ids].float().clamp_min(1.0)
            for name, values in self._reward_component_sums.items():
                self.extras["log"][f"Reward/{name}"] = (
                    values[env_ids] / episode_steps
                ).mean().item()
                values[env_ids] = 0.0
            for values in self._episode_metrics.values():
                values[env_ids] = 0.0
            self._steady_position_sum[env_ids] = 0.0
            self._steady_position_count[env_ids] = 0.0

        self._robot.reset(env_ids)
        self._vicon.reset(env_ids)
        self._ball.reset(env_ids)
        super()._reset_idx(env_ids)
        self._apply_domain_randomization(env_ids, nominal=self._evaluation_state.active)
        count = len(env_ids)
        if self._evaluation_protocol is None:
            curriculum_sample = self._curriculum.sample(count, self.device)
            stage = self._stage_curriculum.sample()
            normalized = curriculum_sample.difficulty / max(self._curriculum.difficulty, 1.0e-6)
            handoff_difficulty = (normalized * stage.initial_difficulty).clamp(0.0, 1.0)
            self._impact_enabled[env_ids] = (
                curriculum_sample.impacted
                & self.cfg.enable_impacts
                & (stage.disturbance_difficulty > 0.0)
                & (torch.rand(count, device=self.device) < stage.impact_probability)
            )
            self._episode_difficulty[env_ids] = (
                normalized * stage.disturbance_difficulty
            ).clamp(0.0, 1.0)
            self._impact_band[env_ids] = curriculum_sample.band
        else:
            self._impact_enabled[env_ids] = True
            self._episode_difficulty[env_ids] = 0.0
            self._impact_band[env_ids] = 0

        difficulty = handoff_difficulty if self._evaluation_protocol is None else self._episode_difficulty[env_ids]
        impact = self._impact_enabled[env_ids]
        pose = self._robot.data.default_root_state[env_ids, :7].clone()
        handoff = sample_handoff_state(count, self.device, difficulty=difficulty)
        pose[:, :3] = self._target_position[env_ids] + handoff.position_error_m
        pose[:, 3:7] = handoff.orientation_wxyz
        velocity = torch.cat((handoff.linear_velocity_mps, handoff.angular_velocity_radps), dim=-1)
        if self._evaluation_state.active:
            pose, velocity = fixed_target_hover_state(
                self._robot.data.default_root_state[env_ids],
                self._target_position[env_ids],
                self._target_quat[env_ids],
            )
        self._robot.write_root_pose_to_sim(pose, env_ids)
        self._robot.write_root_velocity_to_sim(velocity, env_ids)
        bootstrap_position = self._robot.data.root_pos_w.clone()
        bootstrap_quaternion = self._robot.data.root_quat_w.clone()
        bootstrap_position[env_ids] = pose[:, :3]
        bootstrap_quaternion[env_ids] = pose[:, 3:7]
        self._vicon.push(
            bootstrap_position,
            bootstrap_quaternion,
            # ``measurement_delay_s`` is an observation-selection age, never
            # a backdated source timestamp.  Backdating here could violate the
            # Vicon bridge's monotonic timestamp contract after an env reset.
            timestamp_s=self._vicon_time_s,
        )

        ball_pose = self._ball.data.default_root_state[env_ids, :7].clone()
        ball_pose[:, :3] = self.scene.env_origins[env_ids]
        ball_pose[:, 2] = -2.0
        self._ball.write_root_pose_to_sim(ball_pose, env_ids)
        self._ball.write_root_velocity_to_sim(torch.zeros(count, 6, device=self.device), env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._executed_actions[env_ids] = 0.0
        self._action_delay.reset(env_ids)
        self._actuator.reset(env_ids)
        self._betaflight.reset(env_ids)
        self._curriculum_recovery_dwell[env_ids] = 0.0
        self._precision_recovery_dwell[env_ids] = 0.0
        if self._evaluation_protocol is None:
            self._sample_disturbance(env_ids)
        else:
            self._disturbance_force[env_ids] = 0.0
            self._disturbance_torque[env_ids] = 0.0
            self._scheduled_force[env_ids] = 0.0
            self._scheduled_torque[env_ids] = 0.0
            self._impact_happened[env_ids] = False
            self._disturbance_elapsed[env_ids] = 0.0
        initial_feature = torch.cat((self._current_state(noisy=self._evaluation_state.noisy_observations)[env_ids], self._actions[env_ids]), dim=-1)
        self._history.reset(env_ids, initial_feature)
        self._discard_evaluation_episode = False

    def _sample_disturbance(self, env_ids: torch.Tensor):
        sample = sample_impact_wrench(
            self._domain.mass[env_ids],
            self._domain.inertia[env_ids],
            self._episode_difficulty[env_ids],
            self._impact_cfg,
        )
        self._disturbance_start[env_ids] = (sample.start_s / self.step_dt).round().long()
        duration_steps = (sample.duration_s / self.step_dt).round().long().clamp_min(1)
        self._disturbance_end[env_ids] = self._disturbance_start[env_ids] + duration_steps
        self._disturbance_elapsed[env_ids] = 0.0
        self._impact_happened[env_ids] = False
        self._disturbance_force[env_ids] = 0.0
        self._disturbance_torque[env_ids] = 0.0
        self._scheduled_force[env_ids, 0] = sample.force_b
        self._scheduled_torque[env_ids, 0] = sample.torque_b
        self._application_point[env_ids] = sample.application_point_b
        inactive = env_ids[~self._impact_enabled[env_ids]]
        self._scheduled_force[inactive] = 0.0
        self._scheduled_torque[inactive] = 0.0

    def _update_disturbances(self):
        if self._evaluation_protocol is not None:
            time_s = self.episode_length_buf.float() * self.step_dt
            self._disturbance_force.zero_()
            self._disturbance_torque.zero_()
            for impact in self._evaluation_protocol.impacts:
                active = (time_s >= impact.trigger_time_s) & (time_s < impact.end_time_s)
                self._disturbance_force[active, 0] = impact.force_b.to(self.device)
                self._disturbance_torque[active, 0] = impact.equivalent_torque_b.to(self.device)
            first_time = self._evaluation_protocol.impacts[0].trigger_time_s
            happened = time_s >= first_time
            self._impact_happened |= happened
            self._disturbance_elapsed[happened] = time_s[happened] - first_time
            return
        active = (
            self._impact_enabled
            & (self.episode_length_buf >= self._disturbance_start)
            & (self.episode_length_buf < self._disturbance_end)
        )
        self._disturbance_force.zero_()
        self._disturbance_torque.zero_()
        self._disturbance_force[active] = self._scheduled_force[active]
        self._disturbance_torque[active] = self._scheduled_torque[active]
        happened = self._impact_enabled & (self.episode_length_buf >= self._disturbance_start)
        self._impact_happened |= happened
        self._disturbance_elapsed[happened] += self.step_dt

    def begin_fixed_evaluation(self, protocol: FixedImpactProtocol):
        """Enable deterministic, nominal five-impact evaluation for the next reset."""
        self._evaluation_protocol = protocol
        self._evaluation_state.begin_fixed_evaluation(protocol)

    @property
    def evaluation_finished(self) -> bool:
        return self._evaluation_protocol is not None and bool(
            torch.all(
                self.episode_length_buf
                >= evaluation_horizon_steps(
                    self._evaluation_protocol.total_duration_s,
                    self.step_dt,
                    margin_steps=1,
                )
                - 1
            )
        )

    def evaluation_step_metrics(self) -> dict[str, torch.Tensor]:
        """State cached before DirectRLEnv can reset a terminal evaluation environment."""
        return self._evaluation_last_metrics

    def end_fixed_evaluation(self):
        self._evaluation_protocol = None
        self._evaluation_state.end_fixed_evaluation()
        # The next wrapper reset starts training again but must not count the
        # evaluation rollout as a curriculum episode.
        self._discard_evaluation_episode = True

    def apply_impulse(self, env_ids: torch.Tensor, delta_velocity_w: torch.Tensor, delta_angular_velocity_b: torch.Tensor):
        velocity = self._robot.data.root_vel_w[env_ids].clone()
        velocity[:, :3] += delta_velocity_w
        velocity[:, 3:] += delta_angular_velocity_b
        self._robot.write_root_velocity_to_sim(velocity, env_ids)

    def launch_ball(self, env_ids: torch.Tensor, speed: float = 8.0):
        robot_pos = self._robot.data.root_pos_w[env_ids]
        direction = torch.randn(len(env_ids), 3, device=self.device)
        direction[:, 2] *= 0.35
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        distance = 1.5 + torch.rand(len(env_ids), 1, device=self.device)
        origin = robot_pos - direction * distance
        pose = torch.cat((origin, self._ball.data.root_quat_w[env_ids]), dim=-1)
        velocity = torch.zeros(len(env_ids), 6, device=self.device)
        velocity[:, :3] = direction * speed
        self._ball.write_root_pose_to_sim(pose, env_ids)
        self._ball.write_root_velocity_to_sim(velocity, env_ids)

    def launch_targeted_ball(
        self,
        env_ids: torch.Tensor,
        target_point_b: torch.Tensor,
        approach_direction_b: torch.Tensor,
        impact_speed_mps: float | torch.Tensor,
        *,
        flight_time_s: float = 0.5,
        contact_clearance_m: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Launch a gravity-compensated ball through a selected body-frame target."""

        count = len(env_ids)
        if target_point_b.shape != (count, 3) or approach_direction_b.shape != (count, 3):
            raise ValueError("target point and approach direction must both have shape (len(env_ids), 3)")
        if flight_time_s <= 0.0:
            raise ValueError("flight_time_s must be positive")
        if not math.isfinite(contact_clearance_m) or contact_clearance_m < 0.0:
            raise ValueError("contact_clearance_m must be non-negative and finite")
        direction_norm = approach_direction_b.norm(dim=-1, keepdim=True)
        if not torch.allclose(direction_norm, torch.ones_like(direction_norm), atol=1.0e-5, rtol=0.0):
            raise ValueError("approach_direction_b rows must be unit vectors")
        speed = torch.as_tensor(impact_speed_mps, device=self.device, dtype=torch.float32)
        speed = speed.expand(count).reshape(count, 1)
        if not bool(torch.all(torch.isfinite(speed) & (speed > 0.0))):
            raise ValueError("impact_speed_mps must be positive and finite")

        quaternion_w = self._robot.data.root_quat_w[env_ids]
        robot_position_w = self._robot.data.root_pos_w[env_ids]
        direction_w = quat_apply(quaternion_w, approach_direction_b)
        contact_position_w = robot_position_w + quat_apply(quaternion_w, target_point_b)
        target_position_w = contact_position_w - direction_w * contact_clearance_m
        impact_velocity_w = direction_w * speed
        gravity_w = torch.tensor(self.cfg.sim.gravity, device=self.device, dtype=torch.float32).expand(count, 3)
        initial_velocity_w = impact_velocity_w - gravity_w * flight_time_s
        origin_position_w = (
            target_position_w
            - initial_velocity_w * flight_time_s
            - 0.5 * gravity_w * flight_time_s * flight_time_s
        )
        pose = torch.cat((origin_position_w, self._ball.data.root_quat_w[env_ids]), dim=-1)
        velocity = torch.zeros(count, 6, device=self.device)
        velocity[:, :3] = initial_velocity_w
        self._ball.write_root_pose_to_sim(pose, env_ids)
        self._ball.write_root_velocity_to_sim(velocity, env_ids)
        return {
            "origin_position_w": origin_position_w,
            "initial_velocity_w": initial_velocity_w,
            "target_position_w": target_position_w,
            "impact_velocity_w": impact_velocity_w,
        }

    def get_training_state(self) -> dict:
        return {
            "curriculum": self._curriculum.state_dict(),
            "stage_curriculum": {
                "initial_difficulty": self._stage_curriculum.initial_difficulty,
                "disturbance_difficulty": self._stage_curriculum.disturbance_difficulty,
                "impact_probability": self._stage_curriculum.impact_probability,
            },
        }

    def load_training_state(self, state: dict) -> None:
        if state and "curriculum" in state:
            self._curriculum.load_state_dict(state["curriculum"])
        if state and "stage_curriculum" in state:
            stage = state["stage_curriculum"]
            self._stage_curriculum.initial_difficulty = float(stage["initial_difficulty"])
            self._stage_curriculum.disturbance_difficulty = float(stage["disturbance_difficulty"])
            self._stage_curriculum.impact_probability = float(
                stage.get("impact_probability", self._stage_curriculum.impact_probability)
            )
