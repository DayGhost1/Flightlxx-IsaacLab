"""Versioned, deterministic preflight impact-validation protocol generation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


Vector3 = tuple[float, float, float]


def _vector3(value: Sequence[float], name: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    vector = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector  # type: ignore[return-value]


def _unit_vector(value: Sequence[float], name: str) -> Vector3:
    vector = _vector3(value, name)
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _uniform_sphere(rng: random.Random) -> Vector3:
    """Sample uniformly by surface area, using uniform azimuth and z."""

    z = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return (radius * math.cos(azimuth), radius * math.sin(azimuth), z)


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("stratified ranges require at least two samples")
    step = (stop - start) / float(count - 1)
    return [start + index * step for index in range(count)]


@dataclass(frozen=True)
class ImpactCase:
    case_id: str
    group: str
    application_point_b: Vector3
    direction_b: Vector3
    impulse_ns: float
    duration_s: float
    level: str
    template_id: str = ""
    region: str = ""

    @property
    def force_b(self) -> Vector3:
        magnitude = self.impulse_ns / self.duration_s
        return tuple(magnitude * component for component in self.direction_b)  # type: ignore[return-value]

    @property
    def equivalent_torque_b(self) -> Vector3:
        px, py, pz = self.application_point_b
        fx, fy, fz = self.force_b
        return (py * fz - pz * fy, pz * fx - px * fz, px * fy - py * fx)


@dataclass(frozen=True)
class ContinuousEpisode:
    episode_id: str
    recovery_policy: str
    impacts: tuple[ImpactCase, ...]
    intervals_s: tuple[float, ...]


@dataclass(frozen=True)
class BallCase:
    case_id: str
    scenario_id: str
    target_point_b: Vector3
    approach_direction_b: Vector3
    impact_speed_mps: float
    contact_clearance_m: float


@dataclass(frozen=True)
class PreflightManifest:
    protocol_id: str
    seed: int
    mass_kg: float
    arm_length_m: float
    settle_s: float
    preimpact_timeout_s: float
    recovery_window_s: float
    recovery_dwell_s: float
    final_rms_window_s: float
    thresholds: Mapping[str, float]
    structured: tuple[ImpactCase, ...]
    randomized: tuple[ImpactCase, ...]
    continuous: tuple[ContinuousEpisode, ...]
    balls: tuple[BallCase, ...]

    @property
    def episode_count(self) -> int:
        return len(self.structured) + len(self.randomized) + len(self.continuous) + len(self.balls)

    @property
    def impact_count(self) -> int:
        return (
            len(self.structured)
            + len(self.randomized)
            + sum(len(episode.impacts) for episode in self.continuous)
            + len(self.balls)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_structured(config: Mapping[str, Any]) -> tuple[ImpactCase, ...]:
    settings = config["structured"]
    duration_s = float(settings["duration_s"])
    levels = tuple(
        (str(item["name"]), float(item["impulse_ns"])) for item in settings["levels"]
    )
    cases: list[ImpactCase] = []
    for template in settings["templates"]:
        template_id = str(template["template_id"])
        point = _vector3(template["application_point_b"], "application_point_b")
        direction = _unit_vector(template["direction_b"], "direction_b")
        for level, impulse_ns in levels:
            cases.append(
                ImpactCase(
                    case_id=f"S_{template_id}_{level}",
                    group="structured",
                    template_id=template_id,
                    application_point_b=point,
                    direction_b=direction,
                    impulse_ns=impulse_ns,
                    duration_s=duration_s,
                    level=level,
                )
            )
    return tuple(cases)


def _build_random(config: Mapping[str, Any], rng: random.Random) -> tuple[ImpactCase, ...]:
    settings = config["randomized"]
    regions: list[str] = []
    region_bounds: dict[str, tuple[float, float]] = {}
    for item in settings["regions"]:
        name = str(item["name"])
        count = int(item["count"])
        bounds = (float(item["radius_m"][0]), float(item["radius_m"][1]))
        regions.extend([name] * count)
        region_bounds[name] = bounds
    count = len(regions)
    impulse_values = _linspace(
        float(settings["impulse_ns"][0]), float(settings["impulse_ns"][1]), count
    )
    duration_values = _linspace(
        float(settings["duration_s"][0]), float(settings["duration_s"][1]), count
    )
    rng.shuffle(impulse_values)
    rng.shuffle(duration_values)
    z_min, z_max = (float(value) for value in settings["z_offset_m"])

    cases: list[ImpactCase] = []
    for index, region in enumerate(regions):
        radius = rng.uniform(*region_bounds[region])
        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        point = (radius * math.cos(azimuth), radius * math.sin(azimuth), rng.uniform(z_min, z_max))
        impulse_ns = impulse_values[index]
        cases.append(
            ImpactCase(
                case_id=f"R{index + 1:02d}",
                group="randomized",
                application_point_b=point,
                direction_b=_uniform_sphere(rng),
                impulse_ns=impulse_ns,
                duration_s=duration_values[index],
                level="random",
                region=region,
            )
        )
    return tuple(cases)


def _random_impact(
    rng: random.Random,
    *,
    case_id: str,
    level: str,
    impulse_ns: float,
    settings: Mapping[str, Any],
) -> ImpactCase:
    radius = rng.uniform(float(settings["radius_m"][0]), float(settings["radius_m"][1]))
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    z = rng.uniform(float(settings["z_offset_m"][0]), float(settings["z_offset_m"][1]))
    duration = rng.uniform(float(settings["duration_s"][0]), float(settings["duration_s"][1]))
    return ImpactCase(
        case_id=case_id,
        group="continuous",
        application_point_b=(radius * math.cos(azimuth), radius * math.sin(azimuth), z),
        direction_b=_uniform_sphere(rng),
        impulse_ns=impulse_ns,
        duration_s=duration,
        level=level,
        region="full_body",
    )


def _build_continuous(config: Mapping[str, Any], rng: random.Random) -> tuple[ContinuousEpisode, ...]:
    settings = config["continuous"]
    level_impulses = {str(key): float(value) for key, value in settings["level_impulses_ns"].items()}
    base_levels = ["small", "medium", "medium", "large"]
    episodes: list[ContinuousEpisode] = []
    for index in range(int(settings["episode_count"])):
        levels = list(base_levels)
        rng.shuffle(levels)
        separated = index < int(settings["recover_each_count"])
        interval_range = settings["separated_intervals_s"] if separated else settings["stress_intervals_s"]
        episode_id = f"C{index + 1:02d}"
        impacts = tuple(
            _random_impact(
                rng,
                case_id=f"{episode_id}_I{impact_index + 1}",
                level=level,
                impulse_ns=level_impulses[level],
                settings=settings,
            )
            for impact_index, level in enumerate(levels)
        )
        intervals = tuple(
            rng.uniform(float(interval_range[0]), float(interval_range[1])) for _ in range(3)
        )
        episodes.append(
            ContinuousEpisode(
                episode_id=episode_id,
                recovery_policy="each" if separated else "final",
                impacts=impacts,
                intervals_s=intervals,
            )
        )
    return tuple(episodes)


def _build_balls(config: Mapping[str, Any]) -> tuple[BallCase, ...]:
    settings = config["balls"]
    cases: list[BallCase] = []
    for scenario in settings["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        target = _vector3(scenario["target_point_b"], "target_point_b")
        direction = _unit_vector(scenario["approach_direction_b"], "approach_direction_b")
        clearance = float(scenario["contact_clearance_m"])
        for speed in settings["impact_speeds_mps"]:
            speed_value = float(speed)
            cases.append(
                BallCase(
                    case_id=f"P_{scenario_id}_{speed_value:g}mps",
                    scenario_id=scenario_id,
                    target_point_b=target,
                    approach_direction_b=direction,
                    impact_speed_mps=speed_value,
                    contact_clearance_m=clearance,
                )
            )
    return tuple(cases)


def build_preflight_manifest(config: Mapping[str, Any]) -> PreflightManifest:
    """Resolve a compact versioned config into an explicit deterministic manifest."""

    seed = int(config["seed"])
    rng = random.Random(seed)
    thresholds = {key: float(value) for key, value in config["thresholds"].items()}
    required = {"position_error", "linear_speed", "attitude_error_rad", "angular_speed"}
    if set(thresholds) != required:
        raise ValueError(f"thresholds must be exactly {sorted(required)}")
    manifest = PreflightManifest(
        protocol_id=str(config["protocol_id"]),
        seed=seed,
        mass_kg=float(config["mass_kg"]),
        arm_length_m=float(config["arm_length_m"]),
        settle_s=float(config["settle_s"]),
        preimpact_timeout_s=float(config.get("preimpact_timeout_s", config["settle_s"])),
        recovery_window_s=float(config["recovery_window_s"]),
        recovery_dwell_s=float(config["recovery_dwell_s"]),
        final_rms_window_s=float(config["final_rms_window_s"]),
        thresholds=thresholds,
        structured=_build_structured(config),
        randomized=_build_random(config, rng),
        continuous=_build_continuous(config, rng),
        balls=_build_balls(config),
    )
    if manifest.episode_count != 74 or manifest.impact_count != 92:
        raise ValueError(
            "preflight_validation_v1 must resolve to exactly 74 episodes and 92 impacts; "
            f"got {manifest.episode_count} and {manifest.impact_count}"
        )
    return manifest


def load_preflight_manifest(path: str | Path) -> PreflightManifest:
    return build_preflight_manifest(json.loads(Path(path).read_text(encoding="utf-8")))
