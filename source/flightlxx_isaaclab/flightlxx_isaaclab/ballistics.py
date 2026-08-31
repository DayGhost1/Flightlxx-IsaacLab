"""Pure ballistic targeting math used by the Isaac preflight ball trials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


Vector3 = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]


def _values(value: Sequence[float], count: int, name: str) -> tuple[float, ...]:
    if len(value) != count:
        raise ValueError(f"{name} must contain {count} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _rotate(q: QuaternionWXYZ, vector: Vector3) -> Vector3:
    """Rotate a vector by a unit WXYZ quaternion without external dependencies."""

    w, x, y, z = q
    vx, vy, vz = vector
    # q * [0, v] * conjugate(q), expanded to avoid temporary quaternions.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class BallLaunchState:
    origin_position_w: Vector3
    initial_velocity_w: Vector3
    target_position_w: Vector3
    impact_velocity_w: Vector3
    flight_time_s: float


def solve_ball_launch(
    *,
    robot_position_w: Sequence[float],
    robot_quat_w: Sequence[float],
    target_point_b: Sequence[float],
    approach_direction_b: Sequence[float],
    impact_speed_mps: float,
    gravity_w: Sequence[float],
    flight_time_s: float,
    contact_clearance_m: float = 0.0,
) -> BallLaunchState:
    """Solve initial ball state for a requested target and velocity at impact time."""

    position = _values(robot_position_w, 3, "robot_position_w")
    quat = _values(robot_quat_w, 4, "robot_quat_w")
    quat_norm = math.sqrt(sum(component * component for component in quat))
    if not math.isclose(quat_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("robot_quat_w must be a unit quaternion")
    point = _values(target_point_b, 3, "target_point_b")
    direction = _values(approach_direction_b, 3, "approach_direction_b")
    direction_norm = math.sqrt(sum(component * component for component in direction))
    if not math.isclose(direction_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("approach_direction_b must be a unit vector")
    gravity = _values(gravity_w, 3, "gravity_w")
    speed = float(impact_speed_mps)
    duration = float(flight_time_s)
    clearance = float(contact_clearance_m)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("impact_speed_mps must be positive and finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("flight_time_s must be positive and finite")
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("contact_clearance_m must be non-negative and finite")

    point_w = _rotate(quat, point)  # type: ignore[arg-type]
    direction_w = _rotate(quat, direction)  # type: ignore[arg-type]
    impact_velocity = tuple(speed * component for component in direction_w)
    target = tuple(
        position[index] + point_w[index] - direction_w[index] * clearance
        for index in range(3)
    )
    initial_velocity = tuple(
        impact_velocity[index] - gravity[index] * duration for index in range(3)
    )
    origin = tuple(
        target[index]
        - initial_velocity[index] * duration
        - 0.5 * gravity[index] * duration * duration
        for index in range(3)
    )
    return BallLaunchState(
        origin_position_w=origin,  # type: ignore[arg-type]
        initial_velocity_w=initial_velocity,  # type: ignore[arg-type]
        target_position_w=target,  # type: ignore[arg-type]
        impact_velocity_w=impact_velocity,  # type: ignore[arg-type]
        flight_time_s=duration,
    )


def propagate_ball(
    origin_position_w: Sequence[float],
    initial_velocity_w: Sequence[float],
    gravity_w: Sequence[float],
    time_s: float,
) -> tuple[Vector3, Vector3]:
    """Propagate the no-drag ballistic state; primarily used for verification."""

    origin = _values(origin_position_w, 3, "origin_position_w")
    velocity = _values(initial_velocity_w, 3, "initial_velocity_w")
    gravity = _values(gravity_w, 3, "gravity_w")
    time = float(time_s)
    if time < 0.0 or not math.isfinite(time):
        raise ValueError("time_s must be non-negative and finite")
    position = tuple(
        origin[index] + velocity[index] * time + 0.5 * gravity[index] * time * time
        for index in range(3)
    )
    final_velocity = tuple(velocity[index] + gravity[index] * time for index in range(3))
    return position, final_velocity  # type: ignore[return-value]
