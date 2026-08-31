"""Simulator-independent configuration and state for GUI policy playback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def configure_visual_scene(cfg: Any) -> None:
    """Configure a fixed wide view in which translational recovery is visible."""

    cfg.viewer.eye = (8.5, -10.0, 9.5)
    cfg.viewer.lookat = (0.0, 0.0, 5.0)
    cfg.viewer.origin_type = "world"
    cfg.viewer.env_index = 0
    cfg.viewer.asset_name = None
    cfg.robot.spawn.visual_material.diffuse_color = (0.05, 0.85, 1.0)
    terrain_material = getattr(cfg.terrain, "visual_material", None)
    if terrain_material is not None:
        terrain_material.diffuse_color = (0.70, 0.72, 0.76)


def quaternion_from_x_axis(direction: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return normalized wxyz quaternions rotating +X onto ``direction``.

    The closed-form shortest-arc quaternion is singular when the target is
    exactly -X.  That case is assigned a deterministic 180-degree rotation
    around +Y, which keeps marker orientation stable across frames.
    """

    if direction.shape[-1] != 3:
        raise ValueError(f"direction must end in three values, got {direction.shape}")
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if bool(torch.any(norm <= eps)):
        raise ValueError("direction must be non-zero")
    target = direction / norm
    dot = target[..., 0]
    regular = torch.stack(
        (
            1.0 + dot,
            torch.zeros_like(dot),
            -target[..., 2],
            target[..., 1],
        ),
        dim=-1,
    )
    antiparallel = torch.zeros_like(regular)
    antiparallel[..., 2] = 1.0
    quaternion = torch.where((dot <= -1.0 + eps).unsqueeze(-1), antiparallel, regular)
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(eps)


def transform_body_points(
    root_position_w: torch.Tensor,
    root_quaternion_wxyz: torch.Tensor,
    body_points: torch.Tensor,
) -> torch.Tensor:
    """Transform a fixed set of body-frame points by batched root poses."""

    if root_position_w.shape[-1] != 3 or root_quaternion_wxyz.shape[-1] != 4:
        raise ValueError("root pose tensors must end in 3-position and 4-quaternion values")
    if root_position_w.shape[:-1] != root_quaternion_wxyz.shape[:-1]:
        raise ValueError("root position and quaternion batch shapes must match")
    if body_points.ndim != 2 or body_points.shape[-1] != 3:
        raise ValueError("body_points must have shape (N, 3)")
    quaternion = root_quaternion_wxyz / torch.linalg.vector_norm(
        root_quaternion_wxyz,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    count = body_points.shape[0]
    batch_shape = root_position_w.shape[:-1]
    points_b = body_points.reshape((1,) * len(batch_shape) + (count, 3)).expand(batch_shape + (count, 3))
    quaternion = quaternion.unsqueeze(-2).expand(batch_shape + (count, 4))
    vector = quaternion[..., 1:]
    cross = torch.linalg.cross(vector, points_b, dim=-1)
    rotated = points_b + 2.0 * (
        quaternion[..., :1] * cross
        + torch.linalg.cross(vector, cross, dim=-1)
    )
    return root_position_w.unsqueeze(-2) + rotated


@dataclass
class PlaybackState:
    """Small state machine shared by the GUI callbacks and playback loop."""

    phase: str = "loading"
    status: str = "Loading task and renderer..."
    _start_requested: bool = False

    def mark_ready(self) -> None:
        self.phase = "ready"
        self.status = "Ready: click Start / Replay"

    def request_start(self) -> bool:
        if self.phase not in {"ready", "complete"} or self._start_requested:
            return False
        self._start_requested = True
        self.status = "Replay requested..."
        return True

    def consume_start_request(self) -> bool:
        requested = self._start_requested
        self._start_requested = False
        return requested

    def mark_running(self) -> None:
        self.phase = "running"
        self.status = "Running fixed five-impact evaluation"

    def mark_complete(self, *, passed: bool, recovered_count: int) -> None:
        self.phase = "complete"
        outcome = "PASS" if passed else "FAIL"
        self.status = f"{outcome}: recovered {recovered_count}/5 — click Start / Replay to run again"
