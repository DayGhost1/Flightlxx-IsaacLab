from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for WXYZ quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_error(q_reference: torch.Tensor, q_current: torch.Tensor) -> torch.Tensor:
    """Return canonicalized inverse(q_reference) * q_current in WXYZ order."""
    error = quat_mul(quat_conjugate(q_reference), q_current)
    error = error / error.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.where(error[..., :1] < 0.0, -error, error)


def attitude_cost(q_reference: torch.Tensor, q_current: torch.Tensor) -> torch.Tensor:
    """Sign-invariant attitude cost in [0, 1]."""
    dot = torch.sum(q_reference * q_current, dim=-1)
    return 1.0 - dot.square().clamp(max=1.0)


def quat_rotate_inverse(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the body frame using WXYZ q_body_to_world."""
    zeros = torch.zeros_like(vector[..., :1])
    pure = torch.cat((zeros, vector), dim=-1)
    return quat_mul(quat_mul(quat_conjugate(q), pure), q)[..., 1:]

