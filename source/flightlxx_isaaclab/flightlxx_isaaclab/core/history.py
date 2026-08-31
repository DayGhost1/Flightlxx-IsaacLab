from __future__ import annotations

import torch


class VectorizedHistory:
    """Oldest-to-newest GPU history with explicit repeat-fill reset semantics."""

    def __init__(self, num_envs: int, length: int, feature_dim: int, device: torch.device | str):
        self.buffer = torch.zeros(num_envs, length, feature_dim, device=device)

    def reset(self, env_ids: torch.Tensor, initial_feature: torch.Tensor) -> None:
        self.buffer[env_ids] = initial_feature.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)

    def append(self, feature: torch.Tensor) -> None:
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1] = feature

    def latest(self, length: int) -> torch.Tensor:
        return self.buffer[:, -length:]

