from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, padding=self.left_padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y[..., : -self.left_padding] if self.left_padding else y


class CausalTCN(nn.Module):
    """Causal temporal encoder; input is [batch, time, features]."""

    def __init__(self, feature_dim: int, latent_dim: int = 24, channels: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = feature_dim
        for level, out_channels in enumerate(channels):
            layers.extend((CausalConv1d(in_channels, out_channels, 3, 2**level), nn.ReLU()))
            in_channels = out_channels
        self.temporal = nn.Sequential(*layers)
        self.projection = nn.Linear(in_channels, latent_dim)

    def sequence(self, history: torch.Tensor) -> torch.Tensor:
        return self.projection(self.temporal(history.transpose(1, 2)).transpose(1, 2))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.sequence(history)[:, -1]

