"""Minimal FastTD3 network extension for raw fast/slow FlightLxx histories.

The replay buffer deliberately stores raw histories.  Online and target networks
each own a causal encoder because FastTD3 constructs and soft-updates them as
ordinary model parameters.
"""

from __future__ import annotations

import math

import torch

from fast_td3 import Actor as FastTD3Actor
from fast_td3 import Critic as FastTD3Critic

from .core import CausalTCN


STATE_DIM = 13
ACTION_DIM = 4
FEATURE_DIM = 17
FAST_LENGTH = 4
SLOW_LENGTH = 32
LATENT_DIM = 24
POLICY_RAW_DIM = STATE_DIM + FEATURE_DIM * (FAST_LENGTH + SLOW_LENGTH)
POLICY_ENCODED_DIM = STATE_DIM + FEATURE_DIM * FAST_LENGTH + LATENT_DIM


class _HistoryEncoderMixin:
    def _init_history_encoder(self):
        self.history_tcn = CausalTCN(FEATURE_DIM, LATENT_DIM).to(self.device)

    def _encode_policy(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] < POLICY_RAW_DIM:
            raise ValueError(f"Expected at least {POLICY_RAW_DIM} observation values, got {observation.shape[-1]}")
        current_fast_end = STATE_DIM + FEATURE_DIM * FAST_LENGTH
        slow = observation[:, current_fast_end:POLICY_RAW_DIM].reshape(-1, SLOW_LENGTH, FEATURE_DIM)
        return torch.cat((observation[:, :current_fast_end], self.history_tcn(slow)), dim=-1)


class HistoryActor(_HistoryEncoderMixin, FastTD3Actor):
    def __init__(self, n_obs: int, *args, **kwargs):
        if n_obs != POLICY_RAW_DIM:
            raise ValueError(f"FlightLxx actor requires n_obs={POLICY_RAW_DIM}; got {n_obs}")
        super().__init__(n_obs=POLICY_ENCODED_DIM, *args, **kwargs)
        self._init_history_encoder()
        # With std_max=0.20 this gives [0.15, 0.20, 0.20, 0.15] at maximum noise.
        self.register_buffer("action_noise_scale", torch.tensor([0.75, 1.0, 1.0, 0.75], device=self.device))
        self.exploration_noise_rho = 0.85
        self.register_buffer(
            "exploration_noise_state",
            torch.zeros(self.n_envs, self.n_act, device=self.device),
            persistent=False,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return super().forward(self._encode_policy(obs))

    def explore(self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False) -> torch.Tensor:
        if dones is not None and dones.sum() > 0:
            new_scales = torch.rand(self.n_envs, 1, device=obs.device) * (self.std_max - self.std_min) + self.std_min
            self.noise_scales.copy_(torch.where(dones.view(-1, 1) > 0, new_scales, self.noise_scales))
            self.exploration_noise_state[dones.bool()] = 0.0
        action = self(obs)
        if deterministic:
            return action
        innovation_scale = math.sqrt(1.0 - self.exploration_noise_rho**2)
        self.exploration_noise_state.mul_(self.exploration_noise_rho).add_(
            torch.randn_like(action), alpha=innovation_scale
        )
        noisy_action = action + self.exploration_noise_state * self.noise_scales * self.action_noise_scale
        return noisy_action.clamp(-1.0, 1.0)


class HistoryCritic(_HistoryEncoderMixin, FastTD3Critic):
    def __init__(self, n_obs: int, *args, **kwargs):
        if n_obs < POLICY_RAW_DIM:
            raise ValueError(f"FlightLxx critic requires policy prefix of {POLICY_RAW_DIM}; got {n_obs}")
        self.raw_critic_dim = n_obs
        self.privileged_dim = n_obs - POLICY_RAW_DIM
        super().__init__(n_obs=POLICY_ENCODED_DIM + self.privileged_dim, *args, **kwargs)
        self._init_history_encoder()

    def _encode_critic(self, obs: torch.Tensor) -> torch.Tensor:
        policy = self._encode_policy(obs[:, :POLICY_RAW_DIM])
        return torch.cat((policy, obs[:, POLICY_RAW_DIM:]), dim=-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        return super().forward(self._encode_critic(obs), actions)

    def projection(self, obs, actions, rewards, bootstrap, discount):
        return super().projection(self._encode_critic(obs), actions, rewards, bootstrap, discount)
