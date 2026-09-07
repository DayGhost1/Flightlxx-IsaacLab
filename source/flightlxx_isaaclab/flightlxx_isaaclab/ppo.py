"""RSL-RL PPO integration for the existing 625-value CTBR observation contract.

PPO/GAE/optimization are provided by RSL-RL, not reimplemented here.
"""
from __future__ import annotations

import copy
import importlib.metadata
import os
from pathlib import Path

import torch
from torch import nn
from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module

from .core.tcn import CausalTCN

POLICY_RAW_DIM = 625
CRITIC_RAW_DIM = 645
CHECKPOINT_FORMAT = 'flightlxx_rsl_ppo_v1'


def default_ppo_config() -> dict:
    """Old FlightLxx PPO2 settings adapted to vectorized Isaac Lab rollouts."""
    return {
        'num_steps_per_env': 128, 'save_interval': 100,
        'empirical_normalization': True, 'logger': 'tensorboard',
        'policy': {
            'class_name': 'HistoryActorCritic', 'actor_hidden_dims': [64, 64],
            'critic_hidden_dims': [64, 64], 'activation': 'tanh',
            'init_noise_std': 0.3, 'noise_std_type': 'log',
        },
        'algorithm': {
            'class_name': 'PPO', 'value_loss_coef': 0.5,
            'use_clipped_value_loss': True, 'clip_param': 0.2,
            'entropy_coef': 0.0001, 'num_learning_epochs': 5,
            'num_mini_batches': 8, 'learning_rate': 3e-4,
            'schedule': 'adaptive', 'desired_kl': 0.01,
            'gamma': 0.99, 'lam': 0.95, 'max_grad_norm': 0.5,
        },
    }


class HistoryHead(nn.Module):
    def __init__(self, extra_dim, out_dim, hidden_dims, activation):
        super().__init__()
        self.encoder = CausalTCN(17, 24)
        from rsl_rl.utils import resolve_nn_activation
        layers = []
        dim = 13 + 4 * 17 + 24 + extra_dim
        for width in hidden_dims:
            layers += [nn.Linear(dim, width), resolve_nn_activation(activation)]
            dim = width
        layers.append(nn.Linear(dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        nn.init.orthogonal_(self.mlp[-1].weight, gain=0.01 if out_dim == 4 else 1.0)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, obs):
        slow = obs[:, 81:625].reshape(-1, 32, 17)
        features = torch.cat((obs[:, :81], self.encoder(slow), obs[:, 625:]), dim=-1)
        return self.mlp(features)


class HistoryActorCritic(ActorCritic):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=(64, 64), critic_hidden_dims=(64, 64),
                 activation='tanh', init_noise_std=0.3, noise_std_type='log', **kwargs):
        if (num_actor_obs, num_critic_obs, num_actions) != (625, 645, 4):
            raise ValueError('CTBR PPO requires actor=625, critic=645, actions=4')
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         actor_hidden_dims=list(actor_hidden_dims),
                         critic_hidden_dims=list(critic_hidden_dims), activation=activation,
                         init_noise_std=init_noise_std, noise_std_type=noise_std_type, **kwargs)
        self.actor = HistoryHead(0, 4, actor_hidden_dims, activation)
        self.critic = HistoryHead(20, 1, critic_hidden_dims, activation)


# RSL-RL 2.3 resolves policy names in its runner module.
runner_module.HistoryActorCritic = HistoryActorCritic


class _InitialObservationAdapter:
    """RSL-RL 2.3.3 normalizes step outputs but not learn()'s initial observation."""
    def __init__(self, env, actor_norm, critic_norm):
        self.env = env
        self.actor_norm = actor_norm
        self.critic_norm = critic_norm

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def get_observations(self):
        obs, extras = self.env.get_observations()
        def frozen(norm, x):
            if isinstance(norm, nn.Identity):
                return x
            return (x - norm._mean) / (norm._std + norm.eps)
        extras = dict(extras)
        extras['observations'] = dict(extras['observations'])
        if 'critic' in extras['observations']:
            extras['observations']['critic'] = frozen(self.critic_norm, extras['observations']['critic'])
        return frozen(self.actor_norm, obs), extras


class PPOObservationCache:
    """Mixin for Isaac Lab's RslRlVecEnvWrapper; do not advance history on reads."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def get_observations(self):
        return self._ppo_observations

    def reset(self):
        result = super().reset()
        self._ppo_observations = result
        return result

    def step(self, actions):
        obs, rewards, dones, infos = super().step(actions)
        if 'time_outs' in infos:
            infos['time_outs'] = infos['time_outs'].bool() & ~self.unwrapped.reset_terminated.bool()
        self._ppo_observations = (obs, {'observations': infos['observations']})
        return obs, rewards, dones, infos


class PPORunner(OnPolicyRunner):
    """Add portable checkpoints and environment curriculum to the upstream runner."""
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.saved_config = copy.deepcopy(train_cfg)
        self.env_steps = 0
        self._learning = False
        super().__init__(env, copy.deepcopy(train_cfg), log_dir, device)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._start_iteration = self.current_learning_iteration
        self._learning = True
        raw_env = self.env
        self.env = _InitialObservationAdapter(raw_env, self.obs_normalizer, self.privileged_obs_normalizer)
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.env = raw_env
            self._learning = False
        # Upstream stores the last zero-based iteration, not the next iteration.
        self.current_learning_iteration = self._start_iteration + num_learning_iterations
        self.env_steps += num_learning_iterations * self.num_steps_per_env

    def save(self, path, infos=None):
        completed = self.current_learning_iteration + int(self._learning)
        env_steps = self.env_steps
        if self._learning:
            env_steps += (completed - self._start_iteration) * self.num_steps_per_env
        payload = {
            'format': CHECKPOINT_FORMAT, 'train_cfg': self.saved_config,
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'learning_rate': self.alg.learning_rate,
            'iter': completed, 'global_step': env_steps,
            'total_transitions': self.tot_timesteps, 'total_time': self.tot_time,
            'environment_state': self.env.unwrapped.get_training_state(),
            'obs_norm_state_dict': self.obs_normalizer.state_dict(),
            'privileged_obs_norm_state_dict': self.privileged_obs_normalizer.state_dict(),
            'infos': infos, 'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'rsl_rl_version': importlib.metadata.version('rsl-rl-lib'),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + '.tmp')
        torch.save(payload, tmp)
        os.replace(tmp, destination)

    def load(self, path, load_optimizer=True):
        payload = read_ppo_checkpoint(path, self.device)
        if payload['train_cfg'] != self.saved_config:
            raise ValueError('Resume requires the saved PPO configuration; load train_cfg from checkpoint')
        self.alg.policy.load_state_dict(payload['model_state_dict'])
        self.obs_normalizer.load_state_dict(payload['obs_norm_state_dict'])
        self.privileged_obs_normalizer.load_state_dict(payload['privileged_obs_norm_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(payload['optimizer_state_dict'])
            self.alg.learning_rate = payload['learning_rate']
        self.current_learning_iteration = payload['iter']
        self.env_steps = payload['global_step']
        self.tot_timesteps = payload['total_transitions']
        self.tot_time = payload['total_time']
        self.env.unwrapped.load_training_state(payload['environment_state'])
        torch.set_rng_state(payload['torch_rng_state'].cpu())
        if torch.cuda.is_available() and payload['cuda_rng_state']:
            torch.cuda.set_rng_state_all([x.cpu() for x in payload['cuda_rng_state']])
        return payload['infos']


def read_ppo_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Expected a FlightLxx PPO checkpoint; FastTD3 checkpoints cannot resume PPO')
    return payload


class FrozenNormalizer(EmpiricalNormalization):
    def forward(self, x, update=False):
        if update:
            raise ValueError('Evaluation normalizer is frozen')
        return (x - self._mean) / (self._std + self.eps)


class DeterministicActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, x):
        return self.actor(x).clamp(-1.0, 1.0)

    def explore(self, x, deterministic=True, **kwargs):
        if not deterministic:
            raise ValueError('This policy is for deterministic evaluation only')
        return self(x)


def load_ppo_policy(path, device='cpu'):
    payload = read_ppo_checkpoint(path, device)
    cfg = dict(payload['train_cfg']['policy'])
    cfg.pop('class_name')
    net = HistoryActorCritic(625, 645, 4, **cfg).to(device)
    net.load_state_dict(payload['model_state_dict'])
    normalizer = FrozenNormalizer([625]).to(device)
    if payload['train_cfg']['empirical_normalization']:
        normalizer.load_state_dict(payload['obs_norm_state_dict'])
    else:
        # Identity still supports the evaluation update=False call contract.
        normalizer._std.fill_(1.0 - normalizer.eps)
    return DeterministicActor(net.actor).eval(), normalizer.eval(), payload['global_step']
