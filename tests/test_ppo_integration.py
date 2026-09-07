import copy
import torch
import pytest

from flightlxx_isaaclab.ppo import (
    HistoryActorCritic, PPORunner, PPOObservationCache, default_ppo_config, load_ppo_policy,
)


class ToyEnv:
    num_envs = 4
    num_actions = 4
    device = 'cpu'
    max_episode_length = 4
    cfg = {}

    def __init__(self):
        self.unwrapped = self
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.x = torch.randn(4, 625)
        self.state = {'curriculum': {'difficulty': 0.3}}

    def get_observations(self):
        return self.x, {'observations': {'critic': torch.cat((self.x, torch.ones(4, 20)), -1)}}

    def step(self, actions):
        self.episode_length_buf += 1
        done = self.episode_length_buf >= 4
        self.episode_length_buf[done] = 0
        self.x = torch.randn(4, 625)
        obs, info = self.get_observations()
        info['time_outs'] = done.float()
        return obs, -actions.square().sum(-1), done.long(), info

    def get_training_state(self):
        return copy.deepcopy(self.state)

    def load_training_state(self, state):
        self.state = copy.deepcopy(state)


def config():
    c = default_ppo_config()
    c['num_steps_per_env'] = 4
    c['algorithm'].update(num_learning_epochs=1, num_mini_batches=2)
    return c


def test_history_actor_critic_gradients_and_privileged_isolation():
    net = HistoryActorCritic(625, 645, 4)
    obs = torch.randn(3, 625)
    critic_obs = torch.cat((obs, torch.randn(3, 20)), -1)
    action = net.act(obs)
    loss = -net.get_actions_log_prob(action).mean() + net.evaluate(critic_obs).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.actor.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.critic.encoder.parameters())
    assert net.act_inference(obs).shape == (3, 4)
    with pytest.raises(ValueError):
        HistoryActorCritic(624, 645, 4)


def test_real_rsl_ppo_update_save_resume_and_deployment(tmp_path):
    torch.set_num_threads(1)
    env = ToyEnv()
    runner = PPORunner(env, config(), str(tmp_path), device='cpu')
    before = [p.detach().clone() for p in runner.alg.policy.actor.parameters()]
    runner.learn(2)
    assert any(not torch.equal(p, b) for p, b in zip(runner.alg.policy.actor.parameters(), before))
    checkpoint = tmp_path / 'checkpoint.pt'
    runner.save(str(checkpoint))
    policy, normalizer, step = load_ppo_policy(checkpoint, 'cpu')
    x = torch.randn(7, 625)
    runner.eval_mode()
    expected = runner.alg.policy.act_inference(runner.obs_normalizer(x)).clamp(-1, 1)
    count = normalizer.count.clone()
    assert torch.allclose(policy(normalizer(x, update=False)), expected)
    assert torch.equal(normalizer.count, count)
    assert step == 8
    resumed = PPORunner(ToyEnv(), config(), str(tmp_path / 'resume'), device='cpu')
    resumed.load(str(checkpoint))
    assert resumed.current_learning_iteration == 2
    assert resumed.env.state == env.state
    assert resumed.alg.optimizer.state_dict()['state']
    resumed.learn(1)
    assert resumed.current_learning_iteration == 3

    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts/export_ppo_deployment.py'
    spec = importlib.util.spec_from_file_location('export_ppo', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    onnx_path, error = module.export_policy(checkpoint, tmp_path / 'export')
    assert onnx_path.is_file()
    assert error < 2e-5


def test_on_policy_timeouts_bootstrap_but_crashes_do_not(tmp_path):
    r = PPORunner(ToyEnv(), config(), str(tmp_path), device='cpu')
    obs, info = r.env.get_observations()
    r.alg.act(obs, info['observations']['critic'])
    values = r.alg.transition.values.clone().squeeze(-1)
    timeout = torch.tensor([1., 0., 0., 0.])
    r.alg.process_env_step(torch.ones(4), torch.tensor([1, 1, 0, 0]), {'time_outs': timeout})
    expected = torch.ones(4) + r.alg.gamma * values * timeout
    assert torch.allclose(r.alg.storage.rewards[0, :, 0], expected)


def test_legacy_checkpoint_not_silently_treated_as_ppo(tmp_path):
    p = tmp_path / 'old.pt'
    torch.save({'actor_state_dict': {}}, p)
    with pytest.raises(ValueError, match='PPO'):
        load_ppo_policy(p, 'cpu')


def test_resume_first_rollout_uses_saved_normalization(tmp_path):
    r = PPORunner(ToyEnv(), config(), str(tmp_path), device='cpu')
    r.obs_normalizer._mean.fill_(10)
    r.obs_normalizer._std.fill_(2)
    r.privileged_obs_normalizer._mean.fill_(20)
    r.privileged_obs_normalizer._std.fill_(4)
    obs, extras = r.env.get_observations()
    expected = (obs.clone() - 10) / 2.01
    expected_c = (extras['observations']['critic'].clone() - 20) / 4.01
    seen = []
    original = r.alg.act
    def capture(o, c):
        seen.append((o.clone(), c.clone()))
        return original(o, c)
    r.alg.act = capture
    r.learn(1)
    assert torch.allclose(seen[0][0], expected)
    assert torch.allclose(seen[0][1], expected_c)


def test_observation_reads_do_not_advance_history_and_crash_overrides_timeout():
    class Base:
        def __init__(self):
            self.reads = 0
            self.unwrapped = self
            self.reset_terminated = torch.tensor([True, False])
        def reset(self):
            self.reads += 1
            return torch.full((2, 625), self.reads), {'observations': {}}
        def step(self, actions):
            obs = torch.full((2, 625), 42)
            return obs, torch.zeros(2), torch.ones(2), {'observations': {}, 'time_outs': torch.ones(2)}
    class Cached(PPOObservationCache, Base):
        pass
    env = Cached()
    first, _ = env.get_observations()
    second, _ = env.get_observations()
    assert env.reads == 1
    assert torch.equal(first, second)
    obs, _, _, infos = env.step(torch.zeros(2, 4))
    assert torch.equal(env.get_observations()[0], obs)
    assert infos['time_outs'].tolist() == [False, True]
