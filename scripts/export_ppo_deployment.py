#!/usr/bin/env python3
"""Export deterministic PPO inference including its frozen normalization."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from flightlxx_isaaclab.ppo import load_ppo_policy, read_ppo_checkpoint


class DeploymentPolicy(nn.Module):
    def __init__(self, actor, normalizer):
        super().__init__()
        self.actor = actor
        self.register_buffer('mean', normalizer._mean.clone())
        self.register_buffer('std', normalizer._std.clone())
        self.eps = normalizer.eps

    def forward(self, observation):
        return self.actor((observation - self.mean) / (self.std + self.eps))


def export_policy(checkpoint, output_dir):
    import onnxruntime as ort
    actor, normalizer, step = load_ppo_policy(checkpoint, 'cpu')
    model = DeploymentPolicy(actor, normalizer).eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'ppo_actor_step_{step:08d}.onnx'
    generator = torch.Generator().manual_seed(123)
    observations = normalizer._mean + torch.randn(16, 625, generator=generator) * (normalizer._std + normalizer.eps)
    torch.onnx.export(model, observations[:1], str(path), opset_version=17,
                      input_names=['observation'], output_names=['action'],
                      dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}})
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected = model(observations).numpy()
    actual = session.run(None, {'observation': observations.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    error = float(np.max(np.abs(actual - expected)))
    np.savez(out / 'ppo_golden_vectors.npz', observations=observations.numpy(), actions=expected)
    metadata = {
        'interface_version': 'snowyowl3_ppo_v1', 'algorithm': 'PPO',
        'checkpoint': str(Path(checkpoint).resolve()), 'global_step': step,
        'observation_dim': 625, 'action_dim': 4,
        'observation_layout': 'current 13 + fast 4x17 + slow 32x17; oldest to newest',
        'action_layout': ['collective_thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'],
        'action_range': [-1, 1], 'normalization_embedded': True,
        'normalization_epsilon': normalizer.eps, 'onnx_max_abs_error': error,
        'jetson_runtime_verified': False,
        'rsl_rl_version': read_ppo_checkpoint(checkpoint)['rsl_rl_version'],
    }
    platform = Path(checkpoint).resolve().parent.parent / 'platform.json'
    if not platform.exists():
        platform = Path(checkpoint).resolve().parent / 'platform.json'
    if platform.exists():
        metadata['platform'] = json.loads(platform.read_text())
    (out / 'ppo_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return path, error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(export_policy(args.checkpoint, args.output_dir))
