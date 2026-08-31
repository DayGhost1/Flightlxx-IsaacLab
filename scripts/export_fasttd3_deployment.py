#!/usr/bin/env python3
"""Export a FlightLxx HistoryActor checkpoint as one normalized ONNX policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from flightlxx_isaaclab.fast_td3_models import HistoryActor, POLICY_RAW_DIM


ACTION_DIM = 4
NORMALIZATION_EPSILON = 1.0e-2
INTERFACE_VERSION = "snowyowl3_fasttd3_v2"
DEFAULT_PLATFORM_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flightlxx_isaaclab"
    / "flightlxx_isaaclab"
    / "config"
    / "snowyowl3_real_v1.json"
)


class DeploymentPolicy(nn.Module):
    def __init__(self, actor: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.actor = actor
        self.register_buffer("observation_mean", mean.detach().float().reshape(1, POLICY_RAW_DIM))
        self.register_buffer("observation_std", std.detach().float().reshape(1, POLICY_RAW_DIM))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.observation_mean) / (
            self.observation_std + NORMALIZATION_EPSILON
        )
        return self.actor(normalized)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deployment_filenames(global_step: int) -> dict[str, str]:
    if int(global_step) <= 0:
        raise ValueError("global_step must be positive")
    suffix = f"{int(global_step):08d}"
    return {
        "onnx": f"fasttd3_history_actor_step_{suffix}.onnx",
        "engine": f"fasttd3_history_actor_step_{suffix}.engine",
        "vectors": "fasttd3_golden_vectors.npz",
        "metadata": "fasttd3_metadata.json",
        "deployment_checkpoint": f"fasttd3_deployment_checkpoint_step_{suffix}.pt",
    }


def read_platform_contract(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        target = payload["target"]
        motor = payload["motor"]
        propeller = payload["propeller"]
        position = [float(value) for value in target["position_vicon_m"]]
        rpy = [float(value) for value in target["rpy_deg"]]
        thrust_coefficient = float(propeller["thrust_coefficient_n_per_rpm2"])
        official_max_rpm = float(motor["rpm_official_max"])
        policy_fraction = float(motor["policy_rpm_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid SnowyOwl3 platform contract: {path}") from error
    if len(position) != 3 or len(rpy) != 3:
        raise ValueError("target position and RPY must each contain three values")
    if rpy != [0.0, 0.0, 0.0]:
        raise ValueError("deployment target Roll, Pitch, and Yaw must all be zero")
    if thrust_coefficient <= 0.0 or official_max_rpm <= 0.0:
        raise ValueError("thrust coefficient and official maximum RPM must be positive")
    if not 0.0 < policy_fraction <= 1.0:
        raise ValueError("policy RPM fraction must lie in (0, 1]")
    policy_max_rpm = official_max_rpm * policy_fraction
    return {
        "target_position_world": position,
        "target_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "target_is_placeholder": bool(target.get("placeholder", False)),
        "motor": {
            "official_max_rpm": official_max_rpm,
            "policy_rpm_fraction": policy_fraction,
            "policy_max_rpm": policy_max_rpm,
            "thrust_coefficient_n_per_rpm2": thrust_coefficient,
            "policy_max_total_thrust_n": 4.0
            * thrust_coefficient
            * policy_max_rpm**2,
        },
    }


def build_policy(checkpoint: dict) -> DeploymentPolicy:
    saved = checkpoint["args"]
    actor = HistoryActor(
        n_obs=POLICY_RAW_DIM,
        n_act=ACTION_DIM,
        num_envs=int(saved["num_envs"]),
        device="cpu",
        init_scale=float(saved.get("init_scale", 0.01)),
        hidden_dim=int(saved.get("actor_hidden_dim", 512)),
        std_min=float(saved.get("std_min", 0.05)),
        std_max=float(saved.get("std_max", 0.20)),
        sim_type=str(saved.get("sim_type", "")),
        sim_dimension=int(saved.get("sim_dimension", 64)),
        seq_len=int(saved.get("actor_seq_len", 8)),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    normalizer = checkpoint.get("obs_normalizer_state")
    if not normalizer:
        raise ValueError("checkpoint does not contain obs_normalizer_state")
    mean = normalizer["_mean"]
    std = normalizer["_std"]
    if tuple(mean.shape) != (1, POLICY_RAW_DIM) or tuple(std.shape) != (1, POLICY_RAW_DIM):
        raise ValueError(f"unexpected normalizer shapes: mean={tuple(mean.shape)}, std={tuple(std.shape)}")

    policy = DeploymentPolicy(actor, mean, std)
    policy.eval()
    return policy


def make_golden_inputs() -> np.ndarray:
    generator = np.random.default_rng(20260827)
    hover_state = np.zeros(13, dtype=np.float32)
    hover_state[6] = 1.0
    hover_feature = np.concatenate((hover_state, np.zeros(ACTION_DIM, dtype=np.float32)))
    hover_observation = np.concatenate(
        (hover_state, np.tile(hover_feature, 4), np.tile(hover_feature, 32))
    )
    return np.stack(
        (
            hover_observation,
            hover_observation + generator.normal(0.0, 0.01, POLICY_RAW_DIM).astype(np.float32),
            hover_observation + generator.normal(0.0, 0.10, POLICY_RAW_DIM).astype(np.float32),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform-config", type=Path, default=DEFAULT_PLATFORM_CONFIG)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    platform_config_path = args.platform_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_step = int(checkpoint.get("global_step", -1))
    names = deployment_filenames(global_step)
    onnx_path = output_dir / names["onnx"]
    vectors_path = output_dir / names["vectors"]
    metadata_path = output_dir / names["metadata"]
    deployment_checkpoint_path = output_dir / names["deployment_checkpoint"]
    platform_contract = read_platform_contract(platform_config_path)
    policy = build_policy(checkpoint)

    dummy = torch.zeros(1, POLICY_RAW_DIM, dtype=torch.float32)
    with torch.inference_mode():
        dummy_output = policy(dummy)
    if tuple(dummy_output.shape) != (1, ACTION_DIM):
        raise ValueError(f"unexpected policy output shape {tuple(dummy_output.shape)}")

    torch.onnx.export(
        policy,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    golden_inputs = make_golden_inputs()
    with torch.inference_mode():
        golden_actions = policy(torch.from_numpy(golden_inputs)).cpu().numpy().astype(np.float32)
    if golden_actions.shape != (3, ACTION_DIM) or not np.isfinite(golden_actions).all():
        raise ValueError("invalid golden policy output")
    np.savez(vectors_path, observations=golden_inputs, actions=golden_actions)

    # Keep the exact deployable checkpoint parameters without the critic,
    # replay buffer and optimizer states that are only useful for training.
    torch.save(
        {
            "actor_state_dict": checkpoint["actor_state_dict"],
            "args": checkpoint["args"],
            "global_step": checkpoint["global_step"],
            "obs_normalizer_state": checkpoint["obs_normalizer_state"],
            "source_checkpoint_sha256": file_sha256(checkpoint_path),
        },
        deployment_checkpoint_path,
    )

    metadata = {
        "action_dim": ACTION_DIM,
        "action_order": ["collective", "roll_rate", "pitch_rate", "yaw_rate"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "deployment_checkpoint": str(deployment_checkpoint_path),
        "deployment_checkpoint_filename": deployment_checkpoint_path.name,
        "deployment_checkpoint_sha256": file_sha256(deployment_checkpoint_path),
        "engine_filename": names["engine"],
        "engine_sha256": None,
        "global_step": global_step,
        "hardware_qualified": False,
        "history": {
            "fast_length": 4,
            "feature_dim": 17,
            "ordering": "oldest_to_newest",
            "reset": "repeat_first_feature",
            "slow_length": 32,
            "state_dim": 13,
        },
        "interface_version": INTERFACE_VERSION,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "normalizer": {
            "epsilon": NORMALIZATION_EPSILON,
            "location": "baked_into_engine",
        },
        "onnx_opset": args.opset,
        "onnx_sha256": file_sha256(onnx_path),
        "platform_config_sha256": file_sha256(platform_config_path),
        "policy_raw_dim": POLICY_RAW_DIM,
        "precision": "fp32",
        "quaternion_order": "wxyz",
        **platform_contract,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "deployment_checkpoint": str(deployment_checkpoint_path),
        "deployment_checkpoint_sha256": metadata["deployment_checkpoint_sha256"],
        "global_step": metadata["global_step"],
        "expected_engine": str(output_dir / names["engine"]),
        "golden_actions": golden_actions.tolist(),
        "metadata": str(metadata_path),
        "onnx": str(onnx_path),
        "onnx_sha256": metadata["onnx_sha256"],
        "vectors": str(vectors_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
