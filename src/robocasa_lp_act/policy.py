"""One ACT backbone, with only the paired output parameterization changed."""

from __future__ import annotations

import numpy as np
import torch

from .schema import ACTION_DIM, CAMERA_KEYS, EXECUTION_STEPS, IMAGE_SIZE, LP_ACTION_DIM, STATE_DIM


def _tensor_stats(value: dict) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(array, dtype=torch.float32) for key, array in value.items() if key in {"mean", "std", "min", "max"}}


def make_act_policy(mode: str, stats_payload: dict, *, learning_rate: float = 1e-5, device: str = "cuda"):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    if mode not in {"standard", "lp"}:
        raise ValueError("mode must be standard or lp")
    output_dim = ACTION_DIM if mode == "standard" else LP_ACTION_DIM
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMAGE_SIZE)) for key in CAMERA_KEYS},
    }
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(output_dim,))}
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=32,
        n_action_steps=EXECUTION_STEPS,
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        optimizer_lr=learning_rate,
        optimizer_lr_backbone=learning_rate,
        device=device.split(":", maxsplit=1)[0],
        use_amp=False,
        push_to_hub=False,
    )
    features = stats_payload["features"]
    dataset_stats = {
        "observation.state": _tensor_stats(features["observation.state"]),
        "action": _tensor_stats(features["standard_action" if mode == "standard" else "lp_action"]),
    }
    image_stats = _tensor_stats(features["image_imagenet"])
    for key in CAMERA_KEYS:
        dataset_stats[key] = {name: value.clone() for name, value in image_stats.items()}
    return ACTPolicy(config, dataset_stats=dataset_stats)
