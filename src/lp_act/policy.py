"""Shared LeRobot ACT backbone configuration for the two phase-1 policies."""

from __future__ import annotations

from .r1pro import ACTION_DIM, LP_ACTION_DIM, STATE_DIM
from .training_data import CAMERA_KEYS, IMAGE_SIZE


def make_act_policy(mode: str, *, learning_rate: float = 1e-5):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    if mode not in {"standard", "lp"}:
        raise ValueError("mode must be 'standard' or 'lp'")
    output_dim = ACTION_DIM if mode == "standard" else LP_ACTION_DIM
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMAGE_SIZE))
            for key in CAMERA_KEYS
        },
    }
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(output_dim,))}
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=32,
        n_action_steps=32,
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_vae=True,
        optimizer_lr=learning_rate,
        optimizer_lr_backbone=learning_rate,
    )
    return ACTPolicy(config)
