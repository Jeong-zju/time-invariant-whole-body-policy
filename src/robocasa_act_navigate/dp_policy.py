"""LeRobot Diffusion Policy configured for B2 whole-body spatial tokens."""

from __future__ import annotations

import torch

from .schema import CAMERA_KEYS, IMAGE_SIZE, STATE_DIM


def _tensor_stats(value: dict) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(array, dtype=torch.float32)
        for name, array in value.items()
        if name in {"mean", "std", "min", "max"}
    }


def make_dp_policy(
    stats: dict,
    *,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    num_inference_steps: int | None = None,
):
    """Build stock Diffusion Policy with the same B2 observations and labels.

    This intentionally does not reuse ACT's CVAE or geometry loss.  The
    experiment changes only the policy family while keeping the B2 V4 target,
    data split, cameras, state, and physical-space audit fixed.
    """

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    model_spec = stats.get("model", {})
    horizon = int(model_spec.get("chunk_size", 8))
    action_dim = int(model_spec.get("action_dim", len(stats["features"]["action"]["mean"])))
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMAGE_SIZE)) for key in CAMERA_KEYS},
    }
    config = DiffusionConfig(
        n_obs_steps=1,
        horizon=horizon,
        n_action_steps=horizon,
        drop_n_last_frames=0,
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        vision_backbone="resnet18",
        crop_shape=None,
        crop_is_random=False,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        use_separate_rgb_encoder_per_camera=False,
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        num_inference_steps=num_inference_steps,
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        do_mask_loss_for_padding=False,
        optimizer_lr=learning_rate,
        scheduler_name="cosine",
        scheduler_warmup_steps=500,
        device=device.split(":", 1)[0],
        use_amp=False,
        push_to_hub=False,
    )
    features = stats["features"]
    dataset_stats = {
        "observation.state": _tensor_stats(features["observation.state"]),
        "action": _tensor_stats(features["action"]),
    }
    image_stats = _tensor_stats(features["image_imagenet"])
    for key in CAMERA_KEYS:
        dataset_stats[key] = {name: value.clone() for name, value in image_stats.items()}
    return DiffusionPolicy(config, dataset_stats=dataset_stats)


def temporalize_dp_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add the one-step temporal axis expected by LeRobot Diffusion Policy."""

    result = dict(batch)
    result.pop("action_loss_weight", None)
    result["observation.state"] = result["observation.state"].unsqueeze(1)
    for key in CAMERA_KEYS:
        result[key] = result[key].unsqueeze(1)
    return result


@torch.no_grad()
def predict_dp_action_chunk(policy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run deterministic-call-site DP inference without rollout queues."""

    temporal = temporalize_dp_batch(batch)
    temporal = policy.normalize_inputs(temporal)
    temporal = dict(temporal)
    temporal["observation.images"] = torch.stack(
        [temporal[key] for key in policy.config.image_features], dim=-4
    )
    actions = policy.diffusion.generate_actions(temporal)
    return policy.unnormalize_outputs({"action": actions})["action"]
