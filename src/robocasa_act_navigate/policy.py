"""Unmodified ACT action representation with a random ResNet-18 backbone."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .schema import ACTION_DIM, CAMERA_KEYS, CHUNK_SIZE, EXECUTION_STEPS, IMAGE_SIZE, STATE_DIM


def _tensor_stats(value: dict) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(array, dtype=torch.float32) for name, array in value.items() if name in {"mean", "std", "min", "max"}}


def _integrate_se2_increments_torch(increments: torch.Tensor) -> torch.Tensor:
    """Differentiably integrate body-frame SE(2) Lie increments."""

    poses = []
    current = torch.zeros_like(increments[:, 0])
    for index in range(increments.shape[1]):
        delta = increments[:, index]
        vx, vy, theta = delta.unbind(-1)
        small = theta.abs() < 1e-4
        safe_theta = torch.where(small, torch.ones_like(theta), theta)
        a_exact = torch.sin(theta) / safe_theta
        b_exact = (1.0 - torch.cos(theta)) / safe_theta
        a = torch.where(small, 1.0 - theta.square() / 6.0, a_exact)
        b = torch.where(small, theta / 2.0 - theta.pow(3) / 24.0, b_exact)
        tx = a * vx - b * vy
        ty = b * vx + a * vy
        c, s = torch.cos(current[:, 2]), torch.sin(current[:, 2])
        current = torch.stack(
            (current[:, 0] + c * tx - s * ty,
             current[:, 1] + s * tx + c * ty,
             current[:, 2] + theta),
            dim=-1,
        )
        poses.append(current)
    return torch.stack(poses, dim=1)


def make_policy(stats: dict, *, learning_rate: float = 1e-5, device: str = "cuda"):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IMAGE_SIZE)) for key in CAMERA_KEYS},
    }
    model_spec = stats.get("model", {})
    geometry_loss = stats.get("training", {}).get("geometry_loss")
    chunk_size = int(model_spec.get("chunk_size", CHUNK_SIZE))
    action_dim = int(model_spec.get("action_dim", len(stats["features"]["action"]["mean"])))
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=min(EXECUTION_STEPS, chunk_size),
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        use_vae=True,
        optimizer_lr=learning_rate,
        optimizer_lr_backbone=learning_rate,
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
    class WeightedACTPolicy(ACTPolicy):
        """ACT with an optional per-target loss mask supplied by B2 V3.

        LeRobot's stock masked L1 averages padded zeros into the denominator.
        B2 V3 instead supervises STOP on every query and supplies explicit
        per-query/per-channel weights, normalized by their actual sum.
        """

        def forward(self, batch: dict[str, torch.Tensor]):
            loss_weight = batch.get("action_loss_weight")
            physical_target = batch["action"]
            batch = self.normalize_inputs(batch)
            if self.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = [batch[key] for key in self.config.image_features]
            batch = self.normalize_targets(batch)
            actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
            valid = (~batch["action_is_pad"]).unsqueeze(-1).to(actions_hat.dtype)
            if loss_weight is None:
                weight = valid.expand_as(actions_hat)
            else:
                weight = loss_weight.to(actions_hat.dtype) * valid
            l1_loss = (F.l1_loss(batch["action"], actions_hat, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
            loss_dict = {"l1_loss": l1_loss.item()}
            geometry_total = torch.zeros((), device=actions_hat.device, dtype=torch.float32)
            if geometry_loss:
                physical_prediction = self.unnormalize_outputs({"action": actions_hat})["action"].float()
                physical_target_f = physical_target.float()
                pred_path = _integrate_se2_increments_torch(physical_prediction[..., :3])
                true_path = _integrate_se2_increments_torch(physical_target_f[..., :3])
                continue_mask = physical_target_f[..., 11] >= 0.0
                last = continue_mask.sum(1).clamp(min=1) - 1
                rows = torch.arange(len(last), device=last.device)
                pred_end, true_end = pred_path[rows, last], true_path[rows, last]
                endpoint_loss = torch.linalg.vector_norm(pred_end[:, :2] - true_end[:, :2], dim=-1).mean()
                yaw_loss = (1.0 - torch.cos(pred_end[:, 2] - true_end[:, 2])).mean()

                first_pred, first_true = pred_path[:, 0, :2], true_path[:, 0, :2]
                moving = torch.linalg.vector_norm(first_true, dim=-1) > 1e-4
                cosine = F.cosine_similarity(first_pred[moving], first_true[moving], dim=-1) if moving.any() else first_pred.new_ones(1)
                direction_loss = (1.0 - cosine).mean()

                pair_mask = (continue_mask[:, 1:] & continue_mask[:, :-1]).unsqueeze(-1)
                pred_change = physical_prediction[:, 1:, :3] - physical_prediction[:, :-1, :3]
                true_change = physical_target_f[:, 1:, :3] - physical_target_f[:, :-1, :3]
                smoothness_loss = (torch.abs(pred_change - true_change) * pair_mask).sum() / (3.0 * pair_mask.sum().clamp_min(1))

                yaw_radius = float(geometry_loss.get("yaw_radius_m", 0.25))
                pred_metric = torch.sqrt(physical_prediction[..., 0].square() + physical_prediction[..., 1].square() +
                                         (yaw_radius * physical_prediction[..., 2]).square() + 1e-12)
                true_metric = torch.sqrt(physical_target_f[..., 0].square() + physical_target_f[..., 1].square() +
                                         (yaw_radius * physical_target_f[..., 2]).square() + 1e-12)
                progress_loss = (torch.abs(pred_metric - true_metric) * continue_mask).sum() / continue_mask.sum().clamp_min(1)
                terms = {
                    "endpoint_loss": endpoint_loss,
                    "yaw_geometry_loss": yaw_loss,
                    "direction_loss": direction_loss,
                    "smoothness_loss": smoothness_loss,
                    "progress_loss": progress_loss,
                }
                geometry_total = sum(float(geometry_loss.get(name, 0.0)) * value for name, value in terms.items())
                loss_dict.update({name: value.item() for name, value in terms.items()})
                loss_dict["geometry_loss"] = geometry_total.item()
            if self.config.use_vae:
                mu32, log32 = mu_hat.float(), log_sigma_x2_hat.float()
                mean_kld = (-0.5 * (1 + log32 - mu32.pow(2) - log32.exp())).sum(-1).mean()
                loss_dict["kld_loss"] = mean_kld.item()
                return l1_loss + geometry_total + mean_kld * self.config.kl_weight, loss_dict
            return l1_loss + geometry_total, loss_dict

    return WeightedACTPolicy(config, dataset_stats=dataset_stats)
