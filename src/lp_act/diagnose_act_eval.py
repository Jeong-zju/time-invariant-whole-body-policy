"""Compare an official evaluator ACT observation with held-out demo starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .parquet_data import load_turning_data, split_episodes
from .policy import make_act_policy
from .stats import FeatureStats
from .training_data import CAMERA_KEYS, IMAGENET_MEAN, IMAGENET_STD, TurningACTDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-demo-starts", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--execute-prefix", type=int, default=6)
    return parser.parse_args()


def _physical_errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    absolute = np.abs(prediction - target)
    return {
        "all_mae": float(absolute.mean()),
        "base_mae": float(absolute[..., 0:3].mean()),
        "trunk_mae": float(absolute[..., 3:7].mean()),
        "left_arm_mae": float(absolute[..., 7:14].mean()),
        "left_gripper_mae": float(absolute[..., 14].mean()),
        "right_arm_mae": float(absolute[..., 15:22].mean()),
        "right_gripper_mae": float(absolute[..., 22].mean()),
    }


def _write_montage(
    path: Path,
    eval_data: np.lib.npyio.NpzFile,
    samples: list[dict[str, torch.Tensor]],
    episodes: list[int],
) -> None:
    import matplotlib.pyplot as plt

    mean = IMAGENET_MEAN.numpy()
    std = IMAGENET_STD.numpy()
    images = [("public 301", eval_data["raw_zed_link_camera_0"][..., :3])]
    head_key = "observation.rgb.zed_link_camera_0"
    for episode, sample in zip(episodes, samples):
        image = sample[head_key].numpy() * std + mean
        images.append((f"demo ep {episode}", np.clip(image.transpose(1, 2, 0), 0.0, 1.0)))

    columns = 5
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, (title, image) in zip(axes, images):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(images) :]:
        axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("mode") != "standard":
        raise ValueError("diagnose_act_eval requires a standard ACT checkpoint")
    state_stats = FeatureStats.from_dict(checkpoint["stats"]["observation.state"])
    action_stats = FeatureStats.from_dict(checkpoint["stats"]["standard_action"])

    trajectory = load_turning_data(args.data_root)
    train_episodes, validation_episodes = split_episodes(trajectory.episode_bounds)
    selected_episodes = validation_episodes[: args.num_demo_starts]
    all_episodes = sorted(trajectory.episode_bounds)
    dataset = TurningACTDataset(
        args.data_root,
        all_episodes,
        "standard",
        state_stats,
        action_stats,
        video_backend="pyav",
    )
    selected_global_indices = [
        int(trajectory.global_indices[trajectory.episode_bounds[episode][0]]) for episode in selected_episodes
    ]
    samples = [dataset[index] for index in selected_global_indices]

    policy = make_act_policy("standard").to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()
    normalized_predictions = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            batch = {
                key: torch.stack([sample[key] for sample in batch_samples]).to(device)
                for key in ("observation.state", *CAMERA_KEYS)
            }
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                normalized_predictions.append(policy.predict_action_chunk(batch).float().cpu().numpy())
    normalized_prediction = np.concatenate(normalized_predictions)
    normalized_target = np.stack([sample["action"].numpy() for sample in samples])
    prediction = action_stats.denormalize(normalized_prediction)
    target = action_stats.denormalize(normalized_target)

    eval_data = np.load(args.eval_npz)
    eval_state = eval_data["raw_state"].astype(np.float32)
    eval_state_z = state_stats.normalize(eval_state)
    start_indices = np.asarray([trajectory.episode_bounds[episode][0] for episode in all_episodes])
    start_states = trajectory.states[start_indices]
    start_z = state_stats.normalize(start_states)
    distances = np.linalg.norm(start_z - eval_state_z[None], axis=1)
    nearest_order = np.argsort(distances)[:10]

    train_mask = np.isin(trajectory.episode_indices, train_episodes)
    train_actions = trajectory.actions[train_mask]
    eval_first_action = eval_data["action_chunk"][0]
    action_z = action_stats.normalize(eval_first_action)
    outside_train_range = np.logical_or(
        eval_first_action < train_actions.min(axis=0), eval_first_action > train_actions.max(axis=0)
    )

    prefix = args.execute_prefix
    report = {
        "checkpoint_step": int(checkpoint["step"]),
        "num_demo_starts": len(samples),
        "demo_episodes": selected_episodes,
        "demo_start_normalized_l1_full_chunk": float(np.abs(normalized_prediction - normalized_target).mean()),
        "demo_start_normalized_l1_prefix": float(
            np.abs(normalized_prediction[:, :prefix] - normalized_target[:, :prefix]).mean()
        ),
        "demo_start_physical_errors_full_chunk": _physical_errors(prediction, target),
        "demo_start_physical_errors_prefix": _physical_errors(prediction[:, :prefix], target[:, :prefix]),
        "public_state": {
            "normalized_l2": float(np.linalg.norm(eval_state_z)),
            "normalized_abs_max": float(np.abs(eval_state_z).max()),
            "abs_gt_3": int((np.abs(eval_state_z) > 3.0).sum()),
            "abs_gt_5": int((np.abs(eval_state_z) > 5.0).sum()),
            "nearest_demo_starts": [
                {"episode": int(all_episodes[index]), "normalized_l2": float(distances[index])}
                for index in nearest_order
            ],
        },
        "public_first_predicted_action": {
            "values": eval_first_action.tolist(),
            "normalized_abs_max": float(np.abs(action_z).max()),
            "dims_outside_train_min_max": np.flatnonzero(outside_train_range).tolist(),
        },
    }
    with (args.output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_montage(args.output_dir / "public_vs_demo_starts.png", eval_data, samples[:19], selected_episodes[:19])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
