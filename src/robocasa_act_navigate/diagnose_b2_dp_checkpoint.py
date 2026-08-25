"""Audit a B2 Diffusion Policy checkpoint in physical geometric units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .b2_dataset import NavigateB2Dataset
from .b2_labels import RateFreeMetric, increments_to_absolute_path
from .b2_se2 import wrap_angle
from .data import load_data, load_split
from .dp_policy import make_dp_policy, predict_dp_action_chunk


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.sum(left * right, axis=-1) / np.maximum(denominator, 1e-9)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-key", choices=("train", "val"), default="val")
    parser.add_argument("--selected-indices", type=Path)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = make_dp_policy(
        payload["stats"], device=args.device.split(":", 1)[0],
        num_inference_steps=args.num_inference_steps,
    ).to(args.device)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()

    data = load_data(args.data_root)
    split = load_split(args.split)
    metric = RateFreeMetric.from_dict(json.loads(args.metric.read_text()))
    dataset = NavigateB2Dataset(data, split[args.split_key], args.frame_cache, metric)
    if args.selected_indices:
        selected = np.asarray(json.loads(args.selected_indices.read_text()), dtype=np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(dataset), size=min(args.samples, len(dataset)), replace=False))
    loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    predicted_parts, target_parts, valid_parts = [], [], []
    with torch.inference_mode():
        for batch in loader:
            target_parts.append(batch["action"].numpy())
            valid_parts.append(batch["action"][..., 11].numpy() >= 0.0)
            model_batch = {
                key: value.to(args.device, non_blocking=True)
                for key, value in batch.items()
                if key not in {"action", "action_is_pad", "action_loss_weight"}
            }
            predicted_parts.append(predict_dp_action_chunk(policy, model_batch).float().cpu().numpy())

    predicted = np.concatenate(predicted_parts)
    target = np.concatenate(target_parts)
    valid = np.concatenate(valid_parts)
    predicted_absolute = np.stack([increments_to_absolute_path(value[:, :3]) for value in predicted])
    target_absolute = np.stack([increments_to_absolute_path(value[:, :3]) for value in target])
    valid3 = valid[..., None]
    absolute = np.abs(predicted_absolute - target_absolute)
    absolute[..., 2] = np.abs(wrap_angle(predicted_absolute[..., 2] - target_absolute[..., 2]))
    component_mae = (absolute * valid3).sum((0, 1)) / valid.sum()
    last = valid.sum(1) - 1
    rows = np.arange(len(last))
    pred_end = predicted_absolute[rows, last]
    true_end = target_absolute[rows, last]
    first_pred, first_true = predicted_absolute[:, 0, :2], target_absolute[:, 0, :2]
    end_cosine = _safe_cosine(pred_end[:, :2], true_end[:, :2])
    first_cosine = _safe_cosine(first_pred, first_true)
    endpoint_translation_error = np.linalg.norm(pred_end[:, :2] - true_end[:, :2], axis=-1)
    endpoint_yaw_error = np.abs(wrap_angle(pred_end[:, 2] - true_end[:, 2]))
    pred_delta = np.diff(np.concatenate((np.zeros((len(predicted), 1, 2)), predicted_absolute[..., :2]), 1), axis=1)
    true_delta = np.diff(np.concatenate((np.zeros((len(target), 1, 2)), target_absolute[..., :2]), 1), axis=1)
    pred_length = (np.linalg.norm(pred_delta, axis=-1) * valid).sum(1)
    true_length = (np.linalg.norm(true_delta, axis=-1) * valid).sum(1)
    pred_mode, true_mode = predicted[..., 3] >= 0.0, target[..., 3] >= 0.0
    pred_continue, true_continue = predicted[..., 11] >= 0.0, target[..., 11] >= 0.0
    report = {
        "checkpoint": str(args.checkpoint),
        "policy": "DiffusionPolicy",
        "split_key": args.split_key,
        "samples": int(len(predicted)),
        "num_inference_steps": int(policy.diffusion.num_inference_steps),
        "component_mae": {"x_m": float(component_mae[0]), "y_m": float(component_mae[1]),
                          "yaw_rad": float(component_mae[2])},
        "first_anchor_direction_cosine": _summary(first_cosine),
        "first_anchor_reverse_fraction": float((first_cosine < 0.0).mean()),
        "endpoint_direction_cosine": _summary(end_cosine),
        "endpoint_reverse_fraction": float((end_cosine < 0.0).mean()),
        "endpoint_translation_error_m": _summary(endpoint_translation_error),
        "endpoint_yaw_error_rad": _summary(endpoint_yaw_error),
        "predicted_path_length_m": _summary(pred_length),
        "target_path_length_m": _summary(true_length),
        "control_mode_accuracy_valid": float((pred_mode[valid] == true_mode[valid]).mean()),
        "stop_accuracy_all": float((pred_continue == true_continue).mean()),
        "predicted_continue_fraction": float(pred_continue.mean()),
        "target_continue_fraction": float(true_continue.mean()),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
