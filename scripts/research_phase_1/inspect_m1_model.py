#!/usr/bin/env python3
"""Decode one M1 prediction in physical units and plot its 16 state points."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.data.schema import EmbodimentTag  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402
from scripts.research_phase_1.gr00t_m1_finetune import _phase1_valid_steps  # noqa: E402
from whole_body_policy import load_target_batch  # noqa: E402
from whole_body_policy.groot_m1_data_config import UnitreeG1Phase1M1DataConfig  # noqa: E402


def _decoded_matrix(value: object, width: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if width == 1 and array.ndim == 1:
        array = array[:, None]
    if array.shape != (16, width):
        raise ValueError(f"decoded {name} must have shape {(16, width)}, got {array.shape}")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoising-steps", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    LeRobotSingleDataset._get_all_steps = _phase1_valid_steps
    config = UnitreeG1Phase1M1DataConfig()
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=config.modality_config(),
        transforms=config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        video_backend="decord",
    )
    raw = dataset.get_step_data(trajectory_id=0, base_index=0)
    input_keys = [
        *config.video_keys,
        *config.state_keys,
        "annotation.human.task_description",
    ]
    observations = {key: raw[key] for key in input_keys}
    policy = Gr00tPolicy(
        model_path=str(args.model),
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        denoising_steps=args.denoising_steps,
        device="cuda",
    )
    prediction_dict = policy.get_action(observations)
    predicted = np.concatenate(
        [
            _decoded_matrix(
                prediction_dict["action.phase1_upper_body_position"], 28, "upper body"
            ),
            _decoded_matrix(prediction_dict["action.phase1_base_height"], 1, "base height"),
            _decoded_matrix(
                prediction_dict["action.phase1_base_relative_se2"], 3, "base SE(2)"
            ),
        ],
        axis=-1,
    ).astype(np.float64)
    expected = load_target_batch(args.target).targets[0].astype(np.float64)
    if predicted.shape != (16, 32):
        raise ValueError(f"decoded prediction must be (16, 32), got {predicted.shape}")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("decoded prediction contains non-finite values")

    error = predicted - expected
    report = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "model": str(args.model),
        "episode_index": 0,
        "anchor_index": 0,
        "seed": args.seed,
        "denoising_steps": args.denoising_steps,
        "decoded_shape": list(predicted.shape),
        "physical_unit_errors": {
            "upper_body_rmse_rad": float(np.sqrt(np.mean(np.square(error[:, :28])))),
            "base_height_rmse_m": float(np.sqrt(np.mean(np.square(error[:, 28])))),
            "base_xy_rmse_m": float(np.sqrt(np.mean(np.square(error[:, 29:31])))),
            "base_yaw_rmse_rad": float(np.sqrt(np.mean(np.square(error[:, 31])))),
        },
        "predicted_base_endpoint": predicted[-1, 29:32].tolist(),
        "expected_base_endpoint": expected[-1, 29:32].tolist(),
        "all_finite": True,
        "status": "pass",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    times = np.arange(1, 17, dtype=np.float64) * 0.02
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].plot(times, np.sqrt(np.mean(np.square(error[:, :28]), axis=1)))
    axes[0, 0].set(title="Upper-body RMSE", xlabel="query time (s)", ylabel="rad")
    axes[0, 1].plot(times, expected[:, 28], label="expert")
    axes[0, 1].plot(times, predicted[:, 28], label="model")
    axes[0, 1].set(title="Base height", xlabel="query time (s)", ylabel="m")
    axes[0, 1].legend()
    axes[1, 0].plot(expected[:, 29], expected[:, 30], "o-", label="expert")
    axes[1, 0].plot(predicted[:, 29], predicted[:, 30], "o-", label="model")
    axes[1, 0].set(title="Anchor-relative base route", xlabel="x (m)", ylabel="y (m)")
    axes[1, 0].axis("equal")
    axes[1, 0].legend()
    axes[1, 1].plot(times, expected[:, 31], label="expert")
    axes[1, 1].plot(times, predicted[:, 31], label="model")
    axes[1, 1].set(title="Anchor-relative yaw", xlabel="query time (s)", ylabel="rad")
    axes[1, 1].legend()
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_plot, dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
