"""Required pre-training gates for RoboCasa base-only LP-ACT V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lp_act.se2 import between

from .data import load_robocasa_data, split_episodes
from .execution import decode_lp_chunk, fit_controller_calibration
from .labels import LabelConfig, build_base_path_label, fit_metric_config
from .schema import CONTROL_DT, EXECUTION_STEPS, NONBASE_ACTION, NUM_QUERIES
from .stats import time_action_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_robocasa_data(args.data_root)
    train, validation = split_episodes(data.episode_bounds)
    metric = fit_metric_config(data, train)
    calibration = fit_controller_calibration(data, train)
    candidates = []
    for episode in validation:
        start, end = data.episode_bounds[episode]
        candidates.extend(range(start, max(start, end - NUM_QUERIES)))
    rng = np.random.default_rng(args.seed)
    starts = sorted(rng.choice(candidates, size=min(args.num_windows, len(candidates)), replace=False).tolist())

    translation_sq = []
    yaw_sq = []
    calibrated_translation_sq = []
    calibrated_yaw_sq = []
    command_abs = []
    upper_abs = []
    clip_count = 0
    static_count = 0
    examples = []
    for start in starts:
        label = build_base_path_label(data, start, metric, LabelConfig())
        time_actions, _ = time_action_chunk(data, start)
        target = label.hybrid_target(time_actions[:, NONBASE_ACTION])
        decoded = decode_lp_chunk(target, calibration=calibration)
        measured = between(
            data.base_world_poses[start], data.base_world_poses[start : start + EXECUTION_STEPS + 1]
        )
        pose_error = between(decoded.boundary_base_poses, measured)
        calibrated_error = between(decoded.calibrated_command_poses, measured)
        translation_sq.extend(np.square(pose_error[:, :2]).sum(axis=1).tolist())
        yaw_sq.extend(np.square(pose_error[:, 2]).tolist())
        calibrated_translation_sq.extend(np.square(calibrated_error[:, :2]).sum(axis=1).tolist())
        calibrated_yaw_sq.extend(np.square(calibrated_error[:, 2]).tolist())
        command_abs.extend(
            np.abs(decoded.actions[:, :3] - time_actions[:EXECUTION_STEPS, :3]).reshape(-1).tolist()
        )
        upper_abs.extend(
            np.abs(decoded.actions[:, 3:12] - time_actions[:EXECUTION_STEPS, 3:12]).reshape(-1).tolist()
        )
        clip_count += int(np.sum(np.abs(decoded.unclipped_base_actions) > 1.0))
        static_count += int(label.static)
        if len(examples) < 6:
            examples.append((start, measured, decoded.boundary_base_poses, label.anchor_times, label.base_anchors))

    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (start, measured, reconstructed, anchor_times, anchors) in zip(axes.flat, examples, strict=False):
        axis.plot(measured[:, 0], measured[:, 1], "k-", label="measured 20 Hz")
        axis.plot(reconstructed[:, 0], reconstructed[:, 1], "b--", label="decoded")
        axis.scatter(anchors[:, 0], anchors[:, 1], s=8, c=anchor_times, cmap="viridis", label="LP anchors")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"ep={int(data.episode_indices[start])} frame={int(data.frame_indices[start])}")
    axes.flat[0].legend(fontsize=7)
    figure.savefig(args.output_dir / "label_reconstruction_examples.png", dpi=160)
    plt.close(figure)

    dt = np.concatenate([np.diff(data.timestamps[s:e]) for s, e in data.episode_bounds.values()])
    report = {
        "task": "LineUpCondiments",
        "schema": {
            "frames": int(data.actions.shape[0]),
            "episodes": len(data.episode_bounds),
            "state_dim": int(data.states.shape[1]),
            "action_dim": int(data.actions.shape[1]),
            "base_action_indices": [0, 1, 2],
            "nonbase_action_indices": list(range(3, 12)),
            "lp_output_dim": 13,
            "queries_predicted": NUM_QUERIES,
            "execution_steps_before_replan": EXECUTION_STEPS,
        },
        "split": {"train_episodes": train, "validation_episodes": validation},
        "timestamp": {
            "mean_s": float(dt.mean()),
            "std_s": float(dt.std()),
            "min_s": float(dt.min()),
            "max_s": float(dt.max()),
            "off_20hz_over_1ms": int(np.sum(np.abs(dt - CONTROL_DT) > 1e-3)),
        },
        "metric": metric.to_dict(),
        "controller_calibration": calibration.to_dict(),
        "oracle": {
            "windows": len(starts),
            "base_translation_rmse_m": float(np.sqrt(np.mean(translation_sq))),
            "base_yaw_rmse_rad": float(np.sqrt(np.mean(yaw_sq))),
            "calibrated_command_translation_rmse_m": float(np.sqrt(np.mean(calibrated_translation_sq))),
            "calibrated_command_yaw_rmse_rad": float(np.sqrt(np.mean(calibrated_yaw_sq))),
            "base_command_mae_normalized": float(np.mean(command_abs)),
            "upper_identity_mae": float(np.mean(upper_abs)),
            "unclipped_command_values": clip_count,
            "static_windows": static_count,
        },
        "interpretation": "Measured-pose to waypoint reconstruction is primary. Calibrated command integration and command MAE are diagnostics, not ground-truth motion.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "metric_config.json").write_text(json.dumps(metric.to_dict(), indent=2) + "\n")
    (args.output_dir / "controller_calibration.json").write_text(json.dumps(calibration.to_dict(), indent=2) + "\n")
    print(json.dumps(report["oracle"]))


if __name__ == "__main__":
    main()
