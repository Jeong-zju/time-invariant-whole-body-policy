#!/usr/bin/env python3
"""Timestamp/schema, reconstruction, event, and visualization gate for B1/B2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lpwb.geometry import (
    interpolate_base,
    local_base_pose,
    quaternion_between,
    quaternion_slerp,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
    wrap_angle,
)
from lpwb.labels import (
    LabelConfig,
    build_path_time_label,
    build_pose_time_label,
    event_signature,
)


def interpolate_decoded(label, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knot_times = np.concatenate([[0.0], label.arrival_times.astype(np.float64)])
    base_knots = np.vstack([np.zeros(3), label.base_local])
    eef_pos_knots = np.vstack([np.zeros(3), label.eef_position_delta])
    eef_q_knots = np.vstack(
        [np.array([0.0, 0.0, 0.0, 1.0]), rotvec_to_quaternion(label.eef_rotation_delta)]
    )
    base, eef_pos, eef_q = [], [], []
    for target in query_times:
        hi = int(np.searchsorted(knot_times, target, side="right"))
        hi = min(max(hi, 1), len(knot_times) - 1)
        lo = hi - 1
        alpha = float((target - knot_times[lo]) / max(knot_times[hi] - knot_times[lo], 1e-12))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        base.append(interpolate_base(base_knots[lo], base_knots[hi], alpha))
        eef_pos.append((1.0 - alpha) * eef_pos_knots[lo] + alpha * eef_pos_knots[hi])
        eef_q.append(quaternion_slerp(eef_q_knots[lo], eef_q_knots[hi], alpha))
    return np.asarray(base), np.asarray(eef_pos), np.asarray(eef_q)


def validate_chunk(state, action, timestamp, config, label_builder) -> dict[str, float | bool]:
    label = label_builder(state, action, timestamp, config)
    query_times = timestamp[1:] - timestamp[0]
    base_pred, eef_pos_pred, eef_q_pred = interpolate_decoded(label, query_times)
    base_true = local_base_pose(state[:, 0:3], state[:, 3:7])[1:]
    eef_pos_true = state[1:, 7:10] - state[0, 7:10]
    eef_q_true = np.vstack(
        [quaternion_between(state[0, 10:14], q) for q in state[1:, 10:14]]
    )
    eef_rot_error = np.asarray(
        [
            np.linalg.norm(quaternion_to_rotvec(quaternion_between(q_true, q_pred)))
            for q_true, q_pred in zip(eef_q_true, eef_q_pred)
        ]
    )
    return {
        "base_translation_rmse_m": float(
            np.sqrt(np.mean(np.sum(np.square(base_pred[:, :2] - base_true[:, :2]), axis=1)))
        ),
        "base_yaw_rmse_rad": float(
            np.sqrt(np.mean(np.square(wrap_angle(base_pred[:, 2] - base_true[:, 2]))))
        ),
        "eef_translation_rmse_m": float(
            np.sqrt(np.mean(np.sum(np.square(eef_pos_pred - eef_pos_true), axis=1)))
        ),
        "eef_rotation_rmse_rad": float(np.sqrt(np.mean(np.square(eef_rot_error)))),
        "gripper_events_preserved": event_signature(action[:, 11])
        == event_signature(label.gripper),
        "control_events_preserved": event_signature(action[:, 4])
        == event_signature(label.control_mode),
        "duration_error_s": float(abs(label.durations.sum() - (timestamp[-1] - timestamp[0]))),
        "finite": bool(
            np.all(np.isfinite(np.concatenate(list(label.action_dict().values()), axis=1)))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["b1", "b2"], default="b2")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--episodes-per-task", type=int, default=12)
    parser.add_argument("--chunks-per-episode", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LabelConfig(num_segments=32)
    label_builder = (
        build_pose_time_label if args.method == "b1" else build_path_time_label
    )
    report: dict[str, dict] = {}

    for dataset_string in args.dataset:
        dataset = Path(dataset_string)
        task = dataset.parent.name
        with (dataset / "meta" / "info.json").open() as file:
            info = json.load(file)
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(int(info["total_episodes"]))
        boundary = int(np.floor(0.9 * len(order)))
        episodes = np.sort(order[boundary:][: args.episodes_per_task])
        metrics = []
        first_plot = None
        dt_values = []
        quaternion_norm_error = []
        for episode_index in episodes:
            chunk_index = int(episode_index) // int(info["chunks_size"])
            path = dataset / info["data_path"].format(
                episode_chunk=chunk_index, episode_index=int(episode_index)
            )
            frame = pd.read_parquet(path, columns=["observation.state", "action", "timestamp"])
            state = np.vstack(frame["observation.state"].to_numpy()).astype(np.float64)
            action = np.vstack(frame["action"].to_numpy()).astype(np.float64)
            timestamp = frame["timestamp"].to_numpy(dtype=np.float64)
            dt_values.extend(np.diff(timestamp).tolist())
            quaternion_norm_error.extend(
                np.abs(np.linalg.norm(state[:, 3:7], axis=1) - 1.0).tolist()
            )
            quaternion_norm_error.extend(
                np.abs(np.linalg.norm(state[:, 10:14], axis=1) - 1.0).tolist()
            )
            valid_starts = len(frame) - 32
            starts = rng.choice(
                valid_starts,
                size=args.chunks_per_episode,
                replace=valid_starts < args.chunks_per_episode,
            )
            for start in starts:
                state_chunk = state[start : start + 33]
                action_chunk = action[start : start + 32]
                time_chunk = timestamp[start : start + 33]
                metrics.append(
                    validate_chunk(
                        state_chunk, action_chunk, time_chunk, config, label_builder
                    )
                )
                if first_plot is None:
                    first_plot = (
                        state_chunk,
                        label_builder(state_chunk, action_chunk, time_chunk, config),
                    )

        aggregate = {}
        numeric_keys = [key for key, value in metrics[0].items() if not isinstance(value, bool)]
        boolean_keys = [key for key, value in metrics[0].items() if isinstance(value, bool)]
        for key in numeric_keys:
            values = np.asarray([metric[key] for metric in metrics], dtype=np.float64)
            aggregate[key] = {"mean": float(values.mean()), "max": float(values.max())}
        for key in boolean_keys:
            aggregate[key] = bool(all(metric[key] for metric in metrics))
        aggregate["schema"] = {
            "fps": info["fps"],
            "state_dim": 16,
            "action_dim": 12,
            "dt_median_s": float(np.median(dt_values)),
            "dt_max_abs_error_from_0p05_s": float(
                np.max(np.abs(np.asarray(dt_values) - 0.05))
            ),
            "quaternion_norm_max_error": float(np.max(quaternion_norm_error)),
        }
        report[task] = aggregate

        state_plot, label_plot = first_plot
        base_true = local_base_pose(state_plot[:, 0:3], state_plot[:, 3:7])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(base_true[:, 0], base_true[:, 1], "k.-", label="measured")
        axes[0].plot(
            label_plot.base_local[:, 0],
            label_plot.base_local[:, 1],
            "r.-",
            label=f"{args.method.upper()} anchors",
        )
        axes[0].axis("equal")
        axes[0].set_title(f"{task}: base local path")
        axes[0].legend()
        axes[1].plot(state_plot[:, 7], label="measured EEF-x")
        axes[1].plot(
            label_plot.source_indices + 1,
            label_plot.eef_position_delta[:, 0] + state_plot[0, 7],
            ".",
            label=f"{args.method.upper()} anchors",
        )
        axes[1].set_title("EEF A-B-A/event-sensitive trace")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{task}_label.png", dpi=150)
        plt.close(fig)

    thresholds = {
        "base_translation_rmse_m": 0.03,
        "base_yaw_rmse_rad": 0.08,
        "eef_translation_rmse_m": 0.02,
        "eef_rotation_rmse_rad": 0.10,
        "duration_error_s": 1e-5,
    }
    failures = []
    for task, task_report in report.items():
        for key, threshold in thresholds.items():
            if task_report[key]["max"] > threshold:
                failures.append(f"{task}:{key}={task_report[key]['max']}>{threshold}")
        for key in ["gripper_events_preserved", "control_events_preserved", "finite"]:
            if not task_report[key]:
                failures.append(f"{task}:{key}=false")
    report["gate"] = {
        "method": args.method,
        "passed": not failures,
        "failures": failures,
        "thresholds": thresholds,
    }
    with (output_dir / "validation_report.json").open("w") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report["gate"], indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
