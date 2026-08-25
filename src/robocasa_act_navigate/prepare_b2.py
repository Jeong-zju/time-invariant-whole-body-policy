"""Fit the B2 physical metric, validate labels, and compute masked train stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b2_labels import (
    RateFreeConfig,
    build_rate_free_from_arrays,
    build_rate_free_label,
    fixed_rate_free_metric,
    measured_base_poses,
)
from .b2_se2 import between, compose, exp, log
from .data import load_data, load_split
from .schema import NUM_TASKS


class StreamingStats:
    def __init__(self, width: int) -> None:
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total2 = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.total.size)
        self.count += len(values)
        self.total += values.sum(0)
        self.total2 += np.square(values).sum(0)
        self.minimum = np.minimum(self.minimum, values.min(0))
        self.maximum = np.maximum(self.maximum, values.max(0))

    def result(self) -> dict[str, list[float]]:
        mean = self.total / self.count
        variance = np.maximum(self.total2 / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.maximum(np.sqrt(variance), 1e-6).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def _state_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(0).tolist(),
        "std": np.maximum(values.std(0), 1e-6).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
    }


def _frequency_invariance(metric) -> dict[str, float]:
    def sample(hz: int):
        times = np.linspace(0.0, 4.0, 4 * hz + 1)
        poses = np.column_stack((0.18 * times, 0.05 * np.sin(times), 0.12 * times))
        actions = np.zeros((len(times), 12), dtype=np.float64)
        actions[:, 4] = 1.0
        actions[:, 11] = -1.0
        return build_rate_free_from_arrays(times, poses, actions, metric)

    labels = [sample(hz) for hz in (10, 20, 40)]
    reference = labels[1].target[:, :3]
    error = max(float(np.max(np.abs(label.target[:, :3] - reference))) for label in labels)
    if error > 2e-4:
        raise AssertionError(f"frequency-invariance gate failed: max error {error}")
    return {"max_anchor_error": error, "frequencies_hz": [10, 20, 40]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-step-m", type=float, default=0.025)
    parser.add_argument("--num-anchors", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_data(args.data_root)
    split = load_split(args.split)

    dt = np.concatenate([np.diff(data.timestamps[s:e]) for s, e in data.episode_bounds.values()])
    if not np.allclose(np.median(dt), 0.05, atol=1e-6):
        raise AssertionError(f"unexpected timestamp median {np.median(dt)}")
    metric = fixed_rate_free_metric(spatial_step_m=args.spatial_step_m, num_anchors=args.num_anchors)

    # SE(2) convention gate.
    rng = np.random.default_rng(20260821)
    twists = rng.normal(scale=[0.2, 0.2, 0.4], size=(256, 3))
    roundtrip = float(np.max(np.abs(log(exp(twists)) - twists)))
    composition = float(np.max(np.abs(between(np.zeros(3), compose(np.zeros(3), exp(twists))) - exp(twists))))
    if roundtrip > 1e-10 or composition > 1e-10:
        raise AssertionError("SE(2) convention gate failed")

    action_stats = StreamingStats(12)
    valid_counts = []
    reached = 0
    for episode in split["train"]:
        start, end = data.episode_bounds[episode]
        for index in range(start, end):
            label = build_rate_free_label(data, index, metric)
            action_stats.add(label.target)
            valid_counts.append(label.valid_count)
            reached += int(label.reached_extent)
            if not np.all(label.target[:label.valid_count, 11] > 0.0):
                raise AssertionError("valid B2 anchors do not carry CONTINUE")
            if label.valid_count < metric.num_anchors and not np.all(label.target[label.valid_count:, 11] < 0.0):
                raise AssertionError("terminal B2 anchors do not carry STOP")

    train_indices = np.concatenate([np.arange(*data.episode_bounds[e]) for e in split["train"]])
    one_hot = np.eye(NUM_TASKS, dtype=np.float32)[data.task_indices[train_indices]]
    augmented_state = np.concatenate((data.states[train_indices], one_hot), axis=1)
    frequency = _frequency_invariance(metric)
    stats = {
        "task": "NavigateKitchen",
        "representation": "B2_Path_RateFree_SE2Delta_Stop_WholeBody_V4",
        "model": {"chunk_size": metric.num_anchors, "action_dim": 12},
        "training": {"geometry_loss": {
            "yaw_radius_m": metric.yaw_radius_m,
            "endpoint_loss": 1.0,
            "yaw_geometry_loss": 0.2,
            "direction_loss": 0.05,
            "smoothness_loss": 0.25,
            "progress_loss": 1.0,
        }},
        "metric": metric.to_dict(),
        "label_construction": {
            "uses_timestamp": False,
            "uses_frame_count_horizon": False,
            "scans_until": "fixed physical extent or episode end",
            "whole_body": True,
            "upper_body_alignment": "EEF commands and discrete events at the same geometric anchors",
            "terminal_semantics": "channel11 CONTINUE=+1 STOP=-1; terminal hold is supervised",
            "base_parameterization": "body-frame SE2 Lie increments accumulated from the chunk origin",
        },
        "features": {
            "observation.state": _state_stats(augmented_state),
            "action": action_stats.result(),
            "image_imagenet": {
                "mean": np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None].tolist(),
                "std": np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None].tolist(),
            },
        },
    }
    report = {
        "event": "B2_LABEL_GATES_PASSED",
        "schema": {"episodes": len(data.episode_bounds), "frames": len(data.actions), "median_dt_s": float(np.median(dt))},
        "metric": metric.to_dict(),
        "se2": {"roundtrip_max_error": roundtrip, "composition_max_error": composition},
        "frequency_invariance": frequency,
        "labels": {
            "train_samples": len(valid_counts),
            "mean_valid_anchors": float(np.mean(valid_counts)),
            "median_valid_anchors": float(np.median(valid_counts)),
            "full_extent_fraction": reached / len(valid_counts),
            "padding_is_explicit": True,
            "stop_is_predicted": True,
            "predicted_time_duration_velocity_rate": False,
        },
    }
    (args.output_dir / "metric.json").write_text(json.dumps(metric.to_dict(), indent=2) + "\n")
    (args.output_dir / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "LABEL_GATES_PASSED").write_text("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
