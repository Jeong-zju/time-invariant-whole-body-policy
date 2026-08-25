"""Train-split metric and normalization statistics for RoboCasa ACT policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import RoboCasaData, load_robocasa_data, split_episodes
from .labels import LabelConfig, MetricConfig, build_base_path_label, fit_metric_config
from .schema import NONBASE_ACTION, NUM_QUERIES

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def _feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


class _RunningStats:
    def __init__(self, width: int) -> None:
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.sum_sq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.sum.size)
        self.count += values.shape[0]
        self.sum += values.sum(axis=0)
        self.sum_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self) -> dict[str, list]:
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.maximum(np.sqrt(variance), 1e-6).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def time_action_chunk(data: RoboCasaData, start_index: int, length: int = NUM_QUERIES) -> tuple[np.ndarray, np.ndarray]:
    episode = int(data.episode_indices[start_index])
    _, end = data.episode_bounds[episode]
    available = min(length, end - start_index)
    actions = np.empty((length, data.actions.shape[1]), dtype=np.float64)
    actions[:available] = data.actions[start_index : start_index + available]
    actions[available:] = data.actions[end - 1]
    is_pad = np.arange(length) >= available
    return actions, is_pad


def compute_all_stats(
    data: RoboCasaData,
    train_episodes: list[int],
    metric: MetricConfig,
    *,
    max_lp_start_frames: int = 20_000,
) -> tuple[dict, int]:
    train_indices = np.concatenate(
        [np.arange(*data.episode_bounds[episode], dtype=np.int64) for episode in train_episodes]
    )
    lp_stats = _RunningStats(13)
    config = LabelConfig()
    if train_indices.size > max_lp_start_frames:
        sample_positions = np.linspace(0, train_indices.size - 1, max_lp_start_frames, dtype=np.int64)
        lp_indices = train_indices[sample_positions]
    else:
        lp_indices = train_indices
    for index in lp_indices:
        label = build_base_path_label(data, int(index), metric, config)
        chunk, _ = time_action_chunk(data, int(index))
        lp_stats.update(label.hybrid_target(chunk[:, NONBASE_ACTION]))
    result = {
        "observation.state": _feature_stats(data.states[train_indices]),
        "standard_action": _feature_stats(data.actions[train_indices]),
        "lp_action": lp_stats.result(),
        "image_imagenet": {
            "mean": IMAGENET_MEAN.tolist(),
            "std": IMAGENET_STD.tolist(),
        },
    }
    return result, int(lp_indices.size)


def load_stats(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_robocasa_data(args.data_root)
    train, validation = split_episodes(data.episode_bounds)
    metric = fit_metric_config(data, train)
    stats, lp_sample_frames = compute_all_stats(data, train, metric)
    payload = {
        "task": "LineUpCondiments",
        "train_episodes": train,
        "validation_episodes": validation,
        "num_train_frames": int(sum(data.episode_bounds[e][1] - data.episode_bounds[e][0] for e in train)),
        "lp_stats_sample_frames": lp_sample_frames,
        "lp_stats_query_targets": lp_sample_frames * NUM_QUERIES,
        "features": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.metric_output.write_text(json.dumps(metric.to_dict(), indent=2) + "\n")
    print(json.dumps({"stats": str(args.output), "metric": metric.to_dict(), "train": len(train), "val": len(validation)}))


if __name__ == "__main__":
    main()
