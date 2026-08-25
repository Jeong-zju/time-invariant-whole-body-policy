"""Compute train-only state, standard-action, and LP-action normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .labels import LabelConfig, MetricConfig, build_path_label
from .parquet_data import load_turning_data, split_episodes
from .stats import FeatureStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lp-samples", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_turning_data(args.data_root)
    train_episodes, validation_episodes = split_episodes(data.episode_bounds)
    train_mask = np.isin(data.episode_indices, np.asarray(train_episodes))
    state_stats = FeatureStats.from_values(data.states[train_mask])
    standard_stats = FeatureStats.from_values(data.actions[train_mask])

    with args.metric_config.open() as handle:
        metric = MetricConfig.from_dict(json.load(handle))
    candidates = np.flatnonzero(train_mask)
    rng = np.random.default_rng(args.seed)
    starts = rng.choice(candidates, size=min(args.lp_samples, candidates.size), replace=False)
    label_config = LabelConfig(extent_mode="fixed")
    total = np.zeros(24, dtype=np.float64)
    total_sq = np.zeros(24, dtype=np.float64)
    count = 0
    valid_counts = []
    for start in starts:
        label = build_path_label(data, int(start), metric, label_config)
        values = label.target[~label.is_pad].astype(np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += values.shape[0]
        valid_counts.append(label.valid_count)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    lp_stats = FeatureStats(mean=mean, std=np.maximum(np.sqrt(variance), 1e-4))

    report = {
        "task": "turning_on_radio",
        "split": "episode_index % 5 != 0",
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "lp_sample_windows": int(len(starts)),
        "lp_valid_anchors": int(count),
        "lp_valid_anchor_fraction": float(np.sum(valid_counts) / (len(valid_counts) * label_config.num_anchors)),
        "features": {
            "observation.state": state_stats.to_dict(),
            "standard_action": standard_stats.to_dict(),
            "lp_action": lp_stats.to_dict(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"output={args.output}")
    print(
        f"lp_windows={len(starts)} valid_anchors={count} "
        f"valid_fraction={report['lp_valid_anchor_fraction']:.6f}"
    )


if __name__ == "__main__":
    main()
