#!/usr/bin/env python3
"""Build B2 normalization statistics from deterministic measured-state labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lpwb.labels import LabelConfig, build_path_time_label


ACTION_KEYS = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
    "base_motion": 4,
    "control_mode": 1,
}


class StreamingStats:
    def __init__(self, dim: int, reservoir_size: int, seed: int):
        self.dim = dim
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros(dim, dtype=np.float64)
        self.minimum = np.full(dim, np.inf)
        self.maximum = np.full(dim, -np.inf)
        self.reservoir = np.empty((0, dim), dtype=np.float64)
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.dim)
        self.count += len(values)
        self.sum += values.sum(axis=0)
        self.sum_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        # Quantiles are diagnostic only (training uses exact min/max). Keep a
        # bounded deterministic prefix instead of copying a 100k-row reservoir
        # for every 32-row label chunk.
        remaining = self.reservoir_size - len(self.reservoir)
        if remaining > 0:
            self.reservoir = np.concatenate([self.reservoir, values[:remaining]], axis=0)

    def finish(self) -> dict[str, list[float]]:
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "q01": np.quantile(self.reservoir, 0.01, axis=0).tolist(),
            "q99": np.quantile(self.reservoir, 0.99, axis=0).tolist(),
        }


def selected_episodes(total: int, seed: int, fraction: float) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(total)
    return np.sort(order[: int(np.floor(total * fraction))])


def arrays_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.vstack(frame["observation.state"].to_numpy()).astype(np.float64)
    action = np.vstack(frame["action"].to_numpy()).astype(np.float64)
    timestamp = frame["timestamp"].to_numpy(dtype=np.float64)
    return state, action, timestamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--samples-per-episode", type=int, default=32)
    parser.add_argument("--reservoir-size", type=int, default=100000)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LabelConfig(num_segments=32)

    for dataset_string in args.dataset:
        dataset = Path(dataset_string)
        with (dataset / "meta" / "info.json").open() as file:
            info = json.load(file)
        accumulators = {
            key: StreamingStats(dim, args.reservoir_size, args.seed + index)
            for index, (key, dim) in enumerate(ACTION_KEYS.items())
        }
        episode_indices = selected_episodes(
            int(info["total_episodes"]), args.seed, args.train_fraction
        )
        chunks = 0
        for episode_index in episode_indices:
            chunk = int(episode_index) // int(info["chunks_size"])
            path = dataset / info["data_path"].format(
                episode_chunk=chunk, episode_index=int(episode_index)
            )
            frame = pd.read_parquet(path, columns=["observation.state", "action", "timestamp"])
            state, action, timestamp = arrays_from_frame(frame)
            valid_starts = len(frame) - config.num_segments
            if valid_starts <= 0:
                continue
            rng = np.random.default_rng(args.seed + int(episode_index))
            starts = rng.choice(
                valid_starts,
                size=args.samples_per_episode,
                replace=valid_starts < args.samples_per_episode,
            )
            for start in starts:
                label = build_path_time_label(
                    state[start : start + 33],
                    action[start : start + 32],
                    timestamp[start : start + 33],
                    config,
                )
                for key, values in label.action_dict().items():
                    accumulators[key].update(values)
                chunks += 1
        result = {
            "schema_version": 1,
            "dataset": str(dataset),
            "task": dataset.parent.name,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "num_label_chunks": chunks,
            "label_config": config.__dict__,
            "action": {key: accumulator.finish() for key, accumulator in accumulators.items()},
        }
        output_path = output_dir / f"{dataset.parent.name}.json"
        with output_path.open("w") as file:
            json.dump(result, file, indent=2)
        print(f"wrote {output_path} from {chunks} B2 chunks")


if __name__ == "__main__":
    main()
