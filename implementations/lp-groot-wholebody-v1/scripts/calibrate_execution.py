#!/usr/bin/env python3
"""Fit a training-split-only command-to-measured-motion calibration for B2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lpwb.geometry import (
    local_base_pose,
    normalize_quaternion,
    quaternion_between,
    quaternion_to_rotvec,
    wrap_angle,
)


def selected_episodes(total: int, seed: int, train_fraction: float) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(total)
    return np.sort(order[: int(np.floor(total * train_fraction))])


def _base_increments(state: np.ndarray) -> np.ndarray:
    poses = local_base_pose(state[:, 0:3], state[:, 3:7])
    result = []
    for previous, current in zip(poses[:-1], poses[1:]):
        delta = current[:2] - previous[:2]
        cosine, sine = np.cos(previous[2]), np.sin(previous[2])
        result.append(
            [
                cosine * delta[0] + sine * delta[1],
                -sine * delta[0] + cosine * delta[1],
                float(wrap_angle(current[2] - previous[2])),
            ]
        )
    return np.asarray(result, dtype=np.float64)


def _eef_rotation_increments(state: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(state[:, 10:14])
    return np.vstack(
        [
            quaternion_to_rotvec(quaternion_between(previous, current))
            for previous, current in zip(quaternion[:-1], quaternion[1:])
        ]
    )


def _fit_for_lag(
    episode_pairs: list[tuple[np.ndarray, np.ndarray]], lag: int
) -> dict[str, object]:
    command_parts, measured_parts = [], []
    for command, measured in episode_pairs:
        if len(command) <= lag:
            continue
        if lag:
            command_parts.append(command[:-lag])
            measured_parts.append(measured[lag:])
        else:
            command_parts.append(command)
            measured_parts.append(measured)
    command = np.concatenate(command_parts)
    measured = np.concatenate(measured_parts)
    design = np.column_stack([command, np.ones(len(command))])
    coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
    matrix, bias = coefficients[:3], coefficients[3]
    predicted = command @ matrix + bias
    residual = measured - predicted
    denominator = float(np.sum(np.square(measured - measured.mean(axis=0))))
    r2 = 1.0 - float(np.sum(np.square(residual))) / max(denominator, 1e-12)
    return {
        "matrix": matrix.tolist(),
        "bias": bias.tolist(),
        "lag_steps": lag,
        "r2": r2,
        "rmse": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
        "num_samples": int(len(command)),
    }


def _best_fit(
    episode_pairs: list[tuple[np.ndarray, np.ndarray]], max_lag: int
) -> dict[str, object]:
    candidates = [_fit_for_lag(episode_pairs, lag) for lag in range(max_lag + 1)]
    best = max(candidates, key=lambda item: float(item["r2"]))
    best["lag_candidates"] = [
        {"lag_steps": item["lag_steps"], "r2": item["r2"]} for item in candidates
    ]
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--max-episodes-per-task", type=int, default=64)
    parser.add_argument("--max-lag", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must lie strictly between zero and one")

    pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "base": [],
        "eef_position": [],
        "eef_rotation": [],
    }
    provenance = []
    for dataset_string in args.dataset:
        dataset = Path(dataset_string)
        with (dataset / "meta" / "info.json").open() as file:
            info = json.load(file)
        episodes = selected_episodes(
            int(info["total_episodes"]), args.seed, args.train_fraction
        )[: args.max_episodes_per_task]
        used = 0
        for episode_index in episodes:
            path = dataset / info["data_path"].format(
                episode_chunk=int(episode_index) // int(info["chunks_size"]),
                episode_index=int(episode_index),
            )
            frame = pd.read_parquet(path, columns=["observation.state", "action"])
            state = np.vstack(frame["observation.state"]).astype(np.float64)
            action = np.vstack(frame["action"]).astype(np.float64)
            if len(state) < 2 or len(action) < 1:
                continue
            length = min(len(action), len(state) - 1)
            state, action = state[: length + 1], action[:length]
            pairs["base"].append((action[:, 0:3], _base_increments(state)))
            pairs["eef_position"].append(
                (action[:, 5:8], np.diff(state[:, 7:10], axis=0))
            )
            pairs["eef_rotation"].append(
                (action[:, 8:11], _eef_rotation_increments(state))
            )
            used += 1
        provenance.append(
            {
                "dataset": str(dataset),
                "task": dataset.parent.name,
                "selected_training_episodes": [int(value) for value in episodes],
                "episodes_used": used,
            }
        )

    result = {
        "schema_version": 1,
        "description": "measured_increment = native_command @ matrix + bias",
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "max_lag": args.max_lag,
        "provenance": provenance,
        **{name: _best_fit(value, args.max_lag) for name, value in pairs.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({name: result[name] for name in pairs}, indent=2))


if __name__ == "__main__":
    main()
