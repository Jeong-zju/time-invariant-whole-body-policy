#!/usr/bin/env python3
"""Build immutable LP path-time labels and statistics for a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from lp_groot_base import se2
from lp_groot_base.labels import LabelConfig, build_path_time_label
from lp_groot_base.v3_dataset import episode_slice, load_v3_metadata, load_v3_table


def calibration(table, episodes, horizon: int = 16) -> dict:
    trans, yaw, lengths = [], [], []
    per_episode = []
    for episode_index in episodes["episode_index"].astype(int):
        frame = episode_slice(table, episodes, episode_index)
        state = np.vstack(frame["observation.state"])
        action = np.vstack(frame["action"])
        pose = se2.world_xy_quat_to_se2(state[:, :3], state[:, 3:7])
        delta = se2.log(se2.between(pose[:-1], pose[1:]))
        base = action[:-1, :3]
        mode = action[:-1, 4] > 0
        move_t = mode & (np.linalg.norm(base[:, :2], axis=1) > 1e-3)
        move_r = mode & (np.abs(base[:, 2]) > 1e-3)
        trans.append(np.linalg.norm(delta[move_t, :2], axis=1))
        yaw.append(np.abs(delta[move_r, 2]))
        per_episode.append((delta, mode & ((np.linalg.norm(base[:, :2], axis=1) > 1e-3) | (np.abs(base[:, 2]) > 1e-3))))
    trans = np.concatenate(trans)
    yaw = np.concatenate(yaw)
    trans = trans[trans > 1e-6]
    yaw = yaw[yaw > 1e-6]
    if len(trans) < 100 or len(yaw) < 100:
        raise ValueError("Not enough commanded translation and rotation samples")
    l_xy = float(np.quantile(trans, 0.75))
    l_yaw = float(np.quantile(yaw, 0.75))
    for delta, active in per_episode:
        inc = np.sqrt((delta[:, 0] / l_xy) ** 2 + (delta[:, 1] / l_xy) ** 2 + (delta[:, 2] / l_yaw) ** 2)
        if len(inc) < horizon:
            continue
        cumulative = np.r_[0.0, np.cumsum(inc)]
        window = cumulative[horizon:] - cumulative[:-horizon]
        active_count = np.convolve(active.astype(np.int32), np.ones(horizon, dtype=np.int32), mode="valid")
        lengths.append(window[active_count > 0])
    lengths = np.concatenate(lengths)
    return {
        "l_xy": l_xy,
        "l_yaw": l_yaw,
        "path_extent": float(np.median(lengths)),
        "translation_samples": int(len(trans)),
        "yaw_samples": int(len(yaw)),
        "matched_windows": int(len(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=-1)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    info, episodes = load_v3_metadata(root)
    table = load_v3_table(root)
    if args.max_episodes > 0:
        episodes = episodes.iloc[: args.max_episodes]
    cal = calibration(table, episodes)
    config = LabelConfig(l_xy=cal["l_xy"], l_yaw=cal["l_yaw"], path_extent=cal["path_extent"])
    valid_values = []
    action_values = []
    attained = []
    for episode_index in episodes["episode_index"].astype(int):
        frame = episode_slice(table, episodes, episode_index)
        state = np.vstack(frame["observation.state"])
        timestamp = frame["timestamp"].to_numpy(np.float64)
        actions, masks, sigma = [], [], []
        for start in range(len(frame)):
            label = build_path_time_label(state[:, :3], state[:, 3:7], timestamp, start, config)
            actions.append(label.action)
            masks.append(label.valid)
            sigma.append(label.attained_sigma)
            action_values.append(label.action[label.valid > 0.5])
            valid_values.append(int(label.valid.sum()))
            attained.append(label.attained_sigma)
        np.savez_compressed(
            output / f"episode_{episode_index:06d}.npz",
            action=np.stack(actions).astype(np.float32),
            valid=np.stack(masks).astype(np.float32),
            attained_sigma=np.asarray(sigma, dtype=np.float32),
        )
    values = np.concatenate(action_values, axis=0)
    stats = {
        key: value.tolist()
        for key, value in {
            "mean": values.mean(0), "std": values.std(0), "min": values.min(0), "max": values.max(0),
            "q01": np.quantile(values, 0.01, axis=0), "q99": np.quantile(values, 0.99, axis=0),
        }.items()
    }
    manifest = {
        "source_dataset": str(root.resolve()),
        "source_codebase_version": info.get("codebase_version"),
        "num_episodes": int(len(episodes)),
        "num_frames": int(sum(episodes["length"])),
        "label_config": asdict(config),
        "calibration": cal,
        "valid_anchor_mean": float(np.mean(valid_values)),
        "one_anchor_fraction": float(np.mean(np.asarray(valid_values) == 1)),
        "moving_fraction": float(np.mean(np.asarray(attained) >= config.path_extent / config.num_anchors)),
        "action_stats": stats,
    }
    with (output / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
