#!/usr/bin/env python3
"""Identify normalized command-to-measured-body-twist mapping from demonstrations."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lp_groot_base import se2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-mean-r2", type=float, default=0.75)
    args = parser.parse_args()

    frames = [
        pd.read_parquet(path, columns=["observation.state", "action", "episode_index", "timestamp"])
        for path in sorted(glob.glob(f"{args.dataset_root}/data/chunk-*/*.parquet"))
    ]
    table = pd.concat(frames, ignore_index=True)
    state = np.stack(table["observation.state"].to_numpy())
    action = np.stack(table["action"].to_numpy())
    episode = table["episode_index"].to_numpy()
    timestamp = table["timestamp"].to_numpy(dtype=np.float64)
    pose = se2.world_xy_quat_to_se2(state[:, :3], state[:, 3:7])
    dt = np.diff(timestamp)
    valid = (episode[1:] == episode[:-1]) & (dt > 0.0)
    twist = se2.log(se2.between(pose[:-1], pose[1:])) / dt[:, None]

    candidates = []
    for start in range(action.shape[1] - 2):
        command = action[:-1, start : start + 3][valid]
        target = twist[valid]
        design = np.concatenate([command, np.ones((len(command), 1))], axis=1)
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        prediction = design @ weights
        denominator = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
        r2 = 1.0 - np.sum((target - prediction) ** 2, axis=0) / denominator
        candidates.append(
            {
                "start": start,
                "stop": start + 3,
                "mean_r2": float(np.mean(r2)),
                "r2": r2.tolist(),
                "command_to_twist": weights[:3].tolist(),
                "command_offset": weights[3].tolist(),
            }
        )
    best = max(candidates, key=lambda item: item["mean_r2"])
    report = {
        "method": "least_squares_command_t_to_measured_pose_delta_t_plus_1",
        "control_dt_median": float(np.median(dt[valid])),
        "num_transitions": int(np.sum(valid)),
        "selected_action_slice": [best["start"], best["stop"]],
        "mean_r2": best["mean_r2"],
        "r2": best["r2"],
        "command_to_twist": best["command_to_twist"],
        "command_offset": best["command_offset"],
        "all_candidates": candidates,
        "warning": "Commands are normalized controller inputs, not physical pose labels.",
    }
    if best["mean_r2"] < args.min_mean_r2:
        raise RuntimeError(f"No reliable command slice found: best mean R2={best['mean_r2']:.3f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
