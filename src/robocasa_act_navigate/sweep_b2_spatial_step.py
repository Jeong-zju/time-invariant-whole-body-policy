"""Choose B2 spatial resolution using an oracle measured-feedback reconstruction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .b2_execution import BaseCalibration, RateFreeFollower
from .b2_labels import build_rate_free_label, fixed_rate_free_metric, increments_to_absolute_path, measured_base_poses
from .b2_se2 import between, log
from .data import load_data, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-m", default="0.01,0.015,0.02,0.025")
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    data = load_data(args.data_root)
    split = load_split(args.split)
    calibration = BaseCalibration.load(args.calibration)
    candidates = []
    for episode in split["val"]:
        start, end = data.episode_bounds[episode]
        candidates.extend(range(start, end - 1))
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(candidates, size=min(args.samples, len(candidates)), replace=False))
    poses = measured_base_poses(data.states)

    reports = []
    for spatial_step in [float(value) for value in args.steps_m.split(",")]:
        metric = fixed_rate_free_metric(spatial_step_m=spatial_step, num_anchors=args.num_anchors)
        command_errors = []
        increment_errors = []
        direction = []
        saturation = []
        valid_counts = []
        stop_counts = 0
        for index in indices:
            label = build_rate_free_label(data, int(index), metric)
            valid = label.valid_count
            follower = RateFreeFollower(
                calibration,
                lookahead_distance=spatial_step,
                position_tolerance=min(0.006, 0.30 * spatial_step),
                translation_gain=0.25 / spatial_step,
            )
            oracle_path = increments_to_absolute_path(label.target[:, :3])
            follower.set_plan(oracle_path[:valid], data.states[index, :3], data.states[index, 3:7])
            command, _ = follower.command(data.states[index, :3], data.states[index, 3:7])
            demonstrated = data.actions[index, :3]
            actual_increment = log(between(poses[index], poses[index + 1]))
            reconstructed_increment = command[:3] @ calibration.matrix + calibration.bias
            command_errors.append(np.abs(command[:3] - demonstrated))
            increment_errors.append(np.square(reconstructed_increment - actual_increment))
            # B2 labels supervise measured geometry, so the oracle gate must
            # compare reconstructed physical increments with measured physical
            # increments. Native-command disagreement remains a diagnostic and
            # can include controller delay.
            if np.linalg.norm(actual_increment[:2]) > 1e-3 and np.linalg.norm(reconstructed_increment[:2]) > 1e-3:
                direction.append(float(np.dot(reconstructed_increment[:2], actual_increment[:2]) /
                                       (np.linalg.norm(reconstructed_increment[:2]) * np.linalg.norm(actual_increment[:2]))))
            saturation.append(float(np.any(np.abs(command[:3]) >= 0.999)))
            valid_counts.append(valid)
            stop_counts += int(valid < args.num_anchors)
        command_mae = np.mean(command_errors, axis=0)
        increment_rmse = np.sqrt(np.mean(increment_errors, axis=0))
        direction_mean = float(np.mean(direction)) if direction else -1.0
        report = {
            "spatial_step_m": spatial_step,
            "num_anchors": args.num_anchors,
            "max_extent_m": spatial_step * args.num_anchors,
            "command_mae_xyz": command_mae.tolist(),
            "increment_rmse_xyyaw": increment_rmse.tolist(),
            "moving_direction_cosine_mean": direction_mean,
            "moving_direction_reverse_fraction": float(np.mean(np.asarray(direction) < 0.0)) if direction else 1.0,
            "saturation_fraction": float(np.mean(saturation)),
            "mean_valid_anchors": float(np.mean(valid_counts)),
            "stop_fraction": stop_counts / len(indices),
        }
        report["score"] = float(np.linalg.norm(increment_rmse[:2]) + 0.25 * increment_rmse[2] +
                                0.01 * max(0.0, 1.0 - direction_mean))
        reports.append(report)

    acceptable = [row for row in reports if row["moving_direction_cosine_mean"] >= 0.8 and
                  row["moving_direction_reverse_fraction"] <= 0.05]
    if not acceptable:
        raise AssertionError(json.dumps({"event": "B2_SPATIAL_SWEEP_FAILED", "candidates": reports}))
    selected = min(acceptable, key=lambda row: row["score"])
    payload = {"event": "B2_SPATIAL_SWEEP_PASSED", "selected": selected, "candidates": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
