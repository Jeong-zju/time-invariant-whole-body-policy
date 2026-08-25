"""Audit train-only adaptive geometric segmentation for NavigateKitchen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .adaptive_path import geometric_rdp_indices, piecewise_reconstruction_error, wrap_angle
from .b2_labels import measured_base_poses
from .data import load_data, load_split


def common_origin_path(world_poses: np.ndarray, start: int) -> np.ndarray:
    poses = np.asarray(world_poses[start:], dtype=np.float64)
    origin = np.asarray(world_poses[start], dtype=np.float64)
    delta = poses[:, :2] - origin[:2]
    cosine = np.cos(origin[2])
    sine = np.sin(origin[2])
    return np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            wrap_angle(poses[:, 2] - origin[2]),
        )
    )


def quantiles(values: list[float] | list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        str(level): float(np.quantile(array, level))
        for level in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--frequency-check-stride", type=int, default=20)
    args = parser.parse_args()

    data = load_data(args.data_root)
    split = load_split(args.split)
    starts: list[tuple[int, int]] = []
    for episode in split["train"]:
        episode_start, episode_end = data.episode_bounds[episode]
        starts.extend((index, episode_end) for index in range(episode_start, episode_end))
    if args.sample_limit and len(starts) > args.sample_limit:
        generator = np.random.default_rng(args.seed)
        selected = sorted(generator.choice(len(starts), args.sample_limit, replace=False).tolist())
        starts = [starts[index] for index in selected]

    tolerances = ((0.005, 0.01), (0.01, 0.02), (0.02, 0.05))
    budgets = (8, 16, 32)
    statistics: dict[tuple[float, float], dict] = {}
    for tolerance in tolerances:
        statistics[tolerance] = {
            "required_future_knots": [],
            "full_reconstruction_translation_m": [],
            "full_reconstruction_yaw_rad": [],
            "frequency_10hz_knot_count_delta": [],
            "frequency_10hz_cross_translation_m": [],
            "frequency_10hz_cross_yaw_rad": [],
            "budgets": {
                budget: {
                    "terminal_covered": 0,
                    "valid_knots": [],
                    "covered_source_frames": [],
                    "covered_source_seconds_diagnostic_only": [],
                    "covered_se2_length_m_yaw_radius_0p25": [],
                    "endpoint_translation_m": [],
                    "endpoint_abs_yaw_rad": [],
                }
                for budget in budgets
            },
        }

    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ordinal, (start, episode_end) in enumerate(starts, 1):
        episode_key = (int(data.episode_indices[start]), episode_end)
        if episode_key not in cache:
            episode_start, _ = data.episode_bounds[episode_key[0]]
            poses = measured_base_poses(data.states[episode_start:episode_end])
            times = data.timestamps[episode_start:episode_end]
            cache[episode_key] = (poses, times, np.array([episode_start], dtype=np.int64))
        poses, times, origin_holder = cache[episode_key]
        local_start = start - int(origin_holder[0])
        path = common_origin_path(poses, local_start)
        local_times = times[local_start:]
        if len(path) > 1:
            increments = np.diff(path, axis=0)
            increments[:, 2] = wrap_angle(increments[:, 2])
            cumulative_length = np.concatenate(
                (
                    [0.0],
                    np.cumsum(
                        np.sqrt(
                            increments[:, 0] ** 2
                            + increments[:, 1] ** 2
                            + (0.25 * increments[:, 2]) ** 2
                        )
                    ),
                )
            )
        else:
            cumulative_length = np.zeros(1, dtype=np.float64)

        for tolerance in tolerances:
            translation_tolerance, yaw_tolerance = tolerance
            values = statistics[tolerance]
            keep = geometric_rdp_indices(path, translation_tolerance, yaw_tolerance)
            translation_error, yaw_error = piecewise_reconstruction_error(
                path,
                keep,
                translation_tolerance,
                yaw_tolerance,
            )
            if translation_error > translation_tolerance + 1e-10 or yaw_error > yaw_tolerance + 1e-10:
                raise AssertionError("RDP reconstruction exceeded its physical error contract")
            required = len(keep) - 1
            values["required_future_knots"].append(required)
            values["full_reconstruction_translation_m"].append(translation_error)
            values["full_reconstruction_yaw_rad"].append(yaw_error)
            for budget in budgets:
                budget_values = values["budgets"][budget]
                selected_position = min(budget, required)
                covered = int(keep[selected_position]) if required else 0
                terminal = required <= budget
                budget_values["terminal_covered"] += int(terminal)
                budget_values["valid_knots"].append(selected_position)
                budget_values["covered_source_frames"].append(covered)
                budget_values["covered_source_seconds_diagnostic_only"].append(
                    float(local_times[covered] - local_times[0])
                )
                budget_values["covered_se2_length_m_yaw_radius_0p25"].append(
                    float(cumulative_length[covered])
                )
                budget_values["endpoint_translation_m"].append(float(np.linalg.norm(path[covered, :2])))
                budget_values["endpoint_abs_yaw_rad"].append(float(abs(path[covered, 2])))

            if args.frequency_check_stride > 0 and ordinal % args.frequency_check_stride == 1 and len(path) > 2:
                decimated_indices = np.arange(0, len(path), 2, dtype=np.int64)
                if decimated_indices[-1] != len(path) - 1:
                    decimated_indices = np.append(decimated_indices, len(path) - 1)
                decimated_keep = geometric_rdp_indices(
                    path[decimated_indices],
                    translation_tolerance,
                    yaw_tolerance,
                )
                mapped_keep = decimated_indices[decimated_keep]
                cross_translation, cross_yaw = piecewise_reconstruction_error(
                    path,
                    mapped_keep,
                    translation_tolerance,
                    yaw_tolerance,
                )
                values["frequency_10hz_knot_count_delta"].append(int(len(decimated_keep) - len(keep)))
                values["frequency_10hz_cross_translation_m"].append(cross_translation)
                values["frequency_10hz_cross_yaw_rad"].append(cross_yaw)

        if ordinal == 1 or ordinal % 5_000 == 0 or ordinal == len(starts):
            print(json.dumps({"processed": ordinal, "total": len(starts)}), flush=True)

    report = {
        "task": "NavigateKitchen",
        "source": "measured base position and xyzw quaternion",
        "episodes": "fixed train split only",
        "train_episode_count": len(split["train"]),
        "num_start_states": len(starts),
        "label_reference": "every retained pose is T_t^-1 T_j with one common current-base origin",
        "segmentation_rule": "first K future geometric RDP knots; horizon ends when complexity budget is full",
        "forbidden_label_inputs": ["timestamp", "duration", "frame count", "velocity", "command integral"],
        "source_time_is_reported_for_diagnostics_only": True,
        "tolerances": {},
    }
    for tolerance, values in statistics.items():
        key = f"translation_{tolerance[0]}m_yaw_{tolerance[1]}rad"
        output = {
            "required_future_knots": quantiles(values["required_future_knots"]),
            "full_reconstruction_translation_m": quantiles(values["full_reconstruction_translation_m"]),
            "full_reconstruction_yaw_rad": quantiles(values["full_reconstruction_yaw_rad"]),
            "frequency_10hz_diagnostic": {
                "knot_count_delta": quantiles(values["frequency_10hz_knot_count_delta"]),
                "cross_reconstruction_translation_m": quantiles(values["frequency_10hz_cross_translation_m"]),
                "cross_reconstruction_yaw_rad": quantiles(values["frequency_10hz_cross_yaw_rad"]),
            },
            "budgets": {},
        }
        for budget, budget_values in values["budgets"].items():
            output["budgets"][str(budget)] = {
                "terminal_coverage_fraction": budget_values["terminal_covered"] / len(starts),
                "valid_knots": quantiles(budget_values["valid_knots"]),
                "covered_source_frames_diagnostic_only": quantiles(budget_values["covered_source_frames"]),
                "covered_source_seconds_diagnostic_only": quantiles(
                    budget_values["covered_source_seconds_diagnostic_only"]
                ),
                "covered_se2_length_m_yaw_radius_0p25": quantiles(
                    budget_values["covered_se2_length_m_yaw_radius_0p25"]
                ),
                "endpoint_translation_m": quantiles(budget_values["endpoint_translation_m"]),
                "endpoint_abs_yaw_rad": quantiles(budget_values["endpoint_abs_yaw_rad"]),
            }
        report["tolerances"][key] = output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "ADAPTIVE_PATH_AUDIT_COMPLETE", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
