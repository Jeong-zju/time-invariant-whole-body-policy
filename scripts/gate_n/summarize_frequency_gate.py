#!/usr/bin/env python3
"""Aggregate paired Gate N replanning-frequency rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from frequency_metrics import yaw_from_wxyz


def percentile_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    margin = z * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def relative_xy_and_yaw(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(data["root_position_w"], dtype=np.float64)
    quaternion = np.asarray(data["root_quaternion_wxyz"], dtype=np.float64)
    xy = position[:, :2] - position[0, :2]
    yaw = np.unwrap(np.asarray([yaw_from_wxyz(value) for value in quaternion], dtype=np.float64))
    return xy, yaw - yaw[0]


def load_runs(input_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    runs: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(input_root.glob("replan-*/seed-*/rollout.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        key = (int(report["replan_steps"]), int(report["seed"]))
        if key in runs:
            raise ValueError(f"duplicate run for replan_steps={key[0]}, seed={key[1]}")
        npz_path = path.with_name(report["telemetry_npz"])
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        report["_json_path"] = str(path)
        report["_npz_path"] = str(npz_path)
        runs[key] = report
    return runs


def compare_pair(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    with np.load(reference["_npz_path"]) as reference_data, np.load(candidate["_npz_path"]) as candidate_data:
        reference_xy, reference_yaw = relative_xy_and_yaw(reference_data)
        candidate_xy, candidate_yaw = relative_xy_and_yaw(candidate_data)
        prefix_steps = min(
            int(candidate["replan_steps"]),
            len(reference_data["navigate_command"]),
            len(candidate_data["navigate_command"]),
        )
        state_keys = (
            "root_position_w",
            "root_quaternion_wxyz",
            "root_linear_velocity_w",
            "root_angular_velocity_w",
            "joint_position",
            "left_wrist_pose_pelvis",
            "right_wrist_pose_pelvis",
        )
        prefix_differences = [
            np.asarray(candidate_data[key])[: prefix_steps + 1]
            - np.asarray(reference_data[key])[: prefix_steps + 1]
            for key in state_keys
        ]
        prefix_differences.append(
            np.asarray(candidate_data["navigate_command"])[:prefix_steps]
            - np.asarray(reference_data["navigate_command"])[:prefix_steps]
        )
        prebranch_max_abs = max(
            (float(np.max(np.abs(value))) for value in prefix_differences if value.size),
            default=0.0,
        )
    common_states = min(len(reference_xy), len(candidate_xy))
    if common_states < 2:
        raise ValueError("paired rollouts need at least two common states")
    reference_xy = reference_xy[:common_states]
    candidate_xy = candidate_xy[:common_states]
    reference_yaw = reference_yaw[:common_states]
    candidate_yaw = candidate_yaw[:common_states]
    xy_difference = candidate_xy - reference_xy
    yaw_difference = candidate_yaw - reference_yaw
    return {
        "seed": int(reference["seed"]),
        "common_control_steps": int(common_states - 1),
        "root_xy_trajectory_rms_m": float(np.sqrt(np.mean(np.sum(np.square(xy_difference), axis=1)))),
        "root_yaw_trajectory_rms_rad": float(np.sqrt(np.mean(np.square(yaw_difference)))),
        "common_time_endpoint_xy_error_m": float(np.linalg.norm(xy_difference[-1])),
        "common_time_endpoint_yaw_error_rad": float(abs(yaw_difference[-1])),
        "base_arm_progress_phase_mae_difference": float(
            abs(
                candidate["trajectory"]["base_arm_progress_phase_mae"]
                - reference["trajectory"]["base_arm_progress_phase_mae"]
            )
        ),
        "commanded_vs_measured_endpoint_error_difference_m": float(
            abs(
                candidate["trajectory"]["command_vs_measured_endpoint_error_m"]
                - reference["trajectory"]["command_vs_measured_endpoint_error_m"]
            )
        ),
        "reference_success": bool(reference["rollout"]["success"]),
        "candidate_success": bool(candidate["rollout"]["success"]),
        "success_changed": bool(reference["rollout"]["success"] != candidate["rollout"]["success"]),
        "first_action_chunk_identical": bool(
            reference["first_action_chunk_sha256"] == candidate["first_action_chunk_sha256"]
        ),
        "first_policy_observation_identical": bool(
            reference["first_policy_observation_sha256"]
            == candidate["first_policy_observation_sha256"]
        ),
        "prebranch_control_steps": int(prefix_steps),
        "prebranch_physical_telemetry_identical": bool(prebranch_max_abs == 0.0),
        "prebranch_physical_telemetry_max_abs": prebranch_max_abs,
    }


def aggregate(input_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    runs = load_runs(input_root)
    diagnostic = config["closed_loop_diagnostic"]
    expected_steps = [int(value) for value in diagnostic["replan_steps"]]
    expected_seeds = [int(value) for value in diagnostic["seeds"]]
    reference_steps = int(diagnostic["reference_replan_steps"])

    missing = [
        {"replan_steps": replan_steps, "seed": seed}
        for replan_steps in expected_steps
        for seed in expected_seeds
        if (replan_steps, seed) not in runs
    ]

    conditions: dict[str, Any] = {}
    for replan_steps in expected_steps:
        condition_runs = [runs[(replan_steps, seed)] for seed in expected_seeds if (replan_steps, seed) in runs]
        successes = sum(bool(run["rollout"]["success"]) for run in condition_runs)
        total = len(condition_runs)
        conditions[str(replan_steps)] = {
            "replan_steps": replan_steps,
            "nominal_replan_frequency_hz": 50.0 / replan_steps,
            "completed_runs": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": successes / total if total else None,
            "success_rate_wilson_95": wilson_interval(successes, total),
            "steps_executed": percentile_summary(run["rollout"]["steps_executed"] for run in condition_runs),
            "root_path_length_xy_m": percentile_summary(
                run["trajectory"]["root_path_length_xy_m"] for run in condition_runs
            ),
            "command_vs_measured_endpoint_error_m": percentile_summary(
                run["trajectory"]["command_vs_measured_endpoint_error_m"] for run in condition_runs
            ),
            "base_arm_progress_phase_mae": percentile_summary(
                run["trajectory"]["base_arm_progress_phase_mae"] for run in condition_runs
            ),
            "inference_wall_time_median_s": percentile_summary(
                run["inference"]["wall_time_median_s"] for run in condition_runs
            ),
        }

    paired: dict[str, Any] = {}
    reference_success_rate = conditions[str(reference_steps)]["success_rate"]
    for replan_steps in expected_steps:
        if replan_steps == reference_steps:
            continue
        pairs = [
            compare_pair(runs[(reference_steps, seed)], runs[(replan_steps, seed)])
            for seed in expected_seeds
            if (reference_steps, seed) in runs and (replan_steps, seed) in runs
        ]
        candidate_success_rate = conditions[str(replan_steps)]["success_rate"]
        paired[str(replan_steps)] = {
            "candidate_replan_steps": replan_steps,
            "candidate_replan_frequency_hz": 50.0 / replan_steps,
            "reference_replan_steps": reference_steps,
            "reference_replan_frequency_hz": 50.0 / reference_steps,
            "pairs": len(pairs),
            "all_first_action_chunks_identical": bool(
                pairs and all(pair["first_action_chunk_identical"] for pair in pairs)
            ),
            "all_first_policy_observations_identical": bool(
                pairs and all(pair["first_policy_observation_identical"] for pair in pairs)
            ),
            "all_prebranch_physical_telemetry_identical": bool(
                pairs and all(pair["prebranch_physical_telemetry_identical"] for pair in pairs)
            ),
            "prebranch_physical_telemetry_max_abs": percentile_summary(
                pair["prebranch_physical_telemetry_max_abs"] for pair in pairs
            ),
            "success_rate_change": (
                candidate_success_rate - reference_success_rate
                if candidate_success_rate is not None and reference_success_rate is not None
                else None
            ),
            "success_changed_pairs": sum(pair["success_changed"] for pair in pairs),
            "root_xy_trajectory_rms_m": percentile_summary(
                pair["root_xy_trajectory_rms_m"] for pair in pairs
            ),
            "root_yaw_trajectory_rms_rad": percentile_summary(
                pair["root_yaw_trajectory_rms_rad"] for pair in pairs
            ),
            "common_time_endpoint_xy_error_m": percentile_summary(
                pair["common_time_endpoint_xy_error_m"] for pair in pairs
            ),
            "common_time_endpoint_yaw_error_rad": percentile_summary(
                pair["common_time_endpoint_yaw_error_rad"] for pair in pairs
            ),
            "base_arm_progress_phase_mae_difference": percentile_summary(
                pair["base_arm_progress_phase_mae_difference"] for pair in pairs
            ),
            "commanded_vs_measured_endpoint_error_difference_m": percentile_summary(
                pair["commanded_vs_measured_endpoint_error_difference_m"] for pair in pairs
            ),
            "per_seed": pairs,
        }

    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "input_root": str(input_root),
        "expected_runs": len(expected_steps) * len(expected_seeds),
        "completed_runs": len(runs),
        "missing_runs": missing,
        "conditions": conditions,
        "paired_against_reference": paired,
        "interpretation": {
            "changed_variable": "closed_loop_replanning_frequency",
            "fixed_variables": [
                "checkpoint",
                "seed",
                "first_policy_observation",
                "first_policy_inference_rng_seed",
                "physics_hz",
                "control_hz",
                "action_slot_dt_s",
            ],
            "causal_limit": "This report does not yet identify whether any effect is caused by implicit delta-t, discarded chunk suffixes, or changed feedback bandwidth.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = aggregate(args.input_root, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
