"""Reconstruct fixed-rate R1Pro trajectories from LP-ACT path-time anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import PathLabel
from .r1pro import NONBASE_CONTINUOUS_INDICES, NONBASE_GRIPPER_INDICES
from .se2 import between, interpolate, log, wrap_angle


@dataclass(frozen=True)
class Reconstruction:
    times: np.ndarray
    base_poses: np.ndarray
    nonbase_targets: np.ndarray
    base_twists: np.ndarray


def reconstruct_at_times(label: PathLabel, query_times: np.ndarray) -> Reconstruction:
    """Interpolate valid anchors at requested times, prepending the chunk origin."""

    query_times = np.asarray(query_times, dtype=np.float64)
    valid = ~label.is_pad
    anchor_times = label.anchor_times[valid]
    anchor_targets = label.target[valid]
    if anchor_times.size == 0:
        raise ValueError("Cannot reconstruct a label with no valid anchors")

    origin_nonbase = label.raw_nonbase[0]
    times = np.concatenate([[0.0], anchor_times])
    base = np.concatenate([np.zeros((1, 3), dtype=np.float64), anchor_targets[:, :3]], axis=0)
    nonbase = np.concatenate([origin_nonbase[None], anchor_targets[:, 3:23]], axis=0)

    clipped_times = np.clip(query_times, 0.0, times[-1])
    output_base = np.zeros((query_times.size, 3), dtype=np.float64)
    output_nonbase = np.zeros((query_times.size, 20), dtype=np.float64)
    for index, time in enumerate(clipped_times):
        high = int(np.searchsorted(times, time, side="right"))
        high = min(max(high, 1), len(times) - 1)
        low = high - 1
        duration = times[high] - times[low]
        alpha = 0.0 if duration <= 1e-12 else float((time - times[low]) / duration)
        output_base[index] = interpolate(base[low], base[high], alpha)
        output_nonbase[index] = nonbase[low]
        output_nonbase[index, NONBASE_CONTINUOUS_INDICES] = (
            (1.0 - alpha) * nonbase[low, NONBASE_CONTINUOUS_INDICES]
            + alpha * nonbase[high, NONBASE_CONTINUOUS_INDICES]
        )
        output_nonbase[index, NONBASE_GRIPPER_INDICES] = nonbase[low, NONBASE_GRIPPER_INDICES]

    if query_times.size > 1:
        dt = np.diff(query_times)
        relative = between(output_base[:-1], output_base[1:])
        base_twists = log(relative) / dt[:, None]
    else:
        base_twists = np.zeros((0, 3), dtype=np.float64)
    return Reconstruction(
        times=query_times,
        base_poses=output_base,
        nonbase_targets=output_nonbase,
        base_twists=base_twists,
    )


def oracle_metrics(
    label: PathLabel,
    measured_base_twists: np.ndarray,
    command_base_twists: np.ndarray,
) -> dict[str, float]:
    """Measure compression/reconstruction error over one raw label window."""

    reconstruction = reconstruct_at_times(label, label.raw_times)
    pose_error = log(between(label.raw_base_poses, reconstruction.base_poses))
    translation_error = np.linalg.norm(pose_error[:, :2], axis=-1)
    yaw_error = np.abs(wrap_angle(pose_error[:, 2]))

    continuous_error = (
        reconstruction.nonbase_targets[:, NONBASE_CONTINUOUS_INDICES]
        - label.raw_nonbase[:, NONBASE_CONTINUOUS_INDICES]
    )
    gripper_error = (
        reconstruction.nonbase_targets[:, NONBASE_GRIPPER_INDICES]
        - label.raw_nonbase[:, NONBASE_GRIPPER_INDICES]
    )

    interval_count = min(
        reconstruction.base_twists.shape[0],
        measured_base_twists.shape[0],
        command_base_twists.shape[0],
    )
    if interval_count:
        measured_error = reconstruction.base_twists[:interval_count] - measured_base_twists[:interval_count]
        command_error = reconstruction.base_twists[:interval_count] - command_base_twists[:interval_count]
        measured_rmse = float(np.sqrt(np.mean(measured_error**2)))
        command_rmse = float(np.sqrt(np.mean(command_error**2)))
    else:
        measured_rmse = 0.0
        command_rmse = 0.0

    target_duration = float(label.raw_times[-1]) if label.raw_times.size else 0.0
    reconstructed_duration = float(label.anchor_times[~label.is_pad][-1])
    return {
        "base_translation_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
        "base_translation_max_m": float(np.max(translation_error)),
        "base_yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
        "base_yaw_max_rad": float(np.max(yaw_error)),
        "base_final_translation_m": float(translation_error[-1]),
        "base_final_yaw_rad": float(yaw_error[-1]),
        "base_twist_measured_rmse": measured_rmse,
        "base_twist_command_rmse": command_rmse,
        "joint_target_rmse_rad": float(np.sqrt(np.mean(continuous_error**2))),
        "joint_target_max_rad": float(np.max(np.abs(continuous_error))),
        "gripper_command_mae": float(np.mean(np.abs(gripper_error))),
        "gripper_command_mismatch_rate": float(np.mean(np.abs(gripper_error) > 0.5)),
        "duration_abs_error_s": abs(reconstructed_duration - target_duration),
        "attained_fraction": float(label.attained_fraction),
        "valid_anchor_fraction": float(label.valid_count / label.is_pad.size),
        "static": float(label.static),
        "reached_path_limit": float(label.reached_path_limit),
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("No oracle metric rows to aggregate")
    keys = rows[0].keys()
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }
    return summary
