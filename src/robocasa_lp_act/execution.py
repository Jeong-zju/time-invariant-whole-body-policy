"""Decode base-only LP-ACT predictions onto the fixed 20 Hz controller grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lp_act.se2 import between, compose, exp, interpolate, log

from .schema import (
    ACTION_DIM,
    CONTROL_DT,
    EXECUTION_STEPS,
    LP_ACTION_DIM,
    LP_LOG_DURATION,
    LP_NONBASE_ACTION,
    NUM_QUERIES,
)
from .data import RoboCasaData


@dataclass(frozen=True)
class ControllerCalibration:
    """Measured body twist per unit RoboCasa normalized base command."""

    vx_per_action: float = 0.5851011457937928
    vy_per_action: float = 0.6155741333181562
    yaw_rate_per_action: float = 1.214150563095771

    @property
    def scale(self) -> np.ndarray:
        return np.asarray([self.vx_per_action, self.vy_per_action, self.yaw_rate_per_action])

    def to_dict(self) -> dict[str, float]:
        return {
            "vx_per_action": self.vx_per_action,
            "vy_per_action": self.vy_per_action,
            "yaw_rate_per_action": self.yaw_rate_per_action,
        }


def fit_controller_calibration(data: RoboCasaData, episodes: list[int]) -> ControllerCalibration:
    """Fit measured transition twist = scale * action[t] on train episodes."""

    command_parts: list[np.ndarray] = []
    twist_parts: list[np.ndarray] = []
    for episode in episodes:
        start, end = data.episode_bounds[episode]
        if end - start < 2:
            continue
        dt = np.diff(data.timestamps[start:end])
        twist = log(between(data.base_world_poses[start : end - 1], data.base_world_poses[start + 1 : end])) / dt[:, None]
        command_parts.append(data.actions[start : end - 1, :3])
        twist_parts.append(twist)
    command = np.concatenate(command_parts)
    twist = np.concatenate(twist_parts)
    scales = []
    for dimension in range(3):
        active = np.abs(command[:, dimension]) > 0.02
        denominator = float(np.dot(command[active, dimension], command[active, dimension]))
        if denominator <= 1e-12:
            raise ValueError(f"No active controller samples for base dimension {dimension}")
        scales.append(float(np.dot(command[active, dimension], twist[active, dimension]) / denominator))
    return ControllerCalibration(*scales)


@dataclass(frozen=True)
class DecodedChunk:
    actions: np.ndarray
    boundary_times: np.ndarray
    boundary_base_poses: np.ndarray
    calibrated_command_poses: np.ndarray
    unclipped_base_actions: np.ndarray


def decode_lp_chunk(
    target: np.ndarray,
    *,
    horizon_steps: int = EXECUTION_STEPS,
    control_dt: float = CONTROL_DT,
    calibration: ControllerCalibration = ControllerCalibration(),
    min_segment_duration: float = CONTROL_DT / 4.0,
    max_segment_duration: float = 1.0,
    max_total_duration: float = 4.0,
) -> DecodedChunk:
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != LP_ACTION_DIM:
        raise ValueError(f"Expected (K,{LP_ACTION_DIM}), got {target.shape}")
    durations = np.clip(np.exp(target[:, LP_LOG_DURATION]), min_segment_duration, max_segment_duration)
    anchor_times = np.minimum(np.cumsum(durations), max_total_duration)
    keep = np.r_[True, np.diff(anchor_times) > 1e-9]
    anchor_times = anchor_times[keep]
    anchors = target[keep, :3]

    interpolation_times = np.concatenate([[0.0], anchor_times])
    interpolation_poses = np.concatenate([np.zeros((1, 3)), anchors], axis=0)
    boundary_times = np.arange(horizon_steps + 1, dtype=np.float64) * control_dt
    boundary_poses = np.empty((horizon_steps + 1, 3), dtype=np.float64)
    for index, query in enumerate(boundary_times):
        if query >= interpolation_times[-1]:
            boundary_poses[index] = interpolation_poses[-1]
            continue
        high = int(np.searchsorted(interpolation_times, query, side="right"))
        high = min(max(high, 1), len(interpolation_times) - 1)
        low = high - 1
        width = interpolation_times[high] - interpolation_times[low]
        alpha = 0.0 if width <= 1e-12 else float((query - interpolation_times[low]) / width)
        boundary_poses[index] = interpolate(interpolation_poses[low], interpolation_poses[high], alpha)

    body_twist = log(between(boundary_poses[:-1], boundary_poses[1:])) / control_dt
    unclipped = body_twist / calibration.scale
    actions = np.zeros((horizon_steps, ACTION_DIM), dtype=np.float32)
    actions[:, :3] = np.clip(unclipped, -1.0, 1.0).astype(np.float32)
    upper = target[:, LP_NONBASE_ACTION]
    if upper.shape[0] < horizon_steps:
        upper = np.concatenate([upper, np.repeat(upper[-1:], horizon_steps - upper.shape[0], axis=0)])
    actions[:, 3:12] = upper[:horizon_steps].astype(np.float32)
    calibrated_poses = np.zeros_like(boundary_poses)
    for index, command in enumerate(actions[:, :3]):
        calibrated_poses[index + 1] = compose(
            calibrated_poses[index], exp(command.astype(np.float64) * calibration.scale * control_dt)
        )
    return DecodedChunk(actions, boundary_times, boundary_poses, calibrated_poses, unclipped)
