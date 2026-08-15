"""Pure trajectory metrics for the replanning-frequency Gate N."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


def nested_observation_sha256(value: Any, hasher: Any | None = None) -> str:
    """Hash nested arrays/tensors without depending on dictionary order or reprs."""
    if hasher is None:
        hasher = hashlib.sha256()
        nested_observation_sha256(value, hasher)
        return hasher.hexdigest()
    if isinstance(value, dict):
        hasher.update(b"dict\0")
        for key in sorted(value):
            hasher.update(str(key).encode("utf-8") + b"\0")
            nested_observation_sha256(value[key], hasher)
        return ""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(array.dtype.str.encode("ascii") + b"\0")
    hasher.update(json.dumps(array.shape).encode("ascii") + b"\0")
    hasher.update(array.tobytes())
    return ""


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def integrate_body_twist(commands: np.ndarray, measured_yaw: np.ndarray, dt: float) -> np.ndarray:
    pose = np.zeros((len(commands) + 1, 3), dtype=np.float64)
    if not len(commands):
        return pose
    for index, (vx, vy, yaw_rate) in enumerate(commands):
        yaw = float(measured_yaw[index])
        c, s = math.cos(yaw), math.sin(yaw)
        pose[index + 1, 0] = pose[index, 0] + (c * vx - s * vy) * dt
        pose[index + 1, 1] = pose[index, 1] + (s * vx + c * vy) * dt
        pose[index + 1, 2] = pose[index, 2] + yaw_rate * dt
    return pose


def cumulative_progress(increments: np.ndarray) -> np.ndarray:
    progress = np.concatenate([[0.0], np.cumsum(np.maximum(increments, 0.0))])
    if progress[-1] <= 1e-12:
        return np.zeros_like(progress)
    return progress / progress[-1]


def summarize_trajectory(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    root_linear_velocity: np.ndarray,
    navigate_command: np.ndarray,
    left_wrist_pose: np.ndarray,
    right_wrist_pose: np.ndarray,
    control_dt_s: float,
) -> dict[str, Any]:
    if len(root_position) != len(navigate_command) + 1:
        raise ValueError("root telemetry must include the initial state plus one state per command")
    yaw = np.unwrap(np.asarray([yaw_from_wxyz(quat) for quat in root_quaternion]))
    relative_xy = root_position[:, :2] - root_position[0, :2]
    relative_yaw = yaw - yaw[0]
    xy_steps = np.linalg.norm(np.diff(root_position[:, :2], axis=0), axis=1)
    yaw_steps = np.abs(np.diff(yaw))
    base_increments = xy_steps + 0.5 * yaw_steps

    if left_wrist_pose.shape[-2:] == (4, 4) and right_wrist_pose.shape[-2:] == (4, 4):
        left_wrist_position = left_wrist_pose[:, :3, 3]
        right_wrist_position = right_wrist_pose[:, :3, 3]
    else:
        left_wrist_position = left_wrist_pose[:, :3]
        right_wrist_position = right_wrist_pose[:, :3]
    left_steps = np.linalg.norm(np.diff(left_wrist_position, axis=0), axis=1)
    right_steps = np.linalg.norm(np.diff(right_wrist_position, axis=0), axis=1)
    arm_increments = left_steps + right_steps
    base_progress = cumulative_progress(base_increments)
    arm_progress = cumulative_progress(arm_increments)
    base_arm_phase_mae = float(np.mean(np.abs(base_progress - arm_progress)))

    command_pose = integrate_body_twist(navigate_command, yaw[:-1], control_dt_s)
    command_endpoint_xy = command_pose[-1, :2]
    measured_endpoint_xy = relative_xy[-1]

    c, s = np.cos(yaw[1:]), np.sin(yaw[1:])
    measured_body_vx = c * root_linear_velocity[1:, 0] + s * root_linear_velocity[1:, 1]
    measured_body_vy = -s * root_linear_velocity[1:, 0] + c * root_linear_velocity[1:, 1]
    measured_body_velocity = np.stack([measured_body_vx, measured_body_vy], axis=1)
    command_linear_velocity = navigate_command[:, :2]

    return {
        "simulated_duration_s": float(len(navigate_command) * control_dt_s),
        "root_path_length_xy_m": float(xy_steps.sum()),
        "root_net_displacement_xy_m": float(np.linalg.norm(relative_xy[-1])),
        "root_net_yaw_rad": float(relative_yaw[-1]),
        "command_integrated_path_length_xy_m": float(
            np.linalg.norm(np.diff(command_pose[:, :2], axis=0), axis=1).sum()
        ),
        "command_integrated_endpoint_xy_m": command_endpoint_xy.tolist(),
        "measured_endpoint_xy_m": measured_endpoint_xy.tolist(),
        "command_vs_measured_endpoint_error_m": float(
            np.linalg.norm(command_endpoint_xy - measured_endpoint_xy)
        ),
        "command_vs_measured_linear_velocity_rms_mps": float(
            np.sqrt(np.mean(np.square(command_linear_velocity - measured_body_velocity)))
        ),
        "base_arm_progress_phase_mae": base_arm_phase_mae,
        "navigation_command_active_fraction": float(
            np.mean(np.linalg.norm(navigate_command, axis=1) > 1e-5)
        ),
    }
