"""Execution adapter from B2 Path-Time anchors to RoboCasa command chunks.

The B2 network predicts measured paths, not actuator commands. This module
keeps that distinction explicit and uses a recorded, training-data-only linear
calibration to convert desired measured increments back to the native RoboCasa
command interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .geometry import (
    interpolate_base,
    quaternion_between,
    quaternion_slerp,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
    wrap_angle,
)


@dataclass(frozen=True)
class LinearMap:
    """Measured increment = native command @ matrix + bias."""

    matrix: np.ndarray
    bias: np.ndarray
    lag_steps: int = 0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if matrix.shape != (3, 3) or bias.shape != (3,):
            raise ValueError("linear map must contain a 3x3 matrix and 3-vector bias")
        if self.lag_steps < 0:
            raise ValueError("lag_steps must be non-negative")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
            raise ValueError("linear map contains non-finite values")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bias", bias)

    def inverse(self, measured_increment: np.ndarray) -> np.ndarray:
        measured_increment = np.asarray(measured_increment, dtype=np.float64)
        command = (measured_increment - self.bias) @ np.linalg.pinv(self.matrix)
        return np.clip(command, -1.0, 1.0)


@dataclass(frozen=True)
class ExecutionCalibration:
    base: LinearMap
    eef_position: LinearMap
    eef_rotation: LinearMap

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExecutionCalibration":
        def make(name: str) -> LinearMap:
            item = value[name]
            if not isinstance(item, Mapping):
                raise ValueError(f"calibration entry {name!r} must be a mapping")
            return LinearMap(
                matrix=np.asarray(item["matrix"], dtype=np.float64),
                bias=np.asarray(item["bias"], dtype=np.float64),
                lag_steps=int(item.get("lag_steps", 0)),
            )

        return cls(
            base=make("base"),
            eef_position=make("eef_position"),
            eef_rotation=make("eef_rotation"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExecutionCalibration":
        with Path(path).open() as file:
            return cls.from_dict(json.load(file))


def _validate_prediction(prediction: Mapping[str, np.ndarray]) -> int:
    expected = {
        "base_motion": 4,
        "end_effector_position": 3,
        "end_effector_rotation": 3,
        "gripper_close": 1,
        "control_mode": 1,
    }
    lengths = set()
    for key, width in expected.items():
        value = np.asarray(prediction[key], dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{key} must have shape (K, {width}), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains non-finite values")
        lengths.add(len(value))
    if len(lengths) != 1:
        raise ValueError("all predicted action groups must have the same horizon")
    return lengths.pop()


def _interpolate_path(
    prediction: Mapping[str, np.ndarray],
    query_times: np.ndarray,
    min_segment_duration: float,
    max_segment_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon = _validate_prediction(prediction)
    base_prediction = np.asarray(prediction["base_motion"], dtype=np.float64)
    durations = np.clip(
        np.exp(np.clip(base_prediction[:, 3], -20.0, 20.0)),
        min_segment_duration,
        max_segment_duration,
    )
    arrival = np.cumsum(durations)
    knot_times = np.concatenate([[0.0], arrival])
    base_knots = np.vstack([np.zeros(3), base_prediction[:, :3]])
    eef_position_knots = np.vstack(
        [np.zeros(3), np.asarray(prediction["end_effector_position"], dtype=np.float64)]
    )
    eef_quaternion_knots = np.vstack(
        [
            np.array([0.0, 0.0, 0.0, 1.0]),
            rotvec_to_quaternion(prediction["end_effector_rotation"]),
        ]
    )

    base_values, eef_positions, eef_quaternions = [], [], []
    discrete_indices = []
    for target in query_times:
        hi = int(np.searchsorted(knot_times, target, side="right"))
        hi = min(max(hi, 1), horizon)
        lo = hi - 1
        denominator = max(knot_times[hi] - knot_times[lo], 1e-12)
        alpha = float(np.clip((target - knot_times[lo]) / denominator, 0.0, 1.0))
        base_values.append(interpolate_base(base_knots[lo], base_knots[hi], alpha))
        eef_positions.append(
            (1.0 - alpha) * eef_position_knots[lo] + alpha * eef_position_knots[hi]
        )
        eef_quaternions.append(
            quaternion_slerp(eef_quaternion_knots[lo], eef_quaternion_knots[hi], alpha)
        )
        discrete_indices.append(
            min(int(np.searchsorted(arrival, target, side="left")), horizon - 1)
        )

    indices = np.asarray(discrete_indices, dtype=np.int64)
    gripper = np.asarray(prediction["gripper_close"], dtype=np.float64)[indices, 0]
    control_mode = np.asarray(prediction["control_mode"], dtype=np.float64)[indices, 0]
    return (
        np.asarray(base_values),
        np.asarray(eef_positions),
        np.asarray(eef_quaternions),
        gripper,
        control_mode,
        durations,
    )


def _base_increments(base_path: np.ndarray) -> np.ndarray:
    poses = np.vstack([np.zeros(3), np.asarray(base_path, dtype=np.float64)])
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
    return np.asarray(result)


def _rotation_increments(quaternion_path: np.ndarray) -> np.ndarray:
    quaternions = np.vstack(
        [np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(quaternion_path, dtype=np.float64)]
    )
    return np.vstack(
        [
            quaternion_to_rotvec(quaternion_between(previous, current))
            for previous, current in zip(quaternions[:-1], quaternions[1:])
        ]
    )


def _lead_for_lag(values: np.ndarray, lag_steps: int) -> np.ndarray:
    if lag_steps == 0:
        return values
    if lag_steps >= len(values):
        return np.repeat(values[-1:], len(values), axis=0)
    return np.vstack([values[lag_steps:], np.repeat(values[-1:], lag_steps, axis=0)])


def decode_path_time_commands(
    prediction: Mapping[str, np.ndarray],
    calibration: ExecutionCalibration,
    *,
    control_dt: float = 0.05,
    output_horizon: int | None = None,
    min_segment_duration: float = 0.01,
    max_segment_duration: float = 0.50,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Decode one unbatched B2 prediction into native 20 Hz commands."""
    horizon = _validate_prediction(prediction)
    output_horizon = horizon if output_horizon is None else int(output_horizon)
    if control_dt <= 0.0 or output_horizon <= 0:
        raise ValueError("control_dt and output_horizon must be positive")
    query_times = control_dt * np.arange(1, output_horizon + 1, dtype=np.float64)
    base, eef_position, eef_quaternion, gripper, control_mode, durations = (
        _interpolate_path(
            prediction,
            query_times,
            min_segment_duration,
            max_segment_duration,
        )
    )
    base_delta = _lead_for_lag(_base_increments(base), calibration.base.lag_steps)
    eef_delta = np.diff(np.vstack([np.zeros(3), eef_position]), axis=0)
    eef_delta = _lead_for_lag(eef_delta, calibration.eef_position.lag_steps)
    eef_rotation_delta = _lead_for_lag(
        _rotation_increments(eef_quaternion), calibration.eef_rotation.lag_steps
    )
    commands = {
        "base_motion": np.column_stack(
            [calibration.base.inverse(base_delta), np.zeros(output_horizon)]
        ).astype(np.float32),
        "control_mode": np.where(control_mode >= 0.0, 1.0, -1.0)[:, None].astype(
            np.float32
        ),
        "end_effector_position": calibration.eef_position.inverse(eef_delta).astype(
            np.float32
        ),
        "end_effector_rotation": calibration.eef_rotation.inverse(
            eef_rotation_delta
        ).astype(np.float32),
        "gripper_close": np.where(gripper >= 0.0, 1.0, -1.0)[:, None].astype(
            np.float32
        ),
    }
    diagnostics = {
        "predicted_duration_sum_s": float(durations.sum()),
        "base_command_saturation_fraction": float(
            np.mean(np.abs(commands["base_motion"][:, :3]) >= 0.999)
        ),
        "eef_position_saturation_fraction": float(
            np.mean(np.abs(commands["end_effector_position"]) >= 0.999)
        ),
        "eef_rotation_saturation_fraction": float(
            np.mean(np.abs(commands["end_effector_rotation"]) >= 0.999)
        ),
    }
    return commands, diagnostics
