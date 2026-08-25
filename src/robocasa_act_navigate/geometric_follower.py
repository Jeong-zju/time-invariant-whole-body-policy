"""Measured-pose feedback follower for common-origin geometric base paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .adaptive_path import wrap_angle
from .b2_labels import yaw_from_xyzw


@dataclass(frozen=True)
class BaseRateCalibration:
    """Body twist per second = normalized command @ matrix + bias."""

    matrix: np.ndarray
    bias: np.ndarray
    fitted_control_hz: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if matrix.shape != (3, 3) or bias.shape != (3,):
            raise ValueError(f"expected matrix (3,3) and bias (3,), got {matrix.shape}, {bias.shape}")
        if not np.isfinite(matrix).all() or not np.isfinite(bias).all() or self.fitted_control_hz <= 0.0:
            raise ValueError("invalid base-rate calibration")

    @classmethod
    def identity(cls, control_hz: float = 30.0) -> "BaseRateCalibration":
        return cls(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), control_hz)

    @classmethod
    def load(cls, path: str | Path) -> "BaseRateCalibration":
        value = json.loads(Path(path).read_text())["base_rate"]
        return cls(
            np.asarray(value["matrix"], dtype=np.float64),
            np.asarray(value["bias"], dtype=np.float64),
            float(value["fitted_control_hz"]),
        )

    def to_dict(self) -> dict:
        return {
            "matrix": np.asarray(self.matrix, dtype=np.float64).tolist(),
            "bias": np.asarray(self.bias, dtype=np.float64).tolist(),
            "fitted_control_hz": float(self.fitted_control_hz),
            "semantics": "measured_body_twist_per_second = normalized_command @ matrix + bias",
        }

    def inverse(self, desired_body_twist: np.ndarray) -> np.ndarray:
        command = (np.asarray(desired_body_twist, dtype=np.float64) - self.bias) @ np.linalg.pinv(self.matrix)
        return np.clip(command, -1.0, 1.0)


@dataclass
class MeasuredPosePathFollower:
    calibration: BaseRateCalibration
    control_hz: float = 30.0
    yaw_radius_m: float = 0.25
    lookahead_m: float = 0.05
    position_gain: float = 4.0
    yaw_gain: float = 3.0
    max_translation_speed_mps: float = 0.35
    max_yaw_rate_radps: float = 0.9
    max_command_delta_per_tick: float = 0.15
    goal_position_tolerance_m: float = 0.01
    goal_yaw_tolerance_rad: float = 0.02

    def __post_init__(self) -> None:
        if self.control_hz <= 0.0 or self.yaw_radius_m <= 0.0 or self.lookahead_m <= 0.0:
            raise ValueError("controller frequency and geometric scales must be positive")
        self.clear()

    def clear(self) -> None:
        self.world_path = np.empty((0, 3), dtype=np.float64)
        self.metric_path = np.empty((0, 3), dtype=np.float64)
        self.cumulative_progress = np.empty(0, dtype=np.float64)
        self.minimum_progress = 0.0
        self.last_command = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _world_yaw(quaternion_xyzw: np.ndarray) -> float:
        return float(yaw_from_xyzw(np.asarray(quaternion_xyzw, dtype=np.float64).reshape(1, 4))[0])

    def set_plan(
        self,
        local_common_origin_path: np.ndarray,
        world_position: np.ndarray,
        world_quaternion_xyzw: np.ndarray,
        *,
        preserve_last_command: bool = False,
    ) -> None:
        local = np.asarray(local_common_origin_path, dtype=np.float64)
        if local.ndim != 2 or local.shape[1] != 3 or not len(local) or not np.isfinite(local).all():
            raise ValueError(f"expected finite local path (K,3), got {local.shape}")
        origin_xy = np.asarray(world_position, dtype=np.float64).reshape(-1)[:2]
        origin_yaw = self._world_yaw(world_quaternion_xyzw)
        cosine, sine = np.cos(origin_yaw), np.sin(origin_yaw)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        anchors = np.column_stack(
            (
                local[:, :2] @ rotation.T + origin_xy,
                origin_yaw + np.unwrap(local[:, 2]),
            )
        )
        # The current measured base pose is the explicit first point.  Every
        # predicted anchor remains relative to this one common origin.
        self.world_path = np.concatenate(
            (np.asarray([[origin_xy[0], origin_xy[1], origin_yaw]]), anchors),
            axis=0,
        )
        self.metric_path = self.world_path.copy()
        self.metric_path[:, 2] *= self.yaw_radius_m
        increments = np.diff(self.metric_path, axis=0)
        segment_lengths = np.linalg.norm(increments, axis=-1)
        self.cumulative_progress = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        self.minimum_progress = 0.0
        if not preserve_last_command:
            self.last_command = np.zeros(3, dtype=np.float64)

    def _project_progress(self, position_xy: np.ndarray, yaw: float) -> float:
        query_yaw = yaw + 2.0 * np.pi * np.round((self.world_path[:, 2] - yaw) / (2.0 * np.pi))
        best_score = np.inf
        best_progress = self.minimum_progress
        for index in range(len(self.world_path) - 1):
            start_progress = self.cumulative_progress[index]
            finish_progress = self.cumulative_progress[index + 1]
            if finish_progress + 1e-12 < self.minimum_progress:
                continue
            start = self.metric_path[index]
            finish = self.metric_path[index + 1]
            chord = finish - start
            denominator = float(np.dot(chord, chord))
            query = np.asarray([position_xy[0], position_xy[1], self.yaw_radius_m * query_yaw[index]])
            alpha = 0.0 if denominator <= 1e-18 else float(np.clip(np.dot(query - start, chord) / denominator, 0.0, 1.0))
            candidate = start + alpha * chord
            score = float(np.dot(query - candidate, query - candidate))
            progress = start_progress + alpha * (finish_progress - start_progress)
            if progress + 1e-12 >= self.minimum_progress and score < best_score:
                best_score = score
                best_progress = progress
        self.minimum_progress = max(self.minimum_progress, best_progress)
        return self.minimum_progress

    def _pose_at_progress(self, progress: float) -> np.ndarray:
        progress = float(np.clip(progress, 0.0, self.cumulative_progress[-1]))
        high = int(np.searchsorted(self.cumulative_progress, progress, side="right"))
        high = min(max(high, 1), len(self.world_path) - 1)
        low = high - 1
        width = self.cumulative_progress[high] - self.cumulative_progress[low]
        alpha = 0.0 if width <= 1e-18 else (progress - self.cumulative_progress[low]) / width
        return self.world_path[low] + alpha * (self.world_path[high] - self.world_path[low])

    def command(
        self,
        world_position: np.ndarray,
        world_quaternion_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        if len(self.world_path) < 2:
            return np.zeros(4, dtype=np.float32), {"done": True, "progress_m": 0.0}
        position = np.asarray(world_position, dtype=np.float64).reshape(-1)[:2]
        yaw = self._world_yaw(world_quaternion_xyzw)
        endpoint = self.world_path[-1]
        endpoint_position_error = float(np.linalg.norm(endpoint[:2] - position))
        endpoint_yaw_error = abs(float(wrap_angle(endpoint[2] - yaw)))
        done = (
            endpoint_position_error <= self.goal_position_tolerance_m
            and endpoint_yaw_error <= self.goal_yaw_tolerance_rad
        )
        if done:
            self.last_command.fill(0.0)
            return np.zeros(4, dtype=np.float32), {
                "done": True,
                "progress_m": float(self.cumulative_progress[-1]),
                "endpoint_position_error_m": endpoint_position_error,
                "endpoint_yaw_error_rad": endpoint_yaw_error,
            }

        progress = self._project_progress(position, yaw)
        target_progress = min(progress + self.lookahead_m, float(self.cumulative_progress[-1]))
        target = self._pose_at_progress(target_progress)
        error_world = target[:2] - position
        cosine, sine = np.cos(yaw), np.sin(yaw)
        error_body = np.asarray(
            [
                cosine * error_world[0] + sine * error_world[1],
                -sine * error_world[0] + cosine * error_world[1],
            ]
        )
        desired_translation = self.position_gain * error_body
        norm = float(np.linalg.norm(desired_translation))
        if norm > self.max_translation_speed_mps:
            desired_translation *= self.max_translation_speed_mps / norm
        desired_yaw_rate = float(
            np.clip(
                self.yaw_gain * float(wrap_angle(target[2] - yaw)),
                -self.max_yaw_rate_radps,
                self.max_yaw_rate_radps,
            )
        )
        desired_twist = np.asarray([desired_translation[0], desired_translation[1], desired_yaw_rate])
        raw_command = self.calibration.inverse(desired_twist)
        limited_command = np.clip(
            raw_command,
            self.last_command - self.max_command_delta_per_tick,
            self.last_command + self.max_command_delta_per_tick,
        )
        self.last_command = limited_command
        native = np.asarray([limited_command[0], limited_command[1], limited_command[2], 0.0], dtype=np.float32)
        return native, {
            "done": False,
            "progress_m": float(progress),
            "target_progress_m": float(target_progress),
            "endpoint_progress_m": float(self.cumulative_progress[-1]),
            "endpoint_position_error_m": endpoint_position_error,
            "endpoint_yaw_error_rad": endpoint_yaw_error,
            "target_position_error_m": float(np.linalg.norm(error_world)),
            "target_yaw_error_rad": abs(float(wrap_angle(target[2] - yaw))),
        }
