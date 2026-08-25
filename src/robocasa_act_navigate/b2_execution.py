"""Controller-side measured-pose follower for B2 Path-RateFree predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .b2_labels import yaw_from_xyzw
from .b2_se2 import wrap_angle


@dataclass(frozen=True)
class BaseCalibration:
    matrix: np.ndarray
    bias: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "BaseCalibration":
        value = json.loads(Path(path).read_text())["base"]
        return cls(np.asarray(value["matrix"], dtype=np.float64), np.asarray(value["bias"], dtype=np.float64))

    def inverse(self, measured_increment: np.ndarray) -> np.ndarray:
        command = (np.asarray(measured_increment) - self.bias) @ np.linalg.pinv(self.matrix)
        return np.clip(command, -1.0, 1.0)


@dataclass
class RateFreeFollower:
    calibration: BaseCalibration
    control_dt: float = 0.05
    lookahead_distance: float = 0.025
    position_tolerance: float = 0.006
    yaw_tolerance: float = 0.08
    max_translation_speed: float = 0.35
    max_yaw_rate: float = 0.9
    translation_gain: float = 10.0
    yaw_gain: float = 4.0

    def clear(self) -> None:
        self.world_path = np.empty((0, 3), dtype=np.float64)
        self.path_progress = np.empty(0, dtype=np.float64)
        self.anchor_index = 0

    def __post_init__(self) -> None:
        self.clear()

    def set_plan(self, local_path: np.ndarray, world_position: np.ndarray, world_quaternion: np.ndarray) -> None:
        local = np.asarray(local_path, dtype=np.float64)
        if local.ndim != 2 or local.shape[1] != 3 or not len(local) or not np.isfinite(local).all():
            raise ValueError(f"expected finite local path (K,3), got {local.shape}")
        origin = np.asarray(world_position, dtype=np.float64).reshape(-1)[:2]
        yaw = float(yaw_from_xyzw(np.asarray(world_quaternion).reshape(1, 4))[0])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray([[c, -s], [s, c]])
        self.world_path = np.column_stack((local[:, :2] @ rotation.T + origin, wrap_angle(local[:, 2] + yaw)))
        segment = np.linalg.norm(np.diff(self.world_path[:, :2], axis=0), axis=1)
        self.path_progress = np.concatenate(([0.0], np.cumsum(segment)))
        self.anchor_index = 0

    def command(self, world_position: np.ndarray, world_quaternion: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if not len(self.world_path):
            return np.zeros(4, dtype=np.float32), {"anchor_index": 0.0, "target_index": 0.0}
        position = np.asarray(world_position, dtype=np.float64).reshape(-1)[:2]
        yaw = float(yaw_from_xyzw(np.asarray(world_quaternion).reshape(1, 4))[0])
        # Project onto the remaining forward path so a slightly missed dense
        # anchor cannot deadlock progress forever.
        candidates = self.world_path[self.anchor_index:]
        distance = np.linalg.norm(candidates[:, :2] - position, axis=1)
        yaw_distance = np.abs(wrap_angle(candidates[:, 2] - yaw))
        nearest_offset = int(np.argmin(distance + 0.05 * yaw_distance))
        self.anchor_index += nearest_offset
        while self.anchor_index < len(self.world_path) - 1:
            target = self.world_path[self.anchor_index]
            if np.linalg.norm(target[:2] - position) > self.position_tolerance or abs(float(wrap_angle(target[2] - yaw))) > self.yaw_tolerance:
                break
            self.anchor_index += 1
        target_index = self.anchor_index
        desired_progress = self.path_progress[self.anchor_index] + self.lookahead_distance
        while target_index < len(self.world_path) - 1 and self.path_progress[target_index] < desired_progress:
            target_index += 1
        target = self.world_path[target_index]
        error_world = target[:2] - position
        c, s = np.cos(yaw), np.sin(yaw)
        error_local = np.asarray([c * error_world[0] + s * error_world[1], -s * error_world[0] + c * error_world[1]])
        yaw_error = float(wrap_angle(target[2] - yaw))
        desired_increment = np.asarray([
            *np.clip(self.translation_gain * error_local * self.control_dt,
                     -self.max_translation_speed * self.control_dt, self.max_translation_speed * self.control_dt),
            np.clip(self.yaw_gain * yaw_error * self.control_dt,
                    -self.max_yaw_rate * self.control_dt, self.max_yaw_rate * self.control_dt),
        ])
        native = self.calibration.inverse(desired_increment)
        return np.asarray([native[0], native[1], native[2], 0.0], dtype=np.float32), {
            "anchor_index": float(self.anchor_index), "target_index": float(target_index),
            "position_error_m": float(np.linalg.norm(error_world)), "yaw_error_rad": abs(yaw_error),
            "target_world_x": float(target[0]), "target_world_y": float(target[1]),
            "target_world_yaw": float(target[2]),
            "error_local_x": float(error_local[0]), "error_local_y": float(error_local[1]),
        }
