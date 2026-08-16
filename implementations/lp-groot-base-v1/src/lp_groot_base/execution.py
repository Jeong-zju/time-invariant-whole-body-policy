"""Convert LP path-time predictions into normalized RoboCasa base commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import se2


@dataclass(frozen=True)
class ControllerCalibration:
    command_to_twist: np.ndarray
    command_offset: np.ndarray
    control_dt: float = 0.05

    def __post_init__(self) -> None:
        matrix = np.asarray(self.command_to_twist)
        offset = np.asarray(self.command_offset)
        if matrix.shape != (3, 3) or offset.shape != (3,):
            raise ValueError("Expected a 3x3 command matrix and 3-vector offset")
        if abs(np.linalg.det(matrix)) < 1e-6:
            raise ValueError("Command calibration matrix is singular")

    def twist_to_command(self, twist: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=np.float64)
        command = (twist - self.command_offset) @ np.linalg.inv(self.command_to_twist)
        return np.clip(command, -1.0, 1.0).astype(np.float32)


def reconstruct_prediction(path_time: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Interpolate a predicted 32x4 LP path at controller query times."""
    path_time = np.asarray(path_time, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    if path_time.ndim != 2 or path_time.shape[1] != 4:
        raise ValueError(f"Expected path_time shape (K, 4), got {path_time.shape}")
    poses = np.concatenate([np.zeros((1, 3)), path_time[:, :3]], axis=0)
    durations = np.exp(np.clip(path_time[:, 3], np.log(1e-3), np.log(4.0)))
    arrivals = np.concatenate([[0.0], np.cumsum(durations)])
    output = np.empty((len(query_times), 3), dtype=np.float64)
    for index, time in enumerate(query_times):
        if time <= 0.0:
            output[index] = poses[0]
        elif time >= arrivals[-1]:
            output[index] = poses[-1]
        else:
            high = int(np.searchsorted(arrivals, time, side="right"))
            low = high - 1
            alpha = (time - arrivals[low]) / max(arrivals[high] - arrivals[low], 1e-12)
            output[index] = se2.interpolate(poses[low], poses[high], float(alpha))
    return output


def path_to_base_commands(
    path_time: np.ndarray,
    calibration: ControllerCalibration,
    num_control_steps: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce an open-loop command prefix and its desired local SE(2) poses.

    This never integrates the recorded command stream. The desired poses come
    only from the model's measured-pose LP representation; calibration is used
    solely to express successive desired twists in the simulator command units.
    """
    query_times = calibration.control_dt * np.arange(1, num_control_steps + 1)
    desired = reconstruct_prediction(path_time, query_times)
    previous = np.concatenate([np.zeros((1, 3)), desired[:-1]], axis=0)
    increments = se2.log(se2.between(previous, desired)) / calibration.control_dt
    commands = calibration.twist_to_command(increments)
    return commands, desired.astype(np.float32)
