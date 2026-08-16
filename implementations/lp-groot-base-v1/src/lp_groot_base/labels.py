"""Fixed-extent local base path-time labels for LP-GR00T-Base V1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import se2


@dataclass(frozen=True)
class LabelConfig:
    num_anchors: int = 32
    l_xy: float = 0.05
    l_yaw: float = 0.15
    path_extent: float = 16.0
    max_window_seconds: float = 4.0
    epsilon_sigma: float = 1e-8
    min_duration: float = 1e-4

    def __post_init__(self) -> None:
        if self.num_anchors <= 0:
            raise ValueError("num_anchors must be positive")
        if min(self.l_xy, self.l_yaw, self.path_extent, self.max_window_seconds) <= 0:
            raise ValueError("metric scales, path extent, and max window must be positive")


@dataclass
class PathTimeLabel:
    poses: np.ndarray
    log_durations: np.ndarray
    valid: np.ndarray
    complete: bool
    attained_sigma: float
    endpoint_time: float

    @property
    def durations(self) -> np.ndarray:
        return np.exp(self.log_durations)

    @property
    def action(self) -> np.ndarray:
        return np.concatenate([self.poses, self.log_durations], axis=-1)


def metric_increments(local_poses: np.ndarray, config: LabelConfig) -> np.ndarray:
    """Compute adjacent dimensionless base-only path increments."""
    local_poses = np.asarray(local_poses, dtype=np.float64)
    if len(local_poses) < 2:
        return np.empty((0,), dtype=np.float64)
    delta = se2.log(se2.between(local_poses[:-1], local_poses[1:]))
    return np.sqrt(
        (delta[:, 0] / config.l_xy) ** 2
        + (delta[:, 1] / config.l_xy) ** 2
        + (delta[:, 2] / config.l_yaw) ** 2
    )


def _sample_at_sigma(
    poses: np.ndarray,
    times: np.ndarray,
    sigma: np.ndarray,
    target: float,
) -> tuple[np.ndarray, float]:
    """Interpolate pose and arrival time at cumulative metric coordinate target."""
    if target <= sigma[0] + 1e-12:
        return poses[0].copy(), float(times[0])
    if target >= sigma[-1] - 1e-12:
        return poses[-1].copy(), float(times[-1])

    hi = int(np.searchsorted(sigma, target, side="right"))
    hi = min(max(hi, 1), len(sigma) - 1)
    lo = hi - 1
    while hi < len(sigma) and sigma[hi] - sigma[lo] <= 1e-12:
        hi += 1
    if hi >= len(sigma):
        return poses[-1].copy(), float(times[-1])
    denom = sigma[hi] - sigma[lo]
    alpha = float(np.clip((target - sigma[lo]) / denom, 0.0, 1.0))
    pose = se2.interpolate(poses[lo], poses[hi], alpha)
    time = (1.0 - alpha) * times[lo] + alpha * times[hi]
    return pose, float(time)


def build_path_time_label(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    timestamps: np.ndarray,
    start_index: int,
    config: LabelConfig,
) -> PathTimeLabel:
    """Build one fixed-extent 32-anchor label from measured base poses.

    Incomplete windows keep all reached fixed-extent anchors, add one truthful
    terminal anchor, and mask repeated terminal padding. Static windows therefore
    have exactly one valid terminal anchor carrying the real available duration.
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n = len(timestamps)
    if not (len(positions) == len(quaternions_xyzw) == n):
        raise ValueError("positions, quaternions, and timestamps must have equal length")
    if not 0 <= start_index < n:
        raise IndexError(f"start_index {start_index} outside episode length {n}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    t0 = timestamps[start_index]
    end_index = int(np.searchsorted(timestamps, t0 + config.max_window_seconds, side="right") - 1)
    end_index = min(max(end_index, start_index), n - 1)
    world = se2.world_xy_quat_to_se2(
        positions[start_index : end_index + 1],
        quaternions_xyzw[start_index : end_index + 1],
    )
    local = se2.between(world[0], world)
    times = timestamps[start_index : end_index + 1] - t0
    increments = metric_increments(local, config)
    sigma = np.concatenate([[0.0], np.cumsum(increments)])

    complete = bool(sigma[-1] >= config.path_extent - 1e-10)
    if complete:
        endpoint_pose, endpoint_time = _sample_at_sigma(
            local, times, sigma, config.path_extent
        )
        crossing = int(np.searchsorted(sigma, config.path_extent, side="left"))
        path_poses = np.concatenate([local[:crossing], endpoint_pose[None]], axis=0)
        path_times = np.concatenate([times[:crossing], [endpoint_time]])
        path_sigma = np.concatenate([sigma[:crossing], [config.path_extent]])
        attained = config.path_extent
    else:
        path_poses = local
        path_times = times
        path_sigma = sigma
        endpoint_pose = local[-1].copy()
        endpoint_time = float(times[-1])
        attained = float(sigma[-1])

    target_sigma = (
        np.arange(1, config.num_anchors + 1, dtype=np.float64)
        * config.path_extent
        / config.num_anchors
    )
    anchors: list[np.ndarray] = []
    arrival_times: list[float] = []
    for target in target_sigma:
        if target > attained + 1e-10:
            break
        pose, arrival = _sample_at_sigma(path_poses, path_times, path_sigma, float(target))
        anchors.append(pose)
        arrival_times.append(arrival)

    if not complete and len(anchors) < config.num_anchors:
        terminal_is_present = bool(
            anchors
            and np.linalg.norm(se2.log(se2.between(anchors[-1], endpoint_pose))) < 1e-9
            and abs(arrival_times[-1] - endpoint_time) < 1e-9
        )
        if not terminal_is_present:
            anchors.append(endpoint_pose)
            arrival_times.append(endpoint_time)

    if not anchors:
        anchors = [endpoint_pose]
        arrival_times = [endpoint_time]

    valid_count = min(len(anchors), config.num_anchors)
    anchors = anchors[:valid_count]
    arrival = np.asarray(arrival_times[:valid_count], dtype=np.float64)
    durations = np.diff(np.concatenate([[0.0], arrival]))
    durations = np.maximum(durations, config.min_duration)

    pose_array = np.repeat(anchors[-1][None], config.num_anchors, axis=0)
    duration_array = np.full((config.num_anchors,), durations[-1], dtype=np.float64)
    pose_array[:valid_count] = np.asarray(anchors)
    duration_array[:valid_count] = durations
    valid = np.zeros((config.num_anchors,), dtype=np.float32)
    valid[:valid_count] = 1.0

    return PathTimeLabel(
        poses=pose_array.astype(np.float32),
        log_durations=np.log(duration_array).astype(np.float32)[:, None],
        valid=valid,
        complete=complete,
        attained_sigma=float(attained),
        endpoint_time=float(endpoint_time),
    )


def reconstruct_at_times(label: PathTimeLabel, query_times: np.ndarray) -> np.ndarray:
    """Reconstruct desired SE(2) poses at times from zero through the valid anchors."""
    query_times = np.asarray(query_times, dtype=np.float64)
    valid_count = int(np.sum(label.valid > 0.5))
    if valid_count < 1:
        raise ValueError("label has no valid anchors")
    poses = np.concatenate([np.zeros((1, 3)), label.poses[:valid_count]], axis=0)
    arrival = np.concatenate([[0.0], np.cumsum(label.durations[:valid_count, 0])])
    output = np.empty((len(query_times), 3), dtype=np.float64)
    for i, time in enumerate(query_times):
        if time <= 0.0:
            output[i] = poses[0]
        elif time >= arrival[-1]:
            output[i] = poses[-1]
        else:
            hi = int(np.searchsorted(arrival, time, side="right"))
            lo = hi - 1
            alpha = float((time - arrival[lo]) / max(arrival[hi] - arrival[lo], 1e-12))
            output[i] = se2.interpolate(poses[lo], poses[hi], alpha)
    return output

