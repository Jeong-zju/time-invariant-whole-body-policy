"""Spatial-step Path-RateFree labels for NavigateKitchen.

The base target is a measured relative SE(2) path sampled at equal physical
progress.  The representation contains no timestamp, duration, velocity, or
rate.  V2 uses a fixed physical lookahead and scans the future trajectory by
geometry alone; timestamps are not used to construct its labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .b2_se2 import between, compose, exp, interpolate, log

from .data import NavigateData
from .schema import CHUNK_SIZE


def yaw_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero-norm base quaternion")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def measured_base_poses(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"expected state (N,16), got {states.shape}")
    return np.column_stack((states[:, 0], states[:, 1], yaw_from_xyzw(states[:, 3:7])))


@dataclass(frozen=True)
class RateFreeMetric:
    """Physical SE(2) arc metric: ds^2 = dx^2 + dy^2 + (r dtheta)^2."""

    yaw_radius_m: float
    spatial_step_m: float
    num_anchors: int
    construction: str = "fixed_spatial_step"
    reference_horizon_s: float | None = None
    fit_stride: int | None = None

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RateFreeMetric":
        payload = dict(value)
        if "spatial_step_m" not in payload:
            extent = float(payload.pop("path_extent_m"))
            anchors = int(payload.pop("num_anchors", CHUNK_SIZE))
            payload["spatial_step_m"] = extent / anchors
            payload["num_anchors"] = anchors
        payload.setdefault("construction", "fixed_spatial_step")
        return cls(**payload)

    @property
    def path_extent_m(self) -> float:
        return self.spatial_step_m * self.num_anchors


@dataclass(frozen=True)
class RateFreeConfig:
    num_anchors: int = 8
    static_epsilon_m: float = 1e-4


@dataclass(frozen=True)
class RateFreeLabel:
    target: np.ndarray
    is_pad: np.ndarray
    anchor_progress_m: np.ndarray
    source_indices: np.ndarray
    attained_extent_m: float
    reached_extent: bool

    @property
    def valid_count(self) -> int:
        return int((~self.is_pad).sum())


def absolute_path_to_increments(path: np.ndarray) -> np.ndarray:
    """Encode common-origin SE(2) anchors as body-frame Lie increments."""

    absolute = np.asarray(path, dtype=np.float64)
    increments = np.empty_like(absolute)
    previous = np.zeros(3, dtype=np.float64)
    for index, pose in enumerate(absolute):
        increments[index] = log(between(previous, pose))
        previous = pose
    return increments


def increments_to_absolute_path(increments: np.ndarray) -> np.ndarray:
    """Decode body-frame Lie increments to common-origin SE(2) anchors."""

    values = np.asarray(increments, dtype=np.float64)
    path = np.empty_like(values)
    current = np.zeros(3, dtype=np.float64)
    for index, delta in enumerate(values):
        current = compose(current, exp(delta))
        path[index] = current
    return path


def metric_increments(poses: np.ndarray, yaw_radius_m: float) -> np.ndarray:
    delta = log(between(np.asarray(poses[:-1]), np.asarray(poses[1:])))
    return np.sqrt(np.square(delta[:, 0]) + np.square(delta[:, 1]) + np.square(yaw_radius_m * delta[:, 2]))


def fit_rate_free_metric(
    data: NavigateData,
    train_episodes: list[int],
    *,
    yaw_radius_m: float = 0.25,
    reference_horizon_s: float = CHUNK_SIZE / 20.0,
    fit_stride: int = 4,
) -> RateFreeMetric:
    """Fit median moving path coverage over a fixed number of seconds.

    The physical metric constants do not come from adjacent-frame statistics,
    and endpoint lookup uses timestamps rather than a number of frames.
    """

    coverage: list[float] = []
    for episode in train_episodes:
        start, end = data.episode_bounds[episode]
        if end - start < 2:
            continue
        times = data.timestamps[start:end]
        poses = measured_base_poses(data.states[start:end])
        increments = metric_increments(poses, yaw_radius_m)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        for local_start in range(0, len(times) - 1, fit_stride):
            target_time = times[local_start] + reference_horizon_s
            high = int(np.searchsorted(times, target_time, side="left"))
            if high >= len(times):
                continue
            if times[high] == target_time:
                length = cumulative[high] - cumulative[local_start]
                endpoint = poses[high]
            else:
                low = high - 1
                alpha = float((target_time - times[low]) / (times[high] - times[low]))
                length = cumulative[low] - cumulative[local_start] + alpha * increments[low]
                endpoint = interpolate(poses[low], poses[high], alpha)
            net = log(between(poses[local_start], endpoint))
            moving = np.linalg.norm(net[:2]) > 0.01 or abs(net[2]) > 0.02
            if moving and np.isfinite(length) and length > 0.0:
                coverage.append(float(length))
    if not coverage:
        raise ValueError("no moving fixed-second windows available for B2 metric fit")
    extent = float(np.median(np.asarray(coverage)))
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError(f"invalid fitted path extent {extent}")
    return RateFreeMetric(
        yaw_radius_m,
        extent / CHUNK_SIZE,
        CHUNK_SIZE,
        "legacy_time_fitted",
        reference_horizon_s,
        fit_stride,
    )


def fixed_rate_free_metric(
    *, yaw_radius_m: float = 0.25, spatial_step_m: float = 0.025, num_anchors: int = 8
) -> RateFreeMetric:
    """Return a frequency-independent metric parameterized by spatial resolution."""

    if yaw_radius_m <= 0.0 or spatial_step_m <= 0.0 or num_anchors <= 0:
        raise ValueError("fixed geometric metric constants must be positive")
    return RateFreeMetric(yaw_radius_m, spatial_step_m, num_anchors, "fixed_spatial_step")


def _interpolated_target(actions: np.ndarray, low: int, high: int, alpha: float) -> np.ndarray:
    value = np.asarray(actions[low], dtype=np.float64).copy()
    # Native upper-body continuous commands follow the same geometric anchor.
    value[5:11] = (1.0 - alpha) * actions[low, 5:11] + alpha * actions[high, 5:11]
    # control_mode and gripper_close use zero-order hold.
    return value


def build_rate_free_from_arrays(
    timestamps: np.ndarray,
    poses_world: np.ndarray,
    actions: np.ndarray,
    metric: RateFreeMetric,
    config: RateFreeConfig = RateFreeConfig(),
) -> RateFreeLabel:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    poses_world = np.asarray(poses_world, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64).copy()
    if len(timestamps) != len(poses_world) or len(timestamps) != len(actions):
        raise ValueError("timestamp, pose, and action lengths differ")
    if len(timestamps) == 0 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be nonempty and strictly increasing")
    local = between(poses_world[0], poses_world)
    cumulative = np.concatenate(([0.0], np.cumsum(metric_increments(local, metric.yaw_radius_m))))
    reached_indices = np.flatnonzero(cumulative >= metric.path_extent_m)
    reached = bool(reached_indices.size)
    if reached:
        source_last = max(1, int(reached_indices[0]))
    else:
        source_last = len(timestamps) - 1
    timestamps = timestamps[: source_last + 1]
    local = local[: source_last + 1]
    actions = actions[: source_last + 1]
    cumulative = cumulative[: source_last + 1]
    if reached and cumulative[-1] > metric.path_extent_m:
        low, high = len(cumulative) - 2, len(cumulative) - 1
        alpha = float((metric.path_extent_m - cumulative[low]) / max(cumulative[high] - cumulative[low], 1e-12))
        local[-1] = interpolate(local[low], local[high], alpha)
        actions[-1] = _interpolated_target(actions, low, high, alpha)
        cumulative[-1] = metric.path_extent_m

    attained = float(cumulative[-1])
    if config.num_anchors != metric.num_anchors:
        raise ValueError("config/metric anchor count mismatch")
    requested = np.arange(1, config.num_anchors + 1, dtype=np.float64) * metric.spatial_step_m
    target = np.zeros((config.num_anchors, 12), dtype=np.float32)
    is_pad = np.ones(config.num_anchors, dtype=bool)
    progress = np.zeros(config.num_anchors, dtype=np.float64)
    source_indices = np.zeros(config.num_anchors, dtype=np.int64)
    valid = 0
    if attained <= config.static_epsilon_m:
        target[0, 3] = actions[0, 4]
        target[0, 4:10] = actions[0, 5:11]
        target[0, 10] = actions[0, 11]
        valid = 1
    else:
        for anchor, position in enumerate(requested):
            if position > attained + 1e-12:
                break
            high = int(np.searchsorted(cumulative, position, side="left"))
            high = min(max(high, 1), len(cumulative) - 1)
            low = high - 1
            alpha = float((position - cumulative[low]) / max(cumulative[high] - cumulative[low], 1e-12))
            target[anchor, :3] = interpolate(local[low], local[high], alpha)
            source_action = _interpolated_target(actions, low, high, alpha)
            target[anchor, 3] = source_action[4]
            target[anchor, 4:10] = source_action[5:11]
            target[anchor, 10] = source_action[11]
            progress[anchor] = position
            source_indices[anchor] = low
            valid += 1
        # Preserve one truthful incomplete terminal anchor before padding.
        if valid < config.num_anchors and attained > (progress[valid - 1] if valid else 0.0) + 1e-12:
            target[valid, :3] = local[-1]
            target[valid, 3] = actions[-1, 4]
            target[valid, 4:10] = actions[-1, 5:11]
            target[valid, 10] = actions[-1, 11]
            progress[valid] = attained
            source_indices[valid] = len(local) - 1
            valid += 1
    is_pad[:valid] = False
    target[:valid, 11] = 1.0
    if valid < config.num_anchors:
        target[valid:] = target[valid - 1]
        target[valid:, 11] = -1.0
        progress[valid:] = progress[valid - 1]
        source_indices[valid:] = source_indices[valid - 1]
    target[:, :3] = absolute_path_to_increments(target[:, :3]).astype(np.float32)
    return RateFreeLabel(target, is_pad, progress, source_indices, attained, reached)


def build_rate_free_label(
    data: NavigateData,
    index: int,
    metric: RateFreeMetric,
    config: RateFreeConfig = RateFreeConfig(),
) -> RateFreeLabel:
    episode = int(data.episode_indices[index])
    _, episode_end = data.episode_bounds[episode]
    # V2 walks to the requested physical extent or episode end.  There is no
    # frame-count or elapsed-time cutoff in label construction.
    end = episode_end
    return build_rate_free_from_arrays(
        data.timestamps[index:end],
        measured_base_poses(data.states[index:end]),
        data.actions[index:end],
        metric,
        RateFreeConfig(metric.num_anchors, config.static_epsilon_m),
    )
