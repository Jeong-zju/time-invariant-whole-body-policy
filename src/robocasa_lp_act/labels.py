"""Base-only LP-ACT path-time labels from measured global SE(2) poses."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lp_act.se2 import between, interpolate, log

from .data import RoboCasaData
from .schema import CONTROL_DT, LP_ACTION_DIM, NUM_QUERIES


def _active_quantile(values: np.ndarray, quantile: float, floor: float) -> float:
    values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    active = values[values > floor * 0.1]
    if active.size == 0:
        return floor
    return float(max(np.quantile(active, quantile), floor))


@dataclass(frozen=True)
class MetricConfig:
    xy_scale: float
    theta_scale: float
    path_length: float
    scale_quantile: float = 0.9
    horizon_intervals: int = NUM_QUERIES
    moving_translation_threshold_m: float = 0.01
    moving_yaw_threshold_rad: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "MetricConfig":
        return cls(**value)


@dataclass(frozen=True)
class LabelConfig:
    num_anchors: int = NUM_QUERIES
    max_duration_s: float = 4.0
    static_translation_epsilon_m: float = 0.005
    static_yaw_epsilon_rad: float = 0.005
    static_horizon_s: float = NUM_QUERIES * CONTROL_DT
    nominal_dt: float = CONTROL_DT


@dataclass(frozen=True)
class BasePathLabel:
    base_anchors: np.ndarray
    log_durations: np.ndarray
    anchor_times: np.ndarray
    anchor_path_positions: np.ndarray
    attained_length: float
    start_index: int
    source_end_index: int
    static: bool
    reached_path_limit: bool

    def hybrid_target(self, nonbase_time_actions: np.ndarray) -> np.ndarray:
        nonbase_time_actions = np.asarray(nonbase_time_actions, dtype=np.float64)
        if nonbase_time_actions.shape != (self.base_anchors.shape[0], 9):
            raise ValueError(f"Expected non-base target (K,9), got {nonbase_time_actions.shape}")
        target = np.concatenate(
            [self.base_anchors, self.log_durations[:, None], nonbase_time_actions], axis=-1
        )
        if target.shape[1] != LP_ACTION_DIM:
            raise AssertionError(target.shape)
        return target.astype(np.float32)


def base_metric_increments(poses: np.ndarray, metric: MetricConfig) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    delta = log(between(poses[:-1], poses[1:]))
    return np.sqrt(
        (delta[:, 0] / metric.xy_scale) ** 2
        + (delta[:, 1] / metric.xy_scale) ** 2
        + (delta[:, 2] / metric.theta_scale) ** 2
    )


def fit_metric_config(
    data: RoboCasaData,
    train_episodes: list[int],
    *,
    scale_quantile: float = 0.9,
    horizon_intervals: int = NUM_QUERIES,
    sample_stride: int = 8,
) -> MetricConfig:
    delta_parts: list[np.ndarray] = []
    per_episode: list[tuple[np.ndarray, np.ndarray]] = []
    for episode in train_episodes:
        start, end = data.episode_bounds[episode]
        if end - start < 2:
            continue
        poses = data.base_world_poses[start:end]
        delta = log(between(poses[:-1], poses[1:]))
        delta_parts.append(delta)
        per_episode.append((delta, poses))
    if not delta_parts:
        raise ValueError("No measured base increments found")
    all_delta = np.concatenate(delta_parts)
    xy_scale = _active_quantile(all_delta[:, :2], scale_quantile, 1e-4)
    theta_scale = _active_quantile(all_delta[:, 2], scale_quantile, 1e-4)
    provisional = MetricConfig(xy_scale, theta_scale, 1.0, scale_quantile, horizon_intervals)

    horizon_lengths: list[np.ndarray] = []
    for delta, poses in per_episode:
        increments = np.sqrt(
            (delta[:, 0] / xy_scale) ** 2
            + (delta[:, 1] / xy_scale) ** 2
            + (delta[:, 2] / theta_scale) ** 2
        )
        if increments.size < horizon_intervals:
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(increments)])
        starts = np.arange(0, increments.size - horizon_intervals + 1, sample_stride)
        lengths = cumulative[starts + horizon_intervals] - cumulative[starts]
        net = between(poses[starts], poses[starts + horizon_intervals])
        moving_mask = (
            np.linalg.norm(net[:, :2], axis=1) > provisional.moving_translation_threshold_m
        ) | (np.abs(net[:, 2]) > provisional.moving_yaw_threshold_rad)
        moving = lengths[moving_mask]
        if moving.size:
            horizon_lengths.append(moving)
    if not horizon_lengths:
        raise ValueError("No moving 32-step base windows found")
    path_length = float(np.median(np.concatenate(horizon_lengths)))
    return MetricConfig(
        xy_scale=provisional.xy_scale,
        theta_scale=provisional.theta_scale,
        path_length=path_length,
        scale_quantile=scale_quantile,
        horizon_intervals=horizon_intervals,
        moving_translation_threshold_m=provisional.moving_translation_threshold_m,
        moving_yaw_threshold_rad=provisional.moving_yaw_threshold_rad,
    )


def _interpolate_by_time(times: np.ndarray, poses: np.ndarray, queries: np.ndarray) -> np.ndarray:
    result = np.empty((queries.size, 3), dtype=np.float64)
    for index, query in enumerate(queries):
        high = int(np.searchsorted(times, query, side="right"))
        high = min(max(high, 1), len(times) - 1)
        low = high - 1
        width = times[high] - times[low]
        alpha = 0.0 if width <= 1e-12 else float((query - times[low]) / width)
        result[index] = interpolate(poses[low], poses[high], alpha)
    return result


def build_base_path_label(
    data: RoboCasaData,
    start_index: int,
    metric: MetricConfig,
    config: LabelConfig = LabelConfig(),
) -> BasePathLabel:
    episode = int(data.episode_indices[start_index])
    episode_start, episode_end = data.episode_bounds[episode]
    if not episode_start <= start_index < episode_end:
        raise IndexError(start_index)

    relative_episode_times = data.timestamps[start_index:episode_end] - data.timestamps[start_index]
    count = int(np.searchsorted(relative_episode_times, config.max_duration_s + 1e-9, side="right"))
    count = max(1, count)
    source_end = min(start_index + count, episode_end)
    times = data.timestamps[start_index:source_end] - data.timestamps[start_index]
    poses = between(data.base_world_poses[start_index], data.base_world_poses[start_index:source_end])

    if times.size == 1:
        anchor_times = np.arange(1, config.num_anchors + 1, dtype=np.float64) * config.nominal_dt
        anchors = np.zeros((config.num_anchors, 3), dtype=np.float64)
        path_positions = np.zeros(config.num_anchors, dtype=np.float64)
        static = True
        attained = 0.0
        reached = False
    else:
        increments = base_metric_increments(poses, metric)
        cumulative = np.concatenate([[0.0], np.cumsum(increments)])
        attained = float(min(cumulative[-1], metric.path_length))
        static = bool(
            np.max(np.linalg.norm(poses[:, :2], axis=1)) <= config.static_translation_epsilon_m
            and np.max(np.abs(poses[:, 2])) <= config.static_yaw_epsilon_rad
        )
        reached = bool(cumulative[-1] >= metric.path_length)
        if static:
            horizon = min(config.static_horizon_s, float(times[-1]))
            if horizon <= 0.0:
                horizon = config.nominal_dt * config.num_anchors
                anchors = np.zeros((config.num_anchors, 3), dtype=np.float64)
            else:
                anchor_times = np.linspace(horizon / config.num_anchors, horizon, config.num_anchors)
                anchors = _interpolate_by_time(times, poses, anchor_times)
            if "anchor_times" not in locals():
                anchor_times = np.arange(1, config.num_anchors + 1) * config.nominal_dt
            path_positions = np.zeros(config.num_anchors, dtype=np.float64)
        else:
            path_positions = np.linspace(attained / config.num_anchors, attained, config.num_anchors)
            anchor_times = np.empty(config.num_anchors, dtype=np.float64)
            anchors = np.empty((config.num_anchors, 3), dtype=np.float64)
            for anchor_index, path_position in enumerate(path_positions):
                high = int(np.searchsorted(cumulative, path_position, side="left"))
                high = min(max(high, 1), cumulative.size - 1)
                low = high - 1
                width = cumulative[high] - cumulative[low]
                alpha = 0.0 if width <= 1e-12 else float((path_position - cumulative[low]) / width)
                anchor_times[anchor_index] = (1.0 - alpha) * times[low] + alpha * times[high]
                anchors[anchor_index] = interpolate(poses[low], poses[high], alpha)

    durations = np.diff(np.concatenate([[0.0], anchor_times]))
    durations = np.maximum(durations, 1e-4)
    return BasePathLabel(
        base_anchors=anchors,
        log_durations=np.log(durations),
        anchor_times=anchor_times,
        anchor_path_positions=path_positions,
        attained_length=attained,
        start_index=start_index,
        source_end_index=source_end,
        static=static,
        reached_path_limit=reached,
    )
