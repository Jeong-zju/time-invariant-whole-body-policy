"""LP-ACT V1 whole-body metric fitting and path-time label generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .parquet_data import TurningData
from .r1pro import (
    BASE_QVEL_STATE,
    CONTINUOUS_ACTION_INDICES,
    NONBASE_ACTION,
    NONBASE_CONTINUOUS_INDICES,
    NONBASE_GRIPPER_INDICES,
    current_nonbase_target,
)
from .se2 import compose, exp, integrate_body_velocity, interpolate


def _robust_scale(values: np.ndarray, quantile: float, floor: float, axis: int | None = None) -> np.ndarray:
    scale = np.quantile(np.abs(values), quantile, axis=axis)
    return np.maximum(scale, floor)


@dataclass(frozen=True)
class MetricConfig:
    xy_scale: float
    theta_scale: float
    joint_scales: tuple[float, ...]
    joint_weight: float
    path_length: float
    scale_quantile: float = 0.9
    horizon_intervals: int = 32

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "MetricConfig":
        value = dict(value)
        value["joint_scales"] = tuple(value["joint_scales"])
        return cls(**value)


@dataclass(frozen=True)
class LabelConfig:
    num_anchors: int = 32
    max_duration_s: float = 4.0
    static_epsilon: float = 1e-4
    extent_mode: str = "attained"
    nominal_dt: float = 1.0 / 30.0

    def __post_init__(self) -> None:
        if self.num_anchors <= 0:
            raise ValueError("num_anchors must be positive")
        if self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive")
        if self.extent_mode not in {"attained", "fixed"}:
            raise ValueError("extent_mode must be 'attained' or 'fixed'")


@dataclass(frozen=True)
class PathLabel:
    target: np.ndarray
    is_pad: np.ndarray
    anchor_times: np.ndarray
    anchor_path_positions: np.ndarray
    start_index: int
    end_index: int
    attained_length: float
    attained_fraction: float
    static: bool
    reached_path_limit: bool
    raw_times: np.ndarray
    raw_base_poses: np.ndarray
    raw_nonbase: np.ndarray

    @property
    def valid_count(self) -> int:
        return int((~self.is_pad).sum())


def metric_increments(
    base_twist_displacements: np.ndarray,
    joint_deltas: np.ndarray,
    metric: MetricConfig,
) -> np.ndarray:
    base_twist_displacements = np.asarray(base_twist_displacements, dtype=np.float64)
    joint_deltas = np.asarray(joint_deltas, dtype=np.float64)
    if base_twist_displacements.shape[-1] != 3:
        raise ValueError("base_twist_displacements must end in dimension 3")
    if joint_deltas.shape[-1] != len(metric.joint_scales):
        raise ValueError("joint_deltas width does not match metric.joint_scales")
    base_term = (
        (base_twist_displacements[..., 0] / metric.xy_scale) ** 2
        + (base_twist_displacements[..., 1] / metric.xy_scale) ** 2
        + (base_twist_displacements[..., 2] / metric.theta_scale) ** 2
    )
    joint_scales = np.asarray(metric.joint_scales, dtype=np.float64)
    joint_term = metric.joint_weight * np.sum((joint_deltas / joint_scales) ** 2, axis=-1)
    return np.sqrt(base_term + joint_term)


def fit_metric_config(
    data: TurningData,
    train_episodes: list[int],
    *,
    scale_quantile: float = 0.9,
    horizon_intervals: int = 32,
    sample_stride: int = 8,
) -> MetricConfig:
    """Fit robust per-step scales and median 32-interval path coverage on train episodes only."""

    base_parts: list[np.ndarray] = []
    joint_parts: list[np.ndarray] = []
    per_episode_raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode in train_episodes:
        start, end = data.episode_bounds[episode]
        if end - start < 2:
            continue
        dt = np.diff(data.timestamps[start:end])
        valid = np.isfinite(dt) & (dt > 0.0)
        # LeRobotDataWrapper stores obs[t] with the action subsequently applied
        # at t. The resulting measured velocity is therefore in obs[t+1].
        base = data.states[start + 1 : end, BASE_QVEL_STATE] * dt[:, None]
        joints = np.diff(data.actions[start:end, CONTINUOUS_ACTION_INDICES], axis=0)
        base = base[valid]
        joints = joints[valid]
        base_parts.append(base)
        joint_parts.append(joints)
        per_episode_raw[episode] = (base, joints)

    if not base_parts:
        raise ValueError("No valid training increments were found")
    base_all = np.concatenate(base_parts)
    joint_all = np.concatenate(joint_parts)
    xy_scale = float(_robust_scale(base_all[:, :2].reshape(-1), scale_quantile, 1e-4))
    theta_scale = float(_robust_scale(base_all[:, 2], scale_quantile, 1e-4))
    joint_scales = _robust_scale(joint_all, scale_quantile, 1e-4, axis=0)
    joint_weight = 1.0 / joint_all.shape[1]

    provisional = MetricConfig(
        xy_scale=xy_scale,
        theta_scale=theta_scale,
        joint_scales=tuple(float(value) for value in joint_scales),
        joint_weight=joint_weight,
        path_length=1.0,
        scale_quantile=scale_quantile,
        horizon_intervals=horizon_intervals,
    )
    horizon_lengths: list[np.ndarray] = []
    for base, joints in per_episode_raw.values():
        increments = metric_increments(base, joints, provisional)
        if increments.size < horizon_intervals:
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(increments)])
        starts = np.arange(0, increments.size - horizon_intervals + 1, sample_stride)
        horizon_lengths.append(cumulative[starts + horizon_intervals] - cumulative[starts])
    if not horizon_lengths:
        raise ValueError("No full standard-ACT horizon was available to estimate L_sigma")
    path_length = float(np.median(np.concatenate(horizon_lengths)))
    if not np.isfinite(path_length) or path_length <= 0.0:
        raise ValueError(f"Invalid fitted path length: {path_length}")
    return MetricConfig(
        xy_scale=xy_scale,
        theta_scale=theta_scale,
        joint_scales=tuple(float(value) for value in joint_scales),
        joint_weight=joint_weight,
        path_length=path_length,
        scale_quantile=scale_quantile,
        horizon_intervals=horizon_intervals,
    )


def _interpolate_nonbase(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    result = start.copy()
    result[NONBASE_CONTINUOUS_INDICES] = (
        (1.0 - alpha) * start[NONBASE_CONTINUOUS_INDICES]
        + alpha * end[NONBASE_CONTINUOUS_INDICES]
    )
    # Zero-order hold for discrete/continuous controller gripper commands.
    result[NONBASE_GRIPPER_INDICES] = start[NONBASE_GRIPPER_INDICES]
    return result


def build_path_label(
    data: TurningData,
    start_index: int,
    metric: MetricConfig,
    config: LabelConfig,
) -> PathLabel:
    episode = int(data.episode_indices[start_index])
    episode_start, episode_end = data.episode_bounds[episode]
    if not (episode_start <= start_index < episode_end):
        raise IndexError(f"Index {start_index} is outside episode {episode}")

    episode_timestamps = data.timestamps[start_index:episode_end]
    local_max_end = int(
        np.searchsorted(
            episode_timestamps,
            data.timestamps[start_index] + config.max_duration_s + 1e-9,
            side="right",
        )
    )
    max_end = min(start_index + max(local_max_end - 1, 0), episode_end - 1)
    if max_end == start_index and start_index + 1 < episode_end:
        max_end += 1

    raw_times_full = data.timestamps[start_index : max_end + 1] - data.timestamps[start_index]
    dt_full = np.diff(raw_times_full)
    if np.any(dt_full <= 0.0):
        raise ValueError(f"Non-positive dt in episode {episode} at index {start_index}")

    raw_nonbase_full = data.actions[start_index : max_end + 1, NONBASE_ACTION].astype(np.float64).copy()
    raw_nonbase_full[0] = current_nonbase_target(data.states[start_index], data.actions[start_index])
    raw_base_full = integrate_body_velocity(
        data.states[start_index + 1 : max_end + 1, BASE_QVEL_STATE],
        dt_full,
    )
    base_displacements = data.states[start_index + 1 : max_end + 1, BASE_QVEL_STATE] * dt_full[:, None]
    joint_deltas = np.diff(raw_nonbase_full[:, NONBASE_CONTINUOUS_INDICES], axis=0)
    increments = metric_increments(base_displacements, joint_deltas, metric)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])

    reached = np.flatnonzero(cumulative >= metric.path_length)
    if reached.size:
        local_end = max(1, int(reached[0]))
        reached_path_limit = True
    else:
        local_end = len(raw_times_full) - 1
        reached_path_limit = False
    local_end = max(1, local_end) if len(raw_times_full) > 1 else 0

    raw_times = raw_times_full[: local_end + 1]
    raw_base = raw_base_full[: local_end + 1]
    raw_nonbase = raw_nonbase_full[: local_end + 1]
    cumulative = cumulative[: local_end + 1]

    # In fixed-extent mode the supervised trajectory ends exactly at L_sigma,
    # not at the end of the raw control interval that first overshoots it.  The
    # measured base velocity is constant over that interval, so SE(2)
    # interpolation gives the physically consistent fractional endpoint.
    if config.extent_mode == "fixed" and reached_path_limit:
        high = len(cumulative) - 1
        low = high - 1
        denominator = cumulative[high] - cumulative[low]
        alpha = 0.0 if denominator <= 1e-12 else float(
            (metric.path_length - cumulative[low]) / denominator
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        raw_times[-1] = (1.0 - alpha) * raw_times[low] + alpha * raw_times[high]
        raw_base[-1] = interpolate(raw_base[low], raw_base[high], alpha)
        raw_nonbase[-1] = _interpolate_nonbase(raw_nonbase[low], raw_nonbase[high], alpha)
        cumulative[-1] = metric.path_length

    attained = float(cumulative[-1])
    static = attained < config.static_epsilon

    target = np.zeros((config.num_anchors, 24), dtype=np.float32)
    is_pad = np.zeros(config.num_anchors, dtype=bool)
    anchor_times = np.zeros(config.num_anchors, dtype=np.float64)
    anchor_path_positions = np.zeros(config.num_anchors, dtype=np.float64)

    if static:
        available_duration = float(raw_times[-1]) if raw_times.size > 1 else config.nominal_dt * config.num_anchors
        segment_duration = max(available_duration / config.num_anchors, 1e-6)
        target[:, :3] = raw_base[-1]
        target[:, 3:23] = raw_nonbase[-1]
        target[:, 23] = np.log(segment_duration)
        anchor_times = np.arange(1, config.num_anchors + 1, dtype=np.float64) * segment_duration
    else:
        extent = attained if config.extent_mode == "attained" else metric.path_length
        requested = np.arange(1, config.num_anchors + 1, dtype=np.float64) / config.num_anchors * extent
        previous_tau = 0.0
        valid_count = 0
        terminal_inserted = False
        for anchor, path_position in enumerate(requested):
            if path_position > attained + 1e-12:
                # Preserve the incomplete final segment before padding.  This
                # makes a short terminal window reconstructible without
                # pretending that it contains 32 equally spaced valid anchors.
                if config.extent_mode == "fixed" and not terminal_inserted:
                    target[anchor, :3] = raw_base[-1]
                    target[anchor, 3:23] = raw_nonbase[-1]
                    tau = float(raw_times[-1])
                    path_position = attained
                    previous_tau = tau
                    valid_count += 1
                    terminal_inserted = True
                else:
                    is_pad[anchor] = True
                    target[anchor, :3] = raw_base[-1]
                    target[anchor, 3:23] = raw_nonbase[-1]
                    tau = previous_tau + config.nominal_dt
            else:
                high = int(np.searchsorted(cumulative, path_position, side="right"))
                high = min(max(high, 1), len(cumulative) - 1)
                low = high - 1
                denominator = cumulative[high] - cumulative[low]
                alpha = 0.0 if denominator <= 1e-12 else float((path_position - cumulative[low]) / denominator)
                target[anchor, :3] = interpolate(raw_base[low], raw_base[high], alpha)
                target[anchor, 3:23] = _interpolate_nonbase(raw_nonbase[low], raw_nonbase[high], alpha)
                tau = float((1.0 - alpha) * raw_times[low] + alpha * raw_times[high])
                previous_tau = tau
                valid_count += 1
            segment_duration = max(tau - anchor_times[anchor - 1] if anchor > 0 else tau, 1e-6)
            target[anchor, 23] = np.log(segment_duration)
            anchor_times[anchor] = tau
            anchor_path_positions[anchor] = min(path_position, attained)

        if valid_count == 0:
            raise RuntimeError("Non-static path label unexpectedly has no valid anchors")

    return PathLabel(
        target=target,
        is_pad=is_pad,
        anchor_times=anchor_times,
        anchor_path_positions=anchor_path_positions,
        start_index=start_index,
        end_index=start_index + local_end,
        attained_length=attained,
        attained_fraction=attained / metric.path_length,
        static=static,
        reached_path_limit=reached_path_limit,
        raw_times=raw_times,
        raw_base_poses=raw_base,
        raw_nonbase=raw_nonbase,
    )
