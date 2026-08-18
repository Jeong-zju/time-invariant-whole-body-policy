"""Measured whole-body Path-Time labels for the matched B2 pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    interpolate_base,
    local_base_pose,
    normalize_quaternion,
    quaternion_between,
    quaternion_slerp,
    quaternion_to_rotvec,
)


@dataclass(frozen=True)
class LabelConfig:
    num_segments: int = 32
    base_translation_scale: float = 0.05
    base_yaw_scale: float = 0.15
    eef_translation_scale: float = 0.02
    eef_rotation_scale: float = 0.15
    min_duration: float = 1e-4
    static_epsilon: float = 1e-10

    def __post_init__(self) -> None:
        values = (
            self.num_segments,
            self.base_translation_scale,
            self.base_yaw_scale,
            self.eef_translation_scale,
            self.eef_rotation_scale,
            self.min_duration,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all label configuration values must be positive")


@dataclass(frozen=True)
class PathTimeLabel:
    base_local: np.ndarray
    eef_position_delta: np.ndarray
    eef_rotation_delta: np.ndarray
    gripper: np.ndarray
    control_mode: np.ndarray
    log_durations: np.ndarray
    arrival_times: np.ndarray
    source_indices: np.ndarray
    event_times: np.ndarray

    @property
    def durations(self) -> np.ndarray:
        return np.exp(self.log_durations)

    def action_dict(self) -> dict[str, np.ndarray]:
        return {
            "end_effector_position": self.eef_position_delta.astype(np.float32),
            "end_effector_rotation": self.eef_rotation_delta.astype(np.float32),
            "gripper_close": self.gripper[:, None].astype(np.float32),
            "base_motion": np.column_stack([self.base_local, self.log_durations]).astype(
                np.float32
            ),
            "control_mode": self.control_mode[:, None].astype(np.float32),
        }


def _validate_inputs(
    states: np.ndarray, actions: np.ndarray, timestamps: np.ndarray, config: LabelConfig
) -> None:
    if states.shape != (config.num_segments + 1, 16):
        raise ValueError(f"states must have shape ({config.num_segments + 1}, 16)")
    if actions.shape != (config.num_segments, 12):
        raise ValueError(f"actions must have shape ({config.num_segments}, 12)")
    if timestamps.shape != (config.num_segments + 1,):
        raise ValueError(f"timestamps must have shape ({config.num_segments + 1},)")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("state/action contains non-finite values")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps must be strictly increasing")


def metric_increments(states: np.ndarray, config: LabelConfig) -> np.ndarray:
    """Dimensionless adjacent whole-body distance from measured states."""
    base = local_base_pose(states[:, 0:3], states[:, 3:7])
    base_dxy = np.diff(base[:, :2], axis=0)
    base_dyaw = np.diff(np.unwrap(base[:, 2]))
    eef_dpos = np.diff(states[:, 7:10], axis=0)
    eef_q = normalize_quaternion(states[:, 10:14])
    eef_drot = np.vstack(
        [quaternion_to_rotvec(quaternion_between(eef_q[i], eef_q[i + 1])) for i in range(len(eef_q) - 1)]
    )
    return np.sqrt(
        np.sum((base_dxy / config.base_translation_scale) ** 2, axis=1)
        + (base_dyaw / config.base_yaw_scale) ** 2
        + np.sum((eef_dpos / config.eef_translation_scale) ** 2, axis=1)
        + np.sum((eef_drot / config.eef_rotation_scale) ** 2, axis=1)
    )


def _sample_continuous(
    values: np.ndarray,
    quaternions: np.ndarray,
    sigma: np.ndarray,
    timestamps: np.ndarray,
    target_sigma: float,
    target_time_if_static: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if sigma[-1] <= 1e-12:
        hi = int(np.searchsorted(timestamps, target_time_if_static, side="right"))
        hi = min(max(hi, 1), len(timestamps) - 1)
        lo = hi - 1
        alpha = float(
            (target_time_if_static - timestamps[lo])
            / max(timestamps[hi] - timestamps[lo], 1e-12)
        )
    else:
        hi = int(np.searchsorted(sigma, target_sigma, side="right"))
        hi = min(max(hi, 1), len(sigma) - 1)
        lo = hi - 1
        while hi < len(sigma) and sigma[hi] - sigma[lo] <= 1e-12:
            hi += 1
        if hi >= len(sigma):
            hi, lo, alpha = len(sigma) - 1, len(sigma) - 2, 1.0
        else:
            alpha = float((target_sigma - sigma[lo]) / (sigma[hi] - sigma[lo]))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    value = (1.0 - alpha) * values[lo] + alpha * values[hi]
    quat = quaternion_slerp(quaternions[lo], quaternions[hi], alpha)
    arrival = (1.0 - alpha) * timestamps[lo] + alpha * timestamps[hi]
    return value, quat, float(arrival)


def _insert_event_knots(
    arrival: np.ndarray, action_times: np.ndarray, gripper: np.ndarray, control_mode: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    changed = np.flatnonzero(
        (np.abs(np.diff(gripper)) > 1e-8) | (np.abs(np.diff(control_mode)) > 1e-8)
    ) + 1
    event_times = action_times[changed]
    result = arrival.copy()
    locked: set[int] = {len(result) - 1}
    for event_time in event_times:
        if np.any(np.isclose(result, event_time, atol=1e-9, rtol=0.0)):
            locked.add(int(np.argmin(np.abs(result - event_time))))
            continue
        candidates = [i for i in range(len(result) - 1) if i not in locked]
        if not candidates:
            raise ValueError("more event knots than available path anchors")
        index = min(candidates, key=lambda i: abs(result[i] - event_time))
        result[index] = event_time
        locked.add(index)
    result.sort()
    if np.any(np.diff(np.concatenate([[action_times[0]], result])) <= 0.0):
        raise ValueError("event insertion produced non-increasing arrival times")
    return result, event_times


def _interpolate_at_time(
    values: np.ndarray, quaternions: np.ndarray, timestamps: np.ndarray, target: float
) -> tuple[np.ndarray, np.ndarray]:
    hi = int(np.searchsorted(timestamps, target, side="right"))
    hi = min(max(hi, 1), len(timestamps) - 1)
    lo = hi - 1
    alpha = float((target - timestamps[lo]) / max(timestamps[hi] - timestamps[lo], 1e-12))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * values[lo] + alpha * values[hi], quaternion_slerp(
        quaternions[lo], quaternions[hi], alpha
    )


def build_path_time_label(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    config: LabelConfig = LabelConfig(),
) -> PathTimeLabel:
    """Encode one 1.6 s measured trajectory; never integrate commanded base motion."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    _validate_inputs(states, actions, timestamps, config)

    base = local_base_pose(states[:, 0:3], states[:, 3:7])
    eef_position = states[:, 7:10]
    eef_quaternion = normalize_quaternion(states[:, 10:14])
    increments = metric_increments(states, config)
    sigma = np.concatenate([[0.0], np.cumsum(increments)])
    fractions = np.arange(1, config.num_segments + 1, dtype=np.float64) / config.num_segments

    default_arrival = []
    if sigma[-1] <= config.static_epsilon:
        default_arrival = timestamps[0] + fractions * (timestamps[-1] - timestamps[0])
    else:
        for fraction in fractions:
            _, _, time = _sample_continuous(
                eef_position,
                eef_quaternion,
                sigma,
                timestamps,
                fraction * sigma[-1],
                timestamps[0] + fraction * (timestamps[-1] - timestamps[0]),
            )
            default_arrival.append(time)
        default_arrival = np.asarray(default_arrival)

    # Raw action schema: base[0:4], control[4], EEF[5:11], gripper[11].
    gripper = actions[:, 11]
    control_mode = actions[:, 4]
    arrival, event_times = _insert_event_knots(
        np.asarray(default_arrival), timestamps[:-1], gripper, control_mode
    )

    base_anchors = []
    eef_pos_anchors = []
    eef_q_anchors = []
    segment_starts = np.concatenate([[timestamps[0]], arrival[:-1]])
    for target in arrival:
        base_value, _ = _interpolate_at_time(
            base, np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(base), 1)), timestamps, target
        )
        eef_value, eef_q_value = _interpolate_at_time(
            eef_position, eef_quaternion, timestamps, target
        )
        base_anchors.append(base_value)
        eef_pos_anchors.append(eef_value)
        eef_q_anchors.append(eef_q_value)
    source_indices = np.asarray(
        [
            int(
                np.clip(
                    np.searchsorted(timestamps[:-1], start, side="right") - 1,
                    0,
                    len(actions) - 1,
                )
            )
            for start in segment_starts
        ],
        dtype=np.int64,
    )

    base_anchors = np.asarray(base_anchors)
    eef_pos_anchors = np.asarray(eef_pos_anchors)
    eef_q_anchors = np.asarray(eef_q_anchors)
    eef_rotation_delta = np.vstack(
        [quaternion_to_rotvec(quaternion_between(eef_quaternion[0], q)) for q in eef_q_anchors]
    )
    durations = np.diff(np.concatenate([[timestamps[0]], arrival]))
    durations = np.maximum(durations, config.min_duration)

    return PathTimeLabel(
        base_local=base_anchors.astype(np.float32),
        eef_position_delta=(eef_pos_anchors - eef_position[0]).astype(np.float32),
        eef_rotation_delta=eef_rotation_delta.astype(np.float32),
        gripper=gripper[source_indices].astype(np.float32),
        control_mode=control_mode[source_indices].astype(np.float32),
        log_durations=np.log(durations).astype(np.float32),
        arrival_times=(arrival - timestamps[0]).astype(np.float32),
        source_indices=source_indices,
        event_times=(event_times - timestamps[0]).astype(np.float32),
    )


def build_pose_time_label(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    config: LabelConfig = LabelConfig(),
) -> PathTimeLabel:
    """Encode frame-aligned measured poses while preserving the original time grid.

    B1 differs from B2 only in how the continuous anchors are selected: every
    measured state at ``t1...tK`` becomes one anchor. No geometry-progress
    resampling is performed, so an intermediate A-B-A state cannot disappear.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    _validate_inputs(states, actions, timestamps, config)

    base = local_base_pose(states[:, 0:3], states[:, 3:7])
    eef_position = states[:, 7:10]
    eef_quaternion = normalize_quaternion(states[:, 10:14])
    durations = np.diff(timestamps)
    changed = np.flatnonzero(
        (np.abs(np.diff(actions[:, 11])) > 1e-8)
        | (np.abs(np.diff(actions[:, 4])) > 1e-8)
    ) + 1
    eef_rotation_delta = np.vstack(
        [
            quaternion_to_rotvec(quaternion_between(eef_quaternion[0], quaternion))
            for quaternion in eef_quaternion[1:]
        ]
    )
    return PathTimeLabel(
        base_local=base[1:].astype(np.float32),
        eef_position_delta=(eef_position[1:] - eef_position[0]).astype(np.float32),
        eef_rotation_delta=eef_rotation_delta.astype(np.float32),
        gripper=actions[:, 11].astype(np.float32),
        control_mode=actions[:, 4].astype(np.float32),
        log_durations=np.log(np.maximum(durations, config.min_duration)).astype(
            np.float32
        ),
        arrival_times=(timestamps[1:] - timestamps[0]).astype(np.float32),
        source_indices=np.arange(config.num_segments, dtype=np.int64),
        event_times=(timestamps[changed] - timestamps[0]).astype(np.float32),
    )


def event_signature(values: np.ndarray) -> tuple[float, ...]:
    values = np.asarray(values).reshape(-1)
    keep = np.concatenate([[True], np.abs(np.diff(values)) > 1e-8])
    return tuple(float(value) for value in values[keep])
