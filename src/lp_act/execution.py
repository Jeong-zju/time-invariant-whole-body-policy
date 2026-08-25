"""Decode LP-ACT V1 path-time anchors into fixed-rate R1Pro commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .r1pro import ACTION_DIM, LP_ACTION_DIM, NONBASE_CONTINUOUS_INDICES, NONBASE_GRIPPER_INDICES
from .se2 import between, interpolate, log


@dataclass(frozen=True)
class ExecutedChunk:
    actions: np.ndarray
    boundary_times: np.ndarray
    boundary_base_poses: np.ndarray


def decode_lp_chunk(
    target: np.ndarray,
    origin_nonbase: np.ndarray,
    *,
    control_dt: float = 1.0 / 30.0,
    min_segment_duration: float = 1.0 / 120.0,
    max_segment_duration: float = 1.0,
    max_total_duration: float = 4.0,
) -> ExecutedChunk:
    """Convert one predicted 32x24 LP target into 30 Hz controller actions.

    Base velocity is derived from consecutive SE(2) boundary poses. Absolute
    trunk/arm targets are linearly interpolated, while grippers use zero-order
    hold. Duration clipping is an execution safety guard, not a relabeling step.
    """

    target = np.asarray(target, dtype=np.float64)
    origin_nonbase = np.asarray(origin_nonbase, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != LP_ACTION_DIM:
        raise ValueError(f"Expected target (K, {LP_ACTION_DIM}), got {target.shape}")
    if origin_nonbase.shape != (20,):
        raise ValueError(f"Expected origin_nonbase (20,), got {origin_nonbase.shape}")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")

    durations = np.clip(np.exp(target[:, 23]), min_segment_duration, max_segment_duration)
    anchor_times = np.minimum(np.cumsum(durations), max_total_duration)
    keep = np.r_[True, np.diff(anchor_times) > 1e-9]
    anchor_times = anchor_times[keep]
    anchors = target[keep]
    if anchor_times.size == 0 or anchor_times[-1] <= 0.0:
        raise ValueError("Predicted chunk has no positive execution duration")

    interpolation_times = np.concatenate([[0.0], anchor_times])
    interpolation_base = np.concatenate([np.zeros((1, 3)), anchors[:, :3]], axis=0)
    interpolation_nonbase = np.concatenate([origin_nonbase[None], anchors[:, 3:23]], axis=0)

    step_count = max(1, int(np.ceil(anchor_times[-1] / control_dt)))
    boundary_times = np.arange(step_count + 1, dtype=np.float64) * control_dt
    clipped = np.minimum(boundary_times, anchor_times[-1])
    base = np.zeros((step_count + 1, 3), dtype=np.float64)
    nonbase = np.zeros((step_count, 20), dtype=np.float64)

    for boundary, time in enumerate(clipped):
        high = int(np.searchsorted(interpolation_times, time, side="right"))
        high = min(max(high, 1), len(interpolation_times) - 1)
        low = high - 1
        segment = interpolation_times[high] - interpolation_times[low]
        alpha = 0.0 if segment <= 1e-12 else float((time - interpolation_times[low]) / segment)
        base[boundary] = interpolate(interpolation_base[low], interpolation_base[high], alpha)
        if boundary == 0:
            continue
        command = interpolation_nonbase[low].copy()
        command[NONBASE_CONTINUOUS_INDICES] = (
            (1.0 - alpha) * interpolation_nonbase[low, NONBASE_CONTINUOUS_INDICES]
            + alpha * interpolation_nonbase[high, NONBASE_CONTINUOUS_INDICES]
        )
        command[NONBASE_GRIPPER_INDICES] = interpolation_nonbase[low, NONBASE_GRIPPER_INDICES]
        nonbase[boundary - 1] = command

    actions = np.zeros((step_count, ACTION_DIM), dtype=np.float32)
    actions[:, :3] = (log(between(base[:-1], base[1:])) / control_dt).astype(np.float32)
    actions[:, 3:23] = nonbase.astype(np.float32)
    return ExecutedChunk(actions=actions, boundary_times=boundary_times, boundary_base_poses=base)
