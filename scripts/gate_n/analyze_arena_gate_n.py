#!/usr/bin/env python3
"""Offline Gate N diagnostics for the Arena G1 demonstrations.

The script deliberately separates three questions:

1. Does the frozen dataset actually contain timing variation?
2. When two cross-episode observations look physically and visually similar,
   are their fixed-time future labels inconsistent, and does motion-progress
   alignment reduce that inconsistency?
3. At the same sample budget, what do uniform-time, whole-body-progress, and
   paper-faithful ISR resampling keep or discard?

It does not train B1--B5 and does not claim that kinematic proxies are contact
labels. The JSON report records those limitations explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


FPS = 50.0
DT = 1.0 / FPS
ACTION_HORIZON = 16

LEFT_ARM = slice(15, 22)
LEFT_HAND = slice(22, 29)
RIGHT_ARM = slice(29, 36)
RIGHT_HAND = slice(36, 43)
WAIST = slice(12, 15)


@dataclass
class Episode:
    index: int
    timestamp: np.ndarray
    state: np.ndarray
    joint_action: np.ndarray
    eef_observation: np.ndarray
    eef_action: np.ndarray
    base_height: np.ndarray
    navigate: np.ndarray
    base_pose: np.ndarray
    model_action: np.ndarray
    geometry: np.ndarray
    whole_body_progress: np.ndarray
    isr_local_coordinate: np.ndarray


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def robust_scale(values: np.ndarray, minimum: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    q05, q95 = np.nanquantile(values, [0.05, 0.95], axis=0)
    scale = np.maximum(q95 - q05, minimum)
    return center, scale


def normalize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - center) / scale


def integrate_base(navigate: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
    """Integrate robot-frame [vx, vy, yaw_rate] commands into an SE(2) proxy."""
    pose = np.zeros((len(navigate), 3), dtype=np.float64)
    for i in range(1, len(navigate)):
        dt = float(timestamp[i] - timestamp[i - 1])
        theta = pose[i - 1, 2]
        vx, vy, yaw_rate = navigate[i - 1]
        c, s = math.cos(theta), math.sin(theta)
        pose[i, 0] = pose[i - 1, 0] + (c * vx - s * vy) * dt
        pose[i, 1] = pose[i - 1, 1] + (s * vx + c * vy) * dt
        pose[i, 2] = pose[i - 1, 2] + yaw_rate * dt
    return pose


def model_action(joint_action: np.ndarray, base_height: np.ndarray, navigate: np.ndarray) -> np.ndarray:
    """Return the exact 32-D ordering used by UnitreeG1SimWBCDataConfig."""
    return np.concatenate(
        [
            joint_action[:, LEFT_ARM],
            joint_action[:, RIGHT_ARM],
            joint_action[:, LEFT_HAND],
            joint_action[:, RIGHT_HAND],
            base_height,
            navigate,
        ],
        axis=1,
    )


def geometry_waypoint(
    base_pose: np.ndarray, joint_action: np.ndarray, base_height: np.ndarray
) -> np.ndarray:
    """A common geometric target used to compare time and progress labels."""
    return np.concatenate(
        [
            base_pose[:, :2],
            np.sin(base_pose[:, 2:3]),
            np.cos(base_pose[:, 2:3]),
            joint_action[:, LEFT_ARM],
            joint_action[:, RIGHT_ARM],
            joint_action[:, LEFT_HAND],
            joint_action[:, RIGHT_HAND],
            base_height,
        ],
        axis=1,
    )


def whole_body_step_distance(
    base_pose: np.ndarray, joint_action: np.ndarray, base_height: np.ndarray
) -> np.ndarray:
    """Weighted physical displacement used for the B3 progress coordinate.

    Units are approximately metres: base translation and height are already in
    metres, yaw uses a 0.5 m characteristic radius, arm joints use 0.15 m/rad,
    and hand joints use 0.03 m/rad. These weights are frozen diagnostics, not
    learned claims about exact end-effector distance.
    """
    if len(base_pose) < 2:
        return np.zeros(0, dtype=np.float64)
    dbase = np.diff(base_pose[:, :2], axis=0)
    dyaw = np.diff(base_pose[:, 2])[:, None] * 0.5
    darm = np.concatenate(
        [np.diff(joint_action[:, LEFT_ARM], axis=0), np.diff(joint_action[:, RIGHT_ARM], axis=0)], axis=1
    ) * 0.15
    dhand = np.concatenate(
        [np.diff(joint_action[:, LEFT_HAND], axis=0), np.diff(joint_action[:, RIGHT_HAND], axis=0)], axis=1
    ) * 0.03
    dheight = np.diff(base_height[:, 0])[:, None]
    return np.sqrt(
        np.sum(dbase**2, axis=1)
        + np.sum(dyaw**2, axis=1)
        + np.sum(darm**2, axis=1)
        + np.sum(dhand**2, axis=1)
        + np.sum(dheight**2, axis=1)
    )


def eef_positions(eef_pose: np.ndarray) -> np.ndarray:
    return np.concatenate([eef_pose[:, 0:3], eef_pose[:, 7:10]], axis=1)


def eef_acceleration(eef_pose: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
    positions = eef_positions(eef_pose)
    if len(positions) < 3:
        return np.zeros_like(positions)
    dt = np.maximum(np.diff(timestamp), 1e-9)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    acceleration = np.zeros_like(positions)
    avg_dt = np.maximum((dt[1:] + dt[:-1]) * 0.5, 1e-9)
    acceleration[1:-1] = np.diff(velocity, axis=0) / avg_dt[:, None]
    acceleration[0] = acceleration[1]
    acceleration[-1] = acceleration[-2]
    return acceleration


def isr_local_coordinate(
    eef_pose: np.ndarray, timestamp: np.ndarray, lambda_acc: float = 0.01
) -> np.ndarray:
    """Additive local information coordinate used only for label alignment.

    Paper-faithful ISR sample indices are computed by ``isr_indices`` below.
    This monotone coordinate is the local analogue needed to query a fixed
    number of future labels from arbitrary (not necessarily retained) anchors.
    """
    positions = eef_positions(eef_pose)
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    acc = np.linalg.norm(eef_acceleration(eef_pose, timestamp)[:-1], axis=1)
    increments = displacement + lambda_acc * acc
    return np.concatenate([[0.0], np.cumsum(increments)])


def isr_indices(
    positions: np.ndarray,
    timestamp: np.ndarray,
    target_distance: float = 0.05,
    lambda_vel: float = 1.0,
    lambda_acc: float = 0.01,
) -> np.ndarray:
    """Algorithm 1 from ISR (arXiv:2606.22907), generalized from R3 to R6.

    The source paper uses a single 3-D end effector. Arena G1 is bimanual, so
    the two wrist positions are concatenated without changing the cost.
    """
    n = len(positions)
    if n <= 2:
        return np.arange(n, dtype=np.int64)
    dt = np.maximum(np.diff(timestamp), 1e-9)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    avg_dt = np.maximum((dt[1:] + dt[:-1]) * 0.5, 1e-9)
    acceleration = np.zeros((n - 1, positions.shape[1]), dtype=np.float64)
    if len(velocity) > 1:
        acceleration[1:] = np.diff(velocity, axis=0) / avg_dt[:, None]
        acceleration[0] = acceleration[1]
    acc_prefix = np.concatenate([[0.0], np.cumsum(np.linalg.norm(acceleration, axis=1))])

    cost = np.full(n, np.inf, dtype=np.float64)
    predecessor = np.full(n, -1, dtype=np.int64)
    cost[0] = 0.0
    for i in range(1, n):
        delta = positions[i] - positions[:i]
        d_vel = np.linalg.norm(delta, axis=1)
        d_acc = acc_prefix[i] - acc_prefix[:i]
        candidates = cost[:i] + (lambda_vel * d_vel + lambda_acc * d_acc - target_distance) ** 2
        predecessor[i] = int(np.argmin(candidates))
        cost[i] = candidates[predecessor[i]]

    indices: list[int] = []
    current = n - 1
    while current >= 0:
        indices.append(current)
        current = int(predecessor[current])
    return np.asarray(indices[::-1], dtype=np.int64)


def uniform_time_indices(length: int, stride: int = 3) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    indices = np.arange(0, length, stride, dtype=np.int64)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def equidistant_indices(coordinate: np.ndarray, count: int) -> np.ndarray:
    """Choose exactly ``count`` monotone indices approximately equidistant in a coordinate."""
    n = len(coordinate)
    if n == 0 or count <= 0:
        return np.zeros(0, dtype=np.int64)
    count = min(max(count, 2), n)
    total = float(coordinate[-1] - coordinate[0])
    if total <= 1e-12:
        return np.unique(np.rint(np.linspace(0, n - 1, count)).astype(np.int64))
    targets = np.linspace(coordinate[0], coordinate[-1], count)
    right = np.searchsorted(coordinate, targets, side="left")
    right = np.clip(right, 0, n - 1)
    left = np.maximum(right - 1, 0)
    take_left = np.abs(coordinate[left] - targets) <= np.abs(coordinate[right] - targets)
    indices = np.where(take_left, left, right)
    indices[0], indices[-1] = 0, n - 1
    return np.unique(indices.astype(np.int64))


def sample_by_coordinate(values: np.ndarray, coordinate: np.ndarray, targets: np.ndarray) -> np.ndarray | None:
    """Linearly interpolate values on a nondecreasing coordinate."""
    if len(values) == 0 or targets[-1] > coordinate[-1] + 1e-12:
        return None
    keep = np.concatenate([[True], np.diff(coordinate) > 1e-12])
    x = coordinate[keep]
    y = values[keep]
    if len(x) == 1:
        return np.repeat(y[:1], len(targets), axis=0)
    columns = [np.interp(targets, x, y[:, j]) for j in range(y.shape[1])]
    return np.stack(columns, axis=1)


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "mean": float(np.mean(arr)),
    }


def load_episodes(dataset_root: Path) -> list[Episode]:
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
    episodes: list[Episode] = []
    for path in files:
        table = pq.read_table(path)

        def array(name: str, dtype: Any = np.float64) -> np.ndarray:
            return np.asarray(table[name].to_pylist(), dtype=dtype)

        timestamp = array("timestamp")
        state = array("observation.state", np.float32)
        joint_action = array("action", np.float32)
        eef_observation = array("observation.eef_pose")
        eef_action = array("action.eef_pose")
        base_height = array("teleop.base_height_command", np.float32)
        navigate = array("teleop.navigate_command", np.float32)
        base_pose = integrate_base(navigate, timestamp)
        action32 = model_action(joint_action, base_height, navigate)
        geometry = geometry_waypoint(base_pose, joint_action, base_height)
        step_distance = whole_body_step_distance(base_pose, joint_action, base_height)
        progress = np.concatenate([[0.0], np.cumsum(step_distance)])
        local_info = isr_local_coordinate(eef_action, timestamp)
        episodes.append(
            Episode(
                index=int(path.stem.rsplit("_", 1)[-1]),
                timestamp=timestamp,
                state=state,
                joint_action=joint_action,
                eef_observation=eef_observation,
                eef_action=eef_action,
                base_height=base_height,
                navigate=navigate,
                base_pose=base_pose,
                model_action=action32,
                geometry=geometry,
                whole_body_progress=progress,
                isr_local_coordinate=local_info,
            )
        )
    return episodes


def state_feature(episode: Episode) -> np.ndarray:
    """Physical match feature; excludes future labels and lower-body gait phase."""
    q = episode.eef_observation[:, [3, 4, 5, 6, 10, 11, 12, 13]]
    # Dataset quaternions are wxyz; rotvec avoids q/-q discontinuities.
    quat_xyzw = q[:, [1, 2, 3, 0, 5, 6, 7, 4]].reshape(-1, 2, 4)
    rotvec = np.concatenate(
        [Rotation.from_quat(quat_xyzw[:, side]).as_rotvec() for side in range(2)], axis=1
    )
    controlled_state = np.concatenate(
        [
            episode.state[:, LEFT_ARM],
            episode.state[:, RIGHT_ARM],
            episode.state[:, LEFT_HAND],
            episode.state[:, RIGHT_HAND],
            episode.state[:, WAIST],
        ],
        axis=1,
    )
    return np.concatenate(
        [
            episode.base_pose[:, :2],
            np.sin(episode.base_pose[:, 2:3]),
            np.cos(episode.base_pose[:, 2:3]),
            controlled_state,
            eef_positions(episode.eef_observation),
            rotvec,
            episode.base_height,
        ],
        axis=1,
    )


def image_features(dataset_root: Path, sample_rows: list[tuple[int, int]]) -> np.ndarray:
    """Low-resolution RGB descriptors used only to reject visibly different scenes."""
    from decord import VideoReader, cpu

    by_episode: dict[int, list[tuple[int, int]]] = {}
    for row, (episode, frame) in enumerate(sample_rows):
        by_episode.setdefault(episode, []).append((row, frame))
    result = np.zeros((len(sample_rows), 12 * 16 * 3), dtype=np.float32)
    video_root = dataset_root / "videos" / "chunk-000" / "observation.images.ego_view"
    for episode, entries in sorted(by_episode.items()):
        reader = VideoReader(
            str(video_root / f"episode_{episode:06d}.mp4"), ctx=cpu(0), num_threads=1
        )
        for start in range(0, len(entries), 32):
            batch_entries = entries[start : start + 32]
            frames = reader.get_batch([frame for _, frame in batch_entries]).asnumpy()
            for (row, _), frame in zip(batch_entries, frames, strict=True):
                small = cv2.resize(frame, (16, 12), interpolation=cv2.INTER_AREA)
                result[row] = small.astype(np.float32).reshape(-1) / 255.0
    return result


def cross_episode_candidates(
    features: np.ndarray, rows: list[tuple[int, int]], neighbors: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(features)
    distances, indices = tree.query(features, k=min(neighbors, len(features)))
    pairs: dict[tuple[int, int], float] = {}
    for source in range(len(features)):
        source_episode = rows[source][0]
        accepted = 0
        for distance, target in zip(distances[source, 1:], indices[source, 1:], strict=True):
            target = int(target)
            if rows[target][0] == source_episode:
                continue
            key = (source, target) if source < target else (target, source)
            pairs[key] = min(float(distance), pairs.get(key, float("inf")))
            accepted += 1
            if accepted == 3:
                break
    ordered = sorted(pairs.items())
    return np.asarray([key for key, _ in ordered], dtype=np.int64), np.asarray(
        [distance for _, distance in ordered], dtype=np.float64
    )


def rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def pair_metrics(
    episodes: list[Episode],
    rows: list[tuple[int, int]],
    pairs: np.ndarray,
    state_distance: np.ndarray,
    visual_distance: np.ndarray,
    geometry_center: np.ndarray,
    geometry_scale: np.ndarray,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    progress_step: float,
    info_step: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    progress_offsets = np.arange(1, ACTION_HORIZON + 1, dtype=np.float64) * progress_step
    info_offsets = np.arange(1, ACTION_HORIZON + 1, dtype=np.float64) * info_step
    for pair_index, (left_row, right_row) in enumerate(pairs):
        left_episode_index, left_frame = rows[int(left_row)]
        right_episode_index, right_frame = rows[int(right_row)]
        left_ep, right_ep = episodes[left_episode_index], episodes[right_episode_index]
        if left_frame + ACTION_HORIZON >= len(left_ep.timestamp) or right_frame + ACTION_HORIZON >= len(
            right_ep.timestamp
        ):
            continue
        left_geometry = normalize(left_ep.geometry, geometry_center, geometry_scale)
        right_geometry = normalize(right_ep.geometry, geometry_center, geometry_scale)
        left_action = normalize(left_ep.model_action, action_center, action_scale)
        right_action = normalize(right_ep.model_action, action_center, action_scale)

        time_left = left_geometry[left_frame + 1 : left_frame + 1 + ACTION_HORIZON]
        time_right = right_geometry[right_frame + 1 : right_frame + 1 + ACTION_HORIZON]
        raw_left = left_action[left_frame : left_frame + ACTION_HORIZON]
        raw_right = right_action[right_frame : right_frame + ACTION_HORIZON]

        progress_left = sample_by_coordinate(
            left_geometry,
            left_ep.whole_body_progress,
            left_ep.whole_body_progress[left_frame] + progress_offsets,
        )
        progress_right = sample_by_coordinate(
            right_geometry,
            right_ep.whole_body_progress,
            right_ep.whole_body_progress[right_frame] + progress_offsets,
        )
        info_left = sample_by_coordinate(
            left_geometry,
            left_ep.isr_local_coordinate,
            left_ep.isr_local_coordinate[left_frame] + info_offsets,
        )
        info_right = sample_by_coordinate(
            right_geometry,
            right_ep.isr_local_coordinate,
            right_ep.isr_local_coordinate[right_frame] + info_offsets,
        )
        if progress_left is None or progress_right is None or info_left is None or info_right is None:
            continue

        left_ds = np.diff(left_ep.whole_body_progress)
        right_ds = np.diff(right_ep.whole_body_progress)
        speed_left = float(left_ds[left_frame] / max(left_ep.timestamp[left_frame + 1] - left_ep.timestamp[left_frame], 1e-9))
        speed_right = float(
            right_ds[right_frame] / max(right_ep.timestamp[right_frame + 1] - right_ep.timestamp[right_frame], 1e-9)
        )
        direction_left = progress_left[min(3, len(progress_left) - 1)] - left_geometry[left_frame]
        direction_right = progress_right[min(3, len(progress_right) - 1)] - right_geometry[right_frame]
        denom = np.linalg.norm(direction_left) * np.linalg.norm(direction_right)
        direction_cosine = float(np.dot(direction_left, direction_right) / denom) if denom > 1e-12 else 0.0
        output.append(
            {
                "left_episode": left_episode_index,
                "left_frame": left_frame,
                "right_episode": right_episode_index,
                "right_frame": right_frame,
                "state_distance": float(state_distance[pair_index]),
                "visual_rmse": float(visual_distance[pair_index]),
                "speed_left": speed_left,
                "speed_right": speed_right,
                "speed_ratio": float(max(speed_left, speed_right) / max(min(speed_left, speed_right), 1e-4)),
                "direction_cosine": direction_cosine,
                "raw_action_chunk_rms": rms(raw_left, raw_right),
                "time_waypoint_chunk_rms": rms(time_left, time_right),
                "progress_waypoint_chunk_rms": rms(progress_left, progress_right),
                "isr_coordinate_chunk_rms": rms(info_left, info_right),
            }
        )
    return output


def coverage(indices: np.ndarray, events: np.ndarray, radius: int = 1) -> float | None:
    event_indices = np.flatnonzero(events)
    if len(event_indices) == 0:
        return None
    retained = np.zeros(len(events), dtype=bool)
    retained[indices] = True
    covered = [np.any(retained[max(0, i - radius) : min(len(events), i + radius + 1)]) for i in event_indices]
    return float(np.mean(covered))


def reconstruction_rms(geometry: np.ndarray, indices: np.ndarray, scale: np.ndarray) -> float:
    time = np.arange(len(geometry), dtype=np.float64)
    reconstructed = np.stack(
        [np.interp(time, indices, geometry[indices, j]) for j in range(geometry.shape[1])], axis=1
    )
    return rms(geometry / scale, reconstructed / scale)


def resampling_metrics(
    episodes: list[Episode], geometry_scale: np.ndarray, isr_target: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    all_eef_acc = np.concatenate(
        [np.linalg.norm(eef_acceleration(ep.eef_action, ep.timestamp), axis=1) for ep in episodes]
    )
    acc_threshold = float(np.quantile(all_eef_acc, 0.95))
    nav_yaw = np.concatenate([np.abs(ep.navigate[:, 2]) for ep in episodes])
    nav_speed = np.concatenate([np.linalg.norm(ep.navigate[:, :2], axis=1) for ep in episodes])
    yaw_threshold = float(np.quantile(nav_yaw, 0.95))
    nav_threshold = float(np.quantile(nav_speed, 0.75))

    for ep in episodes:
        length = len(ep.timestamp)
        time_indices = uniform_time_indices(length, 3)
        progress_indices = equidistant_indices(ep.whole_body_progress, len(time_indices))
        faithful_isr = isr_indices(
            eef_positions(ep.eef_action), ep.timestamp, target_distance=0.05, lambda_acc=0.01
        )
        matched_isr = isr_indices(
            eef_positions(ep.eef_action), ep.timestamp, target_distance=isr_target, lambda_acc=0.01
        )
        acceleration = np.linalg.norm(eef_acceleration(ep.eef_action, ep.timestamp), axis=1)
        step_distance = np.diff(ep.whole_body_progress, append=ep.whole_body_progress[-1])
        pause = step_distance < 1e-4
        turning = np.abs(ep.navigate[:, 2]) >= yaw_threshold
        base_motion = np.linalg.norm(ep.navigate[:, :2], axis=1) >= nav_threshold
        critical_acc = acceleration >= acc_threshold
        for method, indices in {
            "time_uniform_3x": time_indices,
            "whole_body_progress": progress_indices,
            "isr_paper_default": faithful_isr,
            "isr_budget_matched": matched_isr,
        }.items():
            selected_steps = np.diff(ep.whole_body_progress[indices])
            selected_info = np.diff(ep.isr_local_coordinate[indices])
            records.append(
                {
                    "episode": ep.index,
                    "method": method,
                    "retained": int(len(indices)),
                    "retention_ratio": float(len(indices) / length),
                    "geometry_reconstruction_rms": reconstruction_rms(ep.geometry, indices, geometry_scale),
                    "motion_step_cv": float(np.std(selected_steps) / max(np.mean(selected_steps), 1e-12)),
                    "information_step_cv": float(np.std(selected_info) / max(np.mean(selected_info), 1e-12)),
                    "pause_retention": float(np.mean(pause[indices])) if len(indices) else None,
                    "high_acceleration_coverage": coverage(indices, critical_acc),
                    "turning_coverage": coverage(indices, turning),
                    "base_motion_coverage": coverage(indices, base_motion),
                }
            )

    summary: dict[str, Any] = {
        "thresholds": {
            "eef_acceleration_p95": acc_threshold,
            "absolute_yaw_command_p95": yaw_threshold,
            "translation_command_p75": nav_threshold,
        },
        "isr": {
            "paper_default_target_distance": 0.05,
            "lambda_velocity": 1.0,
            "lambda_acceleration": 0.01,
            "budget_matched_target_distance": isr_target,
            "g1_extension": "concatenate left and right wrist positions into R6",
        },
        "methods": {},
    }
    for method in sorted({record["method"] for record in records}):
        selected = [record for record in records if record["method"] == method]
        summary["methods"][method] = {
            key: percentile_summary(record[key] for record in selected if record[key] is not None)
            for key in [
                "retention_ratio",
                "geometry_reconstruction_rms",
                "motion_step_cv",
                "information_step_cv",
                "pause_retention",
                "high_acceleration_coverage",
                "turning_coverage",
                "base_motion_coverage",
            ]
        }
    return summary, records


def calibrate_isr_target(episodes: list[Episode], desired_ratio: float = 1.0 / 3.0) -> float:
    """Choose one dataset-global ISR distance giving the target median retention."""
    calibration = episodes[: min(20, len(episodes))]
    low, high = 1e-4, 50.0
    for _ in range(16):
        middle = math.sqrt(low * high)
        ratios = [
            len(isr_indices(eef_positions(ep.eef_action), ep.timestamp, target_distance=middle)) / len(ep.timestamp)
            for ep in calibration
        ]
        if float(np.median(ratios)) > desired_ratio:
            low = middle
        else:
            high = middle
    return float(math.sqrt(low * high))


def save_pairs_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def extract_frame(dataset_root: Path, episode: int, frame: int) -> np.ndarray:
    from decord import VideoReader, cpu

    path = (
        dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / f"episode_{episode:06d}.mp4"
    )
    return VideoReader(str(path), ctx=cpu(0), num_threads=1)[frame].asnumpy()


def save_pair_montage(dataset_root: Path, path: Path, records: list[dict[str, Any]], count: int = 12) -> None:
    if not records:
        return
    tiles: list[np.ndarray] = []
    for record in records[:count]:
        left = extract_frame(dataset_root, record["left_episode"], record["left_frame"])
        right = extract_frame(dataset_root, record["right_episode"], record["right_frame"])
        left = cv2.cvtColor(cv2.resize(left, (320, 240)), cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(cv2.resize(right, (320, 240)), cv2.COLOR_RGB2BGR)
        tile = np.concatenate([left, right], axis=1)
        label = (
            f"e{record['left_episode']} f{record['left_frame']} vs "
            f"e{record['right_episode']} f{record['right_frame']}  "
            f"speed x{record['speed_ratio']:.1f}  "
            f"time {record['time_waypoint_chunk_rms']:.3f} -> "
            f"progress {record['progress_waypoint_chunk_rms']:.3f}"
        )
        cv2.rectangle(tile, (0, 0), (640, 28), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = [np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, len(tiles), 2)]
    if rows[-1].shape[1] < rows[0].shape[1]:
        rows[-1] = np.pad(rows[-1], ((0, 0), (0, rows[0].shape[1] - rows[-1].shape[1]), (0, 0)))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-matched-pairs", type=int, default=3000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.dataset_root)
    if len(episodes) != 100:
        raise RuntimeError(f"Expected 100 episodes, found {len(episodes)}")

    all_timestamps = np.concatenate([ep.timestamp for ep in episodes])
    all_dt = np.concatenate([np.diff(ep.timestamp) for ep in episodes])
    all_geometry = np.concatenate([ep.geometry for ep in episodes])
    all_actions = np.concatenate([ep.model_action for ep in episodes])
    geometry_center, geometry_scale = robust_scale(all_geometry)
    action_center, action_scale = robust_scale(all_actions, minimum=0.01)
    positive_progress_steps = np.concatenate([np.diff(ep.whole_body_progress) for ep in episodes])
    positive_progress_steps = positive_progress_steps[positive_progress_steps > 1e-6]
    positive_info_steps = np.concatenate([np.diff(ep.isr_local_coordinate) for ep in episodes])
    positive_info_steps = positive_info_steps[positive_info_steps > 1e-6]
    progress_step = float(np.median(positive_progress_steps))
    info_step = float(np.median(positive_info_steps))

    sample_rows: list[tuple[int, int]] = []
    state_blocks: list[np.ndarray] = []
    for ep in episodes:
        frames = np.arange(0, len(ep.timestamp) - ACTION_HORIZON, args.sample_stride, dtype=np.int64)
        feature = state_feature(ep)
        state_blocks.append(feature[frames])
        sample_rows.extend((ep.index, int(frame)) for frame in frames)
    state_values = np.concatenate(state_blocks)
    state_center, state_scale = robust_scale(state_values)
    active_dimensions = state_scale > 1e-3
    state_normalized = normalize(state_values[:, active_dimensions], state_center[active_dimensions], state_scale[active_dimensions])
    pairs, state_distances = cross_episode_candidates(state_normalized, sample_rows)

    visual_features = image_features(args.dataset_root, sample_rows)
    visual_distances = np.sqrt(np.mean((visual_features[pairs[:, 0]] - visual_features[pairs[:, 1]]) ** 2, axis=1))
    state_rank = np.argsort(np.argsort(state_distances)) / max(len(state_distances) - 1, 1)
    visual_rank = np.argsort(np.argsort(visual_distances)) / max(len(visual_distances) - 1, 1)
    eligible = (state_distances <= np.quantile(state_distances, 0.5)) & (
        visual_distances <= np.quantile(visual_distances, 0.5)
    )
    combined_score = state_rank + visual_rank
    selected = np.flatnonzero(eligible)
    selected = selected[np.argsort(combined_score[selected])[: args.max_matched_pairs]]
    selected_pairs = pairs[selected]

    matched_records = pair_metrics(
        episodes,
        sample_rows,
        selected_pairs,
        state_distances[selected],
        visual_distances[selected],
        geometry_center,
        geometry_scale,
        action_center,
        action_scale,
        progress_step,
        info_step,
    )
    speed_contrast = [
        record
        for record in matched_records
        if record["speed_ratio"] >= 1.5 and record["direction_cosine"] >= 0.8
    ]
    speed_contrast.sort(
        key=lambda record: (
            record["progress_waypoint_chunk_rms"] / max(record["time_waypoint_chunk_rms"], 1e-9),
            record["state_distance"] + record["visual_rmse"],
        )
    )

    def ambiguity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        time = np.asarray([record["time_waypoint_chunk_rms"] for record in records])
        progress = np.asarray([record["progress_waypoint_chunk_rms"] for record in records])
        info = np.asarray([record["isr_coordinate_chunk_rms"] for record in records])
        return {
            "pairs": int(len(records)),
            "speed_ratio": percentile_summary(record["speed_ratio"] for record in records),
            "raw_action_chunk_rms": percentile_summary(record["raw_action_chunk_rms"] for record in records),
            "time_waypoint_chunk_rms": percentile_summary(time),
            "progress_waypoint_chunk_rms": percentile_summary(progress),
            "isr_coordinate_chunk_rms": percentile_summary(info),
            "progress_median_reduction_vs_time": (
                float(1.0 - np.median(progress) / np.median(time)) if len(records) and np.median(time) > 0 else None
            ),
            "isr_median_reduction_vs_time": (
                float(1.0 - np.median(info) / np.median(time)) if len(records) and np.median(time) > 0 else None
            ),
            "fraction_progress_better_than_time": (
                float(np.mean(progress < time)) if len(records) else None
            ),
        }

    isr_target = calibrate_isr_target(episodes)
    resampling_summary, resampling_records = resampling_metrics(episodes, geometry_scale, isr_target)

    durations = np.asarray([ep.timestamp[-1] - ep.timestamp[0] for ep in episodes])
    episode_lengths = np.asarray([len(ep.timestamp) for ep in episodes])
    report = {
        "schema_version": 1,
        "protocol_id": "arena-g1-gate-n-offline-v0",
        "dataset": {
            "episodes": len(episodes),
            "frames": int(sum(len(ep.timestamp) for ep in episodes)),
            "fps_declared": FPS,
            "dt": {
                "count": int(len(all_dt)),
                "unique_rounded_9dp": np.unique(np.round(all_dt, 9)).tolist(),
                "min": float(np.min(all_dt)),
                "max": float(np.max(all_dt)),
            },
            "episode_frames": percentile_summary(episode_lengths),
            "episode_duration_seconds": percentile_summary(durations),
            "timestamp_start_values": np.unique([ep.timestamp[0] for ep in episodes]).tolist(),
            "timestamp_end_global_max": float(np.max(all_timestamps)),
        },
        "matching": {
            "sample_stride": args.sample_stride,
            "sampled_frames": len(sample_rows),
            "candidate_cross_episode_pairs": int(len(pairs)),
            "selected_pairs_before_future_filter": int(len(selected_pairs)),
            "selected_pairs_after_future_filter": int(len(matched_records)),
            "state_feature_active_dimensions": int(np.sum(active_dimensions)),
            "visual_descriptor": "16x12 RGB area-downsample; used only as a rejection filter",
            "state_distance_selected": percentile_summary(record["state_distance"] for record in matched_records),
            "visual_rmse_selected": percentile_summary(record["visual_rmse"] for record in matched_records),
        },
        "label_alignment": {
            "action_horizon": ACTION_HORIZON,
            "time_horizon_seconds": ACTION_HORIZON * DT,
            "whole_body_progress_step": progress_step,
            "isr_local_information_step": info_step,
            "all_matched_pairs": ambiguity_summary(matched_records),
            "same_direction_speed_contrast_pairs": ambiguity_summary(speed_contrast),
            "speed_contrast_definition": "speed ratio >= 1.5 and 4-progress-step direction cosine >= 0.8",
        },
        "resampling": resampling_summary,
        "limitations": [
            "The parquet files contain no object pose, contact force, or contact-state field; high EEF acceleration is only a kinematic proxy.",
            "Base SE(2) is reconstructed by integrating commanded navigation velocity because measured root pose is absent.",
            "The RGB descriptor rejects visibly different frames but is not a learned semantic scene metric.",
            "Paper-faithful ISR is defined for one 3-D end effector; the bimanual diagnostic concatenates both wrists into R6.",
            "These are offline label diagnostics. They do not substitute for matched B1-B5 training and closed-loop evaluation.",
        ],
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_pairs_csv(args.output_dir / "matched-pairs.csv", matched_records)
    save_pairs_csv(args.output_dir / "speed-contrast-pairs.csv", speed_contrast)
    save_pairs_csv(args.output_dir / "resampling-per-episode.csv", resampling_records)
    save_pair_montage(args.dataset_root, args.output_dir / "speed-contrast-pairs.png", speed_contrast)
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
