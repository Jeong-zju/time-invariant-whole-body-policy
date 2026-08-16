"""Phase 1: timestamped whole-body position targets and execution.

The Phase 1 policy keeps GR00T's ``[H, 32]`` output shape but changes the last
three values from body velocity to a relative SE(2) pose.  Rows 0:28 remain
upper-body joint/hand position targets and row 28 remains base height.  Query
times are a separate, explicit contract; array indices never stand in for time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .se2 import compose, exp, interpolate, log, relative


UPPER_BODY_DIM = 28
BASE_HEIGHT_INDEX = 28
BASE_SE2_SLICE = slice(29, 32)
ACTION_DIM = 32
TargetSource = Literal["measured_odometry", "command_integration_proxy"]


def _finite_array(value: np.ndarray, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_timestamps(timestamps_s: np.ndarray) -> np.ndarray:
    timestamps_s = _finite_array(timestamps_s, "timestamps_s", ndim=1)
    if len(timestamps_s) < 2:
        raise ValueError("at least two timestamps are required")
    intervals = np.diff(timestamps_s)
    if np.any(intervals <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing within one episode")
    return timestamps_s


def integrate_body_twist(
    timestamps_s: np.ndarray,
    body_twist: np.ndarray,
    initial_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate piecewise-constant ``[vx, vy, yaw_rate]`` on SE(2).

    ``body_twist[k]`` is held on ``[timestamps[k], timestamps[k + 1])``.
    The final twist row is retained for shape compatibility but is not used.
    """
    timestamps_s = _validate_timestamps(timestamps_s)
    body_twist = _finite_array(body_twist, "body_twist", ndim=2)
    if body_twist.shape != (len(timestamps_s), 3):
        raise ValueError(
            f"body_twist must have shape {(len(timestamps_s), 3)}, got {body_twist.shape}"
        )
    if initial_pose is None:
        initial_pose = np.zeros(3, dtype=np.float64)
    initial_pose = _finite_array(initial_pose, "initial_pose", ndim=1)
    if initial_pose.shape != (3,):
        raise ValueError(f"initial_pose must have shape (3,), got {initial_pose.shape}")

    poses = np.empty((len(timestamps_s), 3), dtype=np.float64)
    poses[0] = initial_pose
    for index, dt_s in enumerate(np.diff(timestamps_s)):
        poses[index + 1] = compose(poses[index], exp(body_twist[index] * dt_s))
    return poses


def _linear_at(timestamps_s: np.ndarray, values: np.ndarray, query_s: float) -> np.ndarray:
    right = int(np.searchsorted(timestamps_s, query_s, side="left"))
    if right == 0:
        return values[0].copy()
    if right == len(timestamps_s):
        return values[-1].copy()
    if timestamps_s[right] == query_s:
        return values[right].copy()
    left = right - 1
    fraction = (query_s - timestamps_s[left]) / (timestamps_s[right] - timestamps_s[left])
    return values[left] + fraction * (values[right] - values[left])


def _pose_at(timestamps_s: np.ndarray, poses: np.ndarray, query_s: float) -> np.ndarray:
    right = int(np.searchsorted(timestamps_s, query_s, side="left"))
    if right == 0:
        return poses[0].copy()
    if right == len(timestamps_s):
        return poses[-1].copy()
    if timestamps_s[right] == query_s:
        return poses[right].copy()
    left = right - 1
    fraction = (query_s - timestamps_s[left]) / (timestamps_s[right] - timestamps_s[left])
    return interpolate(poses[left], poses[right], fraction)


@dataclass(frozen=True)
class MultiHorizonTargets:
    """A leak-free collection of Phase 1 targets from one episode."""

    query_times_s: np.ndarray
    anchor_indices: np.ndarray
    targets: np.ndarray
    source: TargetSource

    def __post_init__(self) -> None:
        query_times = _finite_array(self.query_times_s, "query_times_s", ndim=1)
        anchors = np.asarray(self.anchor_indices, dtype=np.int64)
        targets = _finite_array(self.targets, "targets", ndim=3)
        if len(query_times) == 0 or np.any(query_times <= 0.0) or np.any(np.diff(query_times) <= 0.0):
            raise ValueError("query_times_s must be positive and strictly increasing")
        expected = (len(anchors), len(query_times), ACTION_DIM)
        if targets.shape != expected:
            raise ValueError(f"targets must have shape {expected}, got {targets.shape}")
        if len(anchors) and (np.any(anchors < 0) or np.any(np.diff(anchors) <= 0)):
            raise ValueError("anchor_indices must be non-negative and strictly increasing")
        if self.source not in ("measured_odometry", "command_integration_proxy"):
            raise ValueError(f"unsupported target source {self.source!r}")

    @property
    def samples(self) -> int:
        return len(self.anchor_indices)


def build_multi_horizon_targets(
    *,
    timestamps_s: np.ndarray,
    upper_body_position: np.ndarray,
    base_height: np.ndarray,
    query_times_s: np.ndarray,
    base_pose_se2: np.ndarray | None = None,
    base_twist_body: np.ndarray | None = None,
    source: TargetSource,
) -> MultiHorizonTargets:
    """Build Phase 1 targets using real seconds and one explicit pose source.

    Measured odometry is the preferred source.  Arena's existing LeRobot data
    does not contain root pose, so ``command_integration_proxy`` is supported as
    an explicitly marked fallback that must pass a replay-telemetry gate before
    it is used for training.
    """
    timestamps_s = _validate_timestamps(timestamps_s)
    upper = _finite_array(upper_body_position, "upper_body_position", ndim=2)
    if upper.shape != (len(timestamps_s), UPPER_BODY_DIM):
        raise ValueError(
            f"upper_body_position must have shape {(len(timestamps_s), UPPER_BODY_DIM)}, got {upper.shape}"
        )
    height = _finite_array(base_height, "base_height")
    if height.shape == (len(timestamps_s),):
        height = height[:, None]
    if height.shape != (len(timestamps_s), 1):
        raise ValueError(f"base_height must have shape {(len(timestamps_s), 1)}, got {height.shape}")
    query_times = _finite_array(query_times_s, "query_times_s", ndim=1)
    if len(query_times) == 0 or np.any(query_times <= 0.0) or np.any(np.diff(query_times) <= 0.0):
        raise ValueError("query_times_s must be positive and strictly increasing")

    if source == "measured_odometry":
        if base_pose_se2 is None or base_twist_body is not None:
            raise ValueError("measured_odometry requires only base_pose_se2")
        base_pose = _finite_array(base_pose_se2, "base_pose_se2", ndim=2)
        if base_pose.shape != (len(timestamps_s), 3):
            raise ValueError(
                f"base_pose_se2 must have shape {(len(timestamps_s), 3)}, got {base_pose.shape}"
            )
    elif source == "command_integration_proxy":
        if base_twist_body is None or base_pose_se2 is not None:
            raise ValueError("command_integration_proxy requires only base_twist_body")
        base_pose = integrate_body_twist(timestamps_s, base_twist_body)
    else:
        raise ValueError(f"unsupported target source {source!r}")

    latest_anchor_time = timestamps_s[-1] - query_times[-1]
    anchor_indices = np.flatnonzero(timestamps_s <= latest_anchor_time + 1e-12).astype(np.int64)
    targets = np.empty((len(anchor_indices), len(query_times), ACTION_DIM), dtype=np.float64)
    for sample_index, anchor_index in enumerate(anchor_indices):
        anchor_time = timestamps_s[anchor_index]
        anchor_pose = base_pose[anchor_index]
        for horizon_index, delta_t_s in enumerate(query_times):
            target_time = float(anchor_time + delta_t_s)
            targets[sample_index, horizon_index, :UPPER_BODY_DIM] = _linear_at(
                timestamps_s, upper, target_time
            )
            targets[sample_index, horizon_index, BASE_HEIGHT_INDEX] = _linear_at(
                timestamps_s, height, target_time
            )[0]
            target_pose = _pose_at(timestamps_s, base_pose, target_time)
            targets[sample_index, horizon_index, BASE_SE2_SLICE] = relative(anchor_pose, target_pose)

    return MultiHorizonTargets(
        query_times_s=query_times,
        anchor_indices=anchor_indices,
        targets=targets,
        source=source,
    )


def compare_base_proxy_to_odometry(
    *,
    timestamps_s: np.ndarray,
    body_twist: np.ndarray,
    measured_base_pose_se2: np.ndarray,
    horizon_s: float,
) -> dict[str, float | int]:
    """Compare local command integration with measured odometry over fixed windows."""
    timestamps_s = _validate_timestamps(timestamps_s)
    measured = _finite_array(measured_base_pose_se2, "measured_base_pose_se2", ndim=2)
    if measured.shape != (len(timestamps_s), 3):
        raise ValueError(f"measured_base_pose_se2 has invalid shape {measured.shape}")
    if horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive")
    proxy = integrate_body_twist(timestamps_s, body_twist, initial_pose=measured[0])
    xy_errors: list[float] = []
    yaw_errors: list[float] = []
    for anchor_index, anchor_time in enumerate(timestamps_s):
        target_time = float(anchor_time + horizon_s)
        if target_time > timestamps_s[-1] + 1e-12:
            break
        proxy_delta = relative(proxy[anchor_index], _pose_at(timestamps_s, proxy, target_time))
        measured_delta = relative(measured[anchor_index], _pose_at(timestamps_s, measured, target_time))
        error = log(relative(measured_delta, proxy_delta))
        xy_errors.append(float(np.linalg.norm(error[:2])))
        yaw_errors.append(float(abs(error[2])))
    if not xy_errors:
        raise ValueError("episode is shorter than horizon_s")
    xy = np.asarray(xy_errors)
    yaw = np.asarray(yaw_errors)
    return {
        "windows": len(xy),
        "horizon_s": float(horizon_s),
        "xy_rms_m": float(np.sqrt(np.mean(xy**2))),
        "xy_p95_m": float(np.quantile(xy, 0.95)),
        "yaw_rms_rad": float(np.sqrt(np.mean(yaw**2))),
        "yaw_p95_rad": float(np.quantile(yaw, 0.95)),
    }


def save_target_batch(path: str | Path, batch: MultiHorizonTargets) -> None:
    """Save a target batch with its provenance in a portable NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        query_times_s=batch.query_times_s,
        anchor_indices=batch.anchor_indices,
        targets=batch.targets.astype(np.float32),
        metadata_json=np.asarray(json.dumps({"source": batch.source, "schema_version": 1})),
    )


def load_target_batch(path: str | Path) -> MultiHorizonTargets:
    """Load and validate an NPZ written by :func:`save_target_batch`."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema_version") != 1:
            raise ValueError(f"unsupported target schema {metadata.get('schema_version')!r}")
        return MultiHorizonTargets(
            query_times_s=np.asarray(data["query_times_s"], dtype=np.float64),
            anchor_indices=np.asarray(data["anchor_indices"], dtype=np.int64),
            targets=np.asarray(data["targets"], dtype=np.float64),
            source=metadata["source"],
        )


@dataclass(frozen=True)
class WholeBodyReference:
    upper_body_position: np.ndarray
    base_height: float
    base_pose_se2_w: np.ndarray
    base_twist_body_ff: np.ndarray
    upper_body_velocity_ff: np.ndarray
    base_height_velocity_ff: float
    clamped_to_horizon: bool


@dataclass(frozen=True)
class WholeBodyPlan:
    """A timestamped short route generated by one Phase 1 policy call."""

    query_times_s: np.ndarray
    targets: np.ndarray
    anchor_base_pose_se2_w: np.ndarray
    anchor_upper_body_position: np.ndarray
    anchor_base_height: float

    def __post_init__(self) -> None:
        query = _finite_array(self.query_times_s, "query_times_s", ndim=1)
        targets = _finite_array(self.targets, "targets", ndim=2)
        anchor_pose = _finite_array(self.anchor_base_pose_se2_w, "anchor_base_pose_se2_w", ndim=1)
        anchor_upper = _finite_array(
            self.anchor_upper_body_position, "anchor_upper_body_position", ndim=1
        )
        if len(query) == 0 or np.any(query <= 0.0) or np.any(np.diff(query) <= 0.0):
            raise ValueError("query_times_s must be positive and strictly increasing")
        if targets.shape != (len(query), ACTION_DIM):
            raise ValueError(f"targets must have shape {(len(query), ACTION_DIM)}, got {targets.shape}")
        if anchor_pose.shape != (3,):
            raise ValueError("anchor_base_pose_se2_w must have shape (3,)")
        if anchor_upper.shape != (UPPER_BODY_DIM,):
            raise ValueError(f"anchor_upper_body_position must have shape ({UPPER_BODY_DIM},)")
        if not np.isfinite(self.anchor_base_height):
            raise ValueError("anchor_base_height must be finite")

    @property
    def horizon_s(self) -> float:
        return float(self.query_times_s[-1])

    def sample(self, elapsed_s: float) -> WholeBodyReference:
        """Query the plan by measured wall-clock time.

        The last reference is held with zero feed-forward after the horizon;
        this is intentionally different from holding a non-zero velocity.
        """
        if not np.isfinite(elapsed_s):
            raise ValueError("elapsed_s must be finite")
        elapsed = max(0.0, float(elapsed_s))
        clamped = elapsed >= self.horizon_s

        times = np.concatenate(([0.0], np.asarray(self.query_times_s, dtype=np.float64)))
        upper = np.vstack((self.anchor_upper_body_position, self.targets[:, :UPPER_BODY_DIM]))
        height = np.concatenate(([self.anchor_base_height], self.targets[:, BASE_HEIGHT_INDEX]))
        base_local = np.vstack((np.zeros(3, dtype=np.float64), self.targets[:, BASE_SE2_SLICE]))

        if elapsed >= times[-1]:
            index = len(times) - 1
            upper_ref = upper[index]
            height_ref = float(height[index])
            local_ref = base_local[index]
            base_twist = np.zeros(3, dtype=np.float64)
            upper_velocity = np.zeros(UPPER_BODY_DIM, dtype=np.float64)
            height_velocity = 0.0
        else:
            right = int(np.searchsorted(times, elapsed, side="right"))
            left = right - 1
            duration = float(times[right] - times[left])
            fraction = (elapsed - times[left]) / duration
            upper_ref = upper[left] + fraction * (upper[right] - upper[left])
            height_ref = float(height[left] + fraction * (height[right] - height[left]))
            local_ref = interpolate(base_local[left], base_local[right], fraction)
            base_twist = log(relative(base_local[left], base_local[right])) / duration
            upper_velocity = (upper[right] - upper[left]) / duration
            height_velocity = float((height[right] - height[left]) / duration)

        return WholeBodyReference(
            upper_body_position=np.asarray(upper_ref, dtype=np.float64),
            base_height=height_ref,
            base_pose_se2_w=compose(self.anchor_base_pose_se2_w, local_ref),
            base_twist_body_ff=np.asarray(base_twist, dtype=np.float64),
            upper_body_velocity_ff=np.asarray(upper_velocity, dtype=np.float64),
            base_height_velocity_ff=height_velocity,
            clamped_to_horizon=clamped,
        )


class Phase1ExecutionAdapter:
    """Convert a Phase 1 plan back to Arena's policy-level 32-D command.

    Arena currently accepts upper-body/base-height positions plus navigation
    twist; it has no arm-velocity feed-forward slot.  The arm derivative is
    therefore exposed in diagnostics but only base feed-forward is executable.
    """

    def __init__(
        self,
        *,
        kp_xy_per_s: float = 1.0,
        kp_yaw_per_s: float = 1.0,
        max_abs_base_twist: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        if kp_xy_per_s < 0.0 or kp_yaw_per_s < 0.0:
            raise ValueError("tracker gains must be non-negative")
        limits = np.asarray(max_abs_base_twist, dtype=np.float64)
        if limits.shape != (3,) or np.any(limits <= 0.0):
            raise ValueError("max_abs_base_twist must contain three positive values")
        self.kp_xy_per_s = float(kp_xy_per_s)
        self.kp_yaw_per_s = float(kp_yaw_per_s)
        self.max_abs_base_twist = limits

    def command(
        self,
        plan: WholeBodyPlan,
        *,
        elapsed_s: float,
        measured_base_pose_se2_w: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        reference = plan.sample(elapsed_s)
        measured = _finite_array(measured_base_pose_se2_w, "measured_base_pose_se2_w", ndim=1)
        if measured.shape != (3,):
            raise ValueError("measured_base_pose_se2_w must have shape (3,)")
        pose_error = log(relative(measured, reference.base_pose_se2_w))
        # The path derivative is expressed in the reference body's frame while
        # Arena's navigate command is interpreted in the measured body's frame.
        # Rotate feed-forward before adding feedback; the two frames coincide
        # only when yaw tracking error is zero.
        yaw_reference_in_measured = float(relative(measured, reference.base_pose_se2_w)[2])
        c = np.cos(yaw_reference_in_measured)
        s = np.sin(yaw_reference_in_measured)
        navigate = reference.base_twist_body_ff.copy()
        navigate[:2] = np.asarray(
            [c * navigate[0] - s * navigate[1], s * navigate[0] + c * navigate[1]],
            dtype=np.float64,
        )
        navigate[:2] += self.kp_xy_per_s * pose_error[:2]
        navigate[2] += self.kp_yaw_per_s * pose_error[2]
        navigate = np.clip(navigate, -self.max_abs_base_twist, self.max_abs_base_twist)

        command = np.empty(ACTION_DIM, dtype=np.float64)
        command[:UPPER_BODY_DIM] = reference.upper_body_position
        command[BASE_HEIGHT_INDEX] = reference.base_height
        command[BASE_SE2_SLICE] = navigate
        diagnostic = {
            "position_error_m": float(np.linalg.norm(pose_error[:2])),
            "yaw_error_rad": float(abs(pose_error[2])),
            "upper_velocity_ff_norm": float(np.linalg.norm(reference.upper_body_velocity_ff)),
            "clamped_to_horizon": reference.clamped_to_horizon,
        }
        return command, diagnostic
