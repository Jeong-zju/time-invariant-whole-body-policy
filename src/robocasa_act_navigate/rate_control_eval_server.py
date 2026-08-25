"""Frozen-plan execution-rate trials for native ACT and geometric Point policies.

This module intentionally separates model output from execution rate.  Each
episode predicts exactly once from the reset observation.  Native ACT retimes a
fixed prefix of its 20 Hz velocity chunk; the geometric policy advances an
external progress pointer over one fixed prefix of its predicted path.  No
model re-prediction or inference-latency mismatch is present in this first gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.data.types import ModalityConfig
from gr00t.policy.server_client import PolicyServer

from .adaptive_path import wrap_angle
from .b2_labels import yaw_from_xyzw
from .b2_se2 import compose, exp
from .eval_server import STATE_SOURCES, VIDEO_SOURCES, NavigateACTPolicy, base_only_to_native
from .frequency_eval_server import interval_average_zoh
from .geometric_eval_server import NavigateGeometricPolicy


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def retimed_zoh_command(
    chunk: np.ndarray,
    real_start_s: float,
    real_end_s: float,
    *,
    rate_scale: float,
    source_hz: float,
    source_duration_s: float,
    compensate_amplitude: bool = True,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Retime velocity as ``u_alpha(t)=alpha*u(alpha*t)``.

    The returned command is the exact average over the real control interval,
    including zero padding after the selected source prefix ends.  Before
    actuator clipping, its time integral equals the source-prefix integral.
    """

    if rate_scale <= 0.0 or real_start_s < 0.0 or real_end_s <= real_start_s:
        raise ValueError("invalid retiming interval or rate scale")
    source_start = rate_scale * real_start_s
    source_end = rate_scale * real_end_s
    exhausted = source_start >= source_duration_s - 1e-12
    if exhausted:
        return np.zeros(np.asarray(chunk).shape[-1], dtype=np.float32), {
            "exhausted": True,
            "source_start_s": float(source_start),
            "source_end_s": float(source_start),
            "unclipped_saturation_fraction": 0.0,
        }
    valid_end = min(source_end, source_duration_s)
    source_average = interval_average_zoh(
        chunk,
        source_start,
        valid_end,
        source_hz=source_hz,
    )
    # Native ACT playback changes only how quickly tokens are consumed, so the
    # command magnitude is unchanged.  The optional oracle also multiplies by
    # ``rate_scale`` and therefore preserves the command integral in an ideal
    # unsaturated velocity plant.  Keeping these two cases explicit prevents
    # us from silently giving the baseline a geometric rate adapter.
    active_fraction = (valid_end - source_start) / (
        rate_scale * (real_end_s - real_start_s)
    )
    amplitude_multiplier = rate_scale if compensate_amplitude else 1.0
    command = source_average * active_fraction * amplitude_multiplier
    saturation = float(np.mean(np.abs(command) > 1.0 + 1e-12))
    return np.clip(command, -1.0, 1.0).astype(np.float32), {
        "exhausted": False,
        "source_start_s": float(source_start),
        "source_end_s": float(valid_end),
        "compensate_amplitude": bool(compensate_amplitude),
        "unclipped_saturation_fraction": saturation,
    }


def geometric_prefix(path: np.ndarray, extent_m: float, yaw_radius_m: float = 0.25) -> np.ndarray:
    """Return a truthful common-origin prefix ending at ``extent_m`` progress."""

    value = np.asarray(path, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) == 0:
        raise ValueError(f"expected path (K,3), got {value.shape}")
    if extent_m <= 0.0 or yaw_radius_m <= 0.0 or not np.isfinite(value).all():
        raise ValueError("invalid path or geometric extent")
    origin = np.zeros((1, 3), dtype=np.float64)
    full = np.concatenate((origin, value), axis=0)
    metric = full.copy()
    metric[:, 2] *= yaw_radius_m
    lengths = np.linalg.norm(np.diff(metric, axis=0), axis=-1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= extent_m + 1e-12:
        return value.astype(np.float32)
    high = int(np.searchsorted(cumulative, extent_m, side="right"))
    low = high - 1
    width = cumulative[high] - cumulative[low]
    alpha = 0.0 if width <= 1e-18 else (extent_m - cumulative[low]) / width
    endpoint = full[low] + alpha * (full[high] - full[low])
    # ``full[1:high]`` contains all complete predicted anchors before endpoint.
    return np.concatenate((full[1:high], endpoint[None]), axis=0).astype(np.float32)


def calibrated_command_path(
    chunk: np.ndarray,
    calibration,
    *,
    source_hz: float = 20.0,
) -> np.ndarray:
    """Convert a normalized command chunk into a calibrated predicted SE(2) path.

    This is deliberately named a *predicted* path.  The calibration estimates
    body twist from command; its integral is not ground-truth physical motion.
    """

    commands = np.asarray(chunk, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError(f"expected command chunk (K,3), got {commands.shape}")
    if source_hz <= 0.0 or not np.isfinite(commands).all():
        raise ValueError("invalid command chunk or source frequency")
    twists = commands @ np.asarray(calibration.matrix) + np.asarray(calibration.bias)
    path = np.empty_like(twists)
    current = np.zeros(3, dtype=np.float64)
    for index, twist_per_second in enumerate(twists):
        current = compose(current, exp(twist_per_second / source_hz))
        path[index] = current
    path[:, 2] = np.unwrap(path[:, 2])
    return path.astype(np.float32)


def pointer_command(follower, position: np.ndarray, quaternion: np.ndarray, reference: float):
    """Track one externally supplied geometric progress value with pose feedback."""

    position_xy = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
    yaw = follower._world_yaw(quaternion)
    measured_progress = follower._project_progress(position_xy, yaw)
    endpoint = follower.world_path[-1]
    endpoint_position_error = float(np.linalg.norm(endpoint[:2] - position_xy))
    endpoint_yaw_error = abs(float(wrap_angle(endpoint[2] - yaw)))
    reference = min(float(reference), float(follower.cumulative_progress[-1]))
    target = follower._pose_at_progress(reference)
    error_world = target[:2] - position_xy
    cosine, sine = np.cos(yaw), np.sin(yaw)
    error_body = np.asarray(
        [
            cosine * error_world[0] + sine * error_world[1],
            -sine * error_world[0] + cosine * error_world[1],
        ]
    )
    desired_translation = follower.position_gain * error_body
    norm = float(np.linalg.norm(desired_translation))
    if norm > follower.max_translation_speed_mps:
        desired_translation *= follower.max_translation_speed_mps / norm
    desired_yaw_rate = float(
        np.clip(
            follower.yaw_gain * float(wrap_angle(target[2] - yaw)),
            -follower.max_yaw_rate_radps,
            follower.max_yaw_rate_radps,
        )
    )
    raw = follower.calibration.inverse(
        np.asarray([desired_translation[0], desired_translation[1], desired_yaw_rate])
    )
    limited = np.clip(
        raw,
        follower.last_command - follower.max_command_delta_per_tick,
        follower.last_command + follower.max_command_delta_per_tick,
    )
    follower.last_command = limited
    done = (
        reference >= follower.cumulative_progress[-1] - 1e-12
        and endpoint_position_error <= follower.goal_position_tolerance_m
        and endpoint_yaw_error <= follower.goal_yaw_tolerance_rad
    )
    if done:
        follower.last_command.fill(0.0)
        limited = np.zeros(3, dtype=np.float64)
    native = np.asarray([limited[0], limited[1], limited[2], 0.0], dtype=np.float32)
    return native, {
        "reference_progress_m": reference,
        "measured_progress_m": float(measured_progress),
        "endpoint_progress_m": float(follower.cumulative_progress[-1]),
        "endpoint_position_error_m": endpoint_position_error,
        "endpoint_yaw_error_rad": endpoint_yaw_error,
        "done": bool(done),
    }


def _one_tick_config() -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(VIDEO_SOURCES.values())),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_SOURCES)),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "base_motion",
                "control_mode",
                "end_effector_position",
                "end_effector_rotation",
                "gripper_close",
            ],
        ),
    }


class _TraceMixin:
    def _init_trace(self, trace: Path, method: str, rate_scale: float, control_hz: float) -> None:
        self.rate_trace = trace
        self.rate_trace.parent.mkdir(parents=True, exist_ok=True)
        self.rate_trace.write_text("")
        self.rate_method = method
        self.rate_scale = float(rate_scale)
        self.rate_control_hz = float(control_hz)
        self.rate_episode_id = 0
        self.rate_tick = 0

    def _reset_trace(self) -> None:
        self.rate_episode_id += 1
        self.rate_tick = 0
        self._write_rate_trace({"event": "RESET"})

    def _write_rate_trace(self, row: dict[str, Any]) -> None:
        payload = {
            "episode_id": self.rate_episode_id,
            "method": self.rate_method,
            "rate_scale": self.rate_scale,
            "control_hz": self.rate_control_hz,
            **row,
        }
        with self.rate_trace.open("a") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


class NavigateACTFrozenRetimePolicy(_TraceMixin, NavigateACTPolicy):
    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        device: str,
        *,
        rate_scale: float,
        control_hz: float,
        source_hz: float,
        source_duration_s: float,
        trace: Path,
    ) -> None:
        NavigateACTPolicy.__init__(self, checkpoint, tasks, device)
        if self.action_dim != 3:
            raise ValueError("frozen retiming requires the verified base-only ACT checkpoint")
        if source_duration_s <= 0.0 or source_duration_s > 32.0 / source_hz:
            raise ValueError("source duration exceeds ACT chunk")
        self.source_hz = float(source_hz)
        self.source_duration_s = float(source_duration_s)
        self.chunk: np.ndarray | None = None
        self.chunk_hash: str | None = None
        self._init_trace(trace, "act_frozen_retime", rate_scale, control_hz)

    def get_modality_config(self):
        return _one_tick_config()

    def reset(self, options=None):
        NavigateACTPolicy.reset(self, options)
        self.chunk = None
        self.chunk_hash = None
        self._reset_trace()
        return {}

    def _get_action(self, observation, options=None):
        replanned = self.chunk is None
        if replanned:
            native, _ = NavigateACTPolicy._get_action(self, observation, options)
            self.chunk = np.asarray(native["action.base_motion"][0, :, :3], dtype=np.float32).copy()
            self.chunk_hash = array_sha256(self.chunk)
        assert self.chunk is not None and self.chunk_hash is not None
        start_s = self.rate_tick / self.rate_control_hz
        end_s = (self.rate_tick + 1) / self.rate_control_hz
        command, retime = retimed_zoh_command(
            self.chunk,
            start_s,
            end_s,
            rate_scale=self.rate_scale,
            source_hz=self.source_hz,
            source_duration_s=self.source_duration_s,
        )
        position = self._latest(observation["state.base_position"])[0]
        quaternion = self._latest(observation["state.base_rotation"])[0]
        self._write_rate_trace(
            {
                "event": "ACTION",
                "tick": self.rate_tick,
                "replanned": replanned,
                "output_sha256": self.chunk_hash,
                "position": np.asarray(position).reshape(-1)[:2].tolist(),
                "yaw": float(yaw_from_xyzw(np.asarray(quaternion).reshape(1, 4))[0]),
                "base_command": command.tolist(),
                "retime": retime,
            }
        )
        self.rate_tick += 1
        return base_only_to_native(command[None, None]), retime


class NavigateGeometricFrozenPointerPolicy(_TraceMixin, NavigateGeometricPolicy):
    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        calibration: Path,
        device: str,
        *,
        rate_scale: float,
        control_hz: float,
        nominal_progress_rate_mps: float,
        segment_extent_m: float,
        trace: Path,
    ) -> None:
        NavigateGeometricPolicy.__init__(
            self,
            checkpoint,
            tasks,
            calibration,
            device,
            control_hz=control_hz,
            replan_ticks=10**9,
            trace=None,
        )
        if nominal_progress_rate_mps <= 0.0 or segment_extent_m <= 0.0:
            raise ValueError("pointer rate and segment extent must be positive")
        self.nominal_progress_rate_mps = float(nominal_progress_rate_mps)
        self.segment_extent_m = float(segment_extent_m)
        self.full_path: np.ndarray | None = None
        self.segment_path: np.ndarray | None = None
        self.full_hash: str | None = None
        self.segment_hash: str | None = None
        self._init_trace(trace, "point_frozen_pointer", rate_scale, control_hz)

    def get_modality_config(self):
        return _one_tick_config()

    def reset(self, options=None):
        NavigateGeometricPolicy.reset(self, options)
        self.full_path = None
        self.segment_path = None
        self.full_hash = None
        self.segment_hash = None
        self._reset_trace()
        return {}

    def _pointer_command(self, position: np.ndarray, quaternion: np.ndarray, reference: float):
        return pointer_command(self.follower, position, quaternion, reference)

    def _get_action(self, observation, options=None):
        position = self._latest(observation["state.base_position"])[0]
        quaternion = self._latest(observation["state.base_rotation"])[0]
        replanned = self.full_path is None
        if replanned:
            self.full_path = self._predict(observation)
            self.segment_path = geometric_prefix(
                self.full_path,
                self.segment_extent_m,
                yaw_radius_m=self.follower.yaw_radius_m,
            )
            self.full_hash = array_sha256(self.full_path)
            self.segment_hash = array_sha256(self.segment_path)
            self.follower.set_plan(self.segment_path, position, quaternion)
        assert self.full_hash is not None and self.segment_hash is not None
        reference = (
            (self.rate_tick + 1)
            / self.rate_control_hz
            * self.nominal_progress_rate_mps
            * self.rate_scale
        )
        base, info = self._pointer_command(position, quaternion, reference)
        self._write_rate_trace(
            {
                "event": "ACTION",
                "tick": self.rate_tick,
                "replanned": replanned,
                "output_sha256": self.full_hash,
                "segment_sha256": self.segment_hash,
                "position": np.asarray(position).reshape(-1)[:2].tolist(),
                "yaw": float(yaw_from_xyzw(np.asarray(quaternion).reshape(1, 4))[0]),
                "base_command": base.tolist(),
                "pointer": info,
            }
        )
        self.rate_tick += 1
        native = {
            "action.base_motion": base[None, None],
            "action.control_mode": np.ones((1, 1, 1), dtype=np.float32),
            "action.end_effector_position": np.zeros((1, 1, 3), dtype=np.float32),
            "action.end_effector_rotation": np.zeros((1, 1, 3), dtype=np.float32),
            "action.gripper_close": -np.ones((1, 1, 1), dtype=np.float32),
        }
        return native, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("act", "point"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rate-scale", type=float, required=True)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--source-hz", type=float, default=20.0)
    parser.add_argument("--source-duration-s", type=float, default=0.4)
    parser.add_argument("--nominal-progress-rate-mps", type=float, default=0.25)
    parser.add_argument("--segment-extent-m", type=float, default=0.20)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "act":
        policy = NavigateACTFrozenRetimePolicy(
            args.checkpoint,
            args.tasks,
            args.device,
            rate_scale=args.rate_scale,
            control_hz=args.control_hz,
            source_hz=args.source_hz,
            source_duration_s=args.source_duration_s,
            trace=args.trace,
        )
    else:
        if args.calibration is None:
            parser.error("--calibration is required for point")
        policy = NavigateGeometricFrozenPointerPolicy(
            args.checkpoint,
            args.tasks,
            args.calibration,
            args.device,
            rate_scale=args.rate_scale,
            control_hz=args.control_hz,
            nominal_progress_rate_mps=args.nominal_progress_rate_mps,
            segment_extent_m=args.segment_extent_m,
            trace=args.trace,
        )
    print(
        json.dumps(
            {
                "event": "RATE_CONTROL_SERVER_READY",
                "method": args.method,
                "rate_scale": args.rate_scale,
                "control_hz": args.control_hz,
                "frozen_single_prediction": True,
                "source_duration_s": args.source_duration_s if args.method == "act" else None,
                "nominal_progress_rate_mps": (
                    args.nominal_progress_rate_mps if args.method == "point" else None
                ),
                "segment_extent_m": args.segment_extent_m if args.method == "point" else None,
            }
        ),
        flush=True,
    )
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
