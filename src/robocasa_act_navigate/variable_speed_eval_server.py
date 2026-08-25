"""Closed-loop variable-speed policies for the NavigateKitchen causal study.

All methods observe and replan on the same 0.4 simulated-second schedule while
the simulator runs at 50 Hz.  ``rate_scale`` changes progress *inside* each
freshly predicted chunk; it never changes observation timestamps or the model
replanning schedule.

The four methods isolate two questions:

* ``act_native`` changes only ACT token-consumption speed.
* ``act_retime`` additionally scales velocity magnitude, the strongest direct
  retiming baseline available from ACT's command semantics.
* ``act_calibrated_pointer`` converts the same ACT chunk to a predicted SE(2)
  path and advances an external measured-pose pointer.
* ``learned_point_pointer`` uses the separately learned time-free Point path
  with the identical pointer and controller.

The calibrated ACT path is a model-derived prediction.  It must not be called
ground-truth physical motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.policy.server_client import PolicyServer

from .b2_labels import yaw_from_xyzw
from .eval_server import NavigateACTPolicy, base_only_to_native
from .geometric_eval_server import NavigateGeometricPolicy
from .geometric_follower import BaseRateCalibration, MeasuredPosePathFollower
from .rate_control_eval_server import (
    _TraceMixin,
    _one_tick_config,
    array_sha256,
    calibrated_command_path,
    pointer_command,
    retimed_zoh_command,
)


METHODS = (
    "act_native",
    "act_retime",
    "act_calibrated_pointer",
    "learned_point_pointer",
)


def nominal_path_progress_rate(path: np.ndarray, nominal_duration_s: float) -> float:
    """Return SE(2)-metric path length per nominal source second."""

    value = np.asarray(path, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) == 0:
        raise ValueError(f"expected path (K,3), got {value.shape}")
    if nominal_duration_s <= 0.0 or not np.isfinite(value).all():
        raise ValueError("invalid path or nominal duration")
    full = np.concatenate((np.zeros((1, 3), dtype=np.float64), value), axis=0)
    metric = full.copy()
    metric[:, 2] *= 0.25
    return float(np.linalg.norm(np.diff(metric, axis=0), axis=-1).sum() / nominal_duration_s)


class NavigateACTClosedLoopRetimePolicy(_TraceMixin, NavigateACTPolicy):
    """Replan ACT every fixed interval and consume each fresh chunk at a rate."""

    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        device: str,
        *,
        method: str,
        rate_scale: float,
        control_hz: float,
        source_hz: float,
        replan_seconds: float,
        trace: Path,
    ) -> None:
        NavigateACTPolicy.__init__(self, checkpoint, tasks, device)
        if self.action_dim != 3:
            raise ValueError("closed-loop retiming requires the verified base-only ACT checkpoint")
        if method not in ("act_native", "act_retime"):
            raise ValueError(f"invalid ACT retiming method {method}")
        self.method = method
        self.source_hz = float(source_hz)
        self.replan_ticks = int(round(float(replan_seconds) * float(control_hz)))
        if self.source_hz <= 0.0 or self.replan_ticks <= 0:
            raise ValueError("source rate and replan interval must be positive")
        self.chunk: np.ndarray | None = None
        self.chunk_hash: str | None = None
        self.tick_in_plan = 0
        self.plan_count = 0
        self._init_trace(trace, method, rate_scale, control_hz)

    def get_modality_config(self):
        return _one_tick_config()

    def reset(self, options=None):
        NavigateACTPolicy.reset(self, options)
        self.chunk = None
        self.chunk_hash = None
        self.tick_in_plan = 0
        self.plan_count = 0
        self._reset_trace()
        return {}

    def _get_action(self, observation, options=None):
        replanned = self.chunk is None or self.tick_in_plan >= self.replan_ticks
        if replanned:
            native, _ = NavigateACTPolicy._get_action(self, observation, options)
            self.chunk = np.asarray(native["action.base_motion"][0, :, :3], dtype=np.float32).copy()
            self.chunk_hash = array_sha256(self.chunk)
            self.tick_in_plan = 0
            self.plan_count += 1
        assert self.chunk is not None and self.chunk_hash is not None
        start_s = self.tick_in_plan / self.rate_control_hz
        end_s = (self.tick_in_plan + 1) / self.rate_control_hz
        command, retime = retimed_zoh_command(
            self.chunk,
            start_s,
            end_s,
            rate_scale=self.rate_scale,
            source_hz=self.source_hz,
            source_duration_s=len(self.chunk) / self.source_hz,
            compensate_amplitude=self.method == "act_retime",
        )
        position = self._latest(observation["state.base_position"])[0]
        quaternion = self._latest(observation["state.base_rotation"])[0]
        self._write_rate_trace(
            {
                "event": "ACTION",
                "tick": self.rate_tick,
                "tick_in_plan": self.tick_in_plan,
                "replanned": replanned,
                "plan_count": self.plan_count,
                "output_sha256": self.chunk_hash,
                "position": np.asarray(position).reshape(-1)[:2].tolist(),
                "yaw": float(yaw_from_xyzw(np.asarray(quaternion).reshape(1, 4))[0]),
                "base_command": command.tolist(),
                "retime": retime,
            }
        )
        self.tick_in_plan += 1
        self.rate_tick += 1
        return base_only_to_native(command[None, None]), retime


class _ClosedLoopPointerMixin(_TraceMixin):
    def _init_pointer(
        self,
        calibration: Path,
        *,
        method: str,
        rate_scale: float,
        control_hz: float,
        replan_seconds: float,
        nominal_duration_s: float,
        trace: Path,
    ) -> None:
        calibration_value = BaseRateCalibration.load(calibration)
        if abs(calibration_value.fitted_control_hz - control_hz) > 1e-6:
            raise ValueError("controller calibration frequency mismatch")
        self.follower = MeasuredPosePathFollower(
            calibration_value,
            control_hz=control_hz,
            max_command_delta_per_tick=4.5 / control_hz,
        )
        self.pointer_replan_ticks = int(round(replan_seconds * control_hz))
        if self.pointer_replan_ticks <= 0 or nominal_duration_s <= 0.0:
            raise ValueError("replan interval and nominal duration must be positive")
        self.nominal_duration_s = float(nominal_duration_s)
        self.pointer_path: np.ndarray | None = None
        self.pointer_hash: str | None = None
        self.pointer_rate_mps = 0.0
        self.pointer_tick_in_plan = 0
        self.pointer_plan_count = 0
        self._init_trace(trace, method, rate_scale, control_hz)

    def _reset_pointer(self) -> None:
        self.follower.clear()
        self.pointer_path = None
        self.pointer_hash = None
        self.pointer_rate_mps = 0.0
        self.pointer_tick_in_plan = 0
        self.pointer_plan_count = 0
        self._reset_trace()

    def _set_pointer_plan(
        self,
        path: np.ndarray,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        self.pointer_path = np.asarray(path, dtype=np.float32).copy()
        self.pointer_hash = array_sha256(self.pointer_path)
        self.pointer_rate_mps = nominal_path_progress_rate(
            self.pointer_path, self.nominal_duration_s
        )
        self.follower.set_plan(
            self.pointer_path,
            position,
            quaternion,
            preserve_last_command=self.pointer_plan_count > 0,
        )
        self.pointer_tick_in_plan = 0
        self.pointer_plan_count += 1

    def _pointer_action(self, observation, *, replanned: bool):
        assert self.pointer_path is not None and self.pointer_hash is not None
        position = self._latest(observation["state.base_position"])[0]
        quaternion = self._latest(observation["state.base_rotation"])[0]
        reference = (
            (self.pointer_tick_in_plan + 1)
            / self.rate_control_hz
            * self.pointer_rate_mps
            * self.rate_scale
        )
        base, info = pointer_command(self.follower, position, quaternion, reference)
        self._write_rate_trace(
            {
                "event": "ACTION",
                "tick": self.rate_tick,
                "tick_in_plan": self.pointer_tick_in_plan,
                "replanned": replanned,
                "plan_count": self.pointer_plan_count,
                "output_sha256": self.pointer_hash,
                "nominal_progress_rate_mps": self.pointer_rate_mps,
                "position": np.asarray(position).reshape(-1)[:2].tolist(),
                "yaw": float(yaw_from_xyzw(np.asarray(quaternion).reshape(1, 4))[0]),
                "base_command": base.tolist(),
                "pointer": info,
            }
        )
        self.pointer_tick_in_plan += 1
        self.rate_tick += 1
        native = {
            "action.base_motion": base[None, None],
            "action.control_mode": np.ones((1, 1, 1), dtype=np.float32),
            "action.end_effector_position": np.zeros((1, 1, 3), dtype=np.float32),
            "action.end_effector_rotation": np.zeros((1, 1, 3), dtype=np.float32),
            "action.gripper_close": -np.ones((1, 1, 1), dtype=np.float32),
        }
        return native, info


class NavigateACTClosedLoopPointerPolicy(_ClosedLoopPointerMixin, NavigateACTPolicy):
    """Convert each fresh ACT command chunk to a calibrated predicted path."""

    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        calibration: Path,
        device: str,
        *,
        rate_scale: float,
        control_hz: float,
        source_hz: float,
        replan_seconds: float,
        trace: Path,
    ) -> None:
        NavigateACTPolicy.__init__(self, checkpoint, tasks, device)
        if self.action_dim != 3:
            raise ValueError("ACT pointer requires the verified base-only ACT checkpoint")
        self.source_hz = float(source_hz)
        self.path_calibration = BaseRateCalibration.load(calibration)
        self._init_pointer(
            calibration,
            method="act_calibrated_pointer",
            rate_scale=rate_scale,
            control_hz=control_hz,
            replan_seconds=replan_seconds,
            nominal_duration_s=32.0 / self.source_hz,
            trace=trace,
        )

    def get_modality_config(self):
        return _one_tick_config()

    def reset(self, options=None):
        NavigateACTPolicy.reset(self, options)
        self._reset_pointer()
        return {}

    def _get_action(self, observation, options=None):
        replanned = self.pointer_path is None or self.pointer_tick_in_plan >= self.pointer_replan_ticks
        if replanned:
            native, _ = NavigateACTPolicy._get_action(self, observation, options)
            chunk = np.asarray(native["action.base_motion"][0, :, :3], dtype=np.float32)
            path = calibrated_command_path(chunk, self.path_calibration, source_hz=self.source_hz)
            position = self._latest(observation["state.base_position"])[0]
            quaternion = self._latest(observation["state.base_rotation"])[0]
            self._set_pointer_plan(path, position, quaternion)
        return self._pointer_action(observation, replanned=replanned)


class NavigatePointClosedLoopPointerPolicy(_ClosedLoopPointerMixin, NavigateGeometricPolicy):
    """Advance the learned time-free path with the same closed-loop pointer."""

    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        calibration: Path,
        device: str,
        *,
        rate_scale: float,
        control_hz: float,
        replan_seconds: float,
        nominal_duration_s: float,
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
        self._init_pointer(
            calibration,
            method="learned_point_pointer",
            rate_scale=rate_scale,
            control_hz=control_hz,
            replan_seconds=replan_seconds,
            nominal_duration_s=nominal_duration_s,
            trace=trace,
        )

    def reset(self, options=None):
        NavigateGeometricPolicy.reset(self, options)
        self._reset_pointer()
        return {}

    def _get_action(self, observation, options=None):
        replanned = self.pointer_path is None or self.pointer_tick_in_plan >= self.pointer_replan_ticks
        if replanned:
            path = self._predict(observation)
            position = self._latest(observation["state.base_position"])[0]
            quaternion = self._latest(observation["state.base_rotation"])[0]
            self._set_pointer_plan(path, position, quaternion)
        return self._pointer_action(observation, replanned=replanned)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rate-scale", type=float, choices=(0.5, 1.0, 1.5), required=True)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--source-hz", type=float, default=20.0)
    parser.add_argument("--replan-seconds", type=float, default=0.4)
    parser.add_argument("--point-nominal-duration-s", type=float, default=1.6)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    common = dict(
        rate_scale=args.rate_scale,
        control_hz=args.control_hz,
        replan_seconds=args.replan_seconds,
        trace=args.trace,
    )
    if args.method in ("act_native", "act_retime"):
        policy = NavigateACTClosedLoopRetimePolicy(
            args.checkpoint,
            args.tasks,
            args.device,
            method=args.method,
            source_hz=args.source_hz,
            **common,
        )
    elif args.method == "act_calibrated_pointer":
        if args.calibration is None:
            parser.error("--calibration is required for pointer methods")
        policy = NavigateACTClosedLoopPointerPolicy(
            args.checkpoint,
            args.tasks,
            args.calibration,
            args.device,
            source_hz=args.source_hz,
            **common,
        )
    else:
        if args.calibration is None:
            parser.error("--calibration is required for pointer methods")
        policy = NavigatePointClosedLoopPointerPolicy(
            args.checkpoint,
            args.tasks,
            args.calibration,
            args.device,
            nominal_duration_s=args.point_nominal_duration_s,
            **common,
        )

    print(
        json.dumps(
            {
                "event": "VARIABLE_SPEED_SERVER_READY",
                "method": args.method,
                "rate_scale": args.rate_scale,
                "control_hz": args.control_hz,
                "replan_seconds": args.replan_seconds,
                "rate_changes_progress_inside_chunk": True,
                "fixed_replanning_schedule": True,
            }
        ),
        flush=True,
    )
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
