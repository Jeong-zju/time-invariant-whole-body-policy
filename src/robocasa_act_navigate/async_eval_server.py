"""Asynchronous 50 Hz execution with single-flight, drop-on-busy inference.

The simulator asks for one low-level action per call.  This server paces those
calls at ``control_hz`` while model inference runs in one background worker.
Inference requests never queue: a due request is dropped whenever the worker is
busy, and the latest fully completed command chunk or geometric path continues
to execute.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from gr00t.data.types import ModalityConfig
from gr00t.policy.server_client import PolicyServer

from .b2_labels import yaw_from_xyzw
from .eval_server import STATE_SOURCES, VIDEO_SOURCES, NavigateACTPolicy, base_only_to_native
from .frequency_eval_server import interval_average_zoh
from .geometric_eval_server import NavigateGeometricPolicy


class FractionalRequestScheduler:
    """Emit an average request rate on a fixed-rate low-level tick grid."""

    def __init__(self, request_hz: float, control_hz: float) -> None:
        if request_hz <= 0.0 or control_hz <= 0.0 or request_hz > control_hz:
            raise ValueError("require 0 < request_hz <= control_hz")
        self.increment = float(request_hz / control_hz)
        self.phase = 0.0

    def reset(self) -> None:
        self.phase = 0.0

    def step(self) -> bool:
        self.phase += self.increment
        if self.phase + 1e-12 < 1.0:
            return False
        self.phase -= 1.0
        return True


def _one_tick_modality_config() -> dict[str, ModalityConfig]:
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


class _AsyncSingleFlight:
    """Shared scheduling, pacing, tracing, and lifecycle mechanics."""

    def _init_async(
        self,
        *,
        request_hz: float,
        control_hz: float,
        trace: Path | None,
        exact_output: bool = False,
        exact_warmup_inferences: int = 1,
        exact_pipeline_workers: int = 3,
        exact_output_delay_s: float = 0.10,
    ) -> None:
        self.request_hz = float(request_hz)
        self.control_hz = float(control_hz)
        self.scheduler = FractionalRequestScheduler(self.request_hz, self.control_hz)
        self.exact_output = bool(exact_output)
        if exact_warmup_inferences < 1:
            raise ValueError("exact_warmup_inferences must be at least 1")
        self.exact_warmup_inferences = int(exact_warmup_inferences)
        if exact_pipeline_workers < 1:
            raise ValueError("exact_pipeline_workers must be at least 1")
        if exact_output_delay_s <= 0.0:
            raise ValueError("exact_output_delay_s must be positive")
        self.exact_pipeline_workers = int(exact_pipeline_workers)
        self.exact_output_delay_s = float(exact_output_delay_s)
        self.executor = ThreadPoolExecutor(
            max_workers=self.exact_pipeline_workers if self.exact_output else 1,
            thread_name_prefix="policy-inference",
        )
        self.inflight: Future[dict[str, Any]] | None = None
        self.exact_inflight: deque[Future[dict[str, Any]]] = deque()
        self.ready_result: dict[str, Any] | None = None
        self.trace = trace
        if trace is not None:
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text("")
        self.episode_id = 0
        self.control_tick = 0
        self.accepted_requests = 0
        self.dropped_requests = 0
        self.completed_requests = 0
        self._exact_period_s = 1.0 / self.request_hz
        self._next_exact_capture_s = self._exact_period_s
        self._next_exact_release_s = self.exact_output_delay_s
        self._next_deadline: float | None = None
        self._last_action_start: float | None = None

    def _trace_deadline_miss(self, release_time_s: float) -> None:
        """Record the late result's measured latency before a fatal exact-rate miss."""
        result = None
        if self.exact_inflight:
            result = self.exact_inflight[0].result()
        elif self.ready_result is not None:
            result = self.ready_result
        elif self.inflight is not None:
            result = self.inflight.result()
        latency_s = None if result is None else float(result["latency_s"])
        self._write_trace(
            {
                "event": "OUTPUT_DEADLINE_MISS",
                "tick": self.control_tick,
                "request_hz": self.request_hz,
                "completed_latency_s": latency_s,
                "completed_pipeline_latency_s": (
                    None if result is None else float(result.get("pipeline_latency_s", latency_s))
                ),
                "output_period_s": self._exact_period_s,
                "required_pipeline_budget_s": self.exact_output_delay_s,
                "scheduled_release_time_s": float(release_time_s),
            }
        )

    def _run_warmup_inferences(
        self,
        observation: dict[str, Any],
        infer: Callable[[dict[str, Any], int], dict[str, Any]],
    ) -> tuple[dict[str, Any], list[float]]:
        """Warm the same worker thread that will serve timed inference."""
        count = self.exact_warmup_inferences if self.exact_output else 1
        latencies = []
        result = None
        if self.exact_output:
            for _ in range(count):
                futures = [
                    self.executor.submit(
                        infer,
                        copy.deepcopy(observation),
                        self.control_tick,
                    )
                    for _ in range(self.exact_pipeline_workers)
                ]
                for future in futures:
                    result = future.result()
                    latencies.append(float(result["latency_s"]))
        else:
            result = infer(copy.deepcopy(observation), self.control_tick)
            latencies.append(float(result["latency_s"]))
        assert result is not None
        return result, latencies

    def _wait_and_discard_inflight(self) -> None:
        if self.inflight is not None:
            self.inflight.result()
            self.inflight = None
        self.ready_result = None
        while self.exact_inflight:
            self.exact_inflight.popleft().result()

    def _reset_async(self) -> None:
        self._wait_and_discard_inflight()
        self.scheduler.reset()
        self.episode_id += 1
        self.control_tick = 0
        self.accepted_requests = 0
        self.dropped_requests = 0
        self.completed_requests = 0
        self._next_exact_capture_s = self._exact_period_s
        self._next_exact_release_s = self.exact_output_delay_s
        self._next_deadline = None
        self._last_action_start = None
        self._write_trace({"event": "RESET"})

    def _write_trace(self, value: dict[str, Any]) -> None:
        if self.trace is None:
            return
        row = {"episode_id": self.episode_id, "monotonic_s": time.perf_counter(), **value}
        with self.trace.open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _pace(self) -> dict[str, float | bool]:
        period = 1.0 / self.control_hz
        now = time.perf_counter()
        if self._next_deadline is None:
            action_start = now
            lateness = 0.0
            self._next_deadline = action_start + period
        else:
            deadline = self._next_deadline
            if now < deadline:
                time.sleep(deadline - now)
            action_start = time.perf_counter()
            lateness = max(0.0, action_start - deadline)
            self._next_deadline = deadline + period
            # Never issue a catch-up burst after a slow simulator/render step.
            if action_start > self._next_deadline:
                self._next_deadline = action_start + period
        interval = (
            0.0 if self._last_action_start is None else action_start - self._last_action_start
        )
        self._last_action_start = action_start
        return {
            "control_interval_wall_s": float(interval),
            "deadline_lateness_s": float(lateness),
            "deadline_missed": bool(lateness > 0.002),
        }

    def _poll_completed(self, installer: Callable[[dict[str, Any]], None]) -> dict[str, Any] | None:
        if self.ready_result is None and self.inflight is not None and self.inflight.done():
            result = self.inflight.result()
            self.inflight = None
            # Simulation/rendering may run slower than real time.  Hold a
            # completed result until the 50 Hz virtual control clock says its
            # measured wall-time latency has elapsed, so slow rendering cannot
            # make the GPU appear more available than it would be on a robot.
            latency_ticks = max(1, int(math.ceil(float(result["latency_s"]) * self.control_hz)))
            result["virtual_latency_ticks"] = latency_ticks
            result["virtual_ready_tick"] = int(result["capture_tick"]) + latency_ticks
            self.ready_result = result
        if self.ready_result is None or self.control_tick < int(self.ready_result["virtual_ready_tick"]):
            return None
        result = self.ready_result
        self.ready_result = None
        installer(result)
        self.completed_requests += 1
        return result

    def _exact_event_due(self, next_time_s: float) -> bool:
        now_s = self.control_tick / self.control_hz
        return now_s + 1e-12 >= next_time_s

    def _poll_exact_completed(
        self,
        installer: Callable[[dict[str, Any]], None],
        release_time_s: float,
    ) -> dict[str, Any] | None:
        if not self.exact_inflight or not self.exact_inflight[0].done():
            return None
        result = self.exact_inflight[0].result()
        if float(result["virtual_ready_time_s"]) > release_time_s + 1e-12:
            return None
        self.exact_inflight.popleft()
        result["virtual_latency_ticks"] = max(
            1, int(math.ceil(float(result["latency_s"]) * self.control_hz))
        )
        installer(result)
        self.completed_requests += 1
        return result

    def _submit_exact_request(
        self,
        observation: dict[str, Any],
        infer: Callable[[dict[str, Any], int], dict[str, Any]],
        capture_time_s: float,
    ) -> None:
        snapshot = copy.deepcopy(observation)
        capture_tick = self.control_tick
        submitted_s = time.perf_counter()

        def run() -> dict[str, Any]:
            result = infer(snapshot, capture_tick)
            result["capture_time_s"] = float(capture_time_s)
            result["pipeline_latency_s"] = float(time.perf_counter() - submitted_s)
            result["virtual_ready_time_s"] = (
                float(capture_time_s) + float(result["pipeline_latency_s"])
            )
            return result

        self.exact_inflight.append(self.executor.submit(run))
        self.accepted_requests += 1

    def _submit_due_exact_capture(
        self,
        observation: dict[str, Any],
        infer: Callable[[dict[str, Any], int], dict[str, Any]],
    ) -> bool:
        if not self._exact_event_due(self._next_exact_capture_s):
            return False
        capture_time_s = self._next_exact_capture_s
        self._next_exact_capture_s += self._exact_period_s
        self._submit_exact_request(observation, infer, capture_time_s)
        return True

    def _maybe_request(
        self,
        observation: dict[str, Any],
        infer: Callable[[dict[str, Any], int], dict[str, Any]],
    ) -> tuple[bool, bool, bool]:
        due = self.scheduler.step()
        if not due:
            return False, False, False
        if self.inflight is not None or self.ready_result is not None:
            self.dropped_requests += 1
            return True, False, True
        self._submit_request(observation, infer)
        return True, True, False

    def _submit_request(
        self,
        observation: dict[str, Any],
        infer: Callable[[dict[str, Any], int], dict[str, Any]],
        *,
        capture_time_s: float | None = None,
    ) -> None:
        if self.inflight is not None or self.ready_result is not None:
            raise RuntimeError("single-flight inference is already busy")
        snapshot = copy.deepcopy(observation)
        capture_tick = self.control_tick
        if capture_time_s is None:
            capture_time_s = capture_tick / self.control_hz

        def run() -> dict[str, Any]:
            result = infer(snapshot, capture_tick)
            result["capture_time_s"] = float(capture_time_s)
            return result

        self.inflight = self.executor.submit(run)
        self.accepted_requests += 1


class NavigateACTAsyncPolicy(_AsyncSingleFlight, NavigateACTPolicy):
    """Time-aligned ACT chunk execution under asynchronous inference."""

    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        device: str,
        *,
        request_hz: float,
        control_hz: float = 50.0,
        source_action_hz: float = 20.0,
        trace: Path | None = None,
        exact_output: bool = False,
        exact_warmup_inferences: int = 1,
        exact_pipeline_workers: int = 3,
        exact_output_delay_s: float = 0.10,
    ) -> None:
        NavigateACTPolicy.__init__(self, checkpoint, tasks, device)
        if self.action_dim != 3:
            raise ValueError("asynchronous ACT requires the base-only checkpoint")
        self.source_action_hz = float(source_action_hz)
        self._init_async(
            request_hz=request_hz,
            control_hz=control_hz,
            trace=trace,
            exact_output=exact_output,
            exact_warmup_inferences=exact_warmup_inferences,
            exact_pipeline_workers=exact_pipeline_workers,
            exact_output_delay_s=exact_output_delay_s,
        )
        self.chunk: np.ndarray | None = None
        self.plan_capture_tick = 0
        self.plan_latency_s = 0.0

    def get_modality_config(self):
        return _one_tick_modality_config()

    def reset(self, options=None):
        self._reset_async()
        NavigateACTPolicy.reset(self, options)
        self.chunk = None
        self.plan_capture_tick = 0
        self.plan_latency_s = 0.0
        return {}

    def _infer(self, observation: dict[str, Any], capture_tick: int) -> dict[str, Any]:
        started = time.perf_counter()
        native, _ = NavigateACTPolicy._get_action(self, observation, None)
        chunk = np.asarray(native["action.base_motion"][0, :, :3], dtype=np.float32).copy()
        return {
            "capture_tick": int(capture_tick),
            "latency_s": float(time.perf_counter() - started),
            "chunk": chunk,
        }

    def _install(self, result: dict[str, Any]) -> None:
        self.chunk = result["chunk"]
        self.plan_capture_tick = int(result["capture_tick"])
        self.plan_latency_s = float(result["latency_s"])

    def _command(self) -> tuple[np.ndarray, bool, float]:
        if self.chunk is None:
            raise RuntimeError("ACT warm start did not install a chunk")
        start_s = max(0, self.control_tick - self.plan_capture_tick) / self.control_hz
        end_s = start_s + 1.0 / self.control_hz
        duration_s = len(self.chunk) / self.source_action_hz
        exhausted = start_s >= duration_s - 1e-12
        if exhausted:
            return np.zeros(3, dtype=np.float32), True, start_s
        valid_end = min(end_s, duration_s)
        base = interval_average_zoh(
            self.chunk,
            start_s,
            valid_end,
            source_hz=self.source_action_hz,
        )
        # A partially covered final interval is zero-padded, preserving the
        # learned command integral rather than stretching the final token.
        base *= (valid_end - start_s) / (end_s - start_s)
        return base.astype(np.float32), False, start_s

    def _get_action(self, observation, options=None):
        pacing = self._pace()
        warm = self.chunk is None
        completed = None
        release_time_s = None
        due = accepted = dropped = False
        if warm:
            result, warmup_latencies = self._run_warmup_inferences(observation, self._infer)
            self._install(result)
            self._write_trace(
                {
                    "event": "WARM_START",
                    "latency_s": result["latency_s"],
                    "warmup_inferences": len(warmup_latencies),
                    "warmup_latencies_s": warmup_latencies,
                }
            )
            if self.exact_output:
                self._submit_exact_request(observation, self._infer, capture_time_s=0.0)
                accepted = True
        elif self.exact_output:
            if self._exact_event_due(self._next_exact_release_s):
                release_time_s = self._next_exact_release_s
                self._next_exact_release_s += self._exact_period_s
                due = True
                completed = self._poll_exact_completed(self._install, release_time_s)
                if completed is None:
                    self._trace_deadline_miss(release_time_s)
                    raise RuntimeError(
                        f"exact {self.request_hz:g} Hz output deadline missed at tick {self.control_tick}"
                    )
            accepted = self._submit_due_exact_capture(observation, self._infer)
        else:
            completed = self._poll_completed(self._install)
            due, accepted, dropped = self._maybe_request(observation, self._infer)
        base, exhausted, source_start_s = self._command()
        native = base_only_to_native(base[None, None, :])
        event = {
            "event": "ACTION",
            "tick": self.control_tick,
            "method": "act_time_aligned_async",
            "control_hz": self.control_hz,
            "request_hz": self.request_hz,
            "exact_output": self.exact_output,
            "output_released": bool((warm and not self.exact_output) or completed is not None),
            "bootstrap_output": bool(warm and self.exact_output),
            "output_release_time_s": (
                None if warm else (None if completed is None else release_time_s)
            ),
            "request_due": due,
            "request_accepted": accepted,
            "request_dropped": dropped,
            "inference_completed": completed is not None,
            "completed_latency_s": None if completed is None else completed["latency_s"],
            "completed_pipeline_latency_s": (
                None if completed is None else completed.get("pipeline_latency_s")
            ),
            "completed_virtual_latency_ticks": None if completed is None else completed["virtual_latency_ticks"],
            "plan_capture_tick": self.plan_capture_tick,
            "plan_age_ticks": self.control_tick - self.plan_capture_tick,
            "plan_latency_s": self.plan_latency_s,
            "source_start_s": source_start_s,
            "chunk_exhausted": exhausted,
            "base_command": base.tolist(),
            **pacing,
        }
        self._write_trace(event)
        self.control_tick += 1
        return native, event


class NavigateGeometricAsyncPolicy(_AsyncSingleFlight, NavigateGeometricPolicy):
    """Measured-pose follower whose geometric plans arrive asynchronously."""

    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        calibration: Path,
        device: str,
        *,
        request_hz: float,
        control_hz: float = 50.0,
        trace: Path | None = None,
        exact_output: bool = False,
        exact_warmup_inferences: int = 1,
        exact_pipeline_workers: int = 3,
        exact_output_delay_s: float = 0.10,
    ) -> None:
        NavigateGeometricPolicy.__init__(
            self,
            checkpoint,
            tasks,
            calibration,
            device,
            control_hz=control_hz,
            replan_ticks=1,
            trace=None,
        )
        self._init_async(
            request_hz=request_hz,
            control_hz=control_hz,
            trace=trace,
            exact_output=exact_output,
            exact_warmup_inferences=exact_warmup_inferences,
            exact_pipeline_workers=exact_pipeline_workers,
            exact_output_delay_s=exact_output_delay_s,
        )
        self.plan_capture_tick = 0
        self.plan_latency_s = 0.0

    def get_modality_config(self):
        return _one_tick_modality_config()

    def reset(self, options=None):
        self._reset_async()
        NavigateGeometricPolicy.reset(self, options)
        self.plan_capture_tick = 0
        self.plan_latency_s = 0.0
        return {}

    def _infer(self, observation: dict[str, Any], capture_tick: int) -> dict[str, Any]:
        position = self._latest(observation["state.base_position"])[0].copy()
        quaternion = self._latest(observation["state.base_rotation"])[0].copy()
        started = time.perf_counter()
        path = self._predict(observation)
        return {
            "capture_tick": int(capture_tick),
            "latency_s": float(time.perf_counter() - started),
            "path": path,
            "capture_position": position,
            "capture_quaternion": quaternion,
        }

    def _install(self, result: dict[str, Any], *, preserve: bool = True) -> None:
        self.cached_path = result["path"]
        self.follower.set_plan(
            self.cached_path,
            result["capture_position"],
            result["capture_quaternion"],
            preserve_last_command=preserve,
        )
        self.plan_capture_tick = int(result["capture_tick"])
        self.plan_latency_s = float(result["latency_s"])

    def _get_action(self, observation, options=None):
        pacing = self._pace()
        warm = self.cached_path is None
        completed = None
        release_time_s = None
        due = accepted = dropped = False
        if warm:
            result, warmup_latencies = self._run_warmup_inferences(observation, self._infer)
            self._install(result, preserve=False)
            self._write_trace(
                {
                    "event": "WARM_START",
                    "latency_s": result["latency_s"],
                    "warmup_inferences": len(warmup_latencies),
                    "warmup_latencies_s": warmup_latencies,
                }
            )
            if self.exact_output:
                self._submit_exact_request(observation, self._infer, capture_time_s=0.0)
                accepted = True
        elif self.exact_output:
            if self._exact_event_due(self._next_exact_release_s):
                release_time_s = self._next_exact_release_s
                self._next_exact_release_s += self._exact_period_s
                due = True
                completed = self._poll_exact_completed(self._install, release_time_s)
                if completed is None:
                    self._trace_deadline_miss(release_time_s)
                    raise RuntimeError(
                        f"exact {self.request_hz:g} Hz output deadline missed at tick {self.control_tick}"
                    )
            accepted = self._submit_due_exact_capture(observation, self._infer)
        else:
            completed = self._poll_completed(self._install)
            due, accepted, dropped = self._maybe_request(observation, self._infer)
        position = self._latest(observation["state.base_position"])[0]
        quaternion = self._latest(observation["state.base_rotation"])[0]
        base, follower_info = self.follower.command(position, quaternion)
        native = {
            "action.base_motion": base[None, None],
            "action.control_mode": np.ones((1, 1, 1), dtype=np.float32),
            "action.end_effector_position": np.zeros((1, 1, 3), dtype=np.float32),
            "action.end_effector_rotation": np.zeros((1, 1, 3), dtype=np.float32),
            "action.gripper_close": -np.ones((1, 1, 1), dtype=np.float32),
        }
        event = {
            "event": "ACTION",
            "tick": self.control_tick,
            "method": "path_ratefree_async",
            "control_hz": self.control_hz,
            "request_hz": self.request_hz,
            "exact_output": self.exact_output,
            "output_released": bool((warm and not self.exact_output) or completed is not None),
            "bootstrap_output": bool(warm and self.exact_output),
            "output_release_time_s": (
                None if warm else (None if completed is None else release_time_s)
            ),
            "request_due": due,
            "request_accepted": accepted,
            "request_dropped": dropped,
            "inference_completed": completed is not None,
            "completed_latency_s": None if completed is None else completed["latency_s"],
            "completed_pipeline_latency_s": (
                None if completed is None else completed.get("pipeline_latency_s")
            ),
            "completed_virtual_latency_ticks": None if completed is None else completed["virtual_latency_ticks"],
            "plan_capture_tick": self.plan_capture_tick,
            "plan_age_ticks": self.control_tick - self.plan_capture_tick,
            "plan_latency_s": self.plan_latency_s,
            "position": np.asarray(position).reshape(-1)[:2].tolist(),
            "yaw": float(yaw_from_xyzw(np.asarray(quaternion).reshape(1, 4))[0]),
            "base_command": base.tolist(),
            "follower": follower_info,
            **pacing,
        }
        self._write_trace(event)
        self.control_tick += 1
        return native, event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("act", "geometric"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-hz", type=float, required=True)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--source-action-hz", type=float, default=20.0)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--exact-output", action="store_true")
    parser.add_argument("--exact-warmup-inferences", type=int, default=1)
    parser.add_argument("--exact-pipeline-workers", type=int, default=3)
    parser.add_argument("--exact-output-delay-s", type=float, default=0.10)
    args = parser.parse_args()
    if args.method == "act":
        policy = NavigateACTAsyncPolicy(
            args.checkpoint,
            args.tasks,
            args.device,
            request_hz=args.request_hz,
            control_hz=args.control_hz,
            source_action_hz=args.source_action_hz,
            trace=args.trace,
            exact_output=args.exact_output,
            exact_warmup_inferences=args.exact_warmup_inferences,
            exact_pipeline_workers=args.exact_pipeline_workers,
            exact_output_delay_s=args.exact_output_delay_s,
        )
    else:
        if args.calibration is None:
            parser.error("--calibration is required for geometric")
        policy = NavigateGeometricAsyncPolicy(
            args.checkpoint,
            args.tasks,
            args.calibration,
            args.device,
            request_hz=args.request_hz,
            control_hz=args.control_hz,
            trace=args.trace,
            exact_output=args.exact_output,
            exact_warmup_inferences=args.exact_warmup_inferences,
            exact_pipeline_workers=args.exact_pipeline_workers,
            exact_output_delay_s=args.exact_output_delay_s,
        )
    print(
        json.dumps(
            {
                "event": "ASYNC_POLICY_SERVER_READY",
                "method": args.method,
                "checkpoint": str(args.checkpoint),
                "control_hz": args.control_hz,
                "request_hz": args.request_hz,
                "single_flight": True,
                "drop_on_busy": True,
                "queue_depth": 0,
                "exact_output": args.exact_output,
                "exact_pipeline_workers": args.exact_pipeline_workers,
                "exact_output_delay_s": args.exact_output_delay_s,
            }
        ),
        flush=True,
    )
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
