"""Serve native 20 Hz ACT velocity chunks at a requested control frequency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.data.types import ModalityConfig
from gr00t.policy.server_client import PolicyServer

from .eval_server import STATE_SOURCES, VIDEO_SOURCES, NavigateACTPolicy, base_only_to_native


def interval_average_zoh(
    chunk: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    source_hz: float,
) -> np.ndarray:
    """Average a source-rate zero-order-hold signal over one target interval.

    Averaging by exact interval overlap preserves the command integral when a
    20 Hz velocity chunk is executed at 10, 20, or 50 Hz.  This is not used for
    geometric paths, whose output has no time parameter.
    """

    value = np.asarray(chunk, dtype=np.float64)
    if value.ndim != 2 or len(value) == 0 or not np.isfinite(value).all():
        raise ValueError(f"expected finite chunk (H,D), got {value.shape}")
    if source_hz <= 0.0 or start_s < 0.0 or end_s <= start_s:
        raise ValueError("invalid source frequency or target interval")
    source_edges = np.arange(len(value) + 1, dtype=np.float64) / source_hz
    if end_s > source_edges[-1] + 1e-12:
        raise ValueError("target interval exceeds source chunk duration")
    overlap = np.maximum(
        0.0,
        np.minimum(end_s, source_edges[1:]) - np.maximum(start_s, source_edges[:-1]),
    )
    duration = end_s - start_s
    if abs(float(overlap.sum()) - duration) > 1e-10:
        raise RuntimeError("source chunk does not fully cover target interval")
    return (overlap[:, None] * value).sum(axis=0) / duration


class NavigateACTFrequencyPolicy(NavigateACTPolicy):
    """One-tick policy with time-faithful resampling of native ACT commands."""

    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        device: str,
        *,
        source_action_hz: float = 20.0,
        control_hz: float,
        replan_period_s: float = 0.4,
        trace: Path | None = None,
    ) -> None:
        super().__init__(checkpoint, tasks, device)
        if self.action_dim != 3:
            raise ValueError("frequency baseline requires the verified base-only checkpoint")
        if source_action_hz <= 0.0 or control_hz <= 0.0 or replan_period_s <= 0.0:
            raise ValueError("frequencies and replan period must be positive")
        self.source_action_hz = float(source_action_hz)
        self.control_hz = float(control_hz)
        self.replan_period_s = float(replan_period_s)
        self.replan_ticks = int(round(self.control_hz * self.replan_period_s))
        self.source_replan_steps = int(round(self.source_action_hz * self.replan_period_s))
        if abs(self.replan_ticks / self.control_hz - self.replan_period_s) > 1e-12:
            raise ValueError("replan period is not an integer number of control ticks")
        if abs(self.source_replan_steps / self.source_action_hz - self.replan_period_s) > 1e-12:
            raise ValueError("replan period is not an integer number of source action steps")
        self.cached_base: np.ndarray | None = None
        self.feedback_tick = 0
        self.plan_count = 0
        self.trace = trace
        self.trace_tick = 0
        if self.trace is not None:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            self.trace.write_text("")

    def get_modality_config(self):
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

    def reset(self, options=None):
        super().reset(options)
        self.cached_base = None
        self.feedback_tick = 0
        self.plan_count = 0
        self.trace_tick = 0
        self._write_trace({"event": "RESET"})
        return {}

    def _write_trace(self, value: dict[str, Any]) -> None:
        if self.trace is not None:
            with self.trace.open("a") as handle:
                handle.write(json.dumps(value, separators=(",", ":")) + "\n")

    def _get_action(self, observation, options=None):
        replanned = self.cached_base is None or self.feedback_tick >= self.replan_ticks
        if replanned:
            native_chunk, _ = super()._get_action(observation, options)
            self.cached_base = np.asarray(native_chunk["action.base_motion"][0, :, :3]).copy()
            if len(self.cached_base) < self.source_replan_steps:
                raise ValueError("ACT chunk is shorter than the matched replan period")
            self.feedback_tick = 0
            self.plan_count += 1
        start_s = self.feedback_tick / self.control_hz
        end_s = (self.feedback_tick + 1) / self.control_hz
        base = interval_average_zoh(
            self.cached_base,
            start_s,
            end_s,
            source_hz=self.source_action_hz,
        ).astype(np.float32)
        native = base_only_to_native(base[None, None, :])
        self.feedback_tick += 1
        self.trace_tick += 1
        self._write_trace(
            {
                "event": "ACTION",
                "tick": self.trace_tick,
                "control_hz": self.control_hz,
                "source_action_hz": self.source_action_hz,
                "replanned": replanned,
                "plan_count": self.plan_count,
                "feedback_tick": self.feedback_tick,
                "source_interval_s": [start_s, end_s],
                "base_command": base.tolist(),
            }
        )
        return native, {
            "control_hz": self.control_hz,
            "source_action_hz": self.source_action_hz,
            "plan_count": self.plan_count,
            "feedback_tick": self.feedback_tick,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--source-action-hz", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, required=True)
    parser.add_argument("--replan-period-s", type=float, default=0.4)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    policy = NavigateACTFrequencyPolicy(
        args.checkpoint,
        args.tasks,
        args.device,
        source_action_hz=args.source_action_hz,
        control_hz=args.control_hz,
        replan_period_s=args.replan_period_s,
        trace=args.trace,
    )
    print(
        json.dumps(
            {
                "event": "ACT_FREQUENCY_SERVER_READY",
                "checkpoint": str(args.checkpoint),
                "source_action_hz": args.source_action_hz,
                "control_hz": args.control_hz,
                "replan_period_s": args.replan_period_s,
                "replan_ticks": policy.replan_ticks,
            }
        ),
        flush=True,
    )
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
