"""Serve geometric ACT through a one-tick measured-pose path follower."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.data.types import ModalityConfig
from gr00t.policy import BasePolicy
from gr00t.policy.server_client import PolicyServer

from .b2_labels import yaw_from_xyzw
from .eval_server import STATE_SOURCES, VIDEO_SOURCES, _batch_languages, _normalise_language
from .geometric_follower import BaseRateCalibration, MeasuredPosePathFollower
from .policy import make_policy
from .schema import CHUNK_SIZE, NUM_TASKS


class NavigateGeometricPolicy(BasePolicy):
    def __init__(
        self,
        checkpoint: Path,
        tasks: Path,
        calibration: Path,
        device: str,
        control_hz: float = 30.0,
        replan_ticks: int = 12,
        trace: Path | None = None,
    ) -> None:
        super().__init__(strict=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        protocol = payload.get("protocol", {})
        if int(payload.get("step", -1)) != 50_000:
            raise ValueError(f"expected completed step-50000 checkpoint, got {payload.get('step')}")
        if protocol.get("representation") != "fixed_token_common_origin_measured_se2_path":
            raise ValueError("checkpoint is not the fixed-token geometric representation")
        if bool(protocol.get("predicts_time_duration_velocity_rate", True)):
            raise ValueError("geometric checkpoint unexpectedly predicts time/rate")
        self.device = torch.device(device)
        self.policy = make_policy(payload["stats"], device=self.device.type).to(self.device)
        self.policy.load_state_dict(payload["model"], strict=True)
        self.policy.eval()
        calibration_value = BaseRateCalibration.load(calibration)
        if abs(calibration_value.fitted_control_hz - control_hz) > 1e-6:
            raise ValueError(
                f"calibration fitted at {calibration_value.fitted_control_hz} Hz, requested {control_hz} Hz"
            )
        self.follower = MeasuredPosePathFollower(
            calibration_value,
            control_hz=control_hz,
            # Preserve the evaluated 30 Hz limit of 0.15/tick as a fixed
            # continuous-time slew rate across the frequency sweep.
            max_command_delta_per_tick=4.5 / control_hz,
        )
        self.control_hz = float(control_hz)
        self.replan_ticks = int(replan_ticks)
        if self.replan_ticks <= 0:
            raise ValueError("replan_ticks must be positive")
        self.cached_path: np.ndarray | None = None
        self.feedback_tick = 0
        self.plan_count = 0
        self.trace = trace
        self.trace_tick = 0
        if self.trace is not None:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            self.trace.write_text("")
        self.task_to_index: dict[str, int] = {}
        for line in tasks.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["task_index"])
            if index < NUM_TASKS:
                self.task_to_index[_normalise_language(row["task"])] = index
        if sorted(self.task_to_index.values()) != list(range(NUM_TASKS)):
            raise ValueError("task metadata does not cover NavigateKitchen indices 0..12")

    def check_observation(self, observation):
        return None

    def check_action(self, action):
        return None

    def reset(self, options=None):
        self.policy.reset()
        self.follower.clear()
        self.cached_path = None
        self.feedback_tick = 0
        self.plan_count = 0
        self.trace_tick = 0
        self._write_trace({"event": "RESET"})
        return {}

    def _write_trace(self, value: dict[str, Any]) -> None:
        if self.trace is not None:
            with self.trace.open("a") as handle:
                handle.write(json.dumps(value, separators=(",", ":")) + "\n")

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

    @staticmethod
    def _latest(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim < 2:
            raise ValueError(f"expected batched temporal input, got {array.shape}")
        return array[:, -1]

    @torch.inference_mode()
    def _predict(self, observation: dict[str, Any]) -> np.ndarray:
        first = self._latest(observation[next(iter(VIDEO_SOURCES.values()))])
        if len(first) != 1:
            raise ValueError("measured-pose feedback server requires one environment")
        batch: dict[str, torch.Tensor] = {}
        for target, source in VIDEO_SOURCES.items():
            image = self._latest(observation[source])
            # Some transport decoders expose a read-only view.  PyTorch warns
            # that wrapping such a buffer is undefined even when inference is
            # expected not to mutate it, so make ownership explicit here.
            image = np.array(image, dtype=np.uint8, order="C", copy=True)
            batch[target] = (
                torch.from_numpy(image)
                .permute(0, 3, 1, 2)
                .to(self.device, dtype=torch.float32)
                .div_(255.0)
            )
        robot_state = np.concatenate(
            [self._latest(observation[key]).astype(np.float32) for key in STATE_SOURCES], axis=-1
        )
        language = _batch_languages(observation, 1)[0]
        if language not in self.task_to_index:
            raise ValueError(f"unseen task annotation {language!r}")
        one_hot = np.eye(NUM_TASKS, dtype=np.float32)[[self.task_to_index[language]]]
        batch["observation.state"] = torch.from_numpy(
            np.concatenate((robot_state, one_hot), axis=-1)
        ).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.policy.predict_action_chunk(batch)
        path = prediction.float().cpu().numpy()
        if path.shape != (1, CHUNK_SIZE, 3) or not np.isfinite(path).all():
            raise ValueError(f"invalid geometric prediction {path.shape}")
        return path[0]

    def _get_action(self, observation, options=None):
        position = self._latest(observation["state.base_position"])[0]
        quaternion = self._latest(observation["state.base_rotation"])[0]
        replanned = self.cached_path is None or self.feedback_tick >= self.replan_ticks
        if replanned:
            self.cached_path = self._predict(observation)
            self.follower.set_plan(self.cached_path, position, quaternion)
            self.feedback_tick = 0
            self.plan_count += 1
        base, info = self.follower.command(position, quaternion)
        native = {
            "action.base_motion": base[None, None],
            "action.control_mode": np.ones((1, 1, 1), dtype=np.float32),
            "action.end_effector_position": np.zeros((1, 1, 3), dtype=np.float32),
            "action.end_effector_rotation": np.zeros((1, 1, 3), dtype=np.float32),
            "action.gripper_close": -np.ones((1, 1, 1), dtype=np.float32),
        }
        self.feedback_tick += 1
        self.trace_tick += 1
        yaw = float(yaw_from_xyzw(np.asarray(quaternion).reshape(1, 4))[0])
        self._write_trace(
            {
                "event": "ACTION",
                "tick": self.trace_tick,
                "control_hz": self.control_hz,
                "replanned": replanned,
                "plan_count": self.plan_count,
                "feedback_tick": self.feedback_tick,
                "position": np.asarray(position).reshape(-1)[:2].tolist(),
                "yaw": yaw,
                "plan_first": self.cached_path[0].tolist(),
                "plan_endpoint": self.cached_path[-1].tolist(),
                "base_command": base.tolist(),
                "follower": info,
            }
        )
        return native, {
            "control_hz": self.control_hz,
            "plan_count": self.plan_count,
            "feedback_tick": self.feedback_tick,
            "follower": info,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--replan-ticks", type=int, default=12)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    policy = NavigateGeometricPolicy(
        args.checkpoint,
        args.tasks,
        args.calibration,
        args.device,
        args.control_hz,
        args.replan_ticks,
        args.trace,
    )
    print(
        json.dumps(
            {
                "event": "GEOMETRIC_FEEDBACK_SERVER_READY",
                "checkpoint": str(args.checkpoint),
                "control_hz": args.control_hz,
                "replan_ticks": args.replan_ticks,
            }
        ),
        flush=True,
    )
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
