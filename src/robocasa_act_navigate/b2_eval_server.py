"""Serve ACT-B2 through a one-control-step measured-pose feedback follower."""

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

from .b2_execution import BaseCalibration, RateFreeFollower
from .b2_labels import increments_to_absolute_path, yaw_from_xyzw
from .eval_server import VIDEO_SOURCES, STATE_SOURCES, _batch_languages, _normalise_language
from .policy import make_policy
from .schema import NUM_TASKS


class NavigateB2Policy(BasePolicy):
    def __init__(self, checkpoint: Path, tasks: Path, calibration: Path, device: str,
                 replan_steps: int = 8, trace: Path | None = None) -> None:
        super().__init__(strict=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("step", -1)) != 50_000 or payload.get("protocol", {}).get("representation") != "B2_Path_RateFree_SE2Delta_Stop_WholeBody_V4":
            raise ValueError("checkpoint is not the completed corrected B2 model")
        self.device = torch.device(device)
        self.policy = make_policy(payload["stats"], device=self.device.type).to(self.device)
        self.policy.load_state_dict(payload["model"], strict=True); self.policy.eval()
        spatial_step = float(payload["protocol"]["metric"]["spatial_step_m"])
        self.follower = RateFreeFollower(
            BaseCalibration.load(calibration),
            lookahead_distance=spatial_step,
            position_tolerance=min(0.006, 0.30 * spatial_step),
            translation_gain=0.25 / spatial_step,
        )
        self.replan_steps = int(replan_steps)
        self.cached: np.ndarray | None = None
        self.cached_path: np.ndarray | None = None
        self.feedback_step = 0
        self.plan_count = 0
        self.trace = trace
        self.trace_step = 0
        if self.trace is not None:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            self.trace.write_text("")
        self.task_to_index = {}
        for line in tasks.read_text().splitlines():
            if line.strip():
                row = json.loads(line); index = int(row["task_index"])
                if index < NUM_TASKS: self.task_to_index[_normalise_language(row["task"])] = index
        if sorted(self.task_to_index.values()) != list(range(NUM_TASKS)):
            raise ValueError("task metadata does not cover indices 0..12")

    def check_observation(self, observation): return None
    def check_action(self, action): return None
    def reset(self, options=None):
        self.policy.reset(); self.follower.clear(); self.cached = None; self.cached_path = None; self.feedback_step = 0; self.plan_count = 0
        self.trace_step = 0
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
            "action": ModalityConfig(delta_indices=[0], modality_keys=["base_motion", "control_mode", "end_effector_position", "end_effector_rotation", "gripper_close"]),
        }
    @staticmethod
    def _latest(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim < 2: raise ValueError(f"expected batched temporal input, got {array.shape}")
        return array[:, -1]

    @torch.inference_mode()
    def _predict(self, observation: dict[str, Any]) -> np.ndarray:
        first = self._latest(observation[next(iter(VIDEO_SOURCES.values()))])
        if len(first) != 1: raise ValueError("B2 feedback server requires one environment")
        batch = {}
        for target, source in VIDEO_SOURCES.items():
            image = self._latest(observation[source])
            batch[target] = torch.from_numpy(np.ascontiguousarray(image)).permute(0,3,1,2).to(self.device, dtype=torch.float32).div_(255.0)
        robot_state = np.concatenate([self._latest(observation[key]).astype(np.float32) for key in STATE_SOURCES], axis=-1)
        language = _batch_languages(observation, 1)[0]
        if language not in self.task_to_index: raise ValueError(f"unseen task annotation {language!r}")
        state = np.concatenate((robot_state, np.eye(NUM_TASKS, dtype=np.float32)[[self.task_to_index[language]]]), axis=-1)
        batch["observation.state"] = torch.from_numpy(state).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16): prediction = self.policy.predict_action_chunk(batch)
        value = prediction.float().cpu().numpy()
        expected = int(self.policy.config.chunk_size)
        if value.shape != (1,expected,12) or not np.isfinite(value).all(): raise ValueError(f"invalid B2 prediction {value.shape}")
        return value[0]

    def _get_action(self, observation, options=None):
        position = self._latest(observation["state.base_position"])[0]
        quaternion = self._latest(observation["state.base_rotation"])[0]
        replanned = self.cached is None or self.feedback_step >= self.replan_steps
        if replanned:
            self.cached = self._predict(observation); self.feedback_step = 0; self.plan_count += 1
            stop = np.flatnonzero(self.cached[:, 11] < 0.0)
            valid_count = max(1, int(stop[0]) if stop.size else len(self.cached))
            self.cached_path = increments_to_absolute_path(self.cached[:, :3])
            self.follower.set_plan(self.cached_path[:valid_count], position, quaternion)
        base, info = self.follower.command(position, quaternion)
        # Whole-body commands follow attained progress, never the lookahead
        # target used only by the chassis controller.
        anchor = min(int(info["anchor_index"]), len(self.cached) - 1)
        control_mode = 1.0 if self.cached[anchor,3] >= 0.0 else -1.0
        if control_mode < 0.0: base[:] = 0.0
        native = {
            "action.base_motion": base[None,None],
            "action.control_mode": np.asarray([[[control_mode]]], dtype=np.float32),
            "action.end_effector_position": np.clip(self.cached[anchor,4:7],-1,1)[None,None].astype(np.float32),
            "action.end_effector_rotation": np.clip(self.cached[anchor,7:10],-1,1)[None,None].astype(np.float32),
            "action.gripper_close": np.asarray([[[1.0 if self.cached[anchor,10] >= 0.0 else -1.0]]], dtype=np.float32),
        }
        self.feedback_step += 1
        self.trace_step += 1
        yaw = float(yaw_from_xyzw(np.asarray(quaternion).reshape(1, 4))[0])
        self._write_trace({
            "event": "ACTION", "step": self.trace_step, "replanned": replanned,
            "plan_count": self.plan_count, "feedback_step": self.feedback_step,
            "position": np.asarray(position).reshape(-1)[:2].tolist(), "yaw": yaw,
            "plan_first": self.cached_path[0].tolist(), "plan_endpoint": self.cached_path[-1].tolist(),
            "control_mode": control_mode, "base_command": base.tolist(), "follower": info,
        })
        return native, {"plan_count": self.plan_count, "feedback_step": self.feedback_step, "follower": info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=5562)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    policy = NavigateB2Policy(args.checkpoint, args.tasks, args.calibration, args.device, args.replan_steps, args.trace)
    print(json.dumps({"event":"B2_FEEDBACK_SERVER_READY","checkpoint":str(args.checkpoint),"replan_steps":args.replan_steps}), flush=True)
    with PolicyServer(policy, host=args.host, port=args.port) as server: server.run()


if __name__ == "__main__": main()
