"""Serve the trained ACT checkpoint through the GR00T RoboCasa simulator protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.policy import BasePolicy
from gr00t.policy.server_client import PolicyServer
from gr00t.data.types import ModalityConfig

from .policy import make_policy
from .schema import CAMERA_KEYS, NUM_TASKS


VIDEO_SOURCES = {
    "observation.images.robot0_eye_in_hand": "video.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left": "video.robot0_agentview_left",
    "observation.images.robot0_agentview_right": "video.robot0_agentview_right",
}
STATE_SOURCES = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)
LANGUAGE_SOURCES = (
    "annotation.human.task_description",
    "annotation.human.action.task_description",
)


def _normalise_language(value: Any) -> str:
    while isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        value = value[0]
    return str(value).strip().lower().rstrip(".")


def _batch_languages(observation: dict[str, Any], batch: int) -> list[str]:
    for key in LANGUAGE_SOURCES:
        if key in observation:
            value = observation[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)) and len(value) == batch:
                return [_normalise_language(item) for item in value]
            if batch == 1:
                return [_normalise_language(value)]
    raise KeyError(f"missing language key; expected one of {LANGUAGE_SOURCES}")


def base_only_to_native(action: np.ndarray) -> dict[str, np.ndarray]:
    """Expand learned chassis commands into the verified native action schema."""

    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 3 or action.shape[-1] != 3:
        raise ValueError(f"expected base-only action (B,H,3), got {action.shape}")
    prefix = action.shape[:-1]
    zeros_1 = np.zeros((*prefix, 1), dtype=np.float32)
    zeros_3 = np.zeros((*prefix, 3), dtype=np.float32)
    return {
        "action.base_motion": np.concatenate((action, zeros_1), axis=-1),
        "action.control_mode": np.ones((*prefix, 1), dtype=np.float32),
        "action.end_effector_position": zeros_3.copy(),
        "action.end_effector_rotation": zeros_3.copy(),
        "action.gripper_close": -np.ones((*prefix, 1), dtype=np.float32),
    }


class NavigateACTPolicy(BasePolicy):
    def __init__(self, checkpoint: Path, tasks: Path, device: str) -> None:
        super().__init__(strict=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("step", -1)) != 50_000:
            raise ValueError(f"expected checkpoint step 50000, got {payload.get('step')}")
        self.device = torch.device(device)
        self.policy = make_policy(payload["stats"], device=self.device.type).to(self.device)
        self.policy.load_state_dict(payload["model"], strict=True)
        self.policy.eval()
        self.action_dim = int(len(payload["stats"]["features"]["action"]["mean"]))
        if self.action_dim not in (3, 12):
            raise ValueError(f"unsupported learned action dimension: {self.action_dim}")
        self.task_to_index: dict[str, int] = {}
        for line in tasks.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["task_index"])
            if index < NUM_TASKS:
                self.task_to_index[_normalise_language(row["task"])] = index
        if sorted(self.task_to_index.values()) != list(range(NUM_TASKS)):
            raise ValueError("task metadata does not contain exactly indices 0..12")
        self.seen_languages: set[str] = set()

    def check_observation(self, observation: dict[str, Any]) -> None:
        return None

    def check_action(self, action: dict[str, Any]) -> None:
        return None

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.policy.reset()
        return {}

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Declare the exact single-observation / 32-action horizon contract."""
        return {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(VIDEO_SOURCES.values()),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(STATE_SOURCES),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(32)),
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
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        first_video = self._latest(observation[next(iter(VIDEO_SOURCES.values()))])
        batch_size = int(first_video.shape[0])
        batch: dict[str, torch.Tensor] = {}
        for target, source in VIDEO_SOURCES.items():
            image = self._latest(observation[source])
            if image.shape[1:] != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError(f"{source} expected uint8 (B,256,256,3), got {image.dtype} {image.shape}")
            image = np.array(image, dtype=np.uint8, order="C", copy=True)
            batch[target] = (
                torch.from_numpy(image)
                .permute(0, 3, 1, 2)
                .to(self.device, dtype=torch.float32)
                .div_(255.0)
            )
        state_parts = [self._latest(observation[key]).astype(np.float32) for key in STATE_SOURCES]
        robot_state = np.concatenate(state_parts, axis=-1)
        if robot_state.shape != (batch_size, 16):
            raise ValueError(f"expected native state (B,16), got {robot_state.shape}")
        languages = _batch_languages(observation, batch_size)
        task_indices = []
        for language in languages:
            if language not in self.task_to_index:
                raise ValueError(f"unseen NavigateKitchen annotation: {language!r}")
            task_indices.append(self.task_to_index[language])
            if language not in self.seen_languages:
                print(json.dumps({"event": "TASK_CONDITION", "language": language, "task_index": self.task_to_index[language]}), flush=True)
                self.seen_languages.add(language)
        one_hot = np.eye(NUM_TASKS, dtype=np.float32)[np.asarray(task_indices)]
        state = np.concatenate((robot_state, one_hot), axis=-1)
        batch["observation.state"] = torch.from_numpy(state).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.policy.predict_action_chunk(batch)
        action = np.clip(prediction.float().cpu().numpy(), -1.0, 1.0).astype(np.float32)
        if action.shape[1:] != (32, self.action_dim):
            raise ValueError(f"expected ACT output (B,32,{self.action_dim}), got {action.shape}")
        if self.action_dim == 3:
            native = base_only_to_native(action)
        else:
            native = {
                "action.base_motion": action[:, :, 0:4],
                "action.control_mode": action[:, :, 4:5],
                "action.end_effector_position": action[:, :, 5:8],
                "action.end_effector_rotation": action[:, :, 8:11],
                "action.gripper_close": action[:, :, 11:12],
            }
        return native, {"task_indices": task_indices, "chunk_size": 32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5561)
    args = parser.parse_args()
    policy = NavigateACTPolicy(args.checkpoint, args.tasks, args.device)
    print(json.dumps({"event": "ACT_SERVER_READY", "checkpoint": str(args.checkpoint), "port": args.port}), flush=True)
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
