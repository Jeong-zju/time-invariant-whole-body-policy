"""ACT policy adapter for the official BEHAVIOR 2026 websocket evaluator."""

from __future__ import annotations

import logging
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .execution import decode_lp_chunk
from .policy import make_act_policy
from .r1pro import ACTION_DIM, LP_ACTION_DIM, STATE_DIM
from .stats import FeatureStats
from .training_data import CAMERA_KEYS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


LOGGER = logging.getLogger(__name__)

PROPRIO_SUFFIX = "::proprio"
EVAL_CAMERA_SUFFIXES = {
    "observation.rgb.left_realsense_link_camera_0": "left_realsense_link:Camera:0::rgb",
    "observation.rgb.right_realsense_link_camera_0": "right_realsense_link:Camera:0::rgb",
    "observation.rgb.zed_link_camera_0": "zed_link:Camera:0::rgb",
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unique_suffix_key(obs: dict[str, Any], suffix: str) -> str:
    matches = [key for key in obs if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one observation key ending in {suffix!r}, got {matches}; keys={sorted(obs)}")
    return matches[0]


def _prepare_rgb(value: Any) -> torch.Tensor:
    image = _as_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW RGB image, got {image.shape}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3].transpose(2, 0, 1)
    elif image.shape[0] in (3, 4):
        image = image[:3]
    else:
        raise ValueError(f"Could not identify RGB channel axis for image {image.shape}")

    tensor = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32)
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    if tuple(tensor.shape[-2:]) != IMAGE_SIZE:
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True
        ).squeeze(0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def build_act_observation_batch(
    obs: dict[str, Any], state_stats: FeatureStats, *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Map flattened official evaluator observations to the training feature schema."""

    proprio_key = _unique_suffix_key(obs, PROPRIO_SUFFIX)
    state = _as_numpy(obs[proprio_key]).astype(np.float32, copy=False).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected {STATE_DIM} proprio values at {proprio_key}, got {state.shape}")

    batch = {
        "observation.state": torch.from_numpy(
            state_stats.normalize(state).astype(np.float32, copy=False)
        ).unsqueeze(0)
    }
    for training_key in CAMERA_KEYS:
        evaluator_key = _unique_suffix_key(obs, EVAL_CAMERA_SUFFIXES[training_key])
        batch[training_key] = _prepare_rgb(obs[evaluator_key]).unsqueeze(0)
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_lp_execution_origin(state: np.ndarray, lp_chunk: np.ndarray) -> np.ndarray:
    """Build LP decoder origin from measured joints and predicted gripper convention.

    The evaluator exposes two physical finger qpos values per gripper, whereas
    the controller and training target use one normalized command. The first
    predicted gripper target is therefore the only like-for-like origin value.
    """

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    lp_chunk = np.asarray(lp_chunk, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected state ({STATE_DIM},), got {state.shape}")
    if lp_chunk.shape != (32, LP_ACTION_DIM):
        raise ValueError(f"Expected LP chunk (32, {LP_ACTION_DIM}), got {lp_chunk.shape}")
    return np.concatenate(
        [
            state[53:57],
            state[3:10],
            lp_chunk[0, 14:15],
            state[28:35],
            lp_chunk[0, 22:23],
        ]
    )


class ACTEvalPolicy:
    """Load standard ACT or LP-ACT and execute a rolling 30 Hz prefix."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda:0",
        execute_prefix: int = 6,
        diagnostic_dir: str | Path | None = None,
        mode: str = "standard",
    ):
        if mode not in {"standard", "lp"}:
            raise ValueError("mode must be 'standard' or 'lp'")
        if execute_prefix < 1 or execute_prefix > 32:
            raise ValueError("execute_prefix must be in [1, 32]")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.execute_prefix = execute_prefix
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir is not None else None
        self.mode = mode

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("mode") != mode:
            raise ValueError(f"Expected a {mode!r} checkpoint, got mode={checkpoint.get('mode')!r}")
        self.checkpoint_step = int(checkpoint["step"])
        self.state_stats = FeatureStats.from_dict(checkpoint["stats"]["observation.state"])
        stats_key = "standard_action" if mode == "standard" else "lp_action"
        self.action_stats = FeatureStats.from_dict(checkpoint["stats"][stats_key])

        self.policy = make_act_policy(mode).to(self.device)
        self.policy.load_state_dict(checkpoint["model"], strict=True)
        self.policy.eval()
        self._actions: deque[np.ndarray] = deque()
        self._inference_count = 0
        self._observation_schema_logged = False
        self._diagnostic_saved = False
        LOGGER.info(
            "Loaded %s ACT checkpoint %s at step %d; execute_prefix=%d device=%s",
            self.mode,
            self.checkpoint_path,
            self.checkpoint_step,
            self.execute_prefix,
            self.device,
        )

    def reset(self) -> None:
        self._actions.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @torch.inference_mode()
    def _predict(self, obs: dict[str, Any]) -> None:
        if not self._observation_schema_logged:
            LOGGER.info(
                "Official evaluator observation schema: %s",
                {key: tuple(_as_numpy(value).shape) for key, value in sorted(obs.items())},
            )
            self._observation_schema_logged = True
        batch = build_act_observation_batch(obs, self.state_stats, device=self.device)
        started = time.perf_counter()
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            normalized_chunk = self.policy.predict_action_chunk(batch)
        model_chunk = self.action_stats.denormalize(normalized_chunk[0].float().cpu().numpy()).astype(np.float32)
        expected_dim = ACTION_DIM if self.mode == "standard" else LP_ACTION_DIM
        if model_chunk.shape != (32, expected_dim):
            raise ValueError(f"Expected {self.mode} prediction (32, {expected_dim}), got {model_chunk.shape}")
        if not np.isfinite(model_chunk).all():
            raise FloatingPointError("ACT produced non-finite actions")

        if self.mode == "standard":
            action_chunk = model_chunk.copy()
        else:
            proprio_key = _unique_suffix_key(obs, PROPRIO_SUFFIX)
            state = _as_numpy(obs[proprio_key]).astype(np.float32, copy=False).reshape(-1)
            origin_nonbase = build_lp_execution_origin(state, model_chunk)
            action_chunk = decode_lp_chunk(model_chunk, origin_nonbase).actions
            if action_chunk.shape[0] == 0:
                raise ValueError("LP-ACT decoder produced an empty action chunk")

        # Match the official R1Pro controller's raw action limits. Arm and trunk
        # targets remain unnormalized absolute joint positions, as in the demos.
        action_chunk[:, 0:2] = np.clip(action_chunk[:, 0:2], -0.75, 0.75)
        action_chunk[:, 2] = np.clip(action_chunk[:, 2], -1.0, 1.0)
        action_chunk[:, 14] = np.clip(action_chunk[:, 14], -1.0, 1.0)
        action_chunk[:, 22] = np.clip(action_chunk[:, 22], -1.0, 1.0)
        if self.diagnostic_dir is not None and not self._diagnostic_saved:
            self._save_first_inference(obs, batch, normalized_chunk[0], model_chunk, action_chunk)
        self._actions.extend(action_chunk[: self.execute_prefix])
        self._inference_count += 1
        LOGGER.info(
            "ACT inference %d: %.1f ms, queued=%d, first_base=%s",
            self._inference_count,
            1000.0 * (time.perf_counter() - started),
            len(self._actions),
            np.array2string(action_chunk[0, :3], precision=4),
        )

    def _save_first_inference(
        self,
        obs: dict[str, Any],
        batch: dict[str, torch.Tensor],
        normalized_chunk: torch.Tensor,
        model_chunk: np.ndarray,
        action_chunk: np.ndarray,
    ) -> None:
        assert self.diagnostic_dir is not None
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        proprio_key = _unique_suffix_key(obs, PROPRIO_SUFFIX)
        arrays: dict[str, np.ndarray] = {
            "raw_state": _as_numpy(obs[proprio_key]),
            "normalized_state": batch["observation.state"][0].float().cpu().numpy(),
            "normalized_action_chunk": normalized_chunk.float().cpu().numpy(),
            "model_chunk": model_chunk,
            "action_chunk": action_chunk,
        }
        image_summary = {}
        for training_key in CAMERA_KEYS:
            evaluator_key = _unique_suffix_key(obs, EVAL_CAMERA_SUFFIXES[training_key])
            raw = _as_numpy(obs[evaluator_key])
            slug = training_key.rsplit(".", 1)[-1]
            arrays[f"raw_{slug}"] = raw
            arrays[f"normalized_{slug}"] = batch[training_key][0].float().cpu().numpy()
            image_summary[evaluator_key] = {
                "dtype": str(raw.dtype),
                "shape": list(raw.shape),
                "min": float(raw.min()),
                "max": float(raw.max()),
                "mean": float(raw.mean()),
            }
        np.savez_compressed(self.diagnostic_dir / "first_inference.npz", **arrays)
        state_z = arrays["normalized_state"]
        summary = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_step": self.checkpoint_step,
            "mode": self.mode,
            "execute_prefix": self.execute_prefix,
            "proprio_key": proprio_key,
            "raw_state_min": float(arrays["raw_state"].min()),
            "raw_state_max": float(arrays["raw_state"].max()),
            "normalized_state_abs_max": float(np.abs(state_z).max()),
            "normalized_state_abs_gt_3": int((np.abs(state_z) > 3.0).sum()),
            "normalized_state_abs_gt_5": int((np.abs(state_z) > 5.0).sum()),
            "first_action": action_chunk[0].tolist(),
            "images": image_summary,
        }
        with (self.diagnostic_dir / "first_inference.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self._diagnostic_saved = True
        LOGGER.info("Saved first-inference diagnostic to %s", self.diagnostic_dir)

    def act(self, obs: dict[str, Any]) -> torch.Tensor:
        if not self._actions:
            self._predict(obs)
        action = self._actions.popleft()
        if action.shape != (ACTION_DIM,):
            raise AssertionError(f"Internal action shape mismatch: {action.shape}")
        return torch.from_numpy(action.copy())
