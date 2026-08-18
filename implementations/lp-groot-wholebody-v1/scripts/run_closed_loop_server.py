#!/usr/bin/env python3
"""Serve B0 commands or decoded B1/B2 pose anchors to RoboCasa."""

from __future__ import annotations

import argparse
import importlib.util
from typing import Any

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import PolicyServer
from lpwb.execution import ExecutionCalibration, decode_path_time_commands


VIDEO_SOURCES = {
    "robot0_eye_in_hand": (
        "video.robot0_eye_in_hand",
        "video.res256_image_wrist_0",
    ),
    "robot0_agentview_left": (
        "video.robot0_agentview_left",
        "video.res256_image_side_0",
    ),
    "robot0_agentview_right": (
        "video.robot0_agentview_right",
        "video.res256_image_side_1",
    ),
}
LANGUAGE_SOURCES = (
    "annotation.human.task_description",
    "annotation.human.action.task_description",
)
LANGUAGE_TARGET = "annotation.human.task_description"


def load_module(path: str) -> None:
    spec = importlib.util.spec_from_file_location("lpwb_modality_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _language_item(value: Any) -> str:
    while isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        value = value[0]
    return str(value)


def _first_present(observation: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in observation:
            return observation[key]
    raise KeyError(
        "none of the required observation keys are present: "
        + ", ".join(candidates)
    )


class RoboCasaLpwbPolicy(BasePolicy):
    def __init__(
        self,
        policy: Gr00tPolicy,
        method: str,
        calibration: ExecutionCalibration | None,
        control_dt: float,
    ) -> None:
        super().__init__(strict=False)
        self.policy = policy
        self.method = method
        self.calibration = calibration
        self.control_dt = control_dt

    def check_observation(self, observation: dict[str, Any]) -> None:
        return None

    def check_action(self, action: dict[str, Any]) -> None:
        return None

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def _nested_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        state_keys = self.policy.modality_configs["state"].modality_keys
        language = _first_present(observation, LANGUAGE_SOURCES)
        if isinstance(language, np.ndarray):
            language = list(language)
        return {
            "video": {
                target: np.asarray(_first_present(observation, sources))
                for target, sources in VIDEO_SOURCES.items()
            },
            "state": {
                key: np.asarray(observation[f"state.{key}"], dtype=np.float32)
                for key in state_keys
            },
            "language": {
                LANGUAGE_TARGET: [[_language_item(item)] for item in language]
            },
        }

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        predicted, model_info = self.policy.get_action(
            self._nested_observation(observation), options
        )
        if self.method == "b0":
            native = {key: np.asarray(value, dtype=np.float32) for key, value in predicted.items()}
            return {f"action.{key}": value for key, value in native.items()}, model_info

        if self.calibration is None:
            raise RuntimeError(f"{self.method.upper()} requires an execution calibration")
        batch = len(next(iter(predicted.values())))
        decoded: dict[str, list[np.ndarray]] = {}
        diagnostics = []
        for batch_index in range(batch):
            unbatched = {
                key: np.asarray(value)[batch_index, :32] for key, value in predicted.items()
            }
            commands, item_diagnostics = decode_path_time_commands(
                unbatched,
                self.calibration,
                control_dt=self.control_dt,
                output_horizon=32,
            )
            for key, value in commands.items():
                decoded.setdefault(key, []).append(value)
            diagnostics.append(item_diagnostics)
        native = {
            key: np.stack(value).astype(np.float32) for key, value in decoded.items()
        }
        return (
            {f"action.{key}": value for key, value in native.items()},
            {"model": model_info, "execution": diagnostics},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["b0", "b1", "b2"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality-config-path", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--control-dt", type=float, default=0.05)
    args = parser.parse_args()
    if args.method in ("b1", "b2") and not args.calibration:
        parser.error(f"--calibration is required for {args.method}")

    # The modality registration is shared, but recording the method in the
    # process environment keeps the server launch provenance unambiguous.
    import os

    os.environ["LPWB_METHOD"] = args.method
    load_module(args.modality_config_path)
    model = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.checkpoint,
        device=args.device,
    )
    calibration = (
        ExecutionCalibration.load(args.calibration) if args.calibration else None
    )
    policy = RoboCasaLpwbPolicy(
        model,
        method=args.method,
        calibration=calibration,
        control_dt=args.control_dt,
    )
    PolicyServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
