#!/usr/bin/env python3
"""Run the official Arena policy loop while recording the policy POV camera."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def frame_to_rgb8(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255)
    return np.ascontiguousarray(array.astype(np.uint8))


def start_encoder(output: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main() -> None:
    output = Path(os.environ["ARENA_G1_VIDEO_OUTPUT"])
    camera_name = os.environ.get("ARENA_G1_VIDEO_CAMERA", "robot_head_cam_rgb")
    fps = int(os.environ.get("ARENA_G1_VIDEO_FPS", "50"))

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()
        policy, num_steps = create_policy(args_cli)
        from isaaclab_arena.metrics.metrics import compute_metrics

        initial_frame = frame_to_rgb8(obs["camera_obs"][camera_name][0])
        height, width = initial_frame.shape[:2]
        encoder = start_encoder(output, width, height, fps)
        assert encoder.stdin is not None

        try:
            encoder.stdin.write(initial_frame.tobytes())
            for _ in tqdm.tqdm(range(num_steps)):
                with torch.inference_mode():
                    actions = policy.get_action(env, obs)
                    obs, _, terminated, truncated, _ = env.step(actions)
                encoder.stdin.write(frame_to_rgb8(obs["camera_obs"][camera_name][0]).tobytes())

                if terminated.any() or truncated.any():
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)

            metrics = compute_metrics(env)
            print(f"Metrics: {metrics}")
        finally:
            encoder.stdin.close()
            return_code = encoder.wait()
            env.close()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exited with code {return_code}")

    print(f"Video: {output}", flush=True)


if __name__ == "__main__":
    main()
