#!/usr/bin/env python3
"""Run fixed-seed RoboCasa closed-loop rollouts against a remote policy server."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

# The pinned GR00T RoboCasa fork needs an explicit import for this atomic task.
# RoboCasa365 v1.0.1 moved the task and registers all gym IDs from robocasa.__init__.
try:  # pragma: no cover - only exercised by the legacy simulator environment
    from robocasa.environments.kitchen.single_stage.kitchen_pnp import (  # noqa: F401
        PnPCounterToStove,
    )
except ImportError:
    pass

import robocasa  # noqa: F401,E402

try:  # pragma: no cover - registration side effect for the legacy fork
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401,E402
except ImportError:
    pass

from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.eval.sim.wrapper.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.policy.server_client import PolicyClient


def parse_task(value: str) -> tuple[str, str, int, int, int | None]:
    parts = value.split("::")
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            "task must be NAME::GYM_ID::SEED_START::COUNT[::MAX_EPISODE_STEPS]"
        )
    horizon = int(parts[4]) if len(parts) == 5 else None
    return parts[0], parts[1], int(parts[2]), int(parts[3]), horizon


def make_env(
    gym_id: str,
    video_dir: Path,
    max_episode_steps: int,
    n_action_steps: int,
    steps_per_render: int,
    split: str | None,
):
    env_kwargs = {"enable_render": True}
    # RoboCasa365 v1.0.1 exposes exact target-scene splits through the new
    # robocasa/<Task> gym IDs. The legacy Panda-Omron IDs do not accept split.
    if gym_id.startswith("robocasa/") and split is not None:
        env_kwargs["split"] = split
    raw = gym.make(gym_id, **env_kwargs)
    recorded = VideoRecordingWrapper(
        raw,
        VideoRecorder.create_h264(
            fps=20,
            crf=24,
            thread_type="FRAME",
            thread_count=1,
        ),
        video_dir=video_dir,
        steps_per_render=steps_per_render,
        max_episode_steps=max_episode_steps,
        overlay_text=True,
    )
    return MultiStepWrapper(
        recorded,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        terminate_on_success=True,
    )


def batch_observations(observations: list[dict]) -> dict:
    result = {}
    for key in observations[0]:
        values = [item[key] for item in observations]
        if key.startswith("annotation."):
            result[key] = tuple(str(value) for value in values)
        else:
            result[key] = np.stack(values)
    return result


def any_success(info: dict) -> bool:
    value = info.get("success", False)
    return bool(np.any(np.asarray(value)))


def write_report(path: Path, report: dict) -> None:
    successful = [item for item in report["episodes"] if "success" in item]
    report["aggregate"] = {
        "completed": len(successful),
        "successes": int(sum(bool(item["success"]) for item in successful)),
        "success_rate": (
            float(np.mean([bool(item["success"]) for item in successful]))
            if successful
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(report, file, indent=2)


def run_batch(
    client: PolicyClient,
    method: str,
    task_name: str,
    gym_id: str,
    seeds: list[int],
    output_dir: Path,
    max_episode_steps: int,
    n_action_steps: int,
    steps_per_render: int,
    split: str | None,
) -> list[dict]:
    environments, observations, states = [], [], []
    for seed in seeds:
        video_dir = output_dir / "videos" / task_name / f"seed_{seed:04d}"
        env = make_env(
            gym_id,
            video_dir,
            max_episode_steps,
            n_action_steps,
            steps_per_render,
            split,
        )
        observation, _ = env.reset(seed=seed)
        environments.append(env)
        observations.append(observation)
        states.append(
            {
                "seed": seed,
                "success": False,
                "primitive_steps": 0,
                "chunks": 0,
                "start_time": time.time(),
                "done": False,
            }
        )
    client.reset()

    max_chunks = math.ceil(max_episode_steps / n_action_steps) + 1
    while not all(item["done"] for item in states):
        active = [index for index, item in enumerate(states) if not item["done"]]
        action, _ = client.get_action(batch_observations([observations[i] for i in active]))
        for batch_index, env_index in enumerate(active):
            env_action = {
                key: np.asarray(value)[batch_index, :n_action_steps]
                for key, value in action.items()
            }
            observation, _, terminated, truncated, info = environments[env_index].step(
                env_action
            )
            observations[env_index] = observation
            state = states[env_index]
            state["chunks"] += 1
            state["primitive_steps"] += len(info.get("rewards", []))
            state["success"] = bool(state["success"] or any_success(info))
            state["done"] = bool(terminated or truncated or state["success"])
            if state["chunks"] >= max_chunks:
                state["done"] = True

    records = []
    for env, state in zip(environments, states):
        # A reset closes and labels the just-completed recording.
        env.reset()
        env.close()
        seed = int(state["seed"])
        video_dir = output_dir / "videos" / task_name / f"seed_{seed:04d}"
        videos = sorted(str(path) for path in video_dir.glob("*.mp4"))
        records.append(
            {
                "method": method,
                "task": task_name,
                "gym_id": gym_id,
                "seed": seed,
                "success": bool(state["success"]),
                "primitive_steps": int(state["primitive_steps"]),
                "policy_calls": int(state["chunks"]),
                "elapsed_seconds": float(time.time() - state["start_time"]),
                "videos": videos,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["b0", "b1", "b2"], required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--task", action="append", type=parse_task, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--steps-per-render", type=int, default=4)
    parser.add_argument(
        "--split",
        choices=["target", "pretrain", "all", "none"],
        default="target",
    )
    args = parser.parse_args()
    split = None if args.split == "none" else args.split

    output_dir = Path(args.output_dir)
    report_path = output_dir / "results.json"
    report = {
        "schema_version": 2,
        "method": args.method,
        "policy_server": f"{args.policy_host}:{args.policy_port}",
        "fixed_seed_tasks": [
            {
                "task": name,
                "gym_id": gym_id,
                "seed_start": start,
                "count": count,
                "max_episode_steps": horizon or args.max_episode_steps,
            }
            for name, gym_id, start, count, horizon in args.task
        ],
        "default_max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "steps_per_render": args.steps_per_render,
        "split": split,
        "episodes": [],
    }
    write_report(report_path, report)

    client = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=180000,
        strict=False,
    )
    if not client.ping():
        raise RuntimeError("policy server did not answer ping")
    for task_name, gym_id, seed_start, count, task_horizon in args.task:
        max_episode_steps = task_horizon or args.max_episode_steps
        seeds = list(range(seed_start, seed_start + count))
        for offset in range(0, len(seeds), args.batch_size):
            report["episodes"].extend(
                run_batch(
                    client,
                    args.method,
                    task_name,
                    gym_id,
                    seeds[offset : offset + args.batch_size],
                    output_dir,
                    max_episode_steps,
                    args.n_action_steps,
                    args.steps_per_render,
                    split,
                )
            )
            report["episodes"].sort(key=lambda item: (item["task"], item["seed"]))
            write_report(report_path, report)
            print(json.dumps(report["aggregate"], sort_keys=True), flush=True)
    (output_dir / "CLOSED_LOOP_COMPLETE").touch()


if __name__ == "__main__":
    main()
