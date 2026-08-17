#!/usr/bin/env python3
"""Run fixed-seed NavigateKitchen evaluation for an LP base-only checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.lp_execution_policy import LPBaseExecutionPolicy
from lp_groot_base import se2
from robocasa.utils.dataset_registry_utils import get_task_horizon


def adapt_observation(observation: dict) -> dict:
    """Alias RoboCasa365 keys to the LP training modality names."""
    adapted = dict(observation)
    aliases = {
        "video.res256_image_side_0": "video.robot0_agentview_left",
        "video.res256_image_side_1": "video.robot0_agentview_right",
        "video.res256_image_wrist_0": "video.robot0_eye_in_hand",
        "task": "annotation.human.task_description",
    }
    for target, source in aliases.items():
        if target not in adapted:
            if source not in adapted:
                raise KeyError(f"Cannot populate {target}: source observation {source} is missing")
            adapted[target] = adapted[source]
    return adapted


def extract_success(info: dict) -> bool:
    if "success" in info:
        return bool(np.any(info["success"]))
    if "final_info" in info and info["final_info"][0] is not None:
        return bool(np.any(info["final_info"][0].get("success", False)))
    return False


def current_pose(observation: dict) -> np.ndarray:
    position = np.asarray(observation["state.base_position"])[0, -1]
    quaternion = np.asarray(observation["state.base_rotation"])[0, -1]
    return se2.world_xy_quat_to_se2(position[None], quaternion[None])[0]


def run_seed(policy, seed: int, video_dir: Path, horizon: int) -> dict:
    configs = WrapperConfigs(
        video=VideoConfig(video_dir=str(video_dir), max_episode_steps=horizon),
        multistep=MultiStepConfig(
            n_action_steps=8,
            max_episode_steps=horizon,
            terminate_on_success=True,
        ),
    )

    def make_env():
        return create_eval_env(
            env_name="robocasa/NavigateKitchen",
            env_idx=0,
            total_n_envs=1,
            wrapper_configs=configs,
            split="pretrain",
            gpu_id=0,
        )

    env = gym.vector.SyncVectorEnv([make_env])
    observation, _ = env.reset(seed=seed)
    policy.reset()
    success = False
    chunks = 0
    position_errors = []
    yaw_errors = []
    while chunks * 8 < horizon:
        policy_observation = adapt_observation(observation)
        pose_before = current_pose(policy_observation)
        action, policy_info = policy.get_action(policy_observation)
        observation, _, terminated, truncated, env_info = env.step(action)
        chunks += 1
        success = success or extract_success(env_info)
        desired = np.asarray(policy_info["lp_desired_local_pose"])[0, -1]
        if not (terminated[0] or truncated[0]):
            pose_after = current_pose(observation)
            measured = se2.between(pose_before[None], pose_after[None])[0]
            residual = se2.log(se2.between(desired[None], measured[None]))[0]
            position_errors.append(float(np.linalg.norm(residual[:2])))
            yaw_errors.append(float(abs(residual[2])))
        if terminated[0] or truncated[0]:
            break
    env.close()
    return {
        "seed": seed,
        "success": success,
        "control_chunks": chunks,
        "tracking_position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors))))
        if position_errors
        else None,
        "tracking_yaw_rmse_rad": float(np.sqrt(np.mean(np.square(yaw_errors))))
        if yaw_errors
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    if args.episodes != len(args.seeds):
        raise ValueError("--episodes must equal the number of explicit --seeds")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_policy = Gr00tSimPolicyWrapper(
        Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.model_path,
            device=0,
            strict=False,
        ),
        strict=False,
    )
    policy = LPBaseExecutionPolicy(base_policy, args.calibration, num_control_steps=8)
    horizon = get_task_horizon("NavigateKitchen")
    episodes = [
        run_seed(policy, seed, output / "videos" / f"seed_{seed:02d}", horizon)
        for seed in args.seeds
    ]
    report = {
        "task": "NavigateKitchen",
        "split": "pretrain",
        "model_path": args.model_path,
        "seeds": args.seeds,
        "num_episodes": len(episodes),
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "episodes": episodes,
    }
    (output / "closed_loop_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
