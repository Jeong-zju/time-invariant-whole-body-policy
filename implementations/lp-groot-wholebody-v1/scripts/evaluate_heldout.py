#!/usr/bin/env python3
"""Matched deterministic open-loop evaluation on held-out RoboCasa episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from lpwb.labels import (
    LabelConfig,
    build_path_time_label,
    build_pose_time_label,
    event_signature,
)


def selected_validation_episodes(total: int, seed: int, train_fraction: float) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(total)
    boundary = int(np.floor(train_fraction * total))
    return np.sort(order[boundary:])


def observation_at(loader: LeRobotEpisodeLoader, episode_index: int, step: int) -> tuple[dict, object]:
    meta = loader.episodes_metadata[episode_index]
    episode_id = int(meta["episode_index"])
    frame = loader._load_parquet_data(episode_id)
    video = loader._load_video_data(episode_id, np.asarray([step], dtype=np.int64))
    state = {
        key: np.vstack([frame[f"state.{key}"].iloc[step]]).astype(np.float32)
        for key in loader.modality_configs["state"].modality_keys
    }
    language_key = loader.modality_configs["language"].modality_keys[0]
    text = frame[f"language.{language_key}"].iloc[step]
    observation = {
        "video": {key: np.asarray(values)[None, ...] for key, values in video.items()},
        "state": {key: values[None, ...] for key, values in state.items()},
        "language": {language_key: [[text]]},
    }
    return observation, frame


def b0_target(frame, start: int, action_keys: list[str]) -> dict[str, np.ndarray]:
    return {
        key: np.vstack(frame[f"action.{key}"].iloc[start : start + 32]).astype(np.float32)
        for key in action_keys
    }


def pose_target(frame, start: int, method: str) -> dict[str, np.ndarray]:
    state = np.column_stack(
        [
            np.vstack(frame["state.base_position"].iloc[start : start + 33]),
            np.vstack(frame["state.base_rotation"].iloc[start : start + 33]),
            np.vstack(
                frame["state.end_effector_position_relative"].iloc[start : start + 33]
            ),
            np.vstack(
                frame["state.end_effector_rotation_relative"].iloc[start : start + 33]
            ),
            np.vstack(frame["state.gripper_qpos"].iloc[start : start + 33]),
        ]
    )
    action = np.column_stack(
        [
            np.vstack(frame["action.base_motion"].iloc[start : start + 32]),
            np.vstack(frame["action.control_mode"].iloc[start : start + 32]),
            np.vstack(frame["action.end_effector_position"].iloc[start : start + 32]),
            np.vstack(frame["action.end_effector_rotation"].iloc[start : start + 32]),
            np.vstack(frame["action.gripper_close"].iloc[start : start + 32]),
        ]
    )
    timestamp = frame["lpwb.timestamp"].iloc[start : start + 33].to_numpy(dtype=np.float64)
    builder = build_pose_time_label if method == "b1" else build_path_time_label
    return builder(state, action, timestamp, LabelConfig()).action_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["b0", "b1", "b2"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--samples-per-task", type=int, default=10)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.checkpoint,
        device="cuda",
    )
    modality = policy.get_modality_config()
    all_records = []
    for dataset_string in args.dataset:
        dataset = Path(dataset_string)
        task = dataset.parent.name
        loader_class = LeRobotEpisodeLoader
        if args.method in ("b1", "b2"):
            from lpwb_dataset import PathTimeEpisodeLoader

            loader_class = PathTimeEpisodeLoader
        loader = loader_class(dataset_path=dataset, modality_configs=modality)
        episodes = selected_validation_episodes(len(loader), args.seed, 0.9)[
            : args.samples_per_task
        ]
        rng = np.random.default_rng(args.seed + len(task))
        for episode_index in episodes:
            valid = loader.get_episode_length(int(episode_index)) - 32
            start = int(rng.integers(0, valid))
            observation, frame = observation_at(loader, int(episode_index), start)
            predicted_batched, _ = policy.get_action(observation)
            predicted = {key: np.asarray(value)[0, :32] for key, value in predicted_batched.items()}
            target = (
                b0_target(frame, start, list(modality["action"].modality_keys))
                if args.method == "b0"
                else pose_target(frame, start, args.method)
            )
            group_metrics = {}
            for key in modality["action"].modality_keys:
                error = predicted[key] - target[key]
                group_metrics[key] = {
                    "mse": float(np.mean(np.square(error))),
                    "mae": float(np.mean(np.abs(error))),
                }
            record = {
                "task": task,
                "episode_index": int(episode_index),
                "start": start,
                "groups": group_metrics,
            }
            if args.method in ("b1", "b2"):
                record["path_metrics"] = {
                    "base_endpoint_translation_error_m": float(
                        np.linalg.norm(
                            predicted["base_motion"][-1, :2] - target["base_motion"][-1, :2]
                        )
                    ),
                    "base_endpoint_yaw_error_rad": float(
                        abs(predicted["base_motion"][-1, 2] - target["base_motion"][-1, 2])
                    ),
                    "eef_endpoint_translation_error_m": float(
                        np.linalg.norm(
                            predicted["end_effector_position"][-1]
                            - target["end_effector_position"][-1]
                        )
                    ),
                    "duration_sum_error_s": float(
                        abs(
                            np.exp(predicted["base_motion"][:, 3]).sum()
                            - np.exp(target["base_motion"][:, 3]).sum()
                        )
                    ),
                    "gripper_signature_match": event_signature(predicted["gripper_close"] > 0.0)
                    == event_signature(target["gripper_close"] > 0.0),
                }
            all_records.append(record)

    aggregate = {}
    for task in sorted({record["task"] for record in all_records}):
        task_records = [record for record in all_records if record["task"] == task]
        aggregate[task] = {}
        for key in modality["action"].modality_keys:
            aggregate[task][key] = {
                metric: float(np.mean([record["groups"][key][metric] for record in task_records]))
                for metric in ["mse", "mae"]
            }
        if args.method in ("b1", "b2"):
            path_keys = task_records[0]["path_metrics"].keys()
            aggregate[task]["path_metrics"] = {}
            for key in path_keys:
                values = [record["path_metrics"][key] for record in task_records]
                aggregate[task]["path_metrics"][key] = (
                    bool(all(values)) if isinstance(values[0], bool) else float(np.mean(values))
                )

    report = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "aggregate": aggregate,
        "records": all_records,
    }
    with (output_dir / f"{args.method}_heldout.json").open("w") as file:
        json.dump(report, file, indent=2)

    groups = list(modality["action"].modality_keys)
    tasks = list(aggregate)
    x = np.arange(len(groups))
    width = 0.8 / len(tasks)
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, task in enumerate(tasks):
        ax.bar(
            x + (index - (len(tasks) - 1) / 2) * width,
            [aggregate[task][group]["mae"] for group in groups],
            width,
            label=task,
        )
    ax.set_xticks(x, groups, rotation=20, ha="right")
    ax.set_ylabel("unnormalized held-out MAE")
    ax.set_title(f"{args.method.upper()} held-out open-loop error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{args.method}_heldout.png", dpi=150)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
