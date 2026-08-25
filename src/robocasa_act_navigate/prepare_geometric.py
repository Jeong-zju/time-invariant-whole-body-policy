"""Build fixed-token, common-origin geometric targets from measured base poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .adaptive_path import build_fixed_token_path_target, world_poses_to_common_origin
from .b2_labels import measured_base_poses
from .data import load_data, load_split
from .prepare_b2 import StreamingStats, _state_stats
from .schema import CHUNK_SIZE, NUM_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--translation-tolerance-m", type=float, default=0.01)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.02)
    args = parser.parse_args()

    data = load_data(args.data_root)
    split = load_split(args.split)
    shape = (len(data.actions), CHUNK_SIZE, 3)
    metadata_path = args.cache.with_suffix(args.cache.suffix + ".json")
    if args.cache.exists():
        if not metadata_path.is_file():
            raise FileExistsError(f"cache exists without completion metadata: {args.cache}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("shape") != list(shape):
            raise ValueError("existing geometric cache shape mismatch")
        cache = np.memmap(args.cache, mode="r", dtype=np.float32, shape=shape)
    else:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        cache = np.memmap(args.cache, mode="w+", dtype=np.float32, shape=shape)
        terminals = 0
        maximum_translation_error = 0.0
        maximum_yaw_error = 0.0
        processed = 0
        for episode, (start, end) in data.episode_bounds.items():
            poses = measured_base_poses(data.states[start:end])
            for local_start in range(len(poses)):
                path = world_poses_to_common_origin(poses, local_start)
                target = build_fixed_token_path_target(
                    path,
                    num_tokens=CHUNK_SIZE,
                    translation_tolerance_m=args.translation_tolerance_m,
                    yaw_tolerance_rad=args.yaw_tolerance_rad,
                )
                if target.anchors.shape != (CHUNK_SIZE, 3) or not np.isfinite(target.anchors).all():
                    raise AssertionError("invalid fixed-token target")
                if target.max_translation_error_m > args.translation_tolerance_m + 1e-10:
                    raise AssertionError("translation reconstruction contract violated")
                if target.max_yaw_error_rad > args.yaw_tolerance_rad + 1e-10:
                    raise AssertionError("yaw reconstruction contract violated")
                cache[start + local_start] = target.anchors.astype(np.float32)
                terminals += int(target.terminal)
                maximum_translation_error = max(maximum_translation_error, target.max_translation_error_m)
                maximum_yaw_error = max(maximum_yaw_error, target.max_yaw_error_rad)
                processed += 1
            if episode == 0 or (episode + 1) % 25 == 0 or processed == len(data.actions):
                print(json.dumps({"episodes": episode + 1, "frames": processed}), flush=True)
        cache.flush()
        metadata = {
            "shape": list(shape),
            "dtype": "float32",
            "source": "measured base position and xyzw quaternion",
            "source_frequency_hz": 20,
            "common_origin": "every token is T_t^-1 T_j in the same current-base frame",
            "num_tokens": CHUNK_SIZE,
            "translation_tolerance_m": args.translation_tolerance_m,
            "yaw_tolerance_rad": args.yaw_tolerance_rad,
            "uses_time_frame_velocity_for_selection": False,
            "yaw_parameterization": "continuous_unwrapped_relative_yaw_along_measured_path",
            "terminal_fraction_all_frames": terminals / len(data.actions),
            "max_translation_reconstruction_error_m": maximum_translation_error,
            "max_yaw_reconstruction_error_rad": maximum_yaw_error,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    train_indices = np.concatenate([np.arange(*data.episode_bounds[e]) for e in split["train"]])
    target_stats = StreamingStats(3)
    target_stats.add(np.asarray(cache[train_indices]))
    one_hot = np.eye(NUM_TASKS, dtype=np.float32)[data.task_indices[train_indices]]
    augmented_state = np.concatenate((data.states[train_indices], one_hot), axis=1)
    stats = {
        "task": "NavigateKitchen",
        "representation": "FixedToken_CommonOrigin_MeasuredSE2_Path_RateFree",
        "model": {"chunk_size": CHUNK_SIZE, "action_dim": 3},
        "label_construction": {
            "source_frequency_hz": 20,
            "uses_timestamp": False,
            "uses_frame_count_horizon": False,
            "uses_velocity_or_command_integral": False,
            "selection": "SE2 geometric RDP under physical error bounds; first K future knots",
            "densification": "preserve all retained knots and bisect longest SE2 chords to exactly K",
            "reference": "all K poses share the current measured base frame",
            "yaw_parameterization": "continuous unwrapped relative yaw; no independent +/-pi wrapping per token",
            "predicts_time_duration_velocity_rate_stop_padding": False,
        },
        "execution": {
            "controller_frequency_hz": 30,
            "feedback": "measured pose",
        },
        "features": {
            "observation.state": _state_stats(augmented_state),
            "action": target_stats.result(),
            "image_imagenet": {
                "mean": np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None].tolist(),
                "std": np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None].tolist(),
            },
        },
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    report = {
        "event": "GEOMETRIC_LABEL_CACHE_COMPLETE",
        "data_schema": {"frames": len(data.actions), "episodes": len(data.episode_bounds), "native_hz": 20},
        "train_frames": len(train_indices),
        "cache": metadata,
        "stats_fit": "fixed train split only",
        "action_is_pad": "all false; all 32 geometric tokens are supervised",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
