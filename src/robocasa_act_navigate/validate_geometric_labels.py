"""Independent offline gates and visualizations for fixed-token path labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .adaptive_path import (
    build_fixed_token_path_target,
    world_poses_to_common_origin,
    wrap_angle,
)
from .b2_labels import measured_base_poses
from .data import load_data, load_split
from .schema import CHUNK_SIZE


def local_to_world(local: np.ndarray, origin: np.ndarray) -> np.ndarray:
    local = np.asarray(local, dtype=np.float64)
    cosine, sine = np.cos(origin[2]), np.sin(origin[2])
    return np.column_stack(
        (
            origin[0] + cosine * local[:, 0] - sine * local[:, 1],
            origin[1] + sine * local[:, 0] + cosine * local[:, 1],
            wrap_angle(origin[2] + local[:, 2]),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-tolerance-m", type=float, default=0.01)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    data = load_data(args.data_root)
    split = load_split(args.split)
    cache = np.memmap(
        args.cache,
        mode="r",
        dtype=np.float32,
        shape=(len(data.actions), CHUNK_SIZE, 3),
    )
    train_indices = np.concatenate([np.arange(*data.episode_bounds[e]) for e in split["train"]])
    generator = np.random.default_rng(args.seed)
    chosen = np.sort(
        generator.choice(train_indices, min(args.samples, len(train_indices)), replace=False)
    )
    episode_pose_cache = {
        episode: measured_base_poses(data.states[start:end])
        for episode, (start, end) in data.episode_bounds.items()
        if episode in set(split["train"])
    }
    max_cache_error = 0.0
    max_roundtrip_translation = 0.0
    max_roundtrip_yaw = 0.0
    max_source_translation = 0.0
    max_source_yaw = 0.0
    terminal_count = 0
    examples = []
    for ordinal, index in enumerate(chosen):
        episode = int(data.episode_indices[index])
        episode_start, _ = data.episode_bounds[episode]
        poses = episode_pose_cache[episode]
        local_start = int(index - episode_start)
        raw = world_poses_to_common_origin(poses, local_start)
        rebuilt = build_fixed_token_path_target(
            raw,
            num_tokens=CHUNK_SIZE,
            translation_tolerance_m=args.translation_tolerance_m,
            yaw_tolerance_rad=args.yaw_tolerance_rad,
        )
        cached = np.asarray(cache[index], dtype=np.float64)
        max_cache_error = max(max_cache_error, float(np.max(np.abs(cached - rebuilt.anchors))))
        origin = poses[local_start]
        world = local_to_world(cached, origin)
        roundtrip = world_poses_to_common_origin(np.vstack((origin, world)))[1:]
        max_roundtrip_translation = max(
            max_roundtrip_translation,
            float(np.max(np.linalg.norm(roundtrip[:, :2] - cached[:, :2], axis=-1))),
        )
        max_roundtrip_yaw = max(
            max_roundtrip_yaw,
            float(np.max(np.abs(wrap_angle(roundtrip[:, 2] - cached[:, 2])))),
        )
        max_source_translation = max(max_source_translation, rebuilt.max_translation_error_m)
        max_source_yaw = max(max_source_yaw, rebuilt.max_yaw_error_rad)
        terminal_count += int(rebuilt.terminal)
        if len(examples) < 12 and (ordinal % max(1, len(chosen) // 12) == 0):
            examples.append((int(index), raw, cached, rebuilt.covered_source_index))

    if max_cache_error > 1e-5:
        raise AssertionError(f"cached labels differ from independent rebuild: {max_cache_error}")
    if max_roundtrip_translation > 1e-6 or max_roundtrip_yaw > 1e-6:
        raise AssertionError("common-origin world/local roundtrip failed")
    if max_source_translation > args.translation_tolerance_m + 1e-10:
        raise AssertionError("source translation reconstruction gate failed")
    if max_source_yaw > args.yaw_tolerance_rad + 1e-10:
        raise AssertionError("source yaw reconstruction gate failed")

    figure, axes = plt.subplots(3, 4, figsize=(16, 12), squeeze=False)
    for axis, (index, raw, anchors, covered) in zip(axes.flat, examples):
        source = raw[: covered + 1]
        axis.plot(source[:, 0], source[:, 1], color="0.55", linewidth=1.5, label="measured 20 Hz")
        axis.plot(anchors[:, 0], anchors[:, 1], "o-", markersize=2.5, linewidth=1.0, label="32 geometric tokens")
        axis.scatter([0.0], [0.0], marker="*", s=80, color="black", label="current base")
        axis.set_title(f"global frame {index}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("NavigateKitchen common-origin geometric labels")
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    plot = args.output_dir / "label_examples.png"
    figure.savefig(plot, dpi=160)
    plt.close(figure)

    report = {
        "event": "GEOMETRIC_OFFLINE_GATES_PASSED",
        "samples": len(chosen),
        "split": "train only",
        "native_label_source_hz": 20,
        "controller_hz_is_not_used_in_labels": True,
        "common_origin_roundtrip": {
            "max_translation_error_m": max_roundtrip_translation,
            "max_yaw_error_rad": max_roundtrip_yaw,
        },
        "independent_cache_rebuild_max_abs_error": max_cache_error,
        "source_curve_reconstruction": {
            "max_translation_error_m": max_source_translation,
            "max_yaw_error_rad": max_source_yaw,
            "translation_bound_m": args.translation_tolerance_m,
            "yaw_bound_rad": args.yaw_tolerance_rad,
        },
        "terminal_fraction": terminal_count / len(chosen),
        "all_tokens_supervised": True,
        "predicts_time_duration_velocity_rate": False,
        "visualization": str(plot),
    }
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "OFFLINE_GATES_PASSED").write_text("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
