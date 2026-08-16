#!/usr/bin/env python3
"""Build the measured-odometry Phase 1 dataset from frozen Arena artifacts.

The official LeRobot release contains images, language, joint state and command
columns, but omits the measured root trajectory.  The corresponding generated
HDF5 contains aligned ``obs/robot_pos`` and ``obs/robot_quat`` telemetry.  This
tool joins the two without duplicating video, emits auditable per-episode NPZ
files, evaluates the command-integration proxy, and writes a derived LeRobot
dataset whose target statistics describe the *post-anchor-transform* M1 action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from whole_body_policy import (  # noqa: E402
    build_multi_horizon_targets,
    compare_base_proxy_to_odometry,
    save_target_batch,
)


QUERY_TIMES_S = np.arange(1, 17, dtype=np.float64) * 0.02
EXPECTED_FULL_HDF5_BYTES = 23_362_617_731
EXPECTED_FULL_HDF5_SHA256 = "2732fc655035408217529d0ff8146737b0c619fa917e14b3568c1da06d25b05b"

ANCHOR_COLUMN = "phase1.anchor_base_pose_se2"
UPPER_COLUMN = "phase1.target_upper_body_position"
HEIGHT_COLUMN = "phase1.target_base_height"
BASE_COLUMN = "phase1.target_base_pose_se2_global"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid statistics array {array.shape}")
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def _load_or_calculate_source_statistics(source: Path) -> dict[str, Any]:
    """Load LeRobot statistics, or reproduce GR00T's official calculation.

    Arena's frozen HDF5 converter emits the parquet and modality metadata but
    does not emit ``meta/stats.json``.  Computing from the aligned conversion
    is required; borrowing stats from the unrelated pre-converted release
    would reintroduce the publication mismatch this pipeline is designed to
    prevent.
    """
    stats_path = source / "meta" / "stats.json"
    if stats_path.is_file():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    parquet_paths = sorted(source.glob("data/*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no source parquet files under {source}")
    table = pd.concat((pd.read_parquet(path) for path in parquet_paths), ignore_index=True)
    statistics: dict[str, Any] = {}
    for column in table.columns:
        first = table[column].iloc[0]
        if isinstance(first, str):
            continue
        values = np.vstack([np.asarray(value, dtype=np.float32) for value in table[column]])
        statistics[column] = _statistics(values)
    return statistics


def _yaw_from_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"quaternion must have shape (N, 4), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("zero-norm root quaternion")
    w, x, y, z = (quaternion / norm).T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _upper_body_from_policy_state(policy_state: np.ndarray) -> np.ndarray:
    state = np.asarray(policy_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 43:
        raise ValueError(f"policy state must have shape (N, 43), got {state.shape}")
    # Frozen GR00T action order: left arm, right arm, left hand, right hand.
    return np.concatenate(
        (state[:, 15:22], state[:, 29:36], state[:, 22:29], state[:, 36:43]),
        axis=1,
    )


def _prepare_output(source: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty derived dataset: {output}")
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "meta", output / "meta", dirs_exist_ok=True)
    video_link = output / "videos"
    if not video_link.exists():
        video_link.symlink_to(os.path.relpath(source / "videos", output), target_is_directory=True)


def _update_metadata(
    *,
    source: Path,
    output: Path,
    target_values: np.ndarray,
    anchor_values: np.ndarray,
    total_valid_anchors: int,
) -> None:
    modality_path = output / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    modality["state"]["phase1_anchor_base_pose_se2"] = {
        "start": 0,
        "end": 3,
        "absolute": True,
        "dtype": "float32",
        "original_key": ANCHOR_COLUMN,
    }
    modality["action"]["phase1_upper_body_position"] = {
        "start": 0,
        "end": 28,
        "absolute": True,
        "dtype": "float32",
        "original_key": UPPER_COLUMN,
    }
    modality["action"]["phase1_base_height"] = {
        "start": 0,
        "end": 1,
        "absolute": True,
        "dtype": "float32",
        "original_key": HEIGHT_COLUMN,
    }
    modality["action"]["phase1_base_relative_se2"] = {
        "start": 0,
        "end": 3,
        "absolute": True,
        "dtype": "float32",
        "original_key": BASE_COLUMN,
    }
    _write_json(modality_path, modality)

    stats_path = output / "meta" / "stats.json"
    stats = _load_or_calculate_source_statistics(source)
    stats[ANCHOR_COLUMN] = _statistics(anchor_values)
    stats[UPPER_COLUMN] = _statistics(target_values[:, :28])
    stats[HEIGHT_COLUMN] = _statistics(target_values[:, 28:29])
    # The parquet column stores world poses.  The first transform converts them
    # to anchor-relative poses, so these are deliberately post-transform stats.
    stats[BASE_COLUMN] = _statistics(target_values[:, 29:32])
    _write_json(stats_path, stats)

    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][ANCHOR_COLUMN] = {"dtype": "float32", "shape": [3]}
    info["features"][UPPER_COLUMN] = {"dtype": "float32", "shape": [28]}
    info["features"][HEIGHT_COLUMN] = {"dtype": "float32", "shape": [1]}
    info["features"][BASE_COLUMN] = {"dtype": "float32", "shape": [3]}
    info["phase1_valid_anchor_count"] = int(total_valid_anchors)
    info["phase1_query_times_s"] = QUERY_TIMES_S.tolist()
    info["phase1_terminal_policy"] = "drop_anchor_in_custom_dataset_wrapper"
    info["phase1_statistics_semantics"] = "post_anchor_relative_transform"
    _write_json(info_path, info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--source-lerobot", type=Path, required=True)
    parser.add_argument("--output-lerobot", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--max-xy-p95-m", type=float, default=0.05)
    parser.add_argument("--max-yaw-p95-rad", type=float, default=0.08)
    parser.add_argument("--skip-hdf5-sha256", action="store_true")
    args = parser.parse_args()

    if args.hdf5.stat().st_size != EXPECTED_FULL_HDF5_BYTES:
        raise ValueError(
            f"unexpected full HDF5 size {args.hdf5.stat().st_size}; "
            f"expected {EXPECTED_FULL_HDF5_BYTES}"
        )
    hdf5_sha256 = None if args.skip_hdf5_sha256 else _sha256(args.hdf5)
    if hdf5_sha256 is not None and hdf5_sha256 != EXPECTED_FULL_HDF5_SHA256:
        raise ValueError(f"full HDF5 SHA256 mismatch: {hdf5_sha256}")

    episodes = [
        json.loads(line)
        for line in (args.source_lerobot / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(episodes) != 100:
        raise ValueError(f"expected 100 source episodes, got {len(episodes)}")
    _prepare_output(args.source_lerobot, args.output_lerobot)

    replay_root = args.artifact_root / "replay"
    gate_root = args.artifact_root / "data-gate"
    target_root = args.artifact_root / "targets"
    target_rows: list[np.ndarray] = []
    anchor_rows: list[np.ndarray] = []
    gate_reports: list[dict[str, Any]] = []
    episode_manifests: list[dict[str, Any]] = []

    with h5py.File(args.hdf5, "r") as hdf5_file:
        data_group = hdf5_file["data"]
        trajectory_ids = list(data_group.keys())
        if len(trajectory_ids) != len(episodes):
            raise ValueError(
                f"HDF5 trajectories {len(trajectory_ids)} != source episodes {len(episodes)}"
            )
        for position, episode in enumerate(episodes):
            episode_index = int(episode["episode_index"])
            parquet_rel = Path("data") / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            frame = pd.read_parquet(args.source_lerobot / parquet_rel)
            frame_count = len(frame)
            if frame_count != int(episode["length"]):
                raise ValueError(
                    f"episode {episode_index}: parquet rows {frame_count} != metadata {episode['length']}"
                )

            # Arena's frozen official converter enumerates h5py keys in this
            # exact order (lexicographic for this file), not numeric demo order.
            trajectory_id = trajectory_ids[position]
            group = data_group[trajectory_id]
            hdf5_frame_count = len(group["obs/robot_pos"]) - 1
            if hdf5_frame_count != frame_count:
                raise ValueError(
                    f"episode {episode_index} ({trajectory_id}): HDF5 usable frames "
                    f"{hdf5_frame_count} != LeRobot frames {frame_count}; source is not aligned"
                )
            robot_position = np.asarray(group["obs/robot_pos"][:frame_count], dtype=np.float64)
            robot_quaternion = np.asarray(group["obs/robot_quat"][:frame_count], dtype=np.float64)
            if robot_position.shape != (frame_count, 3) or robot_quaternion.shape != (frame_count, 4):
                raise ValueError(
                    f"episode {episode_index}: invalid root telemetry "
                    f"{robot_position.shape}, {robot_quaternion.shape}"
                )
            base_pose = np.column_stack(
                (robot_position[:, 0], robot_position[:, 1], _yaw_from_wxyz(robot_quaternion))
            )
            timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
            if np.any(np.diff(timestamps) <= 0.0):
                raise ValueError(f"episode {episode_index}: non-increasing timestamps")
            policy_state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
            upper = _upper_body_from_policy_state(policy_state)
            base_height = np.stack(frame["teleop.base_height_command"].to_numpy()).astype(np.float64)
            base_twist = np.stack(frame["teleop.navigate_command"].to_numpy()).astype(np.float64)
            expected_height = np.asarray(group["action/base_height_cmd"][:-1], dtype=np.float64)
            expected_twist = np.asarray(group["action/navigate_cmd"][:-1], dtype=np.float64)
            if not np.allclose(base_height, expected_height, rtol=0.0, atol=1e-6):
                raise ValueError(f"episode {episode_index} ({trajectory_id}): base-height alignment failed")
            if not np.allclose(base_twist, expected_twist, rtol=0.0, atol=1e-6):
                raise ValueError(f"episode {episode_index} ({trajectory_id}): navigation alignment failed")

            replay_path = replay_root / f"episode-{episode_index:03d}.npz"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_metadata = {
                "schema_version": 1,
                "protocol_id": "arena-g1-phase-1-m1-v1",
                "episode_index": episode_index,
                "hdf5_trajectory_id": trajectory_id,
                "timestamp_source": "frozen_lerobot_sim_timestamp_50hz",
                "upper_body_source": "measured_observation_joint_position",
                "base_height_source": "expert_base_height_command_proxy",
                "base_pose_source": "expert_hdf5_observation_root_telemetry",
                "base_twist_source": "expert_navigate_command",
            }
            np.savez_compressed(
                replay_path,
                timestamps_s=timestamps,
                upper_body_position=upper.astype(np.float32),
                base_height=base_height.astype(np.float32),
                base_pose_se2=base_pose.astype(np.float32),
                base_twist_body=base_twist.astype(np.float32),
                episode_index=np.asarray(episode_index, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(replay_metadata)),
            )

            proxy_metrics = compare_base_proxy_to_odometry(
                timestamps_s=timestamps,
                body_twist=base_twist,
                measured_base_pose_se2=base_pose,
                horizon_s=0.32,
            )
            proxy_passed = bool(
                proxy_metrics["xy_p95_m"] <= args.max_xy_p95_m
                and proxy_metrics["yaw_p95_rad"] <= args.max_yaw_p95_rad
            )
            gate_report = {
                "schema_version": 1,
                "protocol_id": "arena-g1-phase-1-m1-v1",
                "episode_index": episode_index,
                "input": str(replay_path),
                "metrics": proxy_metrics,
                "thresholds": {
                    "maximum_xy_p95_m": args.max_xy_p95_m,
                    "maximum_yaw_p95_rad": args.max_yaw_p95_rad,
                },
                "proxy_data_gate_passed": proxy_passed,
                "decision": (
                    "allow_proxy_labeled_m1_smoke"
                    if proxy_passed
                    else "use_captured_measured_root_odometry"
                ),
            }
            _write_json(gate_root / f"episode-{episode_index:03d}.json", gate_report)
            gate_reports.append(gate_report)

            targets = build_multi_horizon_targets(
                timestamps_s=timestamps,
                upper_body_position=upper,
                base_height=base_height,
                query_times_s=QUERY_TIMES_S,
                base_pose_se2=base_pose,
                source="measured_odometry",
            )
            target_path = target_root / f"episode-{episode_index:03d}.npz"
            save_target_batch(target_path, targets)
            target_report = {
                "schema_version": 1,
                "protocol_id": "arena-g1-phase-1-m1-v1",
                "episode_index": episode_index,
                "source": targets.source,
                "samples": targets.samples,
                "query_times_s": targets.query_times_s.tolist(),
                "target_shape": list(targets.targets.shape),
                "terminal_anchors_dropped": frame_count - targets.samples,
            }
            _write_json(target_root / f"episode-{episode_index:03d}.json", target_report)

            derived = frame.copy()
            derived[ANCHOR_COLUMN] = [row.astype(np.float32) for row in base_pose]
            derived[UPPER_COLUMN] = [row.astype(np.float32) for row in upper]
            derived[HEIGHT_COLUMN] = [row.astype(np.float32) for row in base_height]
            derived[BASE_COLUMN] = [row.astype(np.float32) for row in base_pose]
            derived.to_parquet(args.output_lerobot / parquet_rel, index=False)

            flattened = targets.targets.reshape(-1, targets.targets.shape[-1])
            target_rows.append(flattened.astype(np.float32))
            anchor_rows.append(base_pose.astype(np.float32))
            episode_manifests.append(
                {
                    "episode_index": episode_index,
                    "hdf5_trajectory_id": trajectory_id,
                    "frames": frame_count,
                    "valid_anchors": targets.samples,
                    "proxy_gate_passed": proxy_passed,
                    "target_source": targets.source,
                }
            )
            print(
                f"[{position + 1:03d}/{len(episodes):03d}] episode {episode_index:03d}: "
                f"frames={frame_count}, anchors={targets.samples}, proxy_pass={proxy_passed}",
                flush=True,
            )

    all_targets = np.concatenate(target_rows, axis=0)
    all_anchors = np.concatenate(anchor_rows, axis=0)
    total_valid_anchors = sum(item["valid_anchors"] for item in episode_manifests)
    _update_metadata(
        source=args.source_lerobot,
        output=args.output_lerobot,
        target_values=all_targets,
        anchor_values=all_anchors,
        total_valid_anchors=total_valid_anchors,
    )

    proxy_pass_count = sum(bool(report["proxy_data_gate_passed"]) for report in gate_reports)
    aggregate_gate = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "episodes_expected": len(episodes),
        "episodes_reported": len(gate_reports),
        "proxy_pass_count": proxy_pass_count,
        "proxy_fail_count": len(gate_reports) - proxy_pass_count,
        "all_proxy_episodes_passed": proxy_pass_count == len(gate_reports),
        "maximum_xy_p95_m": max(float(item["metrics"]["xy_p95_m"]) for item in gate_reports),
        "maximum_yaw_p95_rad": max(float(item["metrics"]["yaw_p95_rad"]) for item in gate_reports),
        "target_source_selected": "measured_odometry",
        "decision": "train_m1_with_measured_root_odometry",
    }
    _write_json(gate_root / "report.json", aggregate_gate)

    manifest = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "dataset_revision": "97f7b5a4135e4e6a206d394c9b6ec0253f1558a7",
        "full_hdf5": {
            "path": str(args.hdf5),
            "bytes": args.hdf5.stat().st_size,
            "sha256": hdf5_sha256 or "skipped_by_explicit_flag",
            "expected_sha256": EXPECTED_FULL_HDF5_SHA256,
        },
        "source_lerobot": str(args.source_lerobot),
        "source_alignment": {
            "method": "frozen_arena_official_hdf5_converter_then_same_trajectory_root_join",
            "published_preconverted_lerobot_used": False,
            "command_columns_verified_against_hdf5": ["base_height_cmd", "navigate_cmd"],
            "hdf5_trajectory_order": "h5py_data_group_key_order_matching_frozen_converter",
        },
        "derived_lerobot": str(args.output_lerobot),
        "episodes": len(episodes),
        "frames": sum(item["frames"] for item in episode_manifests),
        "valid_anchors": total_valid_anchors,
        "frozen_training_budget": {
            "optimizer_steps": 47_468,
            "batch_size": 4,
            "effective_dataset_passes_after_terminal_drop": (
                47_468 * 4 / total_valid_anchors
            ),
            "note": (
                "The same-revision published LeRobot and HDF5 are not trajectory-aligned. "
                "Optimizer-step fairness is retained; exact 2.0 epochs is mathematically "
                "unavailable on the measured-root aligned source."
            ),
        },
        "query_times_s": QUERY_TIMES_S.tolist(),
        "output_shape": [16, 32],
        "target_source": "measured_odometry",
        "base_height_source": "expert_base_height_command_proxy",
        "terminal_policy": "drop_anchor_if_full_horizon_unavailable",
        "future_observation_as_input": False,
        "statistics_source": "all_100_frozen_training_episodes_post_anchor_transform",
        "target_source_counts": {"measured_odometry": total_valid_anchors},
        "episode_manifest": episode_manifests,
    }
    _write_json(args.artifact_root / "dataset-manifest.json", manifest)
    print(json.dumps({"data_gate": aggregate_gate, "manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
