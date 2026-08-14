#!/usr/bin/env python3
"""Fail-closed validation of the frozen Arena G1 installation and artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PINS = {
    "arena": "f479817431f6a02a6e40e4ea76d89203b59146d4",
    "isaaclab": "3c6e67bb5c7ada942a6d1884ab69338f57596f77",
    "groot": "3bce5530b3af0ce3619d5fda041385f9b732dca8",
    "dataset": "97f7b5a4135e4e6a206d394c9b6ec0253f1558a7",
    "model": "629479fedb1cf97c2f11ddc49eed951c5b750139",
}

MODEL_FILES = {
    "config.json": 1706,
    "model-00001-of-00002.safetensors": 4_999_367_032,
    "model-00002-of-00002.safetensors": 2_586_705_312,
    "model.safetensors.index.json": 104_606,
    "experiment_cfg/metadata.json": 44_558,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    arena = args.project_root / "repos/IsaacLab-Arena"
    isaaclab = args.project_root / "repos/IsaacLab"
    groot = arena / "submodules/Isaac-GR00T"
    commits = {"arena": git_head(arena), "isaaclab": git_head(isaaclab), "groot": git_head(groot)}
    for name, actual in commits.items():
        if actual != PINS[name]:
            raise SystemExit(f"{name} revision mismatch: {actual} != {PINS[name]}")

    model_files = {}
    for relative, expected_size in MODEL_FILES.items():
        path = args.model_root / relative
        if not path.is_file() or path.stat().st_size != expected_size:
            raise SystemExit(f"Model artifact mismatch: {path}")
        model_files[relative] = path.stat().st_size
    model_config = json.loads((args.model_root / "config.json").read_text(encoding="utf-8"))
    if model_config.get("architectures") != ["GR00T_N1_5"]:
        raise SystemExit("Official Arena checkpoint is not GR00T_N1_5")

    lerobot = args.dataset_root / "arena_g1_loco_manipulation_dataset_generated/lerobot"
    info = json.loads((lerobot / "meta/info.json").read_text(encoding="utf-8"))
    expected_dataset = {"total_episodes": 100, "total_frames": 94936, "fps": 50}
    for key, expected in expected_dataset.items():
        if info.get(key) != expected:
            raise SystemExit(f"Dataset {key} mismatch: {info.get(key)} != {expected}")
    small_hdf5 = args.dataset_root / "arena_g1_loco_manipulation_dataset_generated_small.hdf5"
    if not small_hdf5.is_file() or small_hdf5.stat().st_size != 230_115_530:
        raise SystemExit(f"Missing replay dataset: {small_hdf5}")
    parquet_count = len(list((lerobot / "data").rglob("*.parquet")))
    video_count = len(list((lerobot / "videos").rglob("*.mp4")))
    if parquet_count != 100 or video_count != 100:
        raise SystemExit(f"LeRobot file count mismatch: parquet={parquet_count}, video={video_count}")

    import torch

    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "isaacsim", "warp-lang", "flash-attn", "transformers", "peft")
    }
    if (
        packages["torch"] != "2.7.0+cu128"
        or packages["isaacsim"] != "5.1.0.0"
        or packages["warp-lang"] != "1.8.1"
        or packages["flash-attn"] != "2.7.4.post1"
    ):
        raise SystemExit(f"Core package pin mismatch: {packages}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    result = {
        "schema_version": 1,
        "status": "pass",
        "benchmark": "NVIDIA Isaac Lab Arena G1 Loco-Manipulation — Box Pick-and-Place",
        "pins": PINS,
        "commits": commits,
        "packages": packages,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "model_architecture": "GR00T_N1_5",
        "model_files": model_files,
        "dataset": {
            **expected_dataset,
            "parquet_files": parquet_count,
            "video_files": video_count,
            "replay_hdf5_bytes": small_hdf5.stat().st_size,
        },
        "action_contract": {
            "model_chunk_shape": [16, 32],
            "model_action": {
                "left_arm_joint_position": 7,
                "right_arm_joint_position": 7,
                "left_hand_joint_position": 7,
                "right_hand_joint_position": 7,
                "base_height": 1,
                "navigate_vx_vy_yaw_rate": 3,
            },
            "simulator_action_dim": 50,
            "executed_joint_targets": 43,
            "torso_rpy_injected_zeros": 3,
        },
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
