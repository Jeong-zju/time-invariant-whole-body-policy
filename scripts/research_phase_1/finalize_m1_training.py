#!/usr/bin/env python3
"""Validate a completed M1 LoRA run and write its immutable manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TARGET_COLUMNS = (
    "phase1.target_upper_body_position",
    "phase1.target_base_height",
    "phase1.target_base_pose_se2_global",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=47_468)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()

    state_path = args.train_output / "trainer_state.json"
    adapter_path = args.train_output / "adapter_model.safetensors"
    adapter_config = args.train_output / "adapter_config.json"
    for path in (state_path, adapter_path, adapter_config, args.dataset_manifest, args.dataset_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    global_step = int(state["global_step"])
    if global_step != args.expected_steps:
        raise ValueError(f"global_step {global_step} != expected {args.expected_steps}")
    loss_entries = [entry for entry in state.get("log_history", []) if "loss" in entry]
    if not loss_entries:
        raise ValueError("trainer_state contains no logged losses")
    if not all(float(entry["loss"]) >= 0.0 for entry in loss_entries):
        raise ValueError("trainer_state contains an invalid loss")

    all_stats = json.loads(args.dataset_stats.read_text(encoding="utf-8"))
    missing = [key for key in TARGET_COLUMNS if key not in all_stats]
    if missing:
        raise KeyError(f"dataset statistics are missing M1 targets: {missing}")
    target_stats = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "statistics_semantics": "all_train_episodes_post_anchor_relative_transform",
        "columns": {key: all_stats[key] for key in TARGET_COLUMNS},
    }
    _write(args.checkpoint_root / "target-stats.json", target_stats)

    manifest = {
        "schema_version": 1,
        "protocol_id": "arena-g1-phase-1-m1-v1",
        "status": "training_complete",
        "train_output": str(args.train_output),
        "global_step": global_step,
        "reported_epoch": float(state.get("epoch", 0.0)),
        "optimizer_steps_frozen": args.expected_steps,
        "batch_size": 4,
        "learning_rate": 1.0e-4,
        "lora": {"rank": 32, "alpha": 32, "dropout": 0.1},
        "tuned": ["projector", "diffusion_action_head"],
        "frozen": ["llm", "visual_backbone"],
        "first_logged_loss": float(loss_entries[0]["loss"]),
        "last_logged_loss": float(loss_entries[-1]["loss"]),
        "minimum_logged_loss": min(float(entry["loss"]) for entry in loss_entries),
        "adapter_sha256": _sha256(adapter_path),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "upstream": {
            "arena": "f479817431f6a02a6e40e4ea76d89203b59146d4",
            "isaac_lab": "3c6e67bb5c7ada942a6d1884ab69338f57596f77",
            "groot": "3bce5530b3af0ce3619d5fda041385f9b732dca8",
            "dataset": "97f7b5a4135e4e6a206d394c9b6ec0253f1558a7",
            "base_checkpoint": "629479fedb1cf97c2f11ddc49eed951c5b750139",
        },
    }
    _write(args.checkpoint_root / "training-manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
