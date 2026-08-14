#!/usr/bin/env python3
"""Merge an Arena G1 action-head LoRA adapter into a GR00T N1.5 checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from peft import PeftModel

from gr00t.model.gr00t_n1 import GR00T_N1_5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    base = GR00T_N1_5.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    metadata_src = args.adapter / "experiment_cfg" / "metadata.json"
    if metadata_src.exists():
        metadata_dst = args.output / "experiment_cfg" / "metadata.json"
        metadata_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_src, metadata_dst)

    summary = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "output": str(args.output),
        "safe_merge": True,
    }
    (args.output / "merge_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
