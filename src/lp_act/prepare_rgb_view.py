"""Create a symlinked turning_on_radio dataset view with only model inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


RGB_KEYS = (
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.rgb.zed_link_camera_0",
)
def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(f"Existing link {target} points to {target.resolve()}, not {source}")
        return
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing path {target}")
    os.symlink(source, target, target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.source / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    # Hugging Face's parquet loader requires every physical tabular column to
    # remain in the declared schema. Remove only unused video features; those
    # are the expensive modalities decoded by LeRobotDataset.__getitem__.
    info["features"] = {
        key: value
        for key, value in info["features"].items()
        if value["dtype"] != "video" or key in RGB_KEYS
    }
    info["total_episodes"] = 200
    info["total_frames"] = 429928
    info["total_tasks"] = 1
    info["splits"] = {"train": "0:200"}

    meta = args.output / "meta"
    meta.mkdir(exist_ok=True)
    with (meta / "info.json").open("w") as handle:
        json.dump(info, handle, indent=2)
    for name in ("stats.json", "tasks.parquet", "tasks.jsonl"):
        if (args.source / "meta" / name).exists():
            _link(args.source / "meta" / name, meta / name)
    _link(args.source / "meta" / "episodes" / "chunk-000", meta / "episodes" / "chunk-000")
    _link(args.source / "data" / "chunk-000", args.output / "data" / "chunk-000")
    for key in RGB_KEYS:
        _link(args.source / "videos" / key / "chunk-000", args.output / "videos" / key / "chunk-000")

    print(f"output={args.output}")
    print(f"features={sorted(info['features'])}")


if __name__ == "__main__":
    main()
