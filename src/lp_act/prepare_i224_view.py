"""Transcode the RGB-only task view to random-access-friendly 224p video."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .training_data import CAMERA_KEYS


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path {target}")
    os.symlink(source, target, target_is_directory=source.is_dir())


def _frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _transcode(job: tuple[Path, Path]) -> dict:
    source, target = job
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(target.stem + ".partial.mp4")
        if temporary.exists():
            temporary.unlink()
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-an",
                "-vf",
                "scale=224:224:flags=lanczos",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-g",
                "1",
                "-keyint_min",
                "1",
                "-sc_threshold",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "8",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(target)
    return {
        "source": str(source),
        "target": str(target),
        "source_frames": _frame_count(source),
        "target_frames": _frame_count(target),
        "target_bytes": target.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.source / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    for key in CAMERA_KEYS:
        feature = info["features"][key]
        feature["shape"] = [224, 224, 3]
        feature["info"].update(
            {
                "video.height": 224,
                "video.width": 224,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.g": 1,
                "video.crf": 18,
                "video.preset": "veryfast",
            }
        )

    meta = args.output / "meta"
    meta.mkdir(exist_ok=True)
    with (meta / "info.json").open("w") as handle:
        json.dump(info, handle, indent=2)
    for name in ("stats.json", "tasks.parquet", "tasks.jsonl"):
        _link(args.source / "meta" / name, meta / name)
    _link(args.source / "meta" / "episodes" / "chunk-000", meta / "episodes" / "chunk-000")
    _link(args.source / "data" / "chunk-000", args.output / "data" / "chunk-000")

    jobs = []
    for key in CAMERA_KEYS:
        for source in sorted((args.source / "videos" / key / "chunk-000").glob("*.mp4")):
            jobs.append((source, args.output / "videos" / key / "chunk-000" / source.name))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(_transcode, jobs))
    for record in records:
        if record["source_frames"] != record["target_frames"]:
            raise RuntimeError(f"Frame count changed during transcode: {record}")

    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "encoding": "224x224 H.264 all-intra, CRF 18, yuv420p",
        "files": records,
    }
    with (args.output / "transcode_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"output={args.output}")
    print(f"files={len(records)} bytes={sum(record['target_bytes'] for record in records)}")


if __name__ == "__main__":
    main()
