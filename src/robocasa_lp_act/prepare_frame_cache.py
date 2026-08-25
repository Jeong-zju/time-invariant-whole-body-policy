"""Decode RoboCasa episode videos once into a random-access uint8 memmap."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .data import load_robocasa_data
from .schema import CAMERA_KEYS, IMAGE_SIZE


def _decode_one(task: tuple[str, str, tuple[int, ...], int, int, int]) -> tuple[int, int, int]:
    video_path, cache_path, shape, camera_index, start, end = task
    import av

    cache = np.memmap(cache_path, mode="r+", dtype=np.uint8, shape=shape)
    count = 0
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            if count >= end - start:
                raise ValueError(f"Too many frames in {video_path}")
            array = frame.to_ndarray(format="rgb24")
            if array.shape != (*IMAGE_SIZE, 3):
                raise ValueError(f"Unexpected frame shape {array.shape} in {video_path}")
            cache[start + count, camera_index] = array
            count += 1
    if count != end - start:
        raise ValueError(f"{video_path}: decoded {count}, expected {end - start}")
    cache.flush()
    return camera_index, start, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_robocasa_data(args.data_root)
    shape = (data.actions.shape[0], len(CAMERA_KEYS), *IMAGE_SIZE, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = np.memmap(args.output, mode="w+", dtype=np.uint8, shape=shape)
    cache.flush()
    del cache

    tasks = []
    for camera_index, camera_key in enumerate(CAMERA_KEYS):
        camera_dir = args.data_root / "videos" / "chunk-000" / camera_key
        for episode, (start, end) in data.episode_bounds.items():
            video_path = camera_dir / f"episode_{episode:06d}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            tasks.append((str(video_path), str(args.output), shape, camera_index, start, end))

    decoded = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_decode_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            _, _, count = future.result()
            decoded += count
            if completed == 1 or completed % 20 == 0 or completed == len(futures):
                print(json.dumps({"videos": completed, "total_videos": len(futures), "decoded_frames": decoded}), flush=True)

    metadata = {
        "data_root": str(args.data_root),
        "cache": str(args.output),
        "dtype": "uint8",
        "shape": list(shape),
        "camera_keys": list(CAMERA_KEYS),
        "videos": len(tasks),
        "decoded_camera_frames": decoded,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
