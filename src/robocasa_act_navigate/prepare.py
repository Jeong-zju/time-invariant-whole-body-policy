"""Prepare train-only statistics and a sequential RGB frame cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from .data import load_data, load_split
from .schema import CAMERA_KEYS, IMAGE_SIZE, NUM_TASKS


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(0).tolist(),
        "std": np.maximum(values.std(0), 1e-6).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
    }


def _decode(task: tuple[str, str, tuple[int, ...], int, int, int]) -> tuple[int, int]:
    video, output, shape, camera, start, end = task
    import av

    cache = np.memmap(output, mode="r+", dtype=np.uint8, shape=shape)
    count = 0
    with av.open(video) as container:
        for frame in container.decode(video=0):
            if count >= end - start:
                raise ValueError(f"too many frames in {video}")
            image = frame.to_ndarray(format="rgb24")
            if image.shape != (*IMAGE_SIZE, 3):
                raise ValueError(f"unexpected image shape {image.shape} in {video}")
            cache[start + count, camera] = image
            count += 1
    cache.flush()
    if count != end - start:
        raise ValueError(f"{video}: decoded {count}, expected {end - start}")
    return camera, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    data = load_data(args.data_root)
    split = load_split(args.split)
    train_indices = np.concatenate([np.arange(*data.episode_bounds[e]) for e in split["train"]])
    one_hot = np.eye(NUM_TASKS, dtype=np.float32)[data.task_indices[train_indices]]
    augmented_state = np.concatenate((data.states[train_indices], one_hot), axis=1)
    target_actions = data.actions[train_indices, :3] if args.base_only else data.actions[train_indices]
    payload = {
        "task": "NavigateKitchen",
        "split_manifest": str(args.split),
        "train_episodes": split["train"],
        "val_episodes": split["val"],
        "test_episodes": split["test"],
        "num_train_frames": int(len(train_indices)),
        "model": {
            "chunk_size": 32,
            "action_dim": int(target_actions.shape[1]),
        },
        "representation": (
            "native_time_indexed_base_command_3d" if args.base_only
            else "native_time_indexed_whole_body_command_12d"
        ),
        "features": {
            "observation.state": _stats(augmented_state),
            "action": _stats(target_actions),
            "image_imagenet": {
                "mean": np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None].tolist(),
                "std": np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None].tolist(),
            },
        },
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(payload, indent=2) + "\n")

    shape = (len(data.actions), len(CAMERA_KEYS), *IMAGE_SIZE, 3)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.cache.exists():
        metadata = args.cache.with_suffix(args.cache.suffix + ".json")
        if not metadata.is_file():
            raise FileExistsError(f"cache exists without completion metadata: {args.cache}")
        saved = json.loads(metadata.read_text())
        if saved.get("shape") != list(shape):
            raise ValueError("existing cache shape does not match dataset")
        print(json.dumps({"event": "cache_reused", "cache": str(args.cache), "shape": list(shape)}), flush=True)
        return
    cache = np.memmap(args.cache, mode="w+", dtype=np.uint8, shape=shape)
    cache.flush()
    del cache
    tasks = []
    for camera, key in enumerate(CAMERA_KEYS):
        directory = args.data_root / "videos" / "chunk-000" / key
        for episode, (start, end) in data.episode_bounds.items():
            video = directory / f"episode_{episode:06d}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            tasks.append((str(video), str(args.cache), shape, camera, start, end))
    decoded = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_decode, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            _, count = future.result()
            decoded += count
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                print(json.dumps({"videos": completed, "total_videos": len(futures), "decoded_frames": decoded}), flush=True)
    metadata = {"data_root": str(args.data_root), "shape": list(shape), "dtype": "uint8", "camera_keys": list(CAMERA_KEYS), "decoded_camera_frames": decoded}
    args.cache.with_suffix(args.cache.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "PREPARATION_COMPLETE", **metadata}), flush=True)


if __name__ == "__main__":
    main()
