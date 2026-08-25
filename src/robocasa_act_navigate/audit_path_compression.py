"""Audit loss-bounded compression of common-origin NavigateKitchen base paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_data, load_split


def wrap_angle(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_xyzw_to_yaw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def common_origin_path(states: np.ndarray, yaw: np.ndarray, index: int, chunk_size: int) -> np.ndarray:
    """Return every future pose as ``T_index^-1 T_future``."""

    ids = np.arange(index, index + chunk_size)
    delta = states[ids, :2].astype(np.float64) - states[index, :2].astype(np.float64)
    cosine = np.cos(yaw[index])
    sine = np.sin(yaw[index])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    local_yaw = wrap_angle(yaw[ids] - yaw[index])
    result = np.stack((local_x, local_y, local_yaw), axis=-1)
    np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
    return result


def rdp_count(path: np.ndarray, translation_tolerance: float, yaw_tolerance: float) -> int:
    """RDP point count using a geometric, rather than temporal, projection.

    Each pose is embedded as ``[x, y, r * unwrapped_yaw]`` only to find its
    closest progress on the candidate SE(2) chord.  Translation and yaw errors
    are then checked separately in physical units.  In particular, interpolation
    never uses the sample index, timestamp, or source frame rate.
    """

    path = np.asarray(path, dtype=np.float64).copy()
    path[:, 2] = np.unwrap(path[:, 2])
    yaw_radius = translation_tolerance / yaw_tolerance
    embedded = path.copy()
    embedded[:, 2] *= yaw_radius

    keep = {0, len(path) - 1}
    stack = [(0, len(path) - 1)]
    while stack:
        left, right = stack.pop()
        if right <= left + 1:
            continue
        chord = embedded[right] - embedded[left]
        denominator = float(np.dot(chord, chord))
        if denominator <= 1e-18:
            alpha = np.zeros(right - left - 1, dtype=np.float64)
        else:
            offsets = embedded[left + 1 : right] - embedded[left]
            alpha = np.clip(offsets @ chord / denominator, 0.0, 1.0)
        interpolated_xy = path[left, :2] + alpha[:, None] * (path[right, :2] - path[left, :2])
        interpolated_yaw = path[left, 2] + alpha * (path[right, 2] - path[left, 2])
        translation_error = np.linalg.norm(path[left + 1 : right, :2] - interpolated_xy, axis=-1)
        yaw_error = np.abs(path[left + 1 : right, 2] - interpolated_yaw)
        normalized_error = np.maximum(
            translation_error / translation_tolerance,
            yaw_error / yaw_tolerance,
        )
        worst = int(np.argmax(normalized_error))
        if normalized_error[worst] > 1.0:
            split = left + 1 + worst
            keep.add(split)
            stack.extend(((left, split), (split, right)))
    return len(keep)


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        str(level): float(np.quantile(array, level))
        for level in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    data = load_data(args.data_root)
    split = load_split(args.split)
    yaw = quaternion_xyzw_to_yaw(data.states[:, 3:7])
    starts: list[int] = []
    for episode in split["train"]:
        start, end = data.episode_bounds[episode]
        starts.extend(range(start, end - args.chunk_size + 1))
    if args.sample_limit and len(starts) > args.sample_limit:
        generator = np.random.default_rng(args.seed)
        starts = sorted(generator.choice(starts, args.sample_limit, replace=False).tolist())

    tolerances = ((0.005, 0.01), (0.01, 0.02), (0.02, 0.05))
    counts = {tolerance: [] for tolerance in tolerances}
    endpoint_translation = []
    endpoint_yaw = []
    metric_length = []
    for ordinal, index in enumerate(starts, 1):
        path = common_origin_path(data.states, yaw, index, args.chunk_size)
        increments = np.diff(path, axis=0)
        increments[:, 2] = wrap_angle(increments[:, 2])
        endpoint_translation.append(float(np.linalg.norm(path[-1, :2])))
        endpoint_yaw.append(float(abs(path[-1, 2])))
        metric_length.append(
            float(
                np.sqrt(
                    increments[:, 0] ** 2
                    + increments[:, 1] ** 2
                    + (0.25 * increments[:, 2]) ** 2
                ).sum()
            )
        )
        for tolerance in tolerances:
            counts[tolerance].append(rdp_count(path, *tolerance))
        if ordinal == 1 or ordinal % 10_000 == 0 or ordinal == len(starts):
            print(json.dumps({"processed": ordinal, "total": len(starts)}), flush=True)

    payload = {
        "task": "NavigateKitchen",
        "episodes": "train split only",
        "train_episode_count": len(split["train"]),
        "source_fps": 20,
        "chunk_size_frames": args.chunk_size,
        "chunk_duration_s": args.chunk_size / 20.0,
        "num_windows": len(starts),
        "reference_frame": "all targets are T_t^-1 T_t+k for one common B_t origin",
        "endpoint_translation_m": quantiles(endpoint_translation),
        "endpoint_abs_yaw_rad": quantiles(endpoint_yaw),
        "se2_metric_length_yaw_radius_0p25m": quantiles(metric_length),
        "rdp_required_points": {
            f"translation_{translation}_yaw_{rotation}": quantiles(counts[(translation, rotation)])
            for translation, rotation in tolerances
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "COMPRESSION_AUDIT_COMPLETE", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
