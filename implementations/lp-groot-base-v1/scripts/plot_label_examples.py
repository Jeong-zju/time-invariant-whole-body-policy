#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.cache_dir).glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError("No cached episodes")
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for ax, path, fraction in zip(axes.flat, paths[:12], np.linspace(0.05, 0.8, 12)):
        data = np.load(path)
        index = min(int(len(data["action"]) * fraction), len(data["action"]) - 1)
        count = int(data["valid"][index].sum())
        pose = data["action"][index, :count, :3]
        ax.plot(np.r_[0, pose[:, 0]], np.r_[0, pose[:, 1]], "o-", ms=3)
        ax.quiver(pose[:, 0], pose[:, 1], np.cos(pose[:, 2]), np.sin(pose[:, 2]), scale=12)
        ax.set_title(f"{path.stem} t={index}, valid={count}")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
    fig.suptitle("LP-GR00T-Base V1 measured-pose labels (current base frame)")
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
