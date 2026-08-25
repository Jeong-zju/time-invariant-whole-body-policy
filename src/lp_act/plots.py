"""Diagnostic figures required by the LP-ACT label/reconstruction gate."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .labels import PathLabel
from .reconstruction import reconstruct_at_times
from .r1pro import NONBASE_CONTINUOUS_INDICES


def plot_label(label: PathLabel, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruction = reconstruct_at_times(label, label.raw_times)
    valid = ~label.is_pad

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(label.raw_base_poses[:, 0], label.raw_base_poses[:, 1], "k-", label="integrated measured qvel")
    ax.plot(reconstruction.base_poses[:, 0], reconstruction.base_poses[:, 1], "b--", label="30 Hz reconstruction")
    ax.scatter(label.target[valid, 0], label.target[valid, 1], c="tab:red", s=18, label="32 anchors")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x in chunk-origin frame [m]")
    ax.set_ylabel("y in chunk-origin frame [m]")
    ax.legend(fontsize=8)
    ax.set_title("Local base path")

    ax = axes[0, 1]
    ax.plot(label.raw_times, label.raw_base_poses[:, 2], "k-", label="raw yaw")
    ax.plot(label.raw_times, reconstruction.base_poses[:, 2], "b--", label="reconstructed yaw")
    ax.scatter(label.anchor_times[valid], label.target[valid, 2], c="tab:red", s=18, label="anchors")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("yaw [rad]")
    ax.legend(fontsize=8)
    ax.set_title("Base orientation")

    joint_positions = NONBASE_CONTINUOUS_INDICES[[0, 4, 10, 11, 14, 17]]
    ax = axes[1, 0]
    for position in joint_positions:
        ax.plot(label.raw_times, label.raw_nonbase[:, position], alpha=0.7, label=f"q{position} raw")
        ax.plot(label.raw_times, reconstruction.nonbase_targets[:, position], "--", alpha=0.7)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("target [rad]")
    ax.set_title("Representative absolute joint targets")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1, 1]
    durations = np.exp(label.target[:, 23])
    colors = np.where(label.is_pad, "tab:gray", "tab:green")
    ax.bar(np.arange(1, len(durations) + 1), durations, color=colors)
    ax.set_xlabel("path anchor")
    ax.set_ylabel("segment duration [s]")
    ax.set_title(
        f"timing | attained/L_sigma={label.attained_fraction:.2f} | valid={label.valid_count}/{len(durations)}"
    )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
