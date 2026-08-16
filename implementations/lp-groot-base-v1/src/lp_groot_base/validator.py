"""Dataset audit, metric calibration, and oracle reconstruction validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import se2
from .labels import LabelConfig, build_path_time_label, reconstruct_at_times
from .v3_dataset import episode_slice, load_v3_metadata, load_v3_table


@dataclass(frozen=True)
class EpisodeData:
    name: str
    positions: np.ndarray
    quaternions: np.ndarray
    timestamps: np.ndarray
    actions: np.ndarray


def episode_paths(dataset_root: str | Path) -> list[Path]:
    paths = sorted(Path(dataset_root).glob("data/chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot episode parquet files under {dataset_root}")
    return paths


def load_base_episode(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path, columns=["observation.state", "timestamp"])
    state = np.vstack(frame["observation.state"].to_numpy()).astype(np.float64)
    if state.shape[1] < 7:
        raise ValueError(f"Expected state with base xyz+xyzw in first 7 dims, got {state.shape}")
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    return state[:, :3], state[:, 3:7], timestamps


def load_actions(path: str | Path) -> np.ndarray:
    frame = pd.read_parquet(path, columns=["action"])
    action = np.vstack(frame["action"].to_numpy()).astype(np.float64)
    if action.shape[1] < 12:
        raise ValueError(f"Expected RoboCasa 12D action, got {action.shape}")
    return action


def _episode_from_frame(name: str, frame: pd.DataFrame) -> EpisodeData:
    state = np.vstack(frame["observation.state"].to_numpy()).astype(np.float64)
    actions = np.vstack(frame["action"].to_numpy()).astype(np.float64)
    if state.shape[1] < 7:
        raise ValueError(f"Expected state with base xyz+xyzw in first 7 dims, got {state.shape}")
    if actions.shape[1] < 12:
        raise ValueError(f"Expected RoboCasa 12D action, got {actions.shape}")
    return EpisodeData(
        name=name,
        positions=state[:, :3],
        quaternions=state[:, 3:7],
        timestamps=frame["timestamp"].to_numpy(dtype=np.float64),
        actions=actions,
    )


def load_episodes(dataset_root: str | Path) -> list[EpisodeData]:
    """Load either legacy per-episode parquet data or compact LeRobot v3 shards."""
    root = Path(dataset_root)
    paths = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if paths:
        records = []
        for path in paths:
            frame = pd.read_parquet(path)
            records.append(_episode_from_frame(path.name, frame))
        return records

    info_path = root / "meta" / "info.json"
    if info_path.exists():
        _, metadata = load_v3_metadata(root)
        table = load_v3_table(root)
        return [
            _episode_from_frame(
                f"episode_{episode_index:06d}",
                episode_slice(table, metadata, episode_index),
            )
            for episode_index in metadata["episode_index"].astype(int)
        ]

    raise FileNotFoundError(f"No supported LeRobot episode data under {dataset_root}")


def load_action_layout(dataset_root: str | Path) -> dict[str, tuple[int, int]]:
    modality_path = Path(dataset_root) / "meta" / "modality.json"
    if not modality_path.exists():
        info_path = Path(dataset_root) / "meta" / "info.json"
        if info_path.exists():
            with info_path.open() as file:
                info = json.load(file)
            action = info.get("features", {}).get("action", {})
            if info.get("codebase_version") == "v3.0" and action.get("shape") == [12]:
                # This is the exact RoboCasa v3 layout used by build_lp_cache.py:
                # normalized base command xyz at 0:3 and control-mode flag at 4.
                return {"base_motion": (0, 3), "control_mode": (4, 5)}
        raise FileNotFoundError(f"No action layout metadata in {modality_path} or {info_path}")
    with modality_path.open() as file:
        modality = json.load(file)
    required = ["base_motion", "control_mode"]
    missing = [key for key in required if key not in modality.get("action", {})]
    if missing:
        raise KeyError(f"Missing action layout entries {missing} in {modality_path}")
    return {
        key: (int(modality["action"][key]["start"]), int(modality["action"][key]["end"]))
        for key in required
    }


def audit_dataset(dataset_root: str | Path) -> dict:
    episodes = load_episodes(dataset_root)
    frame_counts: list[int] = []
    dts: list[np.ndarray] = []
    invalid_timestamp_episodes: list[str] = []
    for episode in episodes:
        timestamps = episode.timestamps
        frame_counts.append(len(timestamps))
        delta = np.diff(timestamps)
        dts.append(delta)
        if np.any(delta <= 0):
            invalid_timestamp_episodes.append(episode.name)
    all_dt = np.concatenate(dts)
    return {
        "num_episodes": len(episodes),
        "num_frames": int(sum(frame_counts)),
        "frames_min": int(min(frame_counts)),
        "frames_median": float(np.median(frame_counts)),
        "frames_max": int(max(frame_counts)),
        "dt_median_seconds": float(np.median(all_dt)),
        "dt_p01_seconds": float(np.quantile(all_dt, 0.01)),
        "dt_p99_seconds": float(np.quantile(all_dt, 0.99)),
        "nonpositive_dt_count": int(np.sum(all_dt <= 0)),
        "invalid_timestamp_episodes": invalid_timestamp_episodes,
    }


def calibrate_metric(
    dataset_root: str | Path,
    reference_horizon_steps: int = 16,
    positive_quantile: float = 0.75,
) -> dict:
    """Estimate robust per-step metric scales and matched fixed path extent."""
    episodes = load_episodes(dataset_root)
    action_layout = load_action_layout(dataset_root)
    base_start, base_end = action_layout["base_motion"]
    mode_start, mode_end = action_layout["control_mode"]
    if base_end - base_start < 3 or mode_end - mode_start != 1:
        raise ValueError(f"Unexpected RoboCasa action layout: {action_layout}")
    commanded_translations: list[np.ndarray] = []
    commanded_rotations: list[np.ndarray] = []
    episode_deltas: list[np.ndarray] = []
    episode_active: list[np.ndarray] = []
    for episode in episodes:
        pos = episode.positions
        quat = episode.quaternions
        actions = episode.actions
        world = se2.world_xy_quat_to_se2(pos, quat)
        delta = se2.log(se2.between(world[:-1], world[1:]))
        # Recorder alignment is action[t] -> observation[t+1]. Commands are used
        # only to reject static simulator jitter while calibrating metric scales;
        # labels themselves continue to use measured poses exclusively.
        base_action = actions[:-1, base_start : base_start + 3]
        base_mode = actions[:-1, mode_start] > 0.0
        translation_command = np.linalg.norm(base_action[:, :2], axis=-1) > 1e-3
        rotation_command = np.abs(base_action[:, 2]) > 1e-3
        commanded_translations.append(
            np.linalg.norm(delta[base_mode & translation_command, :2], axis=-1)
        )
        commanded_rotations.append(np.abs(delta[base_mode & rotation_command, 2]))
        episode_deltas.append(delta)
        episode_active.append(base_mode & (translation_command | rotation_command))

    trans = np.concatenate(commanded_translations)
    rot = np.concatenate(commanded_rotations)
    trans_positive = trans[trans > 1e-6]
    rot_positive = rot[rot > 1e-6]
    if len(trans_positive) < 100 or len(rot_positive) < 100:
        raise ValueError("Dataset needs both nonzero translation and rotation to calibrate base metric")
    l_xy = float(np.quantile(trans_positive, positive_quantile))
    l_yaw = float(np.quantile(rot_positive, positive_quantile))

    horizon_lengths: list[np.ndarray] = []
    for delta, active in zip(episode_deltas, episode_active):
        increments = np.sqrt(
            (delta[:, 0] / l_xy) ** 2
            + (delta[:, 1] / l_xy) ** 2
            + (delta[:, 2] / l_yaw) ** 2
        )
        if len(increments) < reference_horizon_steps:
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(increments)])
        lengths = cumulative[reference_horizon_steps:] - cumulative[:-reference_horizon_steps]
        active_count = np.convolve(
            active.astype(np.int32),
            np.ones(reference_horizon_steps, dtype=np.int32),
            mode="valid",
        )
        horizon_lengths.append(lengths[active_count > 0])
    lengths = np.concatenate(horizon_lengths)
    path_extent = float(np.median(lengths))
    return {
        "l_xy": l_xy,
        "l_yaw": l_yaw,
        "path_extent": path_extent,
        "reference_horizon_steps": int(reference_horizon_steps),
        "positive_quantile": float(positive_quantile),
        "action_layout": {key: list(value) for key, value in action_layout.items()},
        "scale_translation_sample_count": int(len(trans_positive)),
        "scale_rotation_sample_count": int(len(rot_positive)),
        "commanded_translation_p50_m": float(np.median(trans_positive)),
        "commanded_translation_p99_m": float(np.quantile(trans_positive, 0.99)),
        "commanded_yaw_p50_rad": float(np.median(rot_positive)),
        "commanded_yaw_p99_rad": float(np.quantile(rot_positive, 0.99)),
        "matched_active_window_count": int(len(lengths)),
        "matched_length_p25": float(np.quantile(lengths, 0.25)),
        "matched_length_p50": path_extent,
        "matched_length_p75": float(np.quantile(lengths, 0.75)),
    }


def _one_window_errors(
    positions: np.ndarray,
    quaternions: np.ndarray,
    timestamps: np.ndarray,
    start_index: int,
    config: LabelConfig,
) -> tuple[dict, object]:
    label = build_path_time_label(positions, quaternions, timestamps, start_index, config)
    relative_times = timestamps[start_index:] - timestamps[start_index]
    keep = relative_times <= label.endpoint_time + 1e-8
    query_times = relative_times[keep]
    world = se2.world_xy_quat_to_se2(positions[start_index:][keep], quaternions[start_index:][keep])
    raw = se2.between(world[0], world)
    if query_times[-1] < label.endpoint_time - 1e-8:
        query_times = np.concatenate([query_times, [label.endpoint_time]])
        raw = np.concatenate([raw, label.poses[int(label.valid.sum()) - 1][None]], axis=0)
    reconstructed = reconstruct_at_times(label, query_times)
    error = se2.log(se2.between(raw, reconstructed))
    translation_error = np.linalg.norm(error[:, :2], axis=-1)
    yaw_error = np.abs(error[:, 2])

    if len(query_times) > 1:
        dt = np.diff(query_times)
        raw_twist = se2.log(se2.between(raw[:-1], raw[1:])) / dt[:, None]
        recon_twist = se2.log(se2.between(reconstructed[:-1], reconstructed[1:])) / dt[:, None]
        twist_error = recon_twist - raw_twist
        twist_rmse = np.sqrt(np.mean(twist_error**2, axis=0))
        recon_twist_abs = np.abs(recon_twist)
    else:
        twist_rmse = np.zeros(3)
        recon_twist_abs = np.zeros((1, 3))

    valid_count = int(label.valid.sum())
    duration_sum = float(label.durations[:valid_count].sum())
    return (
        {
            "translation_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
            "translation_max_m": float(np.max(translation_error)),
            "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
            "yaw_max_rad": float(np.max(yaw_error)),
            "twist_rmse_vx": float(twist_rmse[0]),
            "twist_rmse_vy": float(twist_rmse[1]),
            "twist_rmse_yaw_rate": float(twist_rmse[2]),
            "reconstructed_abs_vx_max": float(np.max(recon_twist_abs[:, 0])),
            "reconstructed_abs_vy_max": float(np.max(recon_twist_abs[:, 1])),
            "reconstructed_abs_yaw_rate_max": float(np.max(recon_twist_abs[:, 2])),
            "endpoint_translation_error_m": float(translation_error[-1]),
            "endpoint_yaw_error_rad": float(yaw_error[-1]),
            "duration_error_seconds": abs(duration_sum - label.endpoint_time),
            "valid_anchors": valid_count,
            "complete": bool(label.complete),
            "attained_sigma": float(label.attained_sigma),
        },
        label,
    )


def validate_oracle(
    dataset_root: str | Path,
    config: LabelConfig,
    num_windows: int = 512,
    seed: int = 0,
) -> tuple[dict, list[tuple[str, int, object]]]:
    episodes = load_episodes(dataset_root)
    candidates = [(ep, i) for ep, data in enumerate(episodes) for i in range(len(data.timestamps))]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=min(num_windows, len(candidates)), replace=False)
    metrics: list[dict] = []
    examples: list[tuple[str, int, object]] = []
    for chosen_index in chosen:
        episode_index, start_index = candidates[int(chosen_index)]
        episode = episodes[episode_index]
        metric, label = _one_window_errors(
            episode.positions,
            episode.quaternions,
            episode.timestamps,
            start_index,
            config,
        )
        metrics.append(metric)
        if len(examples) < 12:
            examples.append((episode.name, start_index, label))

    numeric_keys = [
        key
        for key, value in metrics[0].items()
        if isinstance(value, (float, int)) and key not in {"valid_anchors"}
    ]
    aggregate = {}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in metrics], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    valid_counts = np.asarray([row["valid_anchors"] for row in metrics])
    return (
        {
            "num_windows": len(metrics),
            "complete_fraction": float(np.mean([row["complete"] for row in metrics])),
            "fully_static_fraction": float(np.mean(valid_counts == 1)),
            "valid_anchors_mean": float(np.mean(valid_counts)),
            "valid_anchors_min": int(np.min(valid_counts)),
            "valid_anchors_max": int(np.max(valid_counts)),
            "metrics": aggregate,
        },
        examples,
    )


def run_full_validation(
    dataset_root: str | Path,
    output_dir: str | Path,
    num_windows: int = 512,
    reference_horizon_steps: int = 16,
    seed: int = 0,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(dataset_root)
    calibration = calibrate_metric(dataset_root, reference_horizon_steps)
    config = LabelConfig(
        l_xy=calibration["l_xy"],
        l_yaw=calibration["l_yaw"],
        path_extent=calibration["path_extent"],
    )
    oracle, examples = validate_oracle(dataset_root, config, num_windows=num_windows, seed=seed)
    report = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "audit": audit,
        "calibration": calibration,
        "label_config": asdict(config),
        "oracle": oracle,
    }
    with (output_dir / "validation_report.json").open("w") as file:
        json.dump(report, file, indent=2)
    np.savez_compressed(
        output_dir / "label_examples.npz",
        poses=np.stack([example[2].poses for example in examples]),
        log_durations=np.stack([example[2].log_durations for example in examples]),
        valid=np.stack([example[2].valid for example in examples]),
        episode=np.asarray([example[0] for example in examples]),
        start_index=np.asarray([example[1] for example in examples]),
    )
    return report
