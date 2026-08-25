"""CLI for Gates A-D: schema, SE(2), label plots, and oracle reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .labels import LabelConfig, PathLabel, build_path_label, fit_metric_config
from .parquet_data import TurningData, load_turning_data, split_episodes
from .plots import plot_label
from .reconstruction import aggregate_metrics, oracle_metrics
from .r1pro import BASE_ACTION, BASE_QVEL_STATE, CONTINUOUS_ACTION_INDICES


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _timestamp_audit(data: TurningData) -> dict:
    pieces = []
    for start, end in data.episode_bounds.values():
        pieces.append(np.diff(data.timestamps[start:end]))
    dt = np.concatenate(pieces)
    return {
        "count": int(dt.size),
        "mean_s": float(np.mean(dt)),
        "std_s": float(np.std(dt)),
        "min_s": float(np.min(dt)),
        "max_s": float(np.max(dt)),
        "nonpositive_count": int(np.sum(dt <= 0.0)),
        "off_30hz_over_1ms_count": int(np.sum(np.abs(dt - 1.0 / 30.0) > 1e-3)),
    }


def _lag_audit(data: TurningData, episodes: list[int], max_lag: int = 5) -> dict:
    results = {}
    best_lags = []
    for dimension in range(3):
        per_lag = {}
        for lag in range(-max_lag, max_lag + 1):
            commands = []
            measured = []
            for episode in episodes:
                start, end = data.episode_bounds[episode]
                if lag >= 0:
                    command_slice = slice(start, end - lag if lag else end)
                    measured_slice = slice(start + lag, end)
                else:
                    command_slice = slice(start - lag, end)
                    measured_slice = slice(start, end + lag)
                commands.append(data.actions[command_slice, BASE_ACTION][:, dimension])
                measured.append(data.states[measured_slice, BASE_QVEL_STATE][:, dimension])
            command = np.concatenate(commands)
            measurement = np.concatenate(measured)
            if np.std(command) < 1e-12 or np.std(measurement) < 1e-12:
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(command, measurement)[0, 1])
            per_lag[str(lag)] = correlation
        best_lag = max(range(-max_lag, max_lag + 1), key=lambda lag: per_lag[str(lag)])
        best_lags.append(best_lag)
        results[f"dim_{dimension}"] = {"correlation_by_lag": per_lag, "best_lag": best_lag}
    results["best_lags"] = best_lags
    results["lag_definition"] = "corr(command[t], measured_qvel[t + lag])"
    return results


def _sample_starts(data: TurningData, episodes: list[int], count: int, seed: int) -> list[int]:
    candidates = []
    for episode in episodes:
        start, end = data.episode_bounds[episode]
        candidates.extend(range(start, end))
    if count >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(candidates), size=count, replace=False).tolist())


def _window_characteristics(data: TurningData, label: PathLabel) -> dict[str, float]:
    start, end = label.start_index, label.end_index
    base = data.states[start : end + 1, BASE_QVEL_STATE]
    joints = data.actions[start : end + 1, CONTINUOUS_ACTION_INDICES]
    if joints.shape[0] > 1:
        joint_motion = float(np.mean(np.linalg.norm(np.diff(joints, axis=0), axis=-1)))
    else:
        joint_motion = 0.0
    xy_speed = np.linalg.norm(base[:, :2], axis=-1)
    quarter = max(1, len(xy_speed) // 4)
    episode_end = data.episode_bounds[int(data.episode_indices[start])][1]
    return {
        "mean_xy_speed": float(np.mean(xy_speed)),
        "mean_abs_yaw_rate": float(np.mean(np.abs(base[:, 2]))),
        "joint_motion": joint_motion,
        "dwell_then_move": float(np.mean(xy_speed[-quarter:]) - np.mean(xy_speed[:quarter])),
        "frames_to_episode_end": float(episode_end - start),
    }


def _select_plot_examples(characteristics: list[dict[str, float]], labels: list[PathLabel]) -> dict[str, int]:
    arrays = {key: np.asarray([row[key] for row in characteristics]) for key in characteristics[0]}
    attained = np.asarray([label.attained_length for label in labels])
    manipulation_score = arrays["joint_motion"] / (arrays["mean_xy_speed"] + 1e-5)
    return {
        "motion": int(np.argmax(arrays["mean_xy_speed"])),
        "turning": int(np.argmax(arrays["mean_abs_yaw_rate"])),
        "manipulation_only": int(np.argmax(manipulation_score)),
        "static": int(np.argmin(attained)),
        "dwell_then_move": int(np.argmax(arrays["dwell_then_move"])),
        "episode_end": int(np.argmin(arrays["frames_to_episode_end"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extent-mode", choices=("attained", "fixed"), default="attained")
    parser.add_argument("--max-duration-s", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_turning_data(args.data_root)
    train_episodes, validation_episodes = split_episodes(data.episode_bounds)
    metric = fit_metric_config(data, train_episodes)
    label_config = LabelConfig(max_duration_s=args.max_duration_s, extent_mode=args.extent_mode)

    starts = _sample_starts(data, validation_episodes, args.num_windows, args.seed)
    labels: list[PathLabel] = []
    rows: list[dict[str, float]] = []
    characteristics: list[dict[str, float]] = []
    row_records = []
    for start in starts:
        label = build_path_label(data, start, metric, label_config)
        # action[t] is applied after obs[t], and its physical response is stored
        # in the next observation row.
        measured = data.states[label.start_index + 1 : label.end_index + 1, BASE_QVEL_STATE]
        commanded = data.actions[label.start_index : label.end_index, BASE_ACTION]
        metrics = oracle_metrics(label, measured, commanded)
        chars = _window_characteristics(data, label)
        labels.append(label)
        rows.append(metrics)
        characteristics.append(chars)
        row_records.append(
            {
                "episode_index": int(data.episode_indices[start]),
                "frame_index": int(data.frame_indices[start]),
                "start_index": int(start),
                "end_index": int(label.end_index),
                **metrics,
                **chars,
            }
        )

    examples = _select_plot_examples(characteristics, labels)
    for category, index in examples.items():
        record = row_records[index]
        plot_label(
            labels[index],
            args.output_dir / "plots" / f"{category}.png",
            title=(
                f"{category} | episode={record['episode_index']} frame={record['frame_index']} "
                f"mode={args.extent_mode}"
            ),
        )

    report = {
        "task": "turning_on_radio",
        "data_root": str(args.data_root),
        "num_frames": int(data.actions.shape[0]),
        "num_episodes": len(data.episode_bounds),
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "num_validation_windows": len(starts),
        "metric_config": metric.to_dict(),
        "label_config": {
            "num_anchors": label_config.num_anchors,
            "max_duration_s": label_config.max_duration_s,
            "static_epsilon": label_config.static_epsilon,
            "extent_mode": label_config.extent_mode,
            "nominal_dt": label_config.nominal_dt,
        },
        "timestamp_audit": _timestamp_audit(data),
        "base_command_measurement_lag_audit": _lag_audit(data, train_episodes),
        "oracle_summary": aggregate_metrics(rows),
        "plot_examples": {category: row_records[index] for category, index in examples.items()},
        "rows": row_records,
    }
    with (args.output_dir / "report.json").open("w") as file:
        json.dump(report, file, indent=2, default=_json_default)
    with (args.output_dir / "metric_config.json").open("w") as file:
        json.dump(metric.to_dict(), file, indent=2)

    summary = report["oracle_summary"]
    print(f"task=turning_on_radio frames={report['num_frames']} episodes={report['num_episodes']}")
    print(
        f"L_sigma={metric.path_length:.6f} xy_scale={metric.xy_scale:.6g} "
        f"theta_scale={metric.theta_scale:.6g}"
    )
    for key in (
        "base_translation_rmse_m",
        "base_yaw_rmse_rad",
        "base_twist_measured_rmse",
        "base_twist_command_rmse",
        "joint_target_rmse_rad",
        "gripper_command_mismatch_rate",
        "duration_abs_error_s",
    ):
        print(f"{key}: mean={summary[key]['mean']:.6g} p90={summary[key]['p90']:.6g} max={summary[key]['max']:.6g}")
    print(f"report={args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
