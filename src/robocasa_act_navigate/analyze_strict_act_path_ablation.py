"""Analyze the same-ACT-output command/path/pointer causal ablation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .analyze_rate_control_sweep import (
    _frechet,
    _hausdorff,
    _local_curve,
    _resample_progress,
)


RATES = (0.5, 1.0, 1.5)
METHODS = (
    "act_native",
    "act_oracle",
    "act_calibrated_pointer",
    "act_measured_pointer_oracle",
    "learned_point_pointer",
)


def _motion_metrics(curve: np.ndarray) -> dict[str, float]:
    delta = np.diff(curve, axis=0)
    delta[:, 2] *= 0.25
    return {
        "measured_path_length_m_equivalent": float(
            np.linalg.norm(delta, axis=1).sum()
        ),
        "endpoint_displacement_m": float(
            np.linalg.norm(curve[-1, :2] - curve[0, :2])
        ),
    }


def _path_reconstruction(calibrated: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    calibrated_curve = np.vstack((np.zeros((1, 3)), calibrated.astype(np.float64)))
    measured_curve = np.vstack((np.zeros((1, 3)), measured.astype(np.float64)))
    calibrated_sampled = _resample_progress(calibrated_curve)
    measured_sampled = _resample_progress(measured_curve)
    return {
        "endpoint_translation_error_m": float(
            np.linalg.norm(calibrated_curve[-1, :2] - measured_curve[-1, :2])
        ),
        "endpoint_yaw_error_rad": float(
            abs(calibrated_curve[-1, 2] - measured_curve[-1, 2])
        ),
        "frechet_m_equivalent": _frechet(calibrated_sampled, measured_sampled),
        "hausdorff_m_equivalent": _hausdorff(calibrated_sampled, measured_sampled),
        "calibrated_path_length_m_equivalent": _motion_metrics(calibrated_curve)[
            "measured_path_length_m_equivalent"
        ],
        "measured_path_length_m_equivalent": _motion_metrics(measured_curve)[
            "measured_path_length_m_equivalent"
        ],
    }


def _sign_test(wins: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    tail = min(wins, non_ties - wins)
    return min(
        1.0,
        2.0 * sum(math.comb(non_ties, k) for k in range(tail + 1)) / 2**non_ties,
    )


def _paired(per_seed: list[dict], left: str, right: str, metric: str) -> dict:
    left_values = []
    right_values = []
    for seed_row in per_seed:
        left_values.append(
            float(
                np.mean(
                    [
                        seed_row["methods"][left]["rates"][rate][metric]
                        for rate in ("0.5", "1.5")
                    ]
                )
            )
        )
        right_values.append(
            float(
                np.mean(
                    [
                        seed_row["methods"][right]["rates"][rate][metric]
                        for rate in ("0.5", "1.5")
                    ]
                )
            )
        )
    wins = sum(left_value < right_value for left_value, right_value in zip(left_values, right_values))
    non_ties = sum(
        abs(left_value - right_value) >= 1e-12
        for left_value, right_value in zip(left_values, right_values)
    )
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    return {
        "lower_is_better": True,
        "left": left,
        "right": right,
        "left_better_seed_pairs": f"{wins}/{len(left_values)}",
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_reduction_fraction": 1.0 - left_mean / right_mean,
        "exact_two_sided_seed_sign_test_p": _sign_test(wins, non_ties),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=10)
    args = parser.parse_args()
    seeds = list(range(20260818, 20260818 + args.num_seeds))
    per_seed = []
    accum = {method: [] for method in METHODS}
    nominal_motion = {method: [] for method in METHODS}
    reconstruction = []

    for seed in seeds:
        directory = args.root / "seeds" / f"seed_{seed}"
        if not (directory / "STRICT_ACT_PATH_ABLATION_COMPLETE").is_file():
            raise RuntimeError(f"missing completed seed {seed}")
        meta = json.loads((directory / "summary.json").read_text())
        if {x["model_input_sha256"] for x in meta["conditions"]} != {
            meta["capture_input_sha256"]
        }:
            raise RuntimeError(f"model input changed within seed {seed}")
        if max(x["restored_state_max_abs_error"] for x in meta["conditions"]) > 1e-12:
            raise RuntimeError(f"state replay mismatch within seed {seed}")
        reconstruction_row = _path_reconstruction(
            np.load(directory / "act_calibrated_path.npy"),
            np.load(directory / "act_measured_path.npy"),
        )
        reconstruction.append(reconstruction_row)
        seed_row = {
            "seed": seed,
            "language": meta["language"],
            "act_command_path_reconstruction": reconstruction_row,
            "methods": {},
        }
        for method in METHODS:
            traces = {}
            plan_hashes = set()
            for rate in RATES:
                tag = str(rate).replace(".", "p")
                path = directory / f"{method}_rate{tag}/trace.jsonl"
                rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
                if len(rows) != int(meta["steps"]):
                    raise RuntimeError(f"trace length mismatch: {path}")
                plan_hashes.update(x["execution_plan_sha256"] for x in rows)
                traces[rate] = rows
            if len(plan_hashes) != 1:
                raise RuntimeError(f"execution plan changed across rates: {seed} {method}")
            reference_curve = _local_curve(traces[1.0])
            reference_sampled = _resample_progress(reference_curve)
            reference_path_length = _motion_metrics(reference_curve)[
                "measured_path_length_m_equivalent"
            ]
            if reference_path_length <= 1e-12:
                raise RuntimeError(f"zero-motion 1x reference: {seed} {method}")
            method_row = {
                "execution_plan_sha256": next(iter(plan_hashes)),
                "rates": {},
            }
            for rate in RATES:
                rows = traces[rate]
                curve = _local_curve(rows)
                sampled = _resample_progress(curve)
                commands = np.asarray(
                    [x["base_command"][:3] for x in rows], dtype=np.float64
                )
                metrics = {
                    "endpoint_translation_difference_to_1x_m": float(
                        np.linalg.norm(curve[-1, :2] - reference_curve[-1, :2])
                    ),
                    "endpoint_yaw_difference_to_1x_rad": float(
                        abs(curve[-1, 2] - reference_curve[-1, 2])
                    ),
                    "frechet_metric_m_equivalent_to_1x": _frechet(
                        sampled, reference_sampled
                    ),
                    "hausdorff_metric_m_equivalent_to_1x": _hausdorff(
                        sampled, reference_sampled
                    ),
                    "command_saturation_fraction": float(
                        np.mean(np.abs(commands) >= 1.0 - 1e-7)
                    ),
                    **_motion_metrics(curve),
                }
                metrics[
                    "endpoint_translation_difference_fraction_of_1x_path"
                ] = (
                    metrics["endpoint_translation_difference_to_1x_m"]
                    / reference_path_length
                )
                metrics["frechet_fraction_of_1x_path"] = (
                    metrics["frechet_metric_m_equivalent_to_1x"]
                    / reference_path_length
                )
                method_row["rates"][str(rate)] = metrics
                if rate == 1.0:
                    nominal_motion[method].append(
                        metrics["measured_path_length_m_equivalent"]
                    )
                else:
                    accum[method].append(metrics)
            seed_row["methods"][method] = method_row
        per_seed.append(seed_row)

    summary = {}
    for method, rows in accum.items():
        summary[method] = {
            "mean_endpoint_translation_difference_to_1x_m": float(
                np.mean([x["endpoint_translation_difference_to_1x_m"] for x in rows])
            ),
            "mean_endpoint_yaw_difference_to_1x_rad": float(
                np.mean([x["endpoint_yaw_difference_to_1x_rad"] for x in rows])
            ),
            "mean_frechet_to_1x_m_equivalent": float(
                np.mean([x["frechet_metric_m_equivalent_to_1x"] for x in rows])
            ),
            "mean_command_saturation_fraction": float(
                np.mean([x["command_saturation_fraction"] for x in rows])
            ),
            "mean_nominal_1x_measured_path_length_m_equivalent": float(
                np.mean(nominal_motion[method])
            ),
            "mean_endpoint_translation_difference_fraction_of_1x_path": float(
                np.mean([
                    x["endpoint_translation_difference_fraction_of_1x_path"]
                    for x in rows
                ])
            ),
            "mean_frechet_fraction_of_1x_path": float(
                np.mean([x["frechet_fraction_of_1x_path"] for x in rows])
            ),
        }
    reconstruction_summary = {
        key: float(np.mean([row[key] for row in reconstruction]))
        for key in reconstruction[0]
    }
    comparisons = {}
    for metric in (
        "endpoint_translation_difference_to_1x_m",
        "frechet_metric_m_equivalent_to_1x",
        "endpoint_translation_difference_fraction_of_1x_path",
        "frechet_fraction_of_1x_path",
    ):
        comparisons[f"act_calibrated_pointer_vs_act_native/{metric}"] = _paired(
            per_seed, "act_calibrated_pointer", "act_native", metric
        )
        comparisons[f"learned_point_pointer_vs_act_calibrated_pointer/{metric}"] = _paired(
            per_seed, "learned_point_pointer", "act_calibrated_pointer", metric
        )
        comparisons[f"act_measured_pointer_oracle_vs_act_calibrated_pointer/{metric}"] = _paired(
            per_seed, "act_measured_pointer_oracle", "act_calibrated_pointer", metric
        )

    meaningful_motion = {
        method: float(np.mean(values)) >= 0.05
        for method, values in nominal_motion.items()
    }
    payload = {
        "protocol": {
            "task": "NavigateKitchen",
            "seeds": seeds,
            "control_hz": 50,
            "trial_seconds": 5.0,
            "rate_scales": list(RATES),
            "strict_simulator_state_replay": True,
            "act_calibrated_path_semantics": "command @ calibration matrix + bias, then SE(2) integration; predicted motion, not ground truth",
            "act_measured_path_semantics": "actual 1x simulator rollout; non-deployable execution oracle",
            "scope": "base-only frozen-output causal execution ablation, not task success",
        },
        "meaningful_motion_gate_at_1x_ge_0p05_m_equivalent": meaningful_motion,
        "all_meaningful_motion": bool(all(meaningful_motion.values())),
        "act_command_path_reconstruction": reconstruction_summary,
        "summary": summary,
        "paired_comparisons": comparisons,
        "per_seed": per_seed,
    }
    (args.root / "analysis").mkdir(exist_ok=True)
    (args.root / "analysis/strict-act-path-ablation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (args.root / "STRICT_ACT_PATH_ABLATION_SWEEP_COMPLETE").write_text("\n")
    print(json.dumps({"event": "STRICT_ACT_PATH_ABLATION_SWEEP_COMPLETE", "summary": summary}))


if __name__ == "__main__":
    main()
