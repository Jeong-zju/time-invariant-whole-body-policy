"""Analyze exact-state paired rate trials produced by strict_rate_replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .analyze_rate_control_sweep import _frechet, _hausdorff, _local_curve, _resample_progress


RATES = (0.5, 1.0, 1.5)
METHODS = ("act_native", "act_oracle", "point")


def _motion_metrics(curve: np.ndarray) -> dict[str, float]:
    delta = np.diff(curve, axis=0)
    metric_delta = delta.copy()
    metric_delta[:, 2] *= 0.25
    return {
        "measured_path_length_m_equivalent": float(np.linalg.norm(metric_delta, axis=1).sum()),
        "endpoint_displacement_m": float(np.linalg.norm(curve[-1, :2] - curve[0, :2])),
        "absolute_yaw_change_rad": float(abs(curve[-1, 2] - curve[0, 2])),
    }


def _paired_point_comparison(per_seed: list[dict]) -> dict[str, dict]:
    comparisons = {}
    for baseline in ("act_native", "act_oracle"):
        comparison = {}
        for metric in (
            "endpoint_translation_difference_to_1x_m",
            "frechet_metric_m_equivalent_to_1x",
        ):
            baseline_values = []
            point_values = []
            for seed_row in per_seed:
                baseline_values.append(
                    float(np.mean([
                        seed_row["methods"][baseline]["rates"][rate][metric]
                        for rate in ("0.5", "1.5")
                    ]))
                )
                point_values.append(
                    float(np.mean([
                        seed_row["methods"]["point"]["rates"][rate][metric]
                        for rate in ("0.5", "1.5")
                    ]))
                )
            wins = sum(
                point < base for point, base in zip(point_values, baseline_values)
            )
            non_ties = sum(
                abs(point - base) >= 1e-12
                for point, base in zip(point_values, baseline_values)
            )
            smaller_tail = min(wins, non_ties - wins)
            exact_sign_p = min(
                1.0,
                2.0
                * sum(math.comb(non_ties, k) for k in range(smaller_tail + 1))
                / (2**non_ties),
            )
            baseline_mean = float(np.mean(baseline_values))
            point_mean = float(np.mean(point_values))
            comparison[metric] = {
                "point_better_pairs": f"{wins}/{len(point_values)}",
                "baseline_mean": baseline_mean,
                "point_mean": point_mean,
                "point_reduction_fraction": 1.0 - point_mean / baseline_mean,
                "exact_two_sided_sign_test_p": exact_sign_p,
            }
        comparisons[f"point_vs_{baseline}"] = comparison
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=10)
    args = parser.parse_args()
    seeds = list(range(20260818, 20260818 + args.num_seeds))
    per_seed = []
    accum = {method: [] for method in METHODS}
    nominal_motion = {method: [] for method in METHODS}
    for seed in seeds:
        directory = args.root / "seeds" / f"seed_{seed}"
        if not (directory / "STRICT_REPLAY_COMPLETE").is_file():
            raise RuntimeError(f"missing strict replay seed {seed}")
        seed_summary = json.loads((directory / "summary.json").read_text())
        input_hashes = {x["model_input_sha256"] for x in seed_summary["conditions"]}
        if input_hashes != {seed_summary["capture_input_sha256"]}:
            raise RuntimeError(f"seed {seed} did not use one exact observation")
        if max(float(x["restored_state_max_abs_error"]) for x in seed_summary["conditions"]) > 1e-12:
            raise RuntimeError(f"seed {seed} did not restore one exact simulator state")
        seed_row = {"seed": seed, "language": seed_summary["language"], "methods": {}}
        for method in METHODS:
            traces = {}
            hashes = set()
            for rate in RATES:
                tag = str(rate).replace(".", "p")
                rows = [json.loads(x) for x in (directory / f"{method}_rate{tag}/trace.jsonl").read_text().splitlines() if x.strip()]
                expected_steps = int(seed_summary["steps"])
                if len(rows) != expected_steps: raise RuntimeError("trace length mismatch")
                hashes.update(x["output_sha256"] for x in rows)
                traces[rate] = rows
            if len(hashes) != 1: raise RuntimeError("frozen output changed across rates")
            reference_curve = _local_curve(traces[1.0])
            reference = _resample_progress(reference_curve)
            method_row = {"output_sha256": next(iter(hashes)), "rates": {}}
            for rate in RATES:
                rows = traces[rate]
                curve = _local_curve(rows)
                sampled = _resample_progress(curve)
                commands = np.asarray([x["base_command"][:3] for x in rows], dtype=np.float64)
                metrics = {
                    "endpoint_translation_difference_to_1x_m": float(np.linalg.norm(curve[-1,:2]-reference_curve[-1,:2])),
                    "endpoint_yaw_difference_to_1x_rad": float(abs(curve[-1,2]-reference_curve[-1,2])),
                    "frechet_metric_m_equivalent_to_1x": _frechet(sampled, reference),
                    "hausdorff_metric_m_equivalent_to_1x": _hausdorff(sampled, reference),
                    "command_saturation_fraction": float(np.mean(np.abs(commands) >= 1.0-1e-7)),
                    **_motion_metrics(curve),
                }
                if method.startswith("act_"):
                    metrics["unclipped_retime_saturation_fraction"] = float(np.mean([x["retime"]["unclipped_saturation_fraction"] for x in rows]))
                else:
                    metrics["final_reference_progress_m"] = float(rows[-1]["pointer"]["reference_progress_m"])
                    metrics["final_measured_progress_m"] = float(rows[-1]["pointer"]["measured_progress_m"])
                    metrics["final_endpoint_position_error_m"] = float(rows[-1]["pointer"]["endpoint_position_error_m"])
                method_row["rates"][str(rate)] = metrics
                if rate != 1.0: accum[method].append(metrics)
                else: nominal_motion[method].append(metrics["measured_path_length_m_equivalent"])
            seed_row["methods"][method] = method_row
        per_seed.append(seed_row)
    summary = {}
    for method, values in accum.items():
        summary[method] = {
            "exact_same_input_and_output": f"{len(seeds)}/{len(seeds)}",
            "mean_frechet_to_1x_m_equivalent": float(np.mean([x["frechet_metric_m_equivalent_to_1x"] for x in values])),
            "mean_hausdorff_to_1x_m_equivalent": float(np.mean([x["hausdorff_metric_m_equivalent_to_1x"] for x in values])),
            "mean_endpoint_translation_difference_to_1x_m": float(np.mean([x["endpoint_translation_difference_to_1x_m"] for x in values])),
            "mean_endpoint_yaw_difference_to_1x_rad": float(np.mean([x["endpoint_yaw_difference_to_1x_rad"] for x in values])),
            "mean_command_saturation_fraction": float(np.mean([x["command_saturation_fraction"] for x in values])),
            "mean_nominal_1x_measured_path_length_m_equivalent": float(np.mean(nominal_motion[method])),
        }
    meaningful_motion = {
        method: float(np.mean(nominal_motion[method])) >= 0.05 for method in METHODS
    }
    payload = {
        "protocol": {
            "task":"NavigateKitchen", "seeds":seeds, "control_hz":50, "trial_seconds":5.0,
            "rate_scales":list(RATES), "strict_simulator_state_replay":True,
            "one_model_prediction_per_seed_and_method":True,
            "act_native":"full 1.6 s / 32-token velocity chunk played as u(alpha*t), without amplitude compensation",
            "act_oracle":"diagnostic upper bound u_alpha(t)=alpha*u(alpha*t); not native ACT",
            "point":"same 0.40 m-equivalent path, pointer rate alpha*0.25 m-equivalent/s",
            "scope":"base-only frozen-output geometric invariance gate; task success is not the endpoint",
        },
        "gate_passed": bool(all(meaningful_motion.values())),
        "meaningful_motion_gate_at_1x_ge_0p05_m_equivalent": meaningful_motion,
        "summary": summary,
        "paired_comparisons": _paired_point_comparison(per_seed),
        "per_seed": per_seed,
    }
    (args.root / "analysis").mkdir(exist_ok=True)
    (args.root / "analysis/strict-rate-analysis.json").write_text(json.dumps(payload,indent=2)+"\n")
    (args.root / "STRICT_RATE_SWEEP_COMPLETE").write_text("\n")
    print(json.dumps({"event":"STRICT_RATE_SWEEP_COMPLETE","summary":summary}))


if __name__ == "__main__":
    main()
