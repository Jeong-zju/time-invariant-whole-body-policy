"""Analyze exact-state stale-plan alignment replays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .analyze_rate_control_sweep import _frechet, _hausdorff, _local_curve, _resample_progress


DELAYS_S = (0.4, 0.8)


def _trace(path: Path, steps: int) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows) != steps:
        raise RuntimeError(f"trace length mismatch: {path} {len(rows)} != {steps}")
    return rows


def _sign_test(wins: int, total: int) -> float:
    tail = min(wins, total - wins)
    return min(1.0, 2.0 * sum(math.comb(total, k) for k in range(tail + 1)) / 2**total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=10)
    args = parser.parse_args()
    seeds = list(range(20260818, 20260818 + args.num_seeds))
    per_seed = []
    values = {"act": [], "point": []}
    for seed in seeds:
        directory = args.root / "seeds" / f"seed_{seed}"
        if not (directory / "STRICT_ALIGNMENT_REPLAY_COMPLETE").is_file():
            raise RuntimeError(f"missing completed seed {seed}")
        meta = json.loads((directory / "summary.json").read_text())
        if {x["model_input_sha256"] for x in meta["conditions"]} != {
            meta["capture_input_sha256"]
        }:
            raise RuntimeError(f"input changed within seed {seed}")
        if max(x["restored_state_max_abs_error"] for x in meta["conditions"]) > 1e-12:
            raise RuntimeError(f"state replay mismatch within seed {seed}")
        seed_row = {"seed": seed, "methods": {}}
        for method in ("act", "point"):
            reference_name = f"{method}_reference_delay0p0"
            reference_rows = _trace(
                directory / reference_name / "trace.jsonl", meta["steps"]
            )
            reference_curve = _local_curve(reference_rows)
            reference_sampled = _resample_progress(reference_curve)
            adapter = "stale_restart" if method == "act" else "projected_pointer"
            method_row = {"reference_output_sha256": reference_rows[0]["output_sha256"], "delays": {}}
            for delay in DELAYS_S:
                tag = str(delay).replace(".", "p")
                rows = _trace(
                    directory / f"{method}_{adapter}_delay{tag}" / "trace.jsonl",
                    meta["steps"],
                )
                if {x["output_sha256"] for x in rows} != {method_row["reference_output_sha256"]}:
                    raise RuntimeError(f"output changed: seed={seed} method={method}")
                if sum(bool(x["alignment_event"]) for x in rows) != 1:
                    raise RuntimeError(f"alignment event mismatch: seed={seed} method={method}")
                curve = _local_curve(rows)
                sampled = _resample_progress(curve)
                time_delta = curve - reference_curve
                time_delta[:, 2] *= 0.25
                time_error = np.linalg.norm(time_delta, axis=1)
                metrics = {
                    "endpoint_translation_difference_to_reference_m": float(
                        np.linalg.norm(curve[-1, :2] - reference_curve[-1, :2])
                    ),
                    "endpoint_yaw_difference_to_reference_rad": float(
                        abs(curve[-1, 2] - reference_curve[-1, 2])
                    ),
                    "frechet_metric_m_equivalent_to_reference": _frechet(
                        sampled, reference_sampled
                    ),
                    "hausdorff_metric_m_equivalent_to_reference": _hausdorff(
                        sampled, reference_sampled
                    ),
                    "time_aligned_rmse_m_equivalent_to_reference": float(
                        np.sqrt(np.mean(time_error**2))
                    ),
                    "time_aligned_mean_error_m_equivalent_to_reference": float(
                        np.mean(time_error)
                    ),
                    "time_aligned_max_error_m_equivalent_to_reference": float(
                        np.max(time_error)
                    ),
                }
                method_row["delays"][str(delay)] = metrics
                values[method].append(metrics)
            seed_row["methods"][method] = method_row
        per_seed.append(seed_row)

    summary = {}
    for method, rows in values.items():
        summary[method] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
    paired = {}
    for metric in (
        "endpoint_translation_difference_to_reference_m",
        "frechet_metric_m_equivalent_to_reference",
        "time_aligned_rmse_m_equivalent_to_reference",
    ):
        act_values = [
            float(np.mean([
                seed_row["methods"]["act"]["delays"][str(delay)][metric]
                for delay in DELAYS_S
            ]))
            for seed_row in per_seed
        ]
        point_values = [
            float(np.mean([
                seed_row["methods"]["point"]["delays"][str(delay)][metric]
                for delay in DELAYS_S
            ]))
            for seed_row in per_seed
        ]
        wins = sum(point < act for point, act in zip(point_values, act_values))
        paired[metric] = {
            "point_better_pairs": f"{wins}/{len(act_values)}",
            "point_reduction_fraction": 1.0 - np.mean(point_values) / np.mean(act_values),
            "exact_two_sided_sign_test_p": _sign_test(wins, len(act_values)),
        }
    payload = {
        "protocol": {
            "task": "NavigateKitchen",
            "seeds": seeds,
            "control_hz": 50,
            "trial_seconds": 5.0,
            "stale_plan_delays_s": list(DELAYS_S),
            "strict_simulator_state_replay": True,
            "act": "restart byte-identical stale 32-token chunk at token zero",
            "point": "keep stale world path and restore pointer from measured-pose projection",
            "scope": "executor alignment gate, not closed-loop task success",
        },
        "summary": summary,
        "paired_comparison": paired,
        "per_seed": per_seed,
    }
    (args.root / "analysis").mkdir(exist_ok=True)
    (args.root / "analysis/strict-alignment-analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (args.root / "STRICT_ALIGNMENT_SWEEP_COMPLETE").write_text("\n")
    print(json.dumps({"event": "STRICT_ALIGNMENT_SWEEP_COMPLETE", "summary": summary}))


if __name__ == "__main__":
    main()
