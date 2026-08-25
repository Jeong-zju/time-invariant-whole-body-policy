"""Audit frozen-output variable-rate trials and quantify geometric consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RATES = (0.5, 1.0, 1.5)
METHODS = ("act", "point")
YAW_RADIUS_M = 0.25


def _tag(rate: float) -> str:
    return str(rate).replace(".", "p")


def _local_curve(rows: list[dict]) -> np.ndarray:
    xy = np.asarray([row["position"] for row in rows], dtype=np.float64)
    yaw = np.unwrap(np.asarray([row["yaw"] for row in rows], dtype=np.float64))
    origin_xy, origin_yaw = xy[0].copy(), float(yaw[0])
    delta = xy - origin_xy
    c, s = np.cos(origin_yaw), np.sin(origin_yaw)
    local_xy = np.column_stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]))
    return np.column_stack((local_xy, yaw - origin_yaw))


def _resample_progress(curve: np.ndarray, count: int = 101) -> np.ndarray:
    metric = curve.copy()
    metric[:, 2] *= YAW_RADIUS_M
    progress = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(metric, axis=0), axis=-1))))
    if progress[-1] <= 1e-12:
        return np.repeat(metric[:1], count, axis=0)
    keep = np.concatenate(([True], np.diff(progress) > 1e-12))
    progress, metric = progress[keep], metric[keep]
    target = np.linspace(0.0, progress[-1], count)
    return np.column_stack([np.interp(target, progress, metric[:, index]) for index in range(3)])


def _frechet(a: np.ndarray, b: np.ndarray) -> float:
    cache = np.full((len(a), len(b)), np.nan)
    def visit(i: int, j: int) -> float:
        if np.isfinite(cache[i, j]): return float(cache[i, j])
        distance = float(np.linalg.norm(a[i] - b[j]))
        if i == 0 and j == 0: value = distance
        elif i == 0: value = max(visit(0, j - 1), distance)
        elif j == 0: value = max(visit(i - 1, 0), distance)
        else: value = max(min(visit(i - 1, j), visit(i - 1, j - 1), visit(i, j - 1)), distance)
        cache[i, j] = value
        return value
    return visit(len(a) - 1, len(b) - 1)


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    conditions: dict[tuple[str, float], dict[int, dict]] = {}
    for method in METHODS:
        for rate in RATES:
            directory = args.root / "evaluation" / f"{method}_rate{_tag(rate)}_seed20260818_count10"
            if not (directory / "EVALUATION_COMPLETE").is_file():
                raise RuntimeError(f"incomplete condition: {directory}")
            trials = json.loads((directory / "results.json").read_text())["trials"]
            trace = [json.loads(x) for x in (directory / "logs/action-trace.jsonl").read_text().splitlines() if x.strip()]
            by_episode: dict[int, list[dict]] = {}
            for row in trace:
                if row["event"] == "ACTION": by_episode.setdefault(int(row["episode_id"]), []).append(row)
            mapped = {}
            for trial in trials:
                rows = by_episode[int(trial["episode_id"])]
                if len(rows) != 150: raise RuntimeError("trace length mismatch")
                hashes = {row["output_sha256"] for row in rows}
                if len(hashes) != 1: raise RuntimeError("model output changed inside frozen trial")
                curve = _local_curve(rows)
                if not np.isfinite(curve).all(): raise RuntimeError("non-finite measured curve")
                commands = np.asarray([row["base_command"][:3] for row in rows], dtype=np.float64)
                mapped[int(trial["seed"])] = {
                    "hash": next(iter(hashes)), "curve": curve, "commands": commands, "rows": rows,
                }
            conditions[(method, rate)] = mapped

    per_seed = []
    summary = {}
    for method in METHODS:
        hash_match = 0
        comparisons = []
        for seed in range(20260818, 20260828):
            hashes = [conditions[(method, rate)][seed]["hash"] for rate in RATES]
            same = len(set(hashes)) == 1
            hash_match += int(same)
            reference_curve = conditions[(method, 1.0)][seed]["curve"]
            reference = _resample_progress(reference_curve)
            row = {"method":method,"seed":seed,"same_model_output_all_rates":same,"rates":{}}
            for rate in RATES:
                item = conditions[(method, rate)][seed]
                curve = item["curve"]
                sampled = _resample_progress(curve)
                endpoint = curve[-1]
                endpoint_reference = reference_curve[-1]
                metrics = {
                    "endpoint_translation_difference_to_1x_m":float(np.linalg.norm(endpoint[:2]-endpoint_reference[:2])),
                    "endpoint_yaw_difference_to_1x_rad":float(abs(endpoint[2]-endpoint_reference[2])),
                    "frechet_metric_m_equivalent_to_1x":_frechet(sampled,reference),
                    "hausdorff_metric_m_equivalent_to_1x":_hausdorff(sampled,reference),
                    "measured_metric_path_length":float(np.linalg.norm(np.diff(np.column_stack((curve[:,:2],YAW_RADIUS_M*curve[:,2])),axis=0),axis=-1).sum()),
                    "command_saturation_fraction":float(np.mean(np.abs(item['commands']) >= 1.0-1e-7)),
                }
                if method == "act":
                    metrics["retime_unclipped_saturation_fraction"] = float(np.mean([
                        x["retime"]["unclipped_saturation_fraction"] for x in item["rows"]
                    ]))
                else:
                    metrics["final_reference_progress_m"] = float(item["rows"][-1]["pointer"]["reference_progress_m"])
                    metrics["final_measured_progress_m"] = float(item["rows"][-1]["pointer"]["measured_progress_m"])
                    metrics["final_endpoint_position_error_m"] = float(item["rows"][-1]["pointer"]["endpoint_position_error_m"])
                row["rates"][str(rate)] = metrics
                if rate != 1.0: comparisons.append(metrics)
            per_seed.append(row)
        if hash_match != 10:
            raise RuntimeError(f"{method}: only {hash_match}/10 seeds preserve identical model output")
        summary[method] = {
            "identical_model_output_across_rates":f"{hash_match}/10",
            "mean_frechet_to_1x_m_equivalent":float(np.mean([x["frechet_metric_m_equivalent_to_1x"] for x in comparisons])),
            "mean_hausdorff_to_1x_m_equivalent":float(np.mean([x["hausdorff_metric_m_equivalent_to_1x"] for x in comparisons])),
            "mean_endpoint_translation_difference_to_1x_m":float(np.mean([x["endpoint_translation_difference_to_1x_m"] for x in comparisons])),
            "mean_endpoint_yaw_difference_to_1x_rad":float(np.mean([x["endpoint_yaw_difference_to_1x_rad"] for x in comparisons])),
            "mean_command_saturation_fraction":float(np.mean([x["command_saturation_fraction"] for x in comparisons])),
        }
    payload = {
        "protocol": {
            "task":"NavigateKitchen", "seeds":[20260818,20260827], "num_seeds":10,
            "rate_scales":list(RATES), "control_hz":50, "fixed_single_prediction":True,
            "trial_seconds":3.0, "upper_body":"verified fixed hold",
            "act":"20 Hz velocity prefix, u_alpha(t)=alpha*u(alpha*t), source prefix 0.4 s",
            "point":"0.20 m-equivalent geometric prefix, external pointer at alpha*0.25 m-equivalent/s",
            "scope":"base-only execution-rate proof-of-concept; not full-body and not a task-success comparison",
        },
        "gate_passed": True,
        "summary": summary,
        "per_seed": per_seed,
    }
    (args.root / "analysis").mkdir(parents=True, exist_ok=True)
    (args.root / "analysis/rate-control-analysis.json").write_text(json.dumps(payload,indent=2)+"\n")
    (args.root / "RATE_CONTROL_EVALUATION_COMPLETE").write_text("\n")
    print(json.dumps({"event":"RATE_CONTROL_EVALUATION_COMPLETE","summary":summary}))


if __name__ == "__main__":
    main()
