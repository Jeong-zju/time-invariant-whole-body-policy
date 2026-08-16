#!/usr/bin/env python3
"""Measure same-frequency run-to-run noise for the frequency Gate N."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from summarize_frequency_gate import compare_pair, percentile_summary


def load_run(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_json_path"] = str(path)
    report["_npz_path"] = str(path.with_name(report["telemetry_npz"]))
    return report


def summarize(reference_root: Path, repeat_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    audit = config["repeatability_audit"]
    seeds = [int(seed) for seed in audit["seeds"]]
    pairs = []
    missing = []
    for seed in seeds:
        reference_path = reference_root / "replan-16" / f"seed-{seed:03d}" / "rollout.json"
        repeat_path = repeat_root / "repeat-16" / f"seed-{seed:03d}" / "rollout.json"
        if not reference_path.exists() or not repeat_path.exists():
            missing.append(seed)
            continue
        pairs.append(compare_pair(load_run(reference_path), load_run(repeat_path)))

    reference_successes = sum(pair["reference_success"] for pair in pairs)
    repeat_successes = sum(pair["candidate_success"] for pair in pairs)
    count = len(pairs)
    success_rate_change = (repeat_successes - reference_successes) / count if count else None
    valid = bool(
        not missing
        and pairs
        and all(pair["first_action_chunk_identical"] for pair in pairs)
        and all(pair["prebranch_physical_telemetry_identical"] for pair in pairs)
    )
    xy = percentile_summary(pair["root_xy_trajectory_rms_m"] for pair in pairs)
    yaw = percentile_summary(pair["root_yaw_trajectory_rms_rad"] for pair in pairs)
    thresholds = audit["maximum_tolerable_same_frequency_noise"]
    material_noise = bool(
        pairs
        and (
            xy["median"] >= float(thresholds["median_root_xy_trajectory_rms_m"])
            or yaw["median"] >= float(thresholds["median_root_yaw_trajectory_rms_rad"])
            or abs(float(success_rate_change)) >= float(thresholds["absolute_success_rate_change"])
        )
    )
    passed = bool(valid and not material_noise)
    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "audit": "same_frequency_repeatability",
        "expected_pairs": len(seeds),
        "completed_pairs": count,
        "missing_seeds": missing,
        "all_first_action_chunks_identical": bool(
            pairs and all(pair["first_action_chunk_identical"] for pair in pairs)
        ),
        "all_prebranch_physical_telemetry_identical": bool(
            pairs and all(pair["prebranch_physical_telemetry_identical"] for pair in pairs)
        ),
        "root_xy_trajectory_rms_m": xy,
        "root_yaw_trajectory_rms_rad": yaw,
        "reference_successes": reference_successes,
        "repeat_successes": repeat_successes,
        "success_rate_change": success_rate_change,
        "valid": valid,
        "material_same_frequency_noise": material_noise,
        "repeatability_control_passed": passed,
        "decision": (
            "frequency_effect_exceeds_same_frequency_noise"
            if passed
            else "do_not_attribute_effect_to_frequency_without_more_controls"
        ),
        "per_seed": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = summarize(args.reference_root, args.repeat_root, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
