#!/usr/bin/env python3
"""Build the frozen Phase 1 M1 decision artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_summary(root: Path, condition: str, seeds: list[int]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in seeds:
        path = root / condition / f"seed-{seed:03d}" / "rollout.json"
        if not path.exists() or not path.with_suffix(".npz").exists():
            missing.append(seed)
            continue
        reports.append(_load(path))
    successes = sum(bool(report["rollout"]["success"]) for report in reports)
    unsafe = [
        int(report["seed"])
        for report in reports
        if not report.get("safety", {}).get("telemetry_all_finite", False)
        or int(report.get("safety", {}).get("base_twist_cap_violations", 1)) > 0
        or bool(report.get("safety", {}).get("unsuccessful_early_termination", True))
    ]
    frequencies = [
        float(report["replan_schedule"]["realized_frequency_hz"])
        for report in reports
    ]
    return {
        "condition": condition,
        "completed_runs": len(reports),
        "missing_seeds": missing,
        "successes": successes,
        "success_rate": successes / len(reports) if reports else None,
        "realized_frequency_hz": {
            "min": float(np.min(frequencies)) if frequencies else None,
            "median": float(np.median(frequencies)) if frequencies else None,
            "max": float(np.max(frequencies)) if frequencies else None,
        },
        "unsafe_seeds": unsafe,
        "safety_checks_passed": bool(reports and not unsafe),
    }


def evaluate(
    standard: dict[str, Any],
    matched_lora: dict[str, Any],
    config: dict[str, Any],
    extended: dict[str, Any],
    dataset_manifest: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["resistance_thresholds"]
    required = int(config["validity"]["minimum_complete_seed_pairs_per_frequency"])
    checks: dict[str, Any] = {}
    for steps in (8, 4):
        paired = standard["paired_against_reference"][str(steps)]
        xy = float(paired["root_xy_trajectory_rms_m"]["median"])
        yaw = float(paired["root_yaw_trajectory_rms_rad"]["median"])
        success_change = float(paired["success_rate_change"])
        complete = int(paired["pairs"]) >= required
        prefix_valid = bool(
            paired["all_first_action_chunks_identical"]
            and paired["all_prebranch_physical_telemetry_identical"]
        )
        below = bool(
            xy < float(thresholds["maximum_median_root_xy_trajectory_rms_m"])
            and yaw < float(thresholds["maximum_median_root_yaw_trajectory_rms_rad"])
            and abs(success_change)
            < float(thresholds["maximum_absolute_success_rate_change"])
        )
        checks[str(steps)] = {
            "complete": complete,
            "forced_prefix_valid": prefix_valid,
            "median_root_xy_trajectory_rms_m": xy,
            "median_root_yaw_trajectory_rms_rad": yaw,
            "success_rate_change": success_change,
            "below_all_thresholds": below,
            "passed": bool(complete and prefix_valid and below),
        }

    default_success = float(standard["conditions"]["16"]["success_rate"])
    matched_success = float(matched_lora["conditions"]["16"]["success_rate"])
    default_drop = matched_success - default_success
    default_preserved = bool(
        default_drop
        <= float(thresholds["maximum_default_frequency_success_drop_vs_matched_lora"])
    )
    measured_target = bool(
        dataset_manifest.get("target_source") == "measured_odometry"
        and dataset_manifest.get("target_source_counts", {}).get("measured_odometry", 0) > 0
    )
    training_complete = bool(
        int(training_manifest.get("global_step", -1))
        == int(config["frozen_variables"]["optimizer_steps"])
    )
    extended_complete = bool(
        extended
        and all(value["completed_runs"] == required for value in extended.values())
    )
    safety_passed = bool(
        extended
        and all(value["safety_checks_passed"] for value in extended.values())
    )
    primary_passed = bool(all(value["passed"] for value in checks.values()))
    passed = bool(
        primary_passed
        and default_preserved
        and measured_target
        and training_complete
        and extended_complete
        and safety_passed
    )
    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "upstream_revisions": training_manifest.get("upstream", {}),
        "dataset": {
            "revision": dataset_manifest.get("dataset_revision"),
            "episodes": dataset_manifest.get("episodes"),
            "frames": dataset_manifest.get("frames"),
            "valid_anchors": dataset_manifest.get("valid_anchors"),
            "target_source_counts": dataset_manifest.get("target_source_counts", {}),
            "measured_odometry_target_required_and_present": measured_target,
        },
        "training": training_manifest,
        "default_capability": {
            "matched_lora_success_rate": matched_success,
            "m1_success_rate": default_success,
            "success_rate_drop": default_drop,
            "maximum_allowed_drop": thresholds[
                "maximum_default_frequency_success_drop_vs_matched_lora"
            ],
            "passed": default_preserved,
        },
        "primary_frequency_checks": checks,
        "deployment_frequency_checks": extended,
        "material_safety_regression_absent": safety_passed,
        "all_required_artifacts_complete": bool(extended_complete and training_complete),
        "m1_passed": passed,
        "decision": (
            "stop_and_report_action_representation_as_sufficient"
            if passed
            else "stop_and_classify_phase_1_m1_failure_before_any_phase_2_work"
        ),
        "plain_language": (
            "M1 同时守住默认能力、跨频率轨迹/成功率、部署频段完整性和安全门禁；"
            "按预注册路线停止，不进入 Phase 2。"
            if passed
            else "M1 至少一项冻结门禁未通过；先定位数据、归一化、tracker、默认能力或跨频率失败，"
            "不得直接进入 Phase 2。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-report", type=Path, required=True)
    parser.add_argument("--matched-lora-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["frozen_variables"]["seeds"]]
    conditions = [
        "fixed-3.125hz",
        "fixed-6.25hz",
        "fixed-12.5hz",
        "fixed-10hz",
        "fixed-15hz",
        "fixed-20hz",
        "fixed-30hz",
        "jitter-10-30hz",
    ]
    extended = {
        condition: _condition_summary(args.rollout_root, condition, seeds)
        for condition in conditions
    }
    result = evaluate(
        _load(args.standard_report),
        _load(args.matched_lora_report),
        config,
        extended,
        _load(args.dataset_manifest),
        _load(args.training_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
