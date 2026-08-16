#!/usr/bin/env python3
"""Build the frozen Phase 1 M1 decision artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


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
    tracker_metrics = {
        key: _numeric_summary(
            [float(report.get("consumer", {}).get(key, float("nan"))) for report in reports]
        )
        for key in (
            "plan_age_s_median",
            "plan_age_s_max",
            "clamped_to_horizon_steps",
            "position_error_m_median",
            "yaw_error_rad_median",
            "replan_velocity_jump_norm_max",
        )
    }
    return {
        "condition": condition,
        "completed_runs": len(reports),
        "missing_seeds": missing,
        "successes": successes,
        "success_rate": successes / len(reports) if reports else None,
        "success_rate_wilson_95": _wilson_interval(successes, len(reports)),
        "realized_frequency_hz": {
            "min": float(np.min(frequencies)) if frequencies else None,
            "median": float(np.median(frequencies)) if frequencies else None,
            "max": float(np.max(frequencies)) if frequencies else None,
        },
        "unsafe_seeds": unsafe,
        "safety_checks_passed": bool(reports and not unsafe),
        "tracker": tracker_metrics,
    }


def _failure_classification(
    *,
    default_preserved: bool,
    checks: dict[str, Any],
    measured_target: bool,
    training_complete: bool,
    safety_passed: bool,
    extended: dict[str, Any],
    action_horizon_s: float,
) -> dict[str, Any]:
    invalid_prefix_conditions = [
        f"{50 / int(steps):g}hz"
        for steps, value in checks.items()
        if value["complete"] and not value["forced_prefix_valid"]
    ]
    failed_frequency_thresholds = [
        f"{50 / int(steps):g}hz"
        for steps, value in checks.items()
        if value["complete"] and not value["below_all_thresholds"]
    ]
    default_tracker = extended.get("fixed-3.125hz", {}).get("tracker", {})
    median_plan_age = default_tracker.get("plan_age_s_median", {}).get("median")
    median_clamped_steps = default_tracker.get("clamped_to_horizon_steps", {}).get("median")
    clock_domain_candidate = bool(
        median_plan_age is not None
        and median_clamped_steps is not None
        and (
            float(median_plan_age) > action_horizon_s
            or float(median_clamped_steps) > 0.0
        )
    )
    return {
        "confirmed_failure_axes": {
            "default_capability": not default_preserved,
            "frequency_thresholds_crossed": failed_frequency_thresholds,
            "paired_frequency_causal_validity": not invalid_prefix_conditions,
        },
        "passed_or_completed_checks": {
            "measured_odometry_target": measured_target,
            "frozen_training_budget": training_complete,
            "material_safety": safety_passed,
        },
        "cross_frequency_interpretation": {
            "invalid_prefix_conditions": invalid_prefix_conditions,
            "status": (
                "descriptive_differences_only_do_not_make_a_clean_frequency_causal_claim"
                if invalid_prefix_conditions
                else "paired_frequency_comparison_valid"
            ),
        },
        "leading_diagnostic_candidate": {
            "class": "tracker_simulation_clock_domain_mismatch",
            "status": "candidate_not_causal_proof" if clock_domain_candidate else "not_observed",
            "action_horizon_s": action_horizon_s,
            "default_plan_age_s_median_across_seeds": median_plan_age,
            "default_clamped_steps_median_across_seeds": median_clamped_steps,
            "reason": (
                f"The tracker queried a {action_horizon_s:g} s physical-time plan with process "
                "wall clock while the simulator did not advance at one-to-one real time."
                if clock_domain_candidate
                else "The default rollout did not show plan-age or horizon-clamp evidence."
            ),
        },
        "unresolved_competing_classes": [
            "full_dataset_target_or_normalization_generalization",
            "closed_loop_tracker_activation_and_reanchor",
        ],
        "next_action": (
            "run_scoped_phase_1_clock_and_normalization_ablations_before_any_phase_2_work"
        ),
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
    failure_classification = (
        None
        if passed
        else _failure_classification(
            default_preserved=default_preserved,
            checks=checks,
            measured_target=measured_target,
            training_complete=training_complete,
            safety_passed=safety_passed,
            extended=extended,
            action_horizon_s=(
                int(config["frozen_variables"]["action_horizon"])
                * float(config["frozen_variables"]["action_slot_dt_s"])
            ),
        )
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
        "primary_conditions": standard["conditions"],
        "primary_frequency_checks": checks,
        "deployment_frequency_checks": extended,
        "material_safety_regression_absent": safety_passed,
        "all_required_artifacts_complete": bool(extended_complete and training_complete),
        "m1_passed": passed,
        "failure_classification": failure_classification,
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
