#!/usr/bin/env python3
"""Evaluate the complete Gate N simple-baseline stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _frequency_resistance(
    report: dict[str, Any],
    thresholds: dict[str, Any],
    required_pairs: int,
    require_forced_prefix: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for replan_steps in (8, 4):
        paired = report["paired_against_reference"][str(replan_steps)]
        complete = int(paired["pairs"]) >= required_pairs
        forced_prefix_valid = bool(
            paired["all_first_action_chunks_identical"]
            and paired["all_prebranch_physical_telemetry_identical"]
        )
        valid = forced_prefix_valid if require_forced_prefix else True
        xy = float(paired["root_xy_trajectory_rms_m"]["median"])
        yaw = float(paired["root_yaw_trajectory_rms_rad"]["median"])
        success_change = float(paired["success_rate_change"])
        below = bool(
            xy < float(thresholds["maximum_median_root_xy_trajectory_rms_m"])
            and yaw < float(thresholds["maximum_median_root_yaw_trajectory_rms_rad"])
            and abs(success_change) < float(thresholds["maximum_absolute_success_rate_change"])
        )
        checks[str(replan_steps)] = {
            "complete": complete,
            "forced_prefix_valid": forced_prefix_valid,
            "forced_prefix_required": require_forced_prefix,
            "valid": valid,
            "median_root_xy_trajectory_rms_m": xy,
            "median_root_yaw_trajectory_rms_rad": yaw,
            "success_rate_change": success_change,
            "below_all_frequency_sensitivity_thresholds": below,
            "passed": bool(complete and valid and below),
        }
    return {
        "frequency_checks": checks,
        "frequency_resistance_passed": bool(all(check["passed"] for check in checks.values())),
    }


def _baseline_result(
    report: dict[str, Any],
    reference_default_success: float,
    thresholds: dict[str, Any],
    required_pairs: int,
    require_forced_prefix: bool,
) -> dict[str, Any]:
    result = _frequency_resistance(report, thresholds, required_pairs, require_forced_prefix)
    default_success = float(report["conditions"]["16"]["success_rate"])
    drop = float(reference_default_success - default_success)
    preserves_default = drop <= float(thresholds["maximum_default_frequency_success_drop_vs_raw"])
    result.update(
        {
            "reference_default_success_rate": reference_default_success,
            "method_default_success_rate": default_success,
            "default_success_rate_drop": drop,
            "preserves_default_success": preserves_default,
            "resists_replanning_frequency_change": bool(
                result["frequency_resistance_passed"] and preserves_default
            ),
        }
    )
    return result


def evaluate(
    official_raw_report: dict[str, Any],
    matched_lora_report: dict[str, Any],
    dt_report: dict[str, Any],
    se2_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["resistance_thresholds"]
    required_pairs = int(config["validity"]["minimum_complete_seed_pairs_per_frequency"])
    official_default = float(official_raw_report["conditions"]["16"]["success_rate"])
    matched_default = float(matched_lora_report["conditions"]["16"]["success_rate"])

    raw_screen = _frequency_resistance(
        official_raw_report,
        thresholds,
        required_pairs,
        require_forced_prefix=True,
    )
    frequency_problem_observed = any(
        check["complete"]
        and check["valid"]
        and not check["below_all_frequency_sensitivity_thresholds"]
        for check in raw_screen["frequency_checks"].values()
    )

    matched = _baseline_result(
        matched_lora_report,
        official_default,
        thresholds,
        required_pairs,
        require_forced_prefix=True,
    )
    dt = _baseline_result(
        dt_report,
        matched_default,
        thresholds,
        required_pairs,
        require_forced_prefix=False,
    )
    dt["first_chunk_difference_is_intended_treatment"] = True
    se2 = _baseline_result(
        se2_report,
        official_default,
        thresholds,
        required_pairs,
        require_forced_prefix=True,
    )

    sufficient = []
    if matched["resists_replanning_frequency_change"]:
        sufficient.append("matched_unconditioned_lora")
    if dt["resists_replanning_frequency_change"]:
        sufficient.append("replan_dt_conditioned_policy")
    if se2["resists_replanning_frequency_change"]:
        sufficient.append("se2_waypoint")
    all_simple_baselines_failed = not sufficient
    research_warranted = bool(frequency_problem_observed and all_simple_baselines_failed)

    if not frequency_problem_observed:
        plain_language = (
            "Gate N-A 没有观察到越过冻结门槛的实际频率敏感性，因此没有需要这些基线解决的问题，"
            "也不进入新的移动 VLA 策略研究。"
        )
        decision = "stop_no_practical_frequency_problem_observed"
    elif research_warranted:
        plain_language = (
            "Gate N-A 已确认原始 policy 存在实际频率敏感性。Gate N-B 中，同预算普通微调、"
            "显式输入真实重规划 delta-t、相对 SE(2) waypoint 三条简单路线，"
            "都没能在保持默认频率能力的同时，把 6.25 和 12.5 Hz 的轨迹及成功率变化全部压回冻结门槛内。"
            "因此 Gate N-B 允许进入面向移动机器人的 VLA 策略研究，但证据只覆盖当前 Arena 任务与 checkpoint。"
        )
        decision = "proceed_to_mobile_vla_policy_research"
    else:
        plain_language = (
            "Gate N-A 已确认原始 policy 存在实际频率敏感性，但 Gate N-B 中至少一条普通训练或简单接口路线"
            "已经把频率变化压回冻结门槛以内，且没有明显损害默认频率能力。"
            "因此现在没有证据声称必须研究新的移动 VLA 策略，应采用通过的简单路线。"
        )
        decision = "stop_new_mobile_vla_policy_claim_use_sufficient_baseline"

    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "raw_frequency_screen": raw_screen,
        "gate_n_a_frequency_problem_observed": frequency_problem_observed,
        "gate_n_a_passed": frequency_problem_observed,
        "methods": {
            "matched_unconditioned_lora": matched,
            "replan_dt_conditioned_policy": dt,
            "se2_waypoint": se2,
        },
        "sufficient_methods": sufficient,
        "simple_baseline_sufficient": bool(sufficient),
        "all_simple_baselines_failed": all_simple_baselines_failed,
        "gate_n_b_new_policy_necessity_passed": research_warranted,
        "mobile_vla_policy_research_warranted": research_warranted,
        "gate_n_passed": research_warranted,
        "decision": decision,
        "plain_language": plain_language,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-raw-report", type=Path, required=True)
    parser.add_argument("--matched-lora-report", type=Path, required=True)
    parser.add_argument("--dt-report", type=Path, required=True)
    parser.add_argument("--se2-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = evaluate(
        load(args.official_raw_report),
        load(args.matched_lora_report),
        load(args.dt_report),
        load(args.se2_report),
        yaml.safe_load(args.config.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
