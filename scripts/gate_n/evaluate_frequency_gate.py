#!/usr/bin/env python3
"""Apply the frozen frequency Gate N thresholds to an aggregate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def evaluate(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["gate_thresholds"]
    paired = report["paired_against_reference"]
    minimum_pairs = int(thresholds["minimum_complete_seed_pairs_per_condition"])

    conditions: dict[str, Any] = {}
    for name, comparison in paired.items():
        checks = {
            "complete_pairs": {
                "value": int(comparison["pairs"]),
                "operator": ">=",
                "threshold": minimum_pairs,
                "passed": bool(comparison["pairs"] >= minimum_pairs),
            },
            "first_chunk_identity": {
                "value": bool(comparison["all_first_action_chunks_identical"]),
                "operator": "is",
                "threshold": True,
                "passed": bool(comparison["all_first_action_chunks_identical"]),
            },
            "prebranch_physical_telemetry_identity": {
                "value": bool(comparison["all_prebranch_physical_telemetry_identical"]),
                "operator": "is",
                "threshold": True,
                "passed": bool(comparison["all_prebranch_physical_telemetry_identical"]),
            },
        }
        effects = {
            "root_xy_trajectory": {
                "value": float(comparison["root_xy_trajectory_rms_m"]["median"]),
                "operator": ">=",
                "threshold": float(thresholds["minimum_median_root_xy_trajectory_rms_m"]),
            },
            "root_yaw_trajectory": {
                "value": float(comparison["root_yaw_trajectory_rms_rad"]["median"]),
                "operator": ">=",
                "threshold": float(thresholds["minimum_median_root_yaw_trajectory_rms_rad"]),
            },
            "success_rate": {
                "value": abs(float(comparison["success_rate_change"])),
                "operator": ">=",
                "threshold": float(thresholds["minimum_absolute_success_rate_change"]),
            },
        }
        for effect in effects.values():
            effect["passed"] = bool(effect["value"] >= effect["threshold"])
        valid = all(check["passed"] for check in checks.values())
        effect_observed = any(effect["passed"] for effect in effects.values())
        conditions[name] = {
            "checks": checks,
            "effects": effects,
            "valid": valid,
            "effect_observed": effect_observed,
            "passed": bool(valid and effect_observed),
        }

    complete_suite = not report["missing_runs"]
    any_valid_condition = bool(conditions and any(item["valid"] for item in conditions.values()))
    gate_passed = bool(complete_suite and any(item["passed"] for item in conditions.values()))
    if not complete_suite:
        decision = thresholds["incomplete_action"]
        plain_language = "正式套件还有缺失 rollout，当前不能下 Gate 结论；只补齐缺失项，不调整门槛。"
    elif not any_valid_condition:
        decision = thresholds["invalid_action"]
        plain_language = "配对首个 action chunk 或分叉前物理轨迹不一致，因果对照无效；当前结果既不是通过，也不是阴性结论，必须重跑无效配对。"
    elif gate_passed:
        decision = thresholds["pass_action"]
        plain_language = "同一个冻结策略只改变重规划频率，就产生了超过预注册门槛的底盘轨迹、朝向或任务结果变化；频率问题成立，但原因尚未归到 velocity 积分，下一步只做无需重训的时钟感知 consumer 和 SE(2) waypoint 对照。"
    else:
        decision = thresholds["failure_action"]
        plain_language = "在当前 10-seed 套件中，重规划频率变化没有产生达到门槛且可配对复核的影响；停止为这个问题开发新模型。"
    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "source_report_protocol_id": report["protocol_id"],
        "complete_suite": complete_suite,
        "conditions": conditions,
        "any_valid_condition": any_valid_condition,
        "frequency_problem_observed": gate_passed,
        "decision": decision,
        "plain_language": plain_language,
        "causal_limit": "A pass demonstrates replanning-rate sensitivity only; it does not identify implicit delta-t as the cause.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = evaluate(report, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
