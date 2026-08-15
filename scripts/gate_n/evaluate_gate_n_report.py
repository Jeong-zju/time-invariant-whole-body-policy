#!/usr/bin/env python3
"""Apply the frozen Gate N thresholds to an offline diagnostic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def evaluate(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["offline_gate"]
    labels = report["label_alignment"]
    matching = report["matching"]
    speed = labels["same_direction_speed_contrast_pairs"]

    n1_checks = {
        "matched_pairs": {
            "value": matching["selected_pairs_after_future_filter"],
            "operator": ">=",
            "threshold": thresholds["minimum_matched_pairs"],
        },
        "same_direction_speed_contrast_pairs": {
            "value": speed["pairs"],
            "operator": ">=",
            "threshold": thresholds["minimum_same_direction_speed_contrast_pairs"],
        },
        "progress_median_disagreement_reduction_vs_time": {
            "value": speed["progress_median_reduction_vs_time"],
            "operator": ">=",
            "threshold": thresholds["minimum_progress_median_disagreement_reduction_vs_time"],
        },
        "fraction_progress_better_than_time": {
            "value": speed["fraction_progress_better_than_time"],
            "operator": ">=",
            "threshold": thresholds["minimum_fraction_pairs_where_progress_beats_time"],
        },
    }
    for check in n1_checks.values():
        check["passed"] = bool(check["value"] >= check["threshold"])
    n1_passed = all(check["passed"] for check in n1_checks.values())

    methods = report["resampling"]["methods"]
    time = methods["time_uniform_3x"]
    max_coverage_drop = float(thresholds["maximum_critical_event_coverage_drop_vs_time_uniform"])
    max_reconstruction_ratio = float(thresholds["maximum_geometry_reconstruction_rms_ratio_vs_time_uniform"])
    critical_fields = ["high_acceleration_coverage", "turning_coverage", "base_motion_coverage"]

    n2_methods: dict[str, Any] = {}
    for method_name in ["whole_body_progress", "isr_budget_matched"]:
        method = methods[method_name]
        coverage_checks: dict[str, Any] = {}
        for field in critical_fields:
            baseline_value = float(time[field]["median"])
            value = float(method[field]["median"])
            drop = baseline_value - value
            coverage_checks[field] = {
                "time_uniform_median": baseline_value,
                "method_median": value,
                "drop": drop,
                "maximum_drop": max_coverage_drop,
                "passed": bool(drop <= max_coverage_drop),
            }
        reconstruction_ratio = float(
            method["geometry_reconstruction_rms"]["median"]
            / time["geometry_reconstruction_rms"]["median"]
        )
        reconstruction_check = {
            "ratio_vs_time_uniform": reconstruction_ratio,
            "maximum_ratio": max_reconstruction_ratio,
            "passed": bool(reconstruction_ratio <= max_reconstruction_ratio),
        }
        n2_methods[method_name] = {
            "coverage": coverage_checks,
            "geometry_reconstruction": reconstruction_check,
            "passed": bool(
                reconstruction_check["passed"]
                and all(check["passed"] for check in coverage_checks.values())
            ),
        }

    desired_ratio = float(config["offline_diagnostic"]["sample_budget"]["compare_isr_at_dataset_global_median_retention_ratio"])
    paper_ratio = float(methods["isr_paper_default"]["retention_ratio"]["median"])
    paper_isr_budget_check = {
        "median_retention_ratio": paper_ratio,
        "desired_retention_ratio": desired_ratio,
        "absolute_gap": abs(paper_ratio - desired_ratio),
        "maximum_gap": 0.05,
        "passed": bool(abs(paper_ratio - desired_ratio) <= 0.05),
    }
    n2_passed = any(method["passed"] for method in n2_methods.values())
    overall_passed = n1_passed and n2_passed

    failed_checks = [name for name, check in n1_checks.items() if not check["passed"]]
    failed_methods = [name for name, method in n2_methods.items() if not method["passed"]]
    if overall_passed:
        decision = "proceed_to_waypoint_tracker_gate"
    else:
        decision = thresholds["failure_action"]

    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "source_report_protocol_id": report["protocol_id"],
        "n1_label_ambiguity": {
            "checks": n1_checks,
            "passed": n1_passed,
            "failed_checks": failed_checks,
        },
        "n2_resampling": {
            "methods": n2_methods,
            "paper_default_isr_sample_budget": paper_isr_budget_check,
            "passed": n2_passed,
            "failed_methods": failed_methods,
        },
        "offline_gate_passed": overall_passed,
        "decision": decision,
        "plain_language": (
            "当前 Arena 单任务没有给出足够强的时间标签歧义证据，而且进度/ISR 整理仍会漏掉过多转向或底盘移动片段；按预注册规则，先停止 B1-B5 模型训练并重新评估数据集或研究主张。"
            if not overall_passed
            else "时间标签歧义和至少一种安全的数据整理方式都通过了预注册门槛，可以继续 waypoint tracker gate。"
        ),
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
