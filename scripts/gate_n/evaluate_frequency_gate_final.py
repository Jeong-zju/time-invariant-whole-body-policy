#!/usr/bin/env python3
"""Combine the primary frequency screen with the same-frequency control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_final(primary: dict[str, Any], repeatability: dict[str, Any]) -> dict[str, Any]:
    primary_positive = bool(primary["frequency_problem_observed"])
    repeatability_passed = bool(repeatability["repeatability_control_passed"])
    final_passed = bool(primary_positive and repeatability_passed)
    if final_passed:
        decision = "proceed_to_clock_aware_consumer_and_se2_waypoint_baselines"
        plain_language = (
            "跨频率效应越过预注册门槛，同频重复噪声未越过；频率问题成立，下一步进入无需重训的 consumer 与 waypoint 对照。"
        )
    elif primary_positive:
        decision = "stabilize_or_model_same_frequency_variance_before_gate_c"
        plain_language = (
            "跨频率筛查为阳性，但同频重复本身也产生了材料性底盘轨迹差异；当前不能把观察到的变化特异地归因于重规划频率，先稳定推理/渲染或增加重复次数建模方差。"
        )
    else:
        decision = "stop_frequency_method_work"
        plain_language = "跨频率筛查未越过门槛，停止围绕该问题开发接口或模型。"
    return {
        "schema_version": 1,
        "protocol_id": primary["protocol_id"],
        "primary_frequency_screen_positive": primary_positive,
        "same_frequency_repeatability_control_passed": repeatability_passed,
        "frequency_specific_effect_established": final_passed,
        "final_gate_n_passed": final_passed,
        "decision": decision,
        "plain_language": plain_language,
        "primary_decision": primary["decision"],
        "repeatability_decision": repeatability["decision"],
        "causal_limit": (
            "A positive cross-frequency screen is insufficient when same-frequency reruns cross the same material-effect threshold."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--repeatability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_final(
        json.loads(args.primary.read_text(encoding="utf-8")),
        json.loads(args.repeatability.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
