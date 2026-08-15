"""Pure utilities for the Gate N replanning delta-t condition."""

from __future__ import annotations

import math


REPLAN_DT_VALUES_S = (0.08, 0.16, 0.32)
REPLAN_DT_REFERENCE_S = 0.16


def encode_replan_dt(dt_s: float) -> float:
    """Encode 0.08/0.16/0.32 seconds as -1/0/+1."""
    if dt_s <= 0.0:
        raise ValueError("replanning delta-t must be positive")
    return float(math.log2(dt_s / REPLAN_DT_REFERENCE_S))
