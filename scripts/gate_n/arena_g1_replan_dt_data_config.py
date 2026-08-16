"""Arena G1 GR00T data config with an explicit replanning delta-t input.

The demonstrations and action targets stay unchanged.  During training, every
sample is paired with one of the three deployment replanning intervals.  The
scalar is appended after the ordinary Arena state has been normalized, so no
source dataset or statistics file is rewritten.  During inference the true
interval is read from ``ARENA_G1_REPLAN_DT_S``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from pydantic import Field

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from isaaclab_arena_gr00t.data_config import UnitreeG1SimWBCDataConfig

from replan_dt import REPLAN_DT_VALUES_S, encode_replan_dt


class AppendReplanDeltaT(ModalityTransform):
    """Append the real replanning interval to the concatenated policy state."""

    apply_to: list[str] = Field(default_factory=lambda: ["state"])
    dt_values_s: tuple[float, ...] = REPLAN_DT_VALUES_S

    def _dt_s(self) -> float:
        if self.training:
            return float(np.random.choice(self.dt_values_s))
        raw = os.environ.get("ARENA_G1_REPLAN_DT_S")
        if raw is None:
            raise RuntimeError("ARENA_G1_REPLAN_DT_S must be set for delta-t-conditioned inference")
        dt_s = float(raw)
        if not any(abs(dt_s - value) <= 1e-9 for value in self.dt_values_s):
            raise ValueError(f"unsupported replanning delta-t {dt_s}; expected one of {self.dt_values_s}")
        return dt_s

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        state = data.get("state")
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"expected concatenated torch state, got {type(state)}")
        encoded = encode_replan_dt(self._dt_s())
        value = torch.full((*state.shape[:-1], 1), encoded, dtype=state.dtype, device=state.device)
        data["state"] = torch.cat([state, value], dim=-1)
        return data


class UnitreeG1SimWBCReplanDtDataConfig(UnitreeG1SimWBCDataConfig):
    """Official Arena data pipeline plus one normalized replanning-clock scalar."""

    def transform(self) -> ModalityTransform:
        original = super().transform()
        transforms = list(original.transforms)
        # GR00TTransform pads state to max_state_dim=64.  Append the scalar
        # immediately before that padding step.
        transforms.insert(-1, AppendReplanDeltaT())
        return ComposedModalityTransform(transforms=transforms)
