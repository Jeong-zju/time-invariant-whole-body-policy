"""Deterministic replanning schedules on a discrete control clock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplanSchedule:
    """Quantize fixed or jittered wall-clock deadlines to control ticks.

    Quantization is cumulative, so non-integer ratios such as 15 Hz on a
    50 Hz controller do not accumulate phase drift.
    """

    control_dt_s: float
    fixed_steps: int | None = None
    fixed_frequency_hz: float | None = None
    jitter_frequency_hz: tuple[float, float] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive and finite")
        modes = sum(
            value is not None
            for value in (self.fixed_steps, self.fixed_frequency_hz, self.jitter_frequency_hz)
        )
        if modes != 1:
            raise ValueError("select exactly one replanning schedule mode")
        if self.fixed_steps is not None and self.fixed_steps <= 0:
            raise ValueError("fixed_steps must be positive")
        if self.fixed_frequency_hz is not None:
            self._validate_frequency(self.fixed_frequency_hz)
        if self.jitter_frequency_hz is not None:
            low, high = self.jitter_frequency_hz
            self._validate_frequency(low)
            self._validate_frequency(high)
            if low > high:
                raise ValueError("jitter frequency lower bound exceeds upper bound")
        self._rng = np.random.default_rng(self.seed)
        self._continuous_deadline_s = 0.0
        self._last_deadline_tick = 0

    def _validate_frequency(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("replanning frequency must be positive and finite")
        if value > 1.0 / self.control_dt_s:
            raise ValueError("replanning frequency exceeds the control frequency")

    def _next_frequency_hz(self) -> float:
        if self.fixed_frequency_hz is not None:
            return float(self.fixed_frequency_hz)
        assert self.jitter_frequency_hz is not None
        low, high = self.jitter_frequency_hz
        return float(self._rng.uniform(low, high))

    def next_steps(self) -> int:
        if self.fixed_steps is not None:
            return int(self.fixed_steps)
        self._continuous_deadline_s += 1.0 / self._next_frequency_hz()
        deadline_tick = max(
            self._last_deadline_tick + 1,
            int(np.floor(self._continuous_deadline_s / self.control_dt_s + 0.5)),
        )
        steps = deadline_tick - self._last_deadline_tick
        self._last_deadline_tick = deadline_tick
        return steps

    def description(self) -> dict[str, object]:
        if self.fixed_steps is not None:
            return {
                "mode": "fixed_steps",
                "fixed_steps": int(self.fixed_steps),
                "nominal_frequency_hz": 1.0 / (self.control_dt_s * self.fixed_steps),
            }
        if self.fixed_frequency_hz is not None:
            return {
                "mode": "fixed_frequency_hz",
                "requested_frequency_hz": float(self.fixed_frequency_hz),
            }
        assert self.jitter_frequency_hz is not None
        return {
            "mode": "uniform_frequency_jitter_hz",
            "frequency_min_hz": float(self.jitter_frequency_hz[0]),
            "frequency_max_hz": float(self.jitter_frequency_hz[1]),
            "seed": int(self.seed),
        }
