"""Explicit train-split normalization statistics for fair ACT comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_values(cls, values: np.ndarray, floor: float = 1e-4) -> "FeatureStats":
        values = np.asarray(values, dtype=np.float64)
        return cls(mean=values.mean(axis=0), std=np.maximum(values.std(axis=0), floor))

    @classmethod
    def from_dict(cls, value: dict) -> "FeatureStats":
        return cls(mean=np.asarray(value["mean"], dtype=np.float32), std=np.asarray(value["std"], dtype=np.float32))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (value - self.mean) / self.std

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return value * self.std + self.mean


def load_stats(path: str | Path) -> dict[str, FeatureStats]:
    with Path(path).open() as handle:
        raw = json.load(handle)
    return {key: FeatureStats.from_dict(value) for key, value in raw["features"].items()}
