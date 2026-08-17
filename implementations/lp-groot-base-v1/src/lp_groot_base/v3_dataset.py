"""Read the compact LeRobot v3 RoboCasa layout used by task mirrors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_v3_metadata(root: str | Path) -> tuple[dict, pd.DataFrame]:
    root = Path(root)
    with (root / "meta" / "info.json").open() as file:
        info = json.load(file)
    episodes = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return info, episodes


def load_v3_table(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3 parquet files below {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def episode_slice(table: pd.DataFrame, episodes: pd.DataFrame, episode_index: int) -> pd.DataFrame:
    meta = episodes.loc[episodes["episode_index"] == episode_index]
    if len(meta) != 1:
        raise KeyError(f"Expected one metadata row for episode {episode_index}, got {len(meta)}")
    row = meta.iloc[0]
    start = int(row["dataset_from_index"])
    stop = int(row["dataset_to_index"])
    frame = table.iloc[start:stop].reset_index(drop=True)
    if not np.all(frame["episode_index"].to_numpy() == episode_index):
        raise ValueError(f"Episode index mismatch for slice {episode_index}")
    return frame
