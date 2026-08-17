import json

import numpy as np
import pandas as pd

from lp_groot_base.validator import load_action_layout, load_episodes


def test_load_episodes_supports_compact_lerobot_v3(tmp_path):
    (tmp_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": {"action": {"shape": [12]}}})
    )
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "dataset_from_index": [0, 2],
            "dataset_to_index": [2, 5],
        }
    ).to_parquet(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    rows = []
    for episode_index, length in [(0, 2), (1, 3)]:
        for frame_index in range(length):
            rows.append(
                {
                    "episode_index": episode_index,
                    "timestamp": frame_index * 0.05,
                    "observation.state": np.arange(12, dtype=np.float32),
                    "action": np.arange(12, dtype=np.float32),
                }
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")

    episodes = load_episodes(tmp_path)

    assert [episode.name for episode in episodes] == ["episode_000000", "episode_000001"]
    assert [len(episode.timestamps) for episode in episodes] == [2, 3]
    assert episodes[1].positions.shape == (3, 3)
    assert episodes[1].actions.shape == (3, 12)
    assert load_action_layout(tmp_path) == {"base_motion": (0, 3), "control_mode": (4, 5)}
