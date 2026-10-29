"""Real v3 metadata can omit episode-level tasks; use the actual per-frame task IDs."""

from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_policy_latent_act.data import episodes_with_tasks


def test_episode_tasks_recovered_from_frames(tmp_path):
    folder = tmp_path / "data/chunk-000"
    folder.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0, 0, 1, 1], "task_index": [7, 7, 9, 9]}),
        folder / "file-000.parquet",
    )
    meta = SimpleNamespace(
        root=tmp_path,
        episodes=[{"episode_index": 0}, {"episode_index": 1}],
        tasks=pd.DataFrame({"task_index": [7, 9]}, index=["task a", "task b"]),
    )
    rows = episodes_with_tasks(meta)
    assert rows[0]["tasks"] == ["task a"]
    assert rows[1]["tasks"] == ["task b"]
