from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.split_loader import assert_split_isolation, load_master_split, split_frames


def test_master_split_isolation():
    master = ROOT / "data" / "splits" / "master_split.csv"
    if not master.exists():
        return
    df = load_master_split(master)
    assert_split_isolation(df)
    train, val, test = split_frames(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train["image_path"]) & set(test["image_path"]) == set()
