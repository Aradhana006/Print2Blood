from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.split_checks import validate_master_split


def test_master_split_passes_when_present():
    master = ROOT / "data" / "splits" / "master_split.csv"
    if not master.exists():
        return
    result = validate_master_split()
    assert result["overall"] == "PASS"
    assert result["group_leak"] == 0
    assert result["hash_leak"] == 0
    assert result["counts"]["total_images"] == 6000


if __name__ == "__main__":
    test_master_split_passes_when_present()
    print("ok")
