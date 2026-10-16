from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import CLASSES


def test_class_order_matches_notebook_and_api():
    expected = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    assert CLASSES == expected
    mapping = {label: idx for idx, label in enumerate(CLASSES)}
    assert mapping["A+"] == 0
    assert mapping["O-"] == 7


def test_backend_settings_class_labels():
    sys.path.insert(0, str(ROOT / "backend"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.class_labels == CLASSES
