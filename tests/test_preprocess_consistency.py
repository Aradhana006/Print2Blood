from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings
from app.ml.resnet50_service import ResNet50Service


def test_resnet_service_preprocess_runs_on_sample():
    get_settings.cache_clear()
    settings = get_settings()
    weights = settings.model_path
    if not weights.exists():
        return
    sample_dir = ROOT / "dataset" / "dataset_blood_group" / "A+"
    bmp = next(sample_dir.glob("*.BMP"), None)
    if bmp is None:
        return
    svc = ResNet50Service(weights_path=weights, class_labels=settings.class_labels)
    out = svc.predict(str(bmp))
    assert out["prediction"] in settings.class_labels
    probs = out["probabilities"]
    assert len(probs) == 8
    assert abs(sum(probs.values()) - 1.0) < 1e-2
    assert 0 <= out["confidence"] <= 1.0


def test_predict_script_uses_resnet_preprocess():
    text = (ROOT / "predict.py").read_text(encoding="utf-8")
    assert "resnet50.preprocess_input" in text or "applications.resnet50 import preprocess_input" in text
