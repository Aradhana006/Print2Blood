from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.hog import DEFAULT_HOG_PARAMS, extract_hog_from_path, hog_feature_dim


def test_hog_dimension_stable():
    dim = hog_feature_dim(DEFAULT_HOG_PARAMS)
    assert dim > 0
    sample = ROOT / "dataset" / "dataset_blood_group" / "A+"
    bmp = next(sample.glob("*.BMP"), None)
    if bmp is None:
        return
    feat = extract_hog_from_path(bmp, params=DEFAULT_HOG_PARAMS)
    assert feat.shape == (dim,)
    assert np.isfinite(feat).all()
