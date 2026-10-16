from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import CLASSES
from src.evaluation.metrics import compute_multiclass_metrics


def test_scaler_fit_scope_and_metrics():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 12))
    x_test = rng.normal(size=(10, 12))
    y_train = rng.integers(0, 8, size=40)
    y_test = rng.integers(0, 8, size=10)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)
    metrics = compute_multiclass_metrics(y_test, pred, CLASSES)
    assert "macro_f1" in metrics
    assert metrics["confusion_matrix"].shape == (8, 8)
