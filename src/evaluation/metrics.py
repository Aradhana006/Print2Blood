"""Multiclass evaluation metrics for research experiments."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return pd.DataFrame(cm, index=labels, columns=labels)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(len(class_names))))
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(len(class_names))))
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(len(class_names))))
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=list(range(len(class_names))))
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix_df(y_true, y_pred, class_names),
        "support_test": int(len(y_true)),
    }
