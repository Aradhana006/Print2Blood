"""Evaluate deployed ResNet50 weights on master_split test set (offline; does not overwrite weights)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CLASSES, project_root
from src.data.split_loader import assert_split_isolation, label_to_index, load_master_split, split_frames
from src.evaluation.cnn_eval import load_resnet_model, predict_batch
from src.evaluation.metrics import compute_multiclass_metrics
from src.evaluation.reports import new_experiment_dir, write_experiment_bundle


def _report_to_df(report_dict: dict) -> pd.DataFrame:
    rows = []
    for key, val in report_dict.items():
        if isinstance(val, dict) and "precision" in val:
            rows.append(
                {
                    "class": key,
                    "precision": val["precision"],
                    "recall": val["recall"],
                    "f1-score": val["f1-score"],
                    "support": val["support"],
                }
            )
    return pd.DataFrame(rows)


def run_eval(
    weights_path: Path,
    master_split: str = "data/splits/master_split.csv",
    image_size: int = 256,
    batch_size: int = 32,
    max_test_samples: int | None = None,
) -> Path:
    root = project_root()
    master = load_master_split(root / master_split)
    assert_split_isolation(master)
    _, _, test_df = split_frames(master)
    if max_test_samples:
        test_df = test_df.head(max_test_samples).reset_index(drop=True)

    print(f"Loading CNN weights from {weights_path} ...")
    model = load_resnet_model(weights_path)
    paths = test_df["image_path"].tolist()
    y_true = label_to_index(test_df["label"])
    y_pred = predict_batch(model, paths, root, CLASSES, image_size=image_size, batch_size=batch_size)

    metrics = compute_multiclass_metrics(y_true, y_pred, CLASSES)
    out_dir = new_experiment_dir(root / "results/experiments", "cnn_master_split_eval")

    comparison = [
        {
            "model": "resnet50_v1_deployed_weights",
            "split": "test",
            "split_protocol": "master_split.csv",
            "test_accuracy": round(metrics["accuracy"], 6),
            "test_macro_precision": round(metrics["macro_precision"], 6),
            "test_macro_recall": round(metrics["macro_recall"], 6),
            "test_macro_f1": round(metrics["macro_f1"], 6),
            "test_support": metrics["support_test"],
            "historical_notebook_accuracy_random_80_20": 0.8142,
        }
    ]
    metadata = {
        "experiment_type": "cnn_offline_eval",
        "weights_path": str(weights_path.resolve()),
        "weights_modified": False,
        "master_split": master_split,
        "preprocessing": "resnet50.preprocess_input@256",
        "class_order": CLASSES,
        "image_size": image_size,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Metrics reflect master-split test images only. "
            "Historical 81.42% came from random 80/20 notebook split and is not comparable without caveats."
        ),
    }
    report = metrics["classification_report"]
    artifacts = {
        "resnet50_v1": {
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report_csv": _report_to_df(report),
            "classification_report_json": report,
        }
    }
    write_experiment_bundle(out_dir, metadata, comparison, artifacts)

    summary = root / "results/experiments" / "latest_cnn_master_split_eval.json"
    summary.write_text(
        json.dumps({"experiment_dir": out_dir.relative_to(root).as_posix(), **metadata, "comparison": comparison}, indent=2),
        encoding="utf-8",
    )
    print(f"Test accuracy: {metrics['accuracy']:.4f}, macro F1: {metrics['macro_f1']:.4f}")
    print(f"Wrote {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "test" / "model_blood_group_detection_resnet.h5",
    )
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_eval(args.weights, max_test_samples=args.max_test_samples, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
