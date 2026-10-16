"""Train/evaluate SVM, Random Forest, and Gradient Boosting on HOG (optional LBP) features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CLASSES, project_root
from src.data.split_loader import assert_split_isolation, label_to_index, load_master_split, split_frames
from src.evaluation.metrics import compute_multiclass_metrics
from src.evaluation.reports import new_experiment_dir, write_experiment_bundle
from src.features.hog import extract_hog_batch
from src.features.lbp import extract_lbp_batch


def _load_cfg(path: Path | None) -> dict:
    cfg_path = path or ROOT / "configs" / "classical_ml_experiment.json"
    with open(cfg_path, encoding="utf-8") as handle:
        return json.load(handle)


def _build_features(df: pd.DataFrame, root: Path, cfg: dict) -> np.ndarray:
    paths = df["image_path"].tolist()
    rep = cfg.get("feature_representation", "hog")
    hog_params = cfg.get("hog", {})
    if rep == "hog":
        return extract_hog_batch(paths, root, hog_params)
    if rep == "lbp":
        return extract_lbp_batch(paths, root, cfg.get("lbp", {}))
    if rep == "hog_lbp":
        hog_x = extract_hog_batch(paths, root, hog_params)
        lbp_x = extract_lbp_batch(paths, root, cfg.get("lbp", {}))
        return np.hstack([hog_x, lbp_x])
    raise ValueError(f"Unknown feature_representation: {rep}")


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


def _select_best(
    name: str,
    param_grid: list[dict],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[object, dict, float]:
    best_score = -1.0
    best_params: dict = {}
    best_model = None
    for params in param_grid:
        params = dict(params)
        if name == "svm":
            model = SVC(random_state=seed, **params)
        elif name == "random_forest":
            model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
        elif name == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=seed, **params)
        else:
            raise ValueError(name)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        metrics = compute_multiclass_metrics(y_val, pred, CLASSES)
        score = metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
    if best_model is None:
        raise RuntimeError(f"No model selected for {name}")
    return best_model, best_params, best_score


def run_experiment(cfg: dict, max_samples: int | None = None) -> Path:
    root = project_root()
    master = load_master_split(root / cfg["master_split"])
    assert_split_isolation(master)
    train_df, val_df, test_df = split_frames(master)

    if max_samples:
        seed = int(cfg.get("random_seed", 42))

        def _subsample(part: pd.DataFrame, n: int) -> pd.DataFrame:
            if len(part) <= n:
                return part.reset_index(drop=True)
            # Stratified subsample so all classes remain represented when possible.
            groups = []
            per_class = max(1, n // len(CLASSES))
            for label in CLASSES:
                g = part.loc[part["label"] == label]
                if g.empty:
                    continue
                groups.append(g.sample(n=min(len(g), per_class), random_state=seed))
            out = pd.concat(groups, ignore_index=True)
            if len(out) > n:
                out = out.sample(n=n, random_state=seed).reset_index(drop=True)
            return out

        train_df = _subsample(train_df, max_samples)
        val_df = _subsample(val_df, max(16, max_samples // 5))
        test_df = _subsample(test_df, max(16, max_samples // 5))

    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log(f"Extracting features (train, n={len(train_df)})...")
    x_train = _build_features(train_df, root, cfg)
    _log(f"Extracting features (validation, n={len(val_df)})...")
    x_val = _build_features(val_df, root, cfg)
    _log(f"Extracting features (test, n={len(test_df)})...")
    x_test = _build_features(test_df, root, cfg)

    y_train = label_to_index(train_df["label"])
    y_val = label_to_index(val_df["label"])
    y_test = label_to_index(test_df["label"])

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    seed = int(cfg.get("random_seed", 42))
    out_dir = new_experiment_dir(root / cfg["output_base"], "classical_ml")

    comparison: list[dict] = []
    artifacts: dict[str, dict] = {}

    for clf_name, clf_cfg in cfg["classifiers"].items():
        _log(f"Tuning {clf_name} on validation...")
        model, best_params, val_macro_f1 = _select_best(
            clf_name,
            clf_cfg["param_grid"],
            x_train_s,
            y_train,
            x_val_s,
            y_val,
            seed,
        )
        pred_test = model.predict(x_test_s)
        test_metrics = compute_multiclass_metrics(y_test, pred_test, CLASSES)
        comparison.append(
            {
                "model": clf_name,
                "feature_representation": cfg.get("feature_representation"),
                "best_params": json.dumps(best_params),
                "validation_macro_f1": round(val_macro_f1, 6),
                "test_accuracy": round(test_metrics["accuracy"], 6),
                "test_macro_precision": round(test_metrics["macro_precision"], 6),
                "test_macro_recall": round(test_metrics["macro_recall"], 6),
                "test_macro_f1": round(test_metrics["macro_f1"], 6),
                "test_support": test_metrics["support_test"],
            }
        )
        sub_key = clf_name
        report = test_metrics["classification_report"]
        artifacts[sub_key] = {
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report_csv": _report_to_df(report),
            "classification_report_json": report,
        }
        bundle_dir = out_dir / clf_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler, "best_params": best_params}, bundle_dir / "pipeline.joblib")

    metadata = {
        "experiment_type": "classical_ml",
        "master_split": cfg["master_split"],
        "split_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "feature_representation": cfg.get("feature_representation"),
        "hog_params": cfg.get("hog"),
        "lbp_params": cfg.get("lbp"),
        "random_seed": seed,
        "class_order": CLASSES,
        "scaler_fitted_on": "train_only",
        "hyperparameter_selection": "validation_macro_f1",
        "test_set_usage": "final_evaluation_once",
        "disclaimer": cfg.get("disclaimer"),
    }
    write_experiment_bundle(out_dir, metadata, comparison, artifacts)
    summary_path = root / cfg["output_base"] / "latest_classical_ml.json"
    summary_path.write_text(
        json.dumps({"experiment_dir": out_dir.relative_to(root).as_posix(), **metadata}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote experiment to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical ML experiment on master split")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug/smoke subset per split")
    parser.add_argument("--feature", choices=["hog", "lbp", "hog_lbp"], default=None)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    if args.feature:
        cfg["feature_representation"] = args.feature
    run_experiment(cfg, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
