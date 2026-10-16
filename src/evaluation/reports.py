"""Write experiment artifacts (CSV/JSON) without overwriting prior runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def new_experiment_dir(base: Path, prefix: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = base / f"{prefix}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_experiment_bundle(
    out_dir: Path,
    metadata: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    per_model_artifacts: dict[str, dict[str, Any]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(comparison_rows).to_csv(out_dir / "comparison.csv", index=False)

    for model_key, bundle in per_model_artifacts.items():
        sub = out_dir / model_key
        sub.mkdir(parents=True, exist_ok=True)
        if "confusion_matrix" in bundle:
            bundle["confusion_matrix"].to_csv(sub / "confusion_matrix.csv")
        if "classification_report_csv" in bundle:
            bundle["classification_report_csv"].to_csv(sub / "classification_report.csv", index=False)
        if "classification_report_json" in bundle:
            (sub / "classification_report.json").write_text(
                json.dumps(bundle["classification_report_json"], indent=2),
                encoding="utf-8",
            )
    return out_dir
