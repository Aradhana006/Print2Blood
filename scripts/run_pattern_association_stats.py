"""Chi-square and Cramér's V for blood group vs heuristic pattern labels (exploratory only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CLASSES, project_root
from src.data.split_loader import load_master_split
from src.features.pattern_heuristic import estimate_pattern_type, load_gray


def cramers_v(confusion: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.values.sum()
    r, k = confusion.shape
    if n <= 0 or min(r, k) <= 1:
        return float("nan")
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def run(max_samples: int | None = None, master_split: str = "data/splits/master_split.csv") -> Path:
    root = project_root()
    df = load_master_split(root / master_split)
    if max_samples:
        df = df.head(max_samples).reset_index(drop=True)

    patterns: list[str] = []
    for rel in df["image_path"]:
        gray = load_gray(root / rel)
        patterns.append(estimate_pattern_type(gray))
    df = df.copy()
    df["pattern_heuristic"] = patterns

    table = pd.crosstab(df["label"], df["pattern_heuristic"]).reindex(index=CLASSES, fill_value=0)
    chi2, p_value, dof, _ = chi2_contingency(table.values)
    v = cramers_v(table)

    out_dir = root / "results/experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pattern_association_stats.json"
    payload = {
        "test": "chi_square_and_cramers_v",
        "rows": "blood_group_label",
        "columns": "pattern_heuristic",
        "pattern_label_validity": (
            "Heuristic orientation cue only (Arch-like/Loop-like/Whorl-like). "
            "Not AFIS-certified pattern ground truth."
        ),
        "assumptions": [
            "Independence of observations is approximate; images may share acquisition context.",
            "Expected cell counts should be >= 5 for asymptotic chi-square; inspect contingency table.",
        ],
        "n_images": int(len(df)),
        "contingency_table": table.to_dict(),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(v),
    }
    table.to_csv(out_dir / "pattern_contingency_table.csv")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
