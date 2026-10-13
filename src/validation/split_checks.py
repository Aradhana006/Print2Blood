from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset import CLASSES, project_root


def validate_master_split(
    master_path: str | Path | None = None,
    groups_path: str | Path | None = None,
    classes: list[str] | None = None,
) -> dict:
    root = project_root()
    master_path = Path(master_path) if master_path else root / "data" / "splits" / "master_split.csv"
    groups_path = Path(groups_path) if groups_path else root / "data" / "splits" / "split_groups.csv"
    classes = classes or CLASSES

    checks = []
    critical_fail = False

    if not master_path.exists():
        return {
            "overall": "FAIL",
            "critical_fail": True,
            "checks": [
                {
                    "name": "master_split_exists",
                    "status": "FAIL",
                    "detail": f"Missing file: {master_path}",
                    "critical": True,
                }
            ],
            "counts": {},
        }

    master = pd.read_csv(master_path)
    required = {"image_path", "label", "group_id", "split"}
    missing_cols = sorted(required - set(master.columns))
    checks.append(
        {
            "name": "required_columns",
            "status": "FAIL" if missing_cols else "PASS",
            "detail": f"Missing columns: {missing_cols}" if missing_cols else "All required columns present",
            "critical": True,
        }
    )
    if missing_cols:
        critical_fail = True
        return {"overall": "FAIL", "critical_fail": True, "checks": checks, "counts": {}}

    # Duplicate image paths
    dup_paths = int(master["image_path"].duplicated().sum())
    checks.append(
        {
            "name": "duplicate_image_paths",
            "status": "FAIL" if dup_paths else "PASS",
            "detail": f"{dup_paths} duplicate image_path rows",
            "critical": True,
        }
    )
    critical_fail = critical_fail or bool(dup_paths)

    # Invalid labels / splits
    invalid_labels = sorted(set(master["label"]) - set(classes))
    checks.append(
        {
            "name": "invalid_labels",
            "status": "FAIL" if invalid_labels else "PASS",
            "detail": f"Invalid labels: {invalid_labels}" if invalid_labels else "Labels restricted to expected classes",
            "critical": True,
        }
    )
    critical_fail = critical_fail or bool(invalid_labels)

    invalid_splits = sorted(set(master["split"]) - {"train", "validation", "test"})
    checks.append(
        {
            "name": "invalid_splits",
            "status": "FAIL" if invalid_splits else "PASS",
            "detail": f"Invalid splits: {invalid_splits}" if invalid_splits else "Splits restricted to train/validation/test",
            "critical": True,
        }
    )
    critical_fail = critical_fail or bool(invalid_splits)

    # Group leakage: group_id in more than one split
    group_splits = master.groupby("group_id")["split"].nunique()
    leak_groups = int((group_splits > 1).sum())
    checks.append(
        {
            "name": "group_leakage",
            "status": "FAIL" if leak_groups else "PASS",
            "detail": f"{leak_groups} groups appear in multiple splits",
            "critical": True,
        }
    )
    critical_fail = critical_fail or bool(leak_groups)

    # Exact hash leakage if md5 available
    if "md5" in master.columns:
        hash_splits = master.groupby("md5")["split"].nunique()
        hash_leak = int((hash_splits > 1).sum())
    else:
        hash_leak = -1
    checks.append(
        {
            "name": "exact_duplicate_leakage",
            "status": "FAIL" if hash_leak > 0 else ("PASS" if hash_leak == 0 else "WARN"),
            "detail": (
                f"{hash_leak} MD5 hashes appear in multiple splits"
                if hash_leak >= 0
                else "md5 column missing; skipped"
            ),
            "critical": hash_leak > 0,
        }
    )
    critical_fail = critical_fail or (hash_leak > 0)

    def overlap(a: str, b: str) -> int:
        sa = set(master.loc[master["split"] == a, "image_path"])
        sb = set(master.loc[master["split"] == b, "image_path"])
        return len(sa & sb)

    overlaps = {
        "train_test_overlap": overlap("train", "test"),
        "train_validation_overlap": overlap("train", "validation"),
        "validation_test_overlap": overlap("validation", "test"),
    }
    for name, value in overlaps.items():
        checks.append(
            {
                "name": name,
                "status": "FAIL" if value else "PASS",
                "detail": f"{value} overlapping image paths",
                "critical": True,
            }
        )
        critical_fail = critical_fail or bool(value)

    # Missing classes overall / per split
    missing_overall = [c for c in classes if c not in set(master["label"])]
    checks.append(
        {
            "name": "missing_classes_overall",
            "status": "FAIL" if missing_overall else "PASS",
            "detail": f"Missing: {missing_overall}" if missing_overall else "All classes present in dataset index",
            "critical": True,
        }
    )
    critical_fail = critical_fail or bool(missing_overall)

    missing_per_split = {}
    for split in ["train", "validation", "test"]:
        present = set(master.loc[master["split"] == split, "label"])
        missing_per_split[split] = [c for c in classes if c not in present]
    any_missing_split = any(missing_per_split.values())
    checks.append(
        {
            "name": "missing_classes_per_split",
            "status": "FAIL" if any_missing_split else "PASS",
            "detail": str(missing_per_split) if any_missing_split else "All classes appear in every split",
            "critical": False,
        }
    )

    counts = {
        "total_images": int(len(master)),
        "total_groups": int(master["group_id"].nunique()),
        "train": int((master["split"] == "train").sum()),
        "validation": int((master["split"] == "validation").sum()),
        "test": int((master["split"] == "test").sum()),
        "class_counts": master["label"].value_counts().reindex(classes).fillna(0).astype(int).to_dict(),
        "split_class_counts": (
            master.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(classes).to_dict()
        ),
    }

    # Proportions warning only
    total = max(counts["total_images"], 1)
    props = {
        "train": counts["train"] / total,
        "validation": counts["validation"] / total,
        "test": counts["test"] / total,
    }
    checks.append(
        {
            "name": "split_proportions",
            "status": "PASS",
            "detail": (
                f"train={props['train']:.3f}, validation={props['validation']:.3f}, "
                f"test={props['test']:.3f} (targets 0.70/0.15/0.15; group constraints apply)"
            ),
            "critical": False,
        }
    )

    if groups_path.exists():
        groups = pd.read_csv(groups_path)
        mismatch = 0
        if "group_id" in groups.columns and "split" in groups.columns:
            mapped = dict(zip(groups["group_id"], groups["split"]))
            for gid, split in master.groupby("group_id")["split"].first().items():
                if mapped.get(gid) != split:
                    mismatch += 1
        checks.append(
            {
                "name": "split_groups_consistency",
                "status": "FAIL" if mismatch else "PASS",
                "detail": f"{mismatch} group_id split mismatches vs split_groups.csv",
                "critical": True,
            }
        )
        critical_fail = critical_fail or bool(mismatch)

    overall = "FAIL" if critical_fail else "PASS"
    return {
        "overall": overall,
        "critical_fail": critical_fail,
        "checks": checks,
        "counts": counts,
        "overlaps": overlaps,
        "hash_leak": hash_leak,
        "group_leak": leak_groups,
        "missing_per_split": missing_per_split,
    }
