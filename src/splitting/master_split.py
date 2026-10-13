from __future__ import annotations

import pandas as pd
import numpy as np


def _group_table(groups_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, g in groups_df.groupby("group_id"):
        labels = sorted(g["label"].unique().tolist())
        majority = g["label"].value_counts().index[0]
        rows.append(
            {
                "group_id": group_id,
                "group_size": int(len(g)),
                "labels": "|".join(labels),
                "primary_label": majority,
                "n_labels": int(len(labels)),
                "image_paths": "|".join(sorted(g["image_path"].tolist())),
            }
        )
    return pd.DataFrame(rows).sort_values(["primary_label", "group_id"]).reset_index(drop=True)


def assign_group_aware_split(
    groups_df: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    classes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Assign whole proxy groups to train/validation/test.

    Groups are never split. Exact percentages may deviate because group sizes vary.
    """
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(random_seed)
    group_tbl = _group_table(groups_df)
    if classes is None:
        classes = sorted(group_tbl["primary_label"].unique().tolist())

    split_of_group: dict[str, str] = {}

    for label in classes:
        label_groups = group_tbl[group_tbl["primary_label"] == label].copy()
        if label_groups.empty:
            continue

        # Deterministic shuffle of groups within label.
        perm = rng.permutation(len(label_groups))
        label_groups = label_groups.iloc[perm].reset_index(drop=True)
        sizes = label_groups["group_size"].astype(int).tolist()
        gids = label_groups["group_id"].tolist()
        total = int(sum(sizes))
        target_test = int(round(total * test_ratio))
        target_val = int(round(total * validation_ratio))

        assigned = ["train"] * len(gids)
        test_count = 0
        val_count = 0

        # Pass 1: fill test.
        for i, size in enumerate(sizes):
            if test_count >= target_test:
                break
            # Avoid taking every remaining group if later splits would be empty.
            remaining_after = len(gids) - i - 1
            need_val = 1 if target_val > 0 else 0
            need_train = 1
            if remaining_after < need_val + need_train and test_count > 0:
                continue
            assigned[i] = "test"
            test_count += size

        # Pass 2: fill validation from remaining train slots.
        for i, size in enumerate(sizes):
            if assigned[i] != "train":
                continue
            if val_count >= target_val:
                break
            remaining_train_slots = sum(1 for a in assigned if a == "train") - 1
            if remaining_train_slots < 1 and val_count > 0:
                continue
            assigned[i] = "validation"
            val_count += size

        # Guarantee coverage when enough groups exist.
        if len(gids) >= 3:
            if "test" not in assigned:
                # Move last train/val group to test.
                for i in range(len(assigned) - 1, -1, -1):
                    if assigned[i] == "train":
                        assigned[i] = "test"
                        break
            if "validation" not in assigned:
                for i in range(len(assigned) - 1, -1, -1):
                    if assigned[i] == "train":
                        assigned[i] = "validation"
                        break
            if "train" not in assigned:
                for i in range(len(assigned)):
                    if assigned[i] != "train":
                        assigned[i] = "train"
                        break

        for gid, split in zip(gids, assigned):
            split_of_group[gid] = split

    for gid in group_tbl["group_id"]:
        split_of_group.setdefault(gid, "train")

    master = groups_df.copy()
    master["split"] = master["group_id"].map(split_of_group)
    master = master.sort_values(["split", "label", "group_id", "image_path"]).reset_index(drop=True)

    group_tbl = group_tbl.copy()
    group_tbl["split"] = group_tbl["group_id"].map(split_of_group)
    group_tbl = group_tbl.sort_values(["split", "primary_label", "group_id"]).reset_index(drop=True)

    counts = master["split"].value_counts()
    total_images = int(len(master))
    summary = {
        "total_images": total_images,
        "total_groups": int(master["group_id"].nunique()),
        "train_count": int(counts.get("train", 0)),
        "validation_count": int(counts.get("validation", 0)),
        "test_count": int(counts.get("test", 0)),
        "train_pct": float(counts.get("train", 0) / total_images),
        "validation_pct": float(counts.get("validation", 0) / total_images),
        "test_pct": float(counts.get("test", 0) / total_images),
        "random_seed": int(random_seed),
        "target_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
    }
    return master, group_tbl, summary
