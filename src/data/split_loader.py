"""Load train/validation/test assignments from the master split manifest."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import CLASSES, project_root


def load_master_split(master_path: str | Path | None = None) -> pd.DataFrame:
    root = project_root()
    path = Path(master_path) if master_path else root / "data" / "splits" / "master_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Master split not found: {path}")
    df = pd.read_csv(path)
    required = {"image_path", "label", "group_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"master_split missing columns: {sorted(missing)}")
    invalid = set(df["split"]) - {"train", "validation", "test"}
    if invalid:
        raise ValueError(f"Invalid split values: {invalid}")
    invalid_labels = set(df["label"]) - set(CLASSES)
    if invalid_labels:
        raise ValueError(f"Invalid labels: {invalid_labels}")
    return df


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["split"] == "train"].reset_index(drop=True)
    val = df.loc[df["split"] == "validation"].reset_index(drop=True)
    test = df.loc[df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def assert_split_isolation(df: pd.DataFrame) -> None:
    paths = df.groupby("split")["image_path"].apply(set)
    train, val, test = paths["train"], paths["validation"], paths["test"]
    if train & val:
        raise AssertionError("Train/validation image_path overlap")
    if train & test:
        raise AssertionError("Train/test image_path overlap")
    if val & test:
        raise AssertionError("Validation/test image_path overlap")
    for split_name, part in [("train", df[df["split"] == "train"]), ("validation", df[df["split"] == "validation"]), ("test", df[df["split"] == "test"])]:
        leaked = part.groupby("group_id")["split"].nunique()
        if (leaked > 1).any():
            raise AssertionError(f"Group leakage in {split_name}")


def label_to_index(labels: pd.Series, class_order: list[str] | None = None) -> np.ndarray:
    order = class_order or CLASSES
    mapping = {c: i for i, c in enumerate(order)}
    return np.asarray([mapping[l] for l in labels], dtype=np.int64)
