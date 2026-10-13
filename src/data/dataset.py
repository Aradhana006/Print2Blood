from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    root = project_root()
    cfg_path = Path(path) if path else root / "configs" / "split_config.json"
    with open(cfg_path, encoding="utf-8") as handle:
        return json.load(handle)


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return arr


def image_stats(path: Path) -> dict:
    with Image.open(path) as img:
        arr = np.asarray(img)
        mode = img.mode
        width, height = img.size
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
        channels = 1
    else:
        channels = arr.shape[2]
        gray = arr[..., :3].mean(axis=2).astype(np.float32)
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "mode": mode,
        "mean_brightness": float(gray.mean()),
        "std_contrast": float(gray.std()),
    }


def load_image_index(cfg: dict | None = None) -> pd.DataFrame:
    root = project_root()
    cfg = cfg or load_config()
    dataset = root / cfg["dataset_path"]
    rows = []
    for label in cfg.get("classes", CLASSES):
        class_dir = dataset / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            stats = image_stats(path)
            rows.append(
                {
                    "image_path": rel,
                    "filename": path.name,
                    "label": label,
                    "md5": file_md5(path),
                    **stats,
                }
            )
    return pd.DataFrame(rows).sort_values(["label", "filename"]).reset_index(drop=True)
