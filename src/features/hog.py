"""HOG feature extraction for classical ML experiments (offline only)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

DEFAULT_HOG_PARAMS = {
    "image_size": 128,
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
    "feature_vector": True,
}


def _load_gray_array(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb2gray(arr)
    if gray.shape != (image_size, image_size):
        gray = resize(gray, (image_size, image_size), anti_aliasing=True, preserve_range=False)
    return gray.astype(np.float64)


def hog_feature_dim(params: dict | None = None) -> int:
    params = {**DEFAULT_HOG_PARAMS, **(params or {})}
    size = int(params["image_size"])
    dummy = np.zeros((size, size), dtype=np.float64)
    feat = hog(
        dummy,
        orientations=int(params["orientations"]),
        pixels_per_cell=tuple(params["pixels_per_cell"]),
        cells_per_block=tuple(params["cells_per_block"]),
        block_norm=str(params["block_norm"]),
        transform_sqrt=bool(params["transform_sqrt"]),
        feature_vector=bool(params["feature_vector"]),
    )
    return int(feat.shape[0])


def extract_hog_from_path(path: Path, root: Path | None = None, params: dict | None = None) -> np.ndarray:
    params = {**DEFAULT_HOG_PARAMS, **(params or {})}
    full = path if path.is_absolute() else (root / path if root else path)
    gray = _load_gray_array(Path(full), int(params["image_size"]))
    return hog(
        gray,
        orientations=int(params["orientations"]),
        pixels_per_cell=tuple(params["pixels_per_cell"]),
        cells_per_block=tuple(params["cells_per_block"]),
        block_norm=str(params["block_norm"]),
        transform_sqrt=bool(params["transform_sqrt"]),
        feature_vector=bool(params["feature_vector"]),
    )


def extract_hog_batch(
    paths: Sequence[str | Path],
    root: Path,
    params: dict | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    if len(paths) == 0:
        return np.empty((0, hog_feature_dim(params)), dtype=np.float64)
    if n_jobs == 1 or len(paths) < 32:
        rows = [extract_hog_from_path(Path(rel), root=root, params=params) for rel in paths]
        return np.vstack(rows)
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(extract_hog_from_path)(Path(rel), root=root, params=params) for rel in paths
    )
    return np.vstack(rows)
