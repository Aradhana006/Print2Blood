"""Local Binary Pattern histogram features (optional classical ML representation)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.transform import resize

DEFAULT_LBP_PARAMS = {
    "image_size": 128,
    "P": 8,
    "R": 1,
    "method": "uniform",
}


def lbp_histogram_dim(params: dict | None = None) -> int:
    params = {**DEFAULT_LBP_PARAMS, **(params or {})}
    p = int(params["P"])
    # uniform LBP: P + 2 bins
    return p + 2


def _load_gray(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb2gray(arr)
    if gray.shape != (image_size, image_size):
        gray = resize(gray, (image_size, image_size), anti_aliasing=True, preserve_range=False)
    return gray.astype(np.float64)


def extract_lbp_from_path(path: Path, root: Path | None = None, params: dict | None = None) -> np.ndarray:
    params = {**DEFAULT_LBP_PARAMS, **(params or {})}
    size = int(params["image_size"])
    p = int(params["P"])
    r = float(params["R"])
    method = str(params["method"])
    full = path if path.is_absolute() else (root / path if root else path)
    gray = _load_gray(Path(full), size)
    lbp = local_binary_pattern(gray, P=p, R=r, method=method)
    n_bins = p + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float64)


def extract_lbp_batch(
    paths: Sequence[str | Path],
    root: Path,
    params: dict | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    if len(paths) == 0:
        return np.empty((0, lbp_histogram_dim(params)), dtype=np.float64)
    if n_jobs == 1 or len(paths) < 32:
        rows = [extract_lbp_from_path(Path(rel), root=root, params=params) for rel in paths]
        return np.vstack(rows)
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(extract_lbp_from_path)(Path(rel), root=root, params=params) for rel in paths
    )
    return np.vstack(rows)
