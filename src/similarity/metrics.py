from __future__ import annotations

import numpy as np
from PIL import Image


def _resize(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    return np.asarray(img.resize(size, Image.Resampling.BILINEAR), dtype=np.float64)


def ssim_score(gray_a: np.ndarray, gray_b: np.ndarray, size: tuple[int, int] = (64, 64)) -> float:
    """Lightweight global SSIM on resized grayscale images."""
    a = _resize(gray_a, size)
    b = _resize(gray_b, size)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numer = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denom = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float(numer / denom)
