from __future__ import annotations

import numpy as np
from PIL import Image


def _resize_gray(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    return np.asarray(img.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def ahash(gray: np.ndarray, hash_size: int = 16) -> int:
    pixels = _resize_gray(gray, (hash_size, hash_size))
    avg = pixels.mean()
    bits = pixels >= avg
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def dhash(gray: np.ndarray, hash_size: int = 16) -> int:
    pixels = _resize_gray(gray, (hash_size + 1, hash_size))
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _dct_2d(block: np.ndarray) -> np.ndarray:
    """Orthonormal 2D DCT via separable matrix multiply."""
    n = block.shape[0]
    x = np.arange(n, dtype=np.float64)
    k = x.reshape(-1, 1)
    basis = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    basis[0, :] *= 1 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / n)
    return basis @ block.astype(np.float64) @ basis.T


def phash(gray: np.ndarray, hash_size: int = 16, highfreq_factor: int = 4) -> int:
    img_size = hash_size * highfreq_factor
    pixels = _resize_gray(gray, (img_size, img_size))
    dct = _dct_2d(pixels)
    dct_low = dct[:hash_size, :hash_size]
    # Exclude DC term for median threshold, matching common pHash practice.
    med = np.median(dct_low.flatten()[1:])
    bits = dct_low >= med
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def hash_to_hex(value: int, n_bits: int) -> str:
    width = max(1, (n_bits + 3) // 4)
    return f"{value:0{width}x}"
