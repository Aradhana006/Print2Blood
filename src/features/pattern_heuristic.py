"""Orientation-based pattern heuristic (research cue only — not AFIS ground truth)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        gray = np.asarray(ImageOps.exif_transpose(img).convert("L"), dtype=np.float32)
    return gray


def estimate_pattern_type(gray: np.ndarray) -> str:
    gy, gx = np.gradient(gray)
    ang = (np.arctan2(gy, gx) + np.pi) / 2.0
    hist, _ = np.histogram(ang.flatten(), bins=18, range=(0, np.pi), density=True)
    peak = float(np.asarray(hist.max()).item())
    entropy = float(np.asarray(-(hist[hist > 0] * np.log(hist[hist > 0] + 1e-12)).sum()).item())
    doubled = 2 * ang
    c = float(np.asarray(np.mean(np.cos(doubled))).item())
    s = float(np.asarray(np.mean(np.sin(doubled))).item())
    r = float(np.asarray(np.hypot(c, s)).item())
    if peak > 0.14 and r > 0.25:
        return "Arch-like"
    if r < 0.12 and entropy > 2.4:
        return "Whorl-like"
    return "Loop-like"
