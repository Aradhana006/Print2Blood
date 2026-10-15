"""Classical fingerprint image analysis helpers (quality + ridge cues)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _to_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        gray = np.asarray(ImageOps.exif_transpose(img).convert("L"), dtype=np.float32)
    return gray


def _laplacian_var(gray: np.ndarray) -> float:
    # Simple Laplacian kernel variance = sharpness proxy
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(gray, 1, mode="edge")
    out = (
        kernel[0, 1] * pad[:-2, 1:-1]
        + kernel[1, 0] * pad[1:-1, :-2]
        + kernel[1, 1] * pad[1:-1, 1:-1]
        + kernel[1, 2] * pad[1:-1, 2:]
        + kernel[2, 1] * pad[2:, 1:-1]
    )
    return float(out.var())


def estimate_pattern_type(gray: np.ndarray) -> dict:
    """
    Lightweight orientation-histogram heuristic.
    Not a certified AFIS classifier — research cue only.
    """
    # Gradients
    gy, gx = np.gradient(gray)
    ang = (np.arctan2(gy, gx) + np.pi) / 2.0  # ridge orientation ~0..pi
    hist, _ = np.histogram(ang.flatten(), bins=18, range=(0, np.pi), density=True)
    peak = float(np.asarray(hist.max()).item())
    entropy = float(np.asarray(-(hist[hist > 0] * np.log(hist[hist > 0] + 1e-12)).sum()).item())
    # Circular variance of doubled angles (for axial data)
    doubled = 2 * ang
    c = float(np.asarray(np.mean(np.cos(doubled))).item())
    s = float(np.asarray(np.mean(np.sin(doubled))).item())
    R = float(np.asarray(np.hypot(c, s)).item())

    if peak > 0.14 and R > 0.25:
        pattern = "Arch-like"
        reason = "Orientation energy is concentrated in a dominant direction."
    elif R < 0.12 and entropy > 2.4:
        pattern = "Whorl-like"
        reason = "Orientation directions are more evenly spread (circular tendency)."
    else:
        pattern = "Loop-like"
        reason = "Mixed orientation structure typical of loop patterns."

    return {
        "pattern_estimate": pattern,
        "orientation_peak": round(peak, 4),
        "orientation_entropy": round(entropy, 4),
        "orientation_coherence": round(R, 4),
        "note": reason + " This is a heuristic cue, not an AFIS-certified pattern class.",
    }


def _py(value) -> float:
    """Force JSON-safe Python float (avoids numpy scalar encoding errors)."""
    return float(np.asarray(value).item())


def analyze_quality(path: Path) -> dict:
    gray = _to_gray(path)
    h, w = gray.shape
    brightness = _py(gray.mean())
    contrast = _py(gray.std())
    sharp = _py(_laplacian_var(gray))
    # Edge density as ridge-activity proxy
    edges = _py(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())
    # Normalize scores 0-100 with soft caps
    sharpness_score = max(0.0, min(100.0, (sharp / 80.0) * 100.0))
    contrast_score = max(0.0, min(100.0, (contrast / 60.0) * 100.0))
    brightness_score = max(0.0, min(100.0, 100.0 - abs(brightness - 140.0) * 0.7))
    ridge_score = max(0.0, min(100.0, (edges / 25.0) * 100.0))
    overall = 0.35 * sharpness_score + 0.25 * contrast_score + 0.20 * brightness_score + 0.20 * ridge_score

    if overall >= 70:
        verdict = "Good"
        tip = "Image looks usable for model inference."
    elif overall >= 45:
        verdict = "Fair"
        tip = "Try brighter lighting and hold the finger steadier."
    else:
        verdict = "Poor"
        tip = "Recapture: fill the frame, reduce blur, improve contrast."

    pattern = estimate_pattern_type(gray)
    return {
        "width": int(w),
        "height": int(h),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharp, 2),
        "scores": {
            "overall": round(float(overall), 1),
            "sharpness": round(float(sharpness_score), 1),
            "contrast": round(float(contrast_score), 1),
            "brightness": round(float(brightness_score), 1),
            "ridge_activity": round(float(ridge_score), 1),
        },
        "verdict": verdict,
        "tip": tip,
        "pattern": pattern,
    }


def make_enhanced_preview_b64(path: Path) -> str:
    """CLAHE-like contrast boost preview for the report."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.15)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")
