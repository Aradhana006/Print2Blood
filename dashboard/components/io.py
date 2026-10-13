from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_csv(rel_path: str) -> pd.DataFrame | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json(rel_path: str) -> dict | None:
    import json

    path = ROOT / rel_path
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_thumbnail(rel_path: str, size: tuple[int, int] = (128, 128)):
    path = ROOT / rel_path
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail(size)
            return img.copy()
    except Exception:
        return None


def status_badge(ok: bool) -> str:
    return "PASS" if ok else "FAIL"
