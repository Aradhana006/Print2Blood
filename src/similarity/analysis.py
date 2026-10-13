"""Compute fingerprint similarity diagnostics (pHash/dHash/aHash/SSIM)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.data.dataset import load_config, load_gray, load_image_index, project_root
from src.similarity.hashes import ahash, dhash, hamming, phash
from src.similarity.metrics import ssim_score


def compute_hashes(index_df: pd.DataFrame, root: Path, hash_size: int) -> pd.DataFrame:
    rows = []
    for row in index_df.itertuples(index=False):
        gray = load_gray(root / row.image_path)
        rows.append(
            {
                "image_path": row.image_path,
                "label": row.label,
                "phash": phash(gray, hash_size=hash_size),
                "dhash": dhash(gray, hash_size=hash_size),
                "ahash": ahash(gray, hash_size=hash_size),
            }
        )
    return pd.DataFrame(rows)


def analyze_similarity(cfg: dict | None = None) -> dict:
    root = project_root()
    cfg = cfg or load_config()
    out_csv = root / cfg["paths"]["similarity_analysis"]
    examples_dir = root / cfg["paths"]["similarity_examples"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    print("Loading image index...")
    index_df = load_image_index(cfg)
    print(f"Images: {len(index_df)}")
    print("Computing perceptual hashes...")
    hash_df = compute_hashes(index_df, root, cfg["hash_size"])
    hash_df.to_csv(root / "results" / "splits" / "image_hashes.csv", index=False)

    gray_cache = {}
    def get_gray(path: str):
        if path not in gray_cache:
            gray_cache[path] = load_gray(root / path)
        return gray_cache[path]

    n_bits = cfg["hash_size"] ** 2
    thr = cfg["similarity_thresholds"]
    pairs = []

    # Within-label candidate search using pHash/dHash/aHash, confirm with SSIM.
    for label, group in hash_df.groupby("label"):
        items = group.to_dict("records")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                ph = hamming(a["phash"], b["phash"])
                dh = hamming(a["dhash"], b["dhash"])
                ah = hamming(a["ahash"], b["ahash"])

                if ph <= thr["phash_max_hamming"]:
                    ssim = ssim_score(
                        get_gray(a["image_path"]),
                        get_gray(b["image_path"]),
                        size=tuple(cfg["ssim_resize"]),
                    )
                    pairs.append(
                        {
                            "image_a": a["image_path"],
                            "image_b": b["image_path"],
                            "label_a": a["label"],
                            "label_b": b["label"],
                            "same_label": True,
                            "method": "phash",
                            "hamming": ph,
                            "similarity_score": 1.0 - (ph / n_bits),
                            "ssim": ssim,
                        }
                    )
                    if thr.get("confirm_with_ssim", True) and ssim >= thr["confirm_ssim_min_score"]:
                        pairs.append(
                            {
                                "image_a": a["image_path"],
                                "image_b": b["image_path"],
                                "label_a": a["label"],
                                "label_b": b["label"],
                                "same_label": True,
                                "method": "phash_ssim",
                                "hamming": ph,
                                "similarity_score": ssim,
                                "ssim": ssim,
                            }
                        )

                if dh <= thr["dhash_max_hamming"]:
                    pairs.append(
                        {
                            "image_a": a["image_path"],
                            "image_b": b["image_path"],
                            "label_a": a["label"],
                            "label_b": b["label"],
                            "same_label": True,
                            "method": "dhash",
                            "hamming": dh,
                            "similarity_score": 1.0 - (dh / n_bits),
                            "ssim": np.nan,
                        }
                    )

                if ah <= thr["ahash_max_hamming"]:
                    pairs.append(
                        {
                            "image_a": a["image_path"],
                            "image_b": b["image_path"],
                            "label_a": a["label"],
                            "label_b": b["label"],
                            "same_label": True,
                            "method": "ahash",
                            "hamming": ah,
                            "similarity_score": 1.0 - (ah / n_bits),
                            "ssim": np.nan,
                        }
                    )

    # Cross-label diagnostic sample using pHash only (no grouping).
    print("Scanning cross-label pHash diagnostics...")
    by_label = {lab: g.to_dict("records") for lab, g in hash_df.groupby("label")}
    labels = sorted(by_label)
    cross_pairs = 0
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            left = by_label[la]
            right = by_label[lb]
            # Limit comparisons for speed while remaining deterministic.
            for a in left[::3]:
                for b in right[::3]:
                    ph = hamming(a["phash"], b["phash"])
                    if ph <= thr["phash_max_hamming"]:
                        pairs.append(
                            {
                                "image_a": a["image_path"],
                                "image_b": b["image_path"],
                                "label_a": a["label"],
                                "label_b": b["label"],
                                "same_label": False,
                                "method": "phash_cross_label",
                                "hamming": ph,
                                "similarity_score": 1.0 - (ph / n_bits),
                                "ssim": np.nan,
                            }
                        )
                        cross_pairs += 1

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > cfg["max_similarity_pairs_saved"]:
        # Keep the strongest pairs preferentially.
        pairs_df = pairs_df.sort_values(
            ["same_label", "similarity_score"], ascending=[False, False]
        ).head(cfg["max_similarity_pairs_saved"])
    pairs_df = pairs_df.sort_values(
        ["method", "same_label", "similarity_score"], ascending=[True, False, False]
    ).reset_index(drop=True)
    pairs_df.to_csv(out_csv, index=False)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, ax in zip(["phash", "dhash"], axes):
        subset = pairs_df[pairs_df["method"] == method]
        if subset.empty:
            ax.set_title(f"{method}: no candidates")
            continue
        ax.hist(subset["similarity_score"], bins=30, color="#2F6F8F", edgecolor="white")
        ax.set_title(f"{method} candidate similarity")
        ax.set_xlabel("similarity_score = 1 - hamming/n_bits")
        ax.set_ylabel("pairs")
    fig.suptitle("Fingerprint similarity candidate scores (same-label search)")
    fig.tight_layout()
    plot_path = root / "results" / "splits" / "similarity_analysis.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    # Example montages for top confirmed pairs
    confirmed = pairs_df[pairs_df["method"] == "phash_ssim"].head(cfg["max_similarity_examples"])
    if confirmed.empty:
        confirmed = pairs_df[pairs_df["method"] == "phash"].head(cfg["max_similarity_examples"])
    for idx, row in enumerate(confirmed.itertuples(index=False), start=1):
        img_a = Image.open(root / row.image_a).convert("L").resize((160, 160))
        img_b = Image.open(root / row.image_b).convert("L").resize((160, 160))
        canvas = Image.new("RGB", (340, 210), "white")
        canvas.paste(img_a.convert("RGB"), (10, 30))
        canvas.paste(img_b.convert("RGB"), (170, 30))
        draw = ImageDraw.Draw(canvas)
        title = f"{row.method} score={row.similarity_score:.3f} | {row.label_a}/{row.label_b}"
        draw.text((10, 8), title, fill="black")
        draw.text((10, 195), Path(row.image_a).name, fill="black")
        draw.text((170, 195), Path(row.image_b).name, fill="black")
        canvas.save(examples_dir / f"similar_{idx:02d}.png")

    stats = {
        "n_images": int(len(index_df)),
        "n_pairs_saved": int(len(pairs_df)),
        "methods": pairs_df["method"].value_counts().to_dict() if len(pairs_df) else {},
        "same_label_fraction_all_saved": float(pairs_df["same_label"].mean()) if len(pairs_df) else 0.0,
        "phash_ssim_pairs": int((pairs_df["method"] == "phash_ssim").sum()) if len(pairs_df) else 0,
        "cross_label_phash_pairs": int(cross_pairs),
        "thresholds": thr,
        "selected_grouping_edges": "phash_ssim",
        "output_csv": str(out_csv.relative_to(root)).replace("\\", "/"),
        "plot": str(plot_path.relative_to(root)).replace("\\", "/"),
    }
    print(stats)
    return {"pairs": pairs_df, "hashes": hash_df, "stats": stats, "index": index_df}


if __name__ == "__main__":
    analyze_similarity()
