"""Build leakage-controlled proxy groups and the master train/validation/test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_config, project_root
from src.similarity.analysis import analyze_similarity
from src.splitting.grouping import build_proxy_groups
from src.splitting.master_split import assign_group_aware_split


def _plot_distributions(master: pd.DataFrame, group_tbl: pd.DataFrame, out_dir: Path, classes: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution
    class_counts = master["label"].value_counts().reindex(classes).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(class_counts.index.astype(str), class_counts.values, color="#2F6F8F")
    ax.set_title("Dataset class distribution")
    ax.set_ylabel("Images")
    ax.set_xlabel("Blood group")
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 8, str(int(v)), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_class_distribution.png", dpi=140)
    plt.close(fig)

    # Split totals
    split_order = ["train", "validation", "test"]
    split_counts = master["split"].value_counts().reindex(split_order).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(split_counts.index.astype(str), split_counts.values, color=["#2F6F8F", "#C27A3A", "#6B8F71"])
    ax.set_title("Master split image counts")
    ax.set_ylabel("Images")
    for i, v in enumerate(split_counts.values):
        ax.text(i, v + 8, str(int(v)), ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "split_distribution.png", dpi=140)
    plt.close(fig)

    # Per-class split stacked
    pivot = (
        master.groupby(["label", "split"]).size().unstack(fill_value=0)
        .reindex(classes)
        .reindex(columns=split_order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = None
    colors = {"train": "#2F6F8F", "validation": "#C27A3A", "test": "#6B8F71"}
    for split in split_order:
        vals = pivot[split].values
        ax.bar(pivot.index.astype(str), vals, bottom=bottom, label=split, color=colors[split])
        bottom = vals if bottom is None else bottom + vals
    ax.set_title("Per-class train / validation / test distribution")
    ax.set_ylabel("Images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_split_distribution.png", dpi=140)
    plt.close(fig)

    # Group size distribution
    sizes = group_tbl["group_size"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes, bins=range(1, int(sizes.max()) + 2), color="#4C6A7A", edgecolor="white", align="left")
    ax.set_title("Proxy group size distribution")
    ax.set_xlabel("Group size")
    ax.set_ylabel("Number of groups")
    fig.tight_layout()
    fig.savefig(out_dir / "group_size_distribution.png", dpi=140)
    plt.close(fig)

    # Exact duplicate bar
    exact = group_tbl[group_tbl["group_id"].astype(str).str.startswith("EXACT_")]
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(exact):
        ax.bar(exact["group_id"], exact["group_size"], color="#8C4A3A")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.set_title(f"Exact duplicate groups (n={len(exact)})")
        ax.set_ylabel("Files in group")
    else:
        ax.text(0.5, 0.5, "No exact duplicate groups", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_dir / "exact_duplicate_statistics.png", dpi=140)
    plt.close(fig)


def build_master_split(cfg: dict | None = None, reuse_similarity: bool = False) -> dict:
    root = project_root()
    cfg = cfg or load_config()
    paths = cfg["paths"]

    print("Building leakage-controlled proxy groups...")
    print(f"Dataset: {cfg['dataset_path']}")

    sim_csv = root / paths["similarity_analysis"]
    if reuse_similarity and sim_csv.exists():
        print(f"Reusing existing similarity analysis: {sim_csv}")
        from src.data.dataset import load_image_index

        index_df = load_image_index(cfg)
        pairs_df = pd.read_csv(sim_csv)
        sim_stats = {"reused": True}
    else:
        sim = analyze_similarity(cfg)
        index_df = sim["index"]
        pairs_df = sim["pairs"]
        sim_stats = sim["stats"]

    print(f"Dataset: {len(index_df)} images")
    groups_df, exact_df, group_meta = build_proxy_groups(
        index_df,
        pairs_df,
        phash_max_hamming=cfg["similarity_thresholds"]["phash_max_hamming"],
        confirm_ssim_min=cfg["similarity_thresholds"]["confirm_ssim_min_score"],
        confirm_with_ssim=cfg["similarity_thresholds"].get("confirm_with_ssim", True),
    )
    print(f"Exact duplicate groups: {group_meta['exact_duplicate_groups']}")
    print(f"Similarity-merged proxy groups: {group_meta['similarity_merged_groups']}")
    print(f"Proxy groups total: {group_meta['total_groups']}")

    print("Creating master split...")
    master, group_tbl, split_summary = assign_group_aware_split(
        groups_df,
        train_ratio=cfg["train_ratio"],
        validation_ratio=cfg["validation_ratio"],
        test_ratio=cfg["test_ratio"],
        random_seed=cfg["random_seed"],
        classes=cfg["classes"],
    )

    # Persist
    master_path = root / paths["master_split"]
    groups_path = root / paths["split_groups"]
    master_path.parent.mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    master_out = master[
        ["image_path", "label", "group_id", "split", "md5", "group_size"]
    ].copy()
    master_out.to_csv(master_path, index=False)
    group_tbl.to_csv(groups_path, index=False)

    exact_path = root / paths["exact_duplicate_groups"]
    exact_path.parent.mkdir(parents=True, exist_ok=True)
    if len(exact_df):
        # Map to final EXACT_ group ids from groups_df
        exact_final = groups_df[groups_df["group_id"].astype(str).str.startswith("EXACT_")].copy()
        exact_final = exact_final.rename(columns={"md5": "hash"})[
            ["group_id", "image_path", "hash", "label"]
        ]
        exact_final.to_csv(exact_path, index=False)
    else:
        pd.DataFrame(columns=["group_id", "image_path", "hash", "label"]).to_csv(exact_path, index=False)

    # Summary tables
    summary_rows = []
    total = len(master_out)
    for split in ["train", "validation", "test"]:
        for label in cfg["classes"]:
            count = int(((master_out["split"] == split) & (master_out["label"] == label)).sum())
            summary_rows.append(
                {
                    "split": split,
                    "label": label,
                    "count": count,
                    "percentage": count / total if total else 0.0,
                }
            )
    summary_csv = pd.DataFrame(summary_rows)
    summary_csv.to_csv(root / paths["split_summary_csv"], index=False)

    class_counts = master_out["label"].value_counts().reindex(cfg["classes"]).fillna(0).astype(int).to_dict()
    group_counts = {
        "train": int((group_tbl["split"] == "train").sum()),
        "validation": int((group_tbl["split"] == "validation").sum()),
        "test": int((group_tbl["split"] == "test").sum()),
    }
    payload = {
        **split_summary,
        "class_counts": class_counts,
        "group_counts": group_counts,
        "grouping_method": cfg["grouping_method"],
        "similarity_method": cfg["selected_similarity_method"],
        "threshold": cfg["similarity_thresholds"],
        "group_meta": group_meta,
        "similarity_stats": sim_stats,
        "disclaimer": (
            "This grouping strategy attempts to prevent obvious duplicate or highly similar "
            "fingerprint images from crossing train/validation/test boundaries. Because "
            "participant identifiers are unavailable, the groups cannot be interpreted as "
            "confirmed human subjects."
        ),
    }
    with open(root / paths["split_summary_json"], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    _plot_distributions(master_out, group_tbl, root / "results" / "splits", cfg["classes"])

    print(
        f"\nTrain: {payload['train_count']}\n"
        f"Validation: {payload['validation_count']}\n"
        f"Test: {payload['test_count']}\n"
    )
    print(f"Master split saved:\n{master_path}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--reuse-similarity", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    build_master_split(cfg, reuse_similarity=args.reuse_similarity)


if __name__ == "__main__":
    main()
