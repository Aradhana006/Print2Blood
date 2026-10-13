"""
Dataset and project audit for Fingerprint-Based Blood Group Detection.

Does NOT train or modify models. Produces:
  results/audit/dataset_summary.csv
  results/audit/dataset_summary.json
  results/audit/class_distribution.png
  results/audit/sample_grid.png
  results/audit/image_dimensions.png
  results/audit/brightness_contrast.png
  results/audit/duplicate_hashes.csv
  results/audit/near_duplicate_pairs.csv
  results/audit/filename_analysis.json
  results/audit/audit_report.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "dataset_blood_group"
OUT = ROOT / "results" / "audit"
CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
CLUSTER_TO_CLASS = {
    0: "A+",
    1: "A-",
    2: "B+",
    3: "B-",
    4: "AB+",
    5: "AB-",
    6: "O+",
    7: "O-",
}


def parse_name(path: Path):
    match = re.match(r"cluster_(\d+)_(\d+)\.(.+)$", path.name, re.I)
    if not match:
        return None
    return {
        "cluster": int(match.group(1)),
        "image_id": int(match.group(2)),
        "ext": match.group(3).lower(),
    }


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(arr_gray: np.ndarray, hash_size: int = 8) -> str:
    """Simple average hash for near-duplicate screening."""
    img = Image.fromarray(arr_gray).resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(img, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels >= avg).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def collect_rows():
    rows = []
    corrupted = []
    for class_name in CLASSES:
        class_dir = DATASET / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            parsed = parse_name(path)
            row = {
                "filepath": str(path.relative_to(ROOT)).replace("\\", "/"),
                "filename": path.name,
                "label": class_name,
                "bytes": path.stat().st_size,
                "cluster": parsed["cluster"] if parsed else None,
                "image_id": parsed["image_id"] if parsed else None,
                "ext": parsed["ext"] if parsed else path.suffix.lower().lstrip("."),
                "name_ok": parsed is not None,
                "cluster_matches_label": (
                    parsed is not None and CLUSTER_TO_CLASS.get(parsed["cluster"]) == class_name
                ),
            }
            try:
                with Image.open(path) as img:
                    img.load()
                    arr = np.asarray(img)
                if arr.ndim == 2:
                    gray = arr.astype(np.float32)
                    channels = 1
                else:
                    channels = arr.shape[2]
                    gray = arr.mean(axis=2).astype(np.float32)
                row.update(
                    {
                        "width": int(arr.shape[1]),
                        "height": int(arr.shape[0]),
                        "channels": int(channels),
                        "dtype": str(arr.dtype),
                        "mean_brightness": float(gray.mean()),
                        "std_contrast": float(gray.std()),
                        "min_pixel": float(gray.min()),
                        "max_pixel": float(gray.max()),
                        "md5": md5_file(path),
                        "ahash": perceptual_hash(gray.astype(np.uint8)),
                        "readable": True,
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                corrupted.append({"filepath": row["filepath"], "error": str(exc)})
                row.update(
                    {
                        "width": None,
                        "height": None,
                        "channels": None,
                        "dtype": None,
                        "mean_brightness": None,
                        "std_contrast": None,
                        "min_pixel": None,
                        "max_pixel": None,
                        "md5": md5_file(path),
                        "ahash": None,
                        "readable": False,
                        "error": str(exc),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows), corrupted


def find_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby("md5")["filepath"].apply(list)
    out = []
    for digest, paths in groups.items():
        if len(paths) > 1:
            out.append({"md5": digest, "count": len(paths), "filepaths": " | ".join(paths)})
    return pd.DataFrame(out)


def find_near_duplicates(df: pd.DataFrame, max_distance: int = 5, max_pairs: int = 5000):
    """
    Group by average-hash first, then compare nearby hashes.
    Near-duplicate grouping is NOT equivalent to true subject IDs.
    """
    usable = df.dropna(subset=["ahash"]).copy()
    by_hash = defaultdict(list)
    for _, row in usable.iterrows():
        by_hash[row["ahash"]].append(row)

    pairs = []
    hashes = list(by_hash.keys())

    # exact ahash collisions across different files
    for digest, items in by_hash.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                pairs.append(
                    {
                        "file_a": a["filepath"],
                        "file_b": b["filepath"],
                        "label_a": a["label"],
                        "label_b": b["label"],
                        "ahash_a": a["ahash"],
                        "ahash_b": b["ahash"],
                        "hamming": 0,
                        "same_label": a["label"] == b["label"],
                        "note": "identical_ahash",
                    }
                )

    # compare distinct hashes with small Hamming distance
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming(hashes[i], hashes[j])
            if dist == 0 or dist > max_distance:
                continue
            for a in by_hash[hashes[i]]:
                for b in by_hash[hashes[j]]:
                    pairs.append(
                        {
                            "file_a": a["filepath"],
                            "file_b": b["filepath"],
                            "label_a": a["label"],
                            "label_b": b["label"],
                            "ahash_a": a["ahash"],
                            "ahash_b": b["ahash"],
                            "hamming": dist,
                            "same_label": a["label"] == b["label"],
                            "note": "near_ahash",
                        }
                    )
                    if len(pairs) >= max_pairs:
                        return pd.DataFrame(pairs)
    return pd.DataFrame(pairs)


def plot_class_distribution(df: pd.DataFrame, path: Path):
    counts = df["label"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#2F6F8F")
    ax.set_xlabel("Blood group")
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset class distribution (n=6,000)")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 10, str(int(value)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for ax, label in zip(axes.flat, CLASSES):
        sample = df[df["label"] == label].iloc[0]
        img = np.asarray(Image.open(ROOT / sample["filepath"]))
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Sample fingerprint per blood-group class")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_dimensions(df: pd.DataFrame, path: Path):
    usable = df.dropna(subset=["width", "height"])
    size_counts = (
        usable.assign(size=usable["width"].astype(int).astype(str) + "x" + usable["height"].astype(int).astype(str))
        .groupby("size")
        .size()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(size_counts.index.astype(str), size_counts.values, color="#6B8F71")
    ax.set_xlabel("Width x Height")
    ax.set_ylabel("Count")
    ax.set_title("Native image dimensions")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_brightness_contrast(df: pd.DataFrame, path: Path):
    usable = df.dropna(subset=["mean_brightness", "std_contrast"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(usable["mean_brightness"], bins=40, color="#2F6F8F", edgecolor="white")
    axes[0].set_title("Mean brightness")
    axes[0].set_xlabel("Pixel mean")
    axes[0].set_ylabel("Images")
    axes[1].hist(usable["std_contrast"], bins=40, color="#C27A3A", edgecolor="white")
    axes[1].set_title("Contrast (pixel std)")
    axes[1].set_xlabel("Pixel std")
    axes[1].set_ylabel("Images")
    fig.suptitle("Image intensity statistics")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def filename_analysis(df: pd.DataFrame) -> dict:
    ids = df["image_id"].dropna().astype(int)
    id_to_labels = defaultdict(set)
    for _, row in df.dropna(subset=["image_id"]).iterrows():
        id_to_labels[int(row["image_id"])].add(row["label"])
    multi = {str(k): sorted(v) for k, v in id_to_labels.items() if len(v) > 1}
    return {
        "naming_pattern": "cluster_<class_code>_<image_id>.BMP",
        "cluster_to_class": CLUSTER_TO_CLASS,
        "all_names_match_pattern": bool(df["name_ok"].all()),
        "all_clusters_match_folder_labels": bool(df["cluster_matches_label"].all()),
        "unique_filenames": int(df["filename"].nunique()),
        "total_files": int(len(df)),
        "unique_image_ids": int(ids.nunique()),
        "image_id_min": int(ids.min()) if len(ids) else None,
        "image_id_max": int(ids.max()) if len(ids) else None,
        "image_ids_spanning_multiple_classes": len(multi),
        "subject_id_field_present": False,
        "subject_id_evidence": (
            "No explicit subject/person ID field was found. "
            "Filenames use cluster_<0-7>_<image_id>, where cluster maps 1:1 to blood-group folder. "
            "Numeric image_ids are unique across the whole dataset (0..5999 style indexing), "
            "so they behave like global sample indices, not verified participant IDs. "
            "Near-duplicate aHash grouping is an imperfect proxy and must not be treated as true subjects."
        ),
        "example_multi_class_ids": dict(list(multi.items())[:10]),
    }


def existing_project_facts() -> dict:
    return {
        "reported_resnet50_accuracy_source": {
            "file": "code/Resnet34/Resnet.ipynb",
            "metric_text": "Test Accuracy: 81.42%",
            "test_loss": 0.58375,
            "macro_f1": 0.82,
            "evaluation_set_size": 1200,
            "notes": (
                "Value comes from saved notebook cell output after model.evaluate(valid_gen). "
                "valid_gen is built from the 20% test split, which is also passed as validation_data during fit()."
            ),
        },
        "split_strategy": {
            "method": "sklearn.model_selection.train_test_split",
            "test_size": 0.20,
            "random_state_on_split": 42,
            "stratify": None,
            "prior_shuffle": "data.sample(frac=1) with NO random_state, then reset_index",
            "deterministic_across_runs": False,
            "reason_not_fully_deterministic": (
                "Unseeded DataFrame.sample(frac=1) reorders rows before train_test_split. "
                "Even with random_state=42, the content mapped to shuffled indices changes across runs."
            ),
            "validation_set": "None dedicated; test split used as validation_data during training",
            "subject_aware": False,
        },
        "preprocessing_by_model": {
            "all_models": "tensorflow.keras.applications.resnet50.preprocess_input",
            "augmentation": "None (ImageDataGenerator only applies preprocess_input)",
            "lenet_image_size": [224, 224],
            "alexnet_image_size": [256, 256],
            "vgg16_image_size": [256, 256],
            "resnet50_image_size": [256, 256],
            "fingerprint_specific_pipeline": False,
        },
        "training_hyperparameters": {
            "optimizer": "adam (Keras default learning rate, typically 0.001)",
            "explicit_learning_rate": None,
            "loss": "categorical_crossentropy",
            "batch_size": 32,
            "epochs": 20,
            "class_weight": None,
            "metrics": ["accuracy"],
        },
        "resnet50_architecture": {
            "backbone": "ResNet50",
            "folder_name_mismatch": "Folder is Resnet34 but code uses ResNet50",
            "weights": "imagenet",
            "include_top": False,
            "pooling": "avg",
            "backbone_trainable": False,
            "head": ["Dense(128, relu)", "Dense(128, relu)", "Dense(8, softmax)"],
            "dropout_in_head": False,
            "batchnorm_in_head": False,
        },
        "leakage_and_evaluation_risks": [
            "Random image-level 80/20 split with no subject constraint",
            "No verified subject IDs in filenames or metadata",
            "Test split used as validation during training (no held-out final test)",
            "Unseeded shuffle before split reduces reproducibility of the exact 81.42% figure",
            "Hardcoded label index map in inference cells may disagree with train_gen.class_indices order",
            "Prediction confidence is not a calibrated probability",
        ],
        "other_model_notebook_results": {
            "LeNet": {"accuracy": 0.775, "loss": 1.15884},
            "VGG16": {"accuracy": 0.7375, "loss": 1.04504},
            "AlexNet": {"accuracy": 0.1633, "loss": 2.0696},
            "ResNet50": {"accuracy": 0.8142, "loss": 0.58375},
        },
    }


def write_audit_report(summary: dict, path: Path):
    facts = summary["project_facts"]
    ds = summary["dataset"]
    leak = summary["leakage_proxy"]
    lines = [
        "# Project Audit Report",
        "",
        "Accuracy improvement is **not** the goal of this audit. Scientific validity and reproducibility are.",
        "",
        "## 1. Existing architecture",
        "",
        "- Four CNN notebooks under `code/`: LeNet, AlexNet, VGG16, ResNet50 (folder named `Resnet34`).",
        "- Best reported model: **ResNet50 + frozen ImageNet backbone + Dense(128)-Dense(128)-Dense(8)**.",
        "- Inference prototype scripts: `predict.py` (ResNet `.h5`) and `run_lenet.py` (training helper).",
        "- System task: 8-class fingerprint image classification into ABO/Rh labels.",
        "- Scientific caveat: the system does **not** detect antigens; it learns visual correlations in this dataset.",
        "",
        "### ResNet50 head",
        "",
        "```text",
        "ResNet50(ImageNet, include_top=False, pooling='avg', trainable=False)",
        "  -> Dense(128, relu)",
        "  -> Dense(128, relu)",
        "  -> Dense(8, softmax)",
        "```",
        "",
        "## 2. Existing preprocessing",
        "",
        "- Only `resnet50.preprocess_input` via `ImageDataGenerator`.",
        "- No CLAHE, denoise, ridge enhancement, or fingerprint segmentation.",
        "- No geometric/color augmentation.",
        f"- ResNet/VGG/AlexNet resize to **256x256**; LeNet to **224x224**.",
        f"- Native images are almost entirely **{ds['dominant_size']}** with **{ds['dominant_channels']}-channel** arrays.",
        "",
        "## 3. Existing split strategy",
        "",
        "- `train_test_split(..., test_size=0.20, random_state=42)`.",
        "- **Not stratified.**",
        "- Preceded by `data.sample(frac=1)` **without a seed** => split is **not fully reproducible** across notebook reruns.",
        "- No dedicated validation set.",
        "- The 20% split is used both as `validation_data` in `fit()` and as the set for final `evaluate()` / classification report.",
        "- **Not subject-aware.**",
        "",
        "## 4. Existing leakage risks",
        "",
        "### Confirmed",
        "",
        "- Image-level random split can place highly similar fingerprints from the same acquisition context into both train and test.",
        "- Test images influence early-stopping-style monitoring because they are the validation stream during training.",
        "- No subject ID metadata exists to prevent person-level leakage.",
        "",
        "### Evidence from this audit",
        "",
        f"- Exact MD5 duplicates: **{leak['exact_duplicate_groups']}** groups.",
        f"- Average-hash identical pairs: **{leak['identical_ahash_pairs']}**.",
        f"- Near-duplicate pairs (Hamming <= 5): **{leak['near_duplicate_pairs']}**.",
        f"- Same-label share among near-duplicate pairs: **{leak['near_duplicate_same_label_fraction']:.3f}**.",
        "",
        "Near-duplicate aHash groups are a **proxy only**. They are **not** verified subject IDs.",
        "",
        "### Filename / subject analysis",
        "",
        f"- Pattern: `{summary['filename_analysis']['naming_pattern']}`.",
        f"- Cluster codes map 1:1 to class folders: `{summary['filename_analysis']['cluster_to_class']}`.",
        f"- Unique numeric `image_id`s: **{summary['filename_analysis']['unique_image_ids']}** / {summary['filename_analysis']['total_files']}.",
        f"- IDs spanning multiple classes: **{summary['filename_analysis']['image_ids_spanning_multiple_classes']}**.",
        f"- Subject ID field present: **{summary['filename_analysis']['subject_id_field_present']}**.",
        f"- Interpretation: {summary['filename_analysis']['subject_id_evidence']}",
        "",
        "## 5. Existing model parameters",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Optimizer | Adam (default LR, typically 1e-3; not set explicitly) |",
        "| Loss | categorical_crossentropy |",
        "| Batch size | 32 |",
        "| Epochs | 20 |",
        "| Class weights | none |",
        "| ImageNet weights | yes (ResNet50/VGG16) |",
        "| Frozen layers | entire ResNet50/VGG16 backbone |",
        "| Trainable layers | classification head only |",
        "| Augmentation | none |",
        "",
        "### Source of reported 81.42%",
        "",
        f"- Notebook: `{facts['reported_resnet50_accuracy_source']['file']}`",
        f"- Printed metric: `{facts['reported_resnet50_accuracy_source']['metric_text']}`",
        f"- Test loss in same output: `{facts['reported_resnet50_accuracy_source']['test_loss']}`",
        f"- Support: `{facts['reported_resnet50_accuracy_source']['evaluation_set_size']}` images",
        f"- Notes: {facts['reported_resnet50_accuracy_source']['notes']}",
        "",
        "Other notebook-reported accuracies:",
        "",
        "| Model | Accuracy | Loss |",
        "|---|---:|---:|",
        "| ResNet50 | 81.42% | 0.58375 |",
        "| LeNet | 77.50% | 1.15884 |",
        "| VGG16 | 73.75% | 1.04504 |",
        "| AlexNet | 16.33% | 2.06960 |",
        "",
        "## 6. Dataset problems",
        "",
        f"- Total images: **{ds['total_images']}**",
        f"- Corrupted/unreadable: **{ds['corrupted_images']}**",
        f"- Class imbalance: A- has **{ds['class_counts'].get('A-', 0)}**, A+ has **{ds['class_counts'].get('A+', 0)}**",
        f"- Dominant native size: **{ds['dominant_size']}**",
        f"- Distinct native sizes: **{ds['num_distinct_sizes']}**",
        f"- Mean brightness range: **{ds['brightness_min']:.2f} – {ds['brightness_max']:.2f}**",
        f"- Contrast (std) range: **{ds['contrast_min']:.2f} – {ds['contrast_max']:.2f}**",
        "- No participant metadata / demographics / finger-type labels.",
        "- Cluster prefix encodes class, which is fine for labeling but provides no subject grouping.",
        "- Possible acquisition/artifact bias has not yet been ruled out.",
        "",
        "## 7. Recommended next experiment",
        "",
        "Do **not** optimize accuracy next.",
        "",
        "1. Create a **master reproducible split config** with a fixed global seed and no unseeded `sample()`.",
        "2. Build a **leakage-controlled experiment**:",
        "   - Experiment A: reproduce random image split baseline (target ~81.42%).",
        "   - Experiment B: subject-proxy / near-duplicate-aware split, clearly labeled as a proxy.",
        "3. If true subject IDs become available, replace the proxy with a real subject-aware split.",
        "4. Keep a frozen copy of the current ResNet50 as the official baseline; do not overwrite it while auditing.",
        "5. Only after leakage-controlled numbers exist, improve preprocessing / fine-tuning / EfficientNet.",
        "",
        "### Suggested evaluation table for the next phase",
        "",
        "| Evaluation | Accuracy | Macro-F1 | Status |",
        "|---|---:|---:|---|",
        "| Random image split (current notebooks) | 81.42% | 0.82 | recorded |",
        "| Reproduced random split (seeded, no unseeded shuffle) | TBD | TBD | next |",
        "| Leakage-controlled / subject-proxy split | TBD | TBD | next |",
        "| External new-subject data | TBD | TBD | later |",
        "",
        "## Reproducibility config",
        "",
        "See `configs/baseline_experiment.json` created alongside this audit.",
        "",
        "## Generated artifacts",
        "",
        "- `results/audit/dataset_summary.csv`",
        "- `results/audit/dataset_summary.json`",
        "- `results/audit/class_distribution.png`",
        "- `results/audit/sample_grid.png`",
        "- `results/audit/image_dimensions.png`",
        "- `results/audit/brightness_contrast.png`",
        "- `results/audit/duplicate_hashes.csv`",
        "- `results/audit/near_duplicate_pairs.csv`",
        "- `results/audit/filename_analysis.json`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    OUT.mkdir(parents=True, exist_ok=True)

    print("Scanning dataset...")
    df, corrupted = collect_rows()
    df.to_csv(OUT / "dataset_summary.csv", index=False)

    exact_dupes = find_exact_duplicates(df)
    exact_dupes.to_csv(OUT / "duplicate_hashes.csv", index=False)

    print("Computing near-duplicate pairs (aHash proxy)...")
    near = find_near_duplicates(df, max_distance=5, max_pairs=5000)
    near.to_csv(OUT / "near_duplicate_pairs.csv", index=False)

    names = filename_analysis(df)
    with open(OUT / "filename_analysis.json", "w", encoding="utf-8") as handle:
        json.dump(names, handle, indent=2)

    usable = df[df["readable"]]
    size_counts = (
        usable.assign(size=usable["width"].astype(int).astype(str) + "x" + usable["height"].astype(int).astype(str))
        ["size"]
        .value_counts()
    )
    class_counts = df["label"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()

    summary = {
        "dataset": {
            "dataset_path": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
            "total_images": int(len(df)),
            "readable_images": int(df["readable"].sum()),
            "corrupted_images": int(len(corrupted)),
            "corrupted_examples": corrupted[:20],
            "class_counts": class_counts,
            "majority_class": max(class_counts, key=class_counts.get),
            "majority_baseline_accuracy": float(max(class_counts.values()) / max(len(df), 1)),
            "dominant_size": size_counts.index[0] if len(size_counts) else None,
            "num_distinct_sizes": int(len(size_counts)),
            "size_counts": size_counts.to_dict(),
            "dominant_channels": int(usable["channels"].mode().iloc[0]) if len(usable) else None,
            "brightness_min": float(usable["mean_brightness"].min()) if len(usable) else None,
            "brightness_max": float(usable["mean_brightness"].max()) if len(usable) else None,
            "brightness_mean": float(usable["mean_brightness"].mean()) if len(usable) else None,
            "contrast_min": float(usable["std_contrast"].min()) if len(usable) else None,
            "contrast_max": float(usable["std_contrast"].max()) if len(usable) else None,
            "contrast_mean": float(usable["std_contrast"].mean()) if len(usable) else None,
            "extensions": df["ext"].value_counts().to_dict(),
        },
        "leakage_proxy": {
            "exact_duplicate_groups": int(len(exact_dupes)),
            "exact_duplicate_extra_files": int(exact_dupes["count"].sum() - len(exact_dupes)) if len(exact_dupes) else 0,
            "identical_ahash_pairs": int(((near["hamming"] == 0).sum()) if len(near) else 0),
            "near_duplicate_pairs": int(len(near)),
            "near_duplicate_same_label_fraction": float(near["same_label"].mean()) if len(near) else 0.0,
            "near_duplicate_cross_label_pairs": int((~near["same_label"]).sum()) if len(near) else 0,
            "disclaimer": (
                "Near-duplicate grouping uses average hash similarity. "
                "It is not equivalent to verified subject IDs."
            ),
        },
        "filename_analysis": names,
        "project_facts": existing_project_facts(),
    }

    with open(OUT / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Writing plots...")
    plot_class_distribution(df, OUT / "class_distribution.png")
    plot_sample_grid(df, OUT / "sample_grid.png")
    plot_dimensions(df, OUT / "image_dimensions.png")
    plot_brightness_contrast(df, OUT / "brightness_contrast.png")
    write_audit_report(summary, OUT / "audit_report.md")

    print(json.dumps({
        "total_images": summary["dataset"]["total_images"],
        "class_counts": summary["dataset"]["class_counts"],
        "exact_duplicate_groups": summary["leakage_proxy"]["exact_duplicate_groups"],
        "near_duplicate_pairs": summary["leakage_proxy"]["near_duplicate_pairs"],
        "majority_baseline": round(summary["dataset"]["majority_baseline_accuracy"], 4),
        "audit_report": str((OUT / "audit_report.md").relative_to(ROOT)).replace("\\", "/"),
    }, indent=2))


if __name__ == "__main__":
    main()
