"""
Fingerprint Blood Group Prediction
Research Dataset & Leakage Audit Dashboard

Read-only. Does not train models or regenerate splits automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Path is used by explorer/duplicate pages for filenames.

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.io import load_csv, load_json, load_thumbnail
from src.validation.split_checks import validate_master_split

st.set_page_config(
    page_title="Fingerprint Blood Group Research Audit",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
h1, h2, h3 {font-family: "IBM Plex Sans", "Segoe UI", sans-serif;}
.metric-card {
  background: linear-gradient(180deg, #f7f4ef 0%, #efe8dc 100%);
  border: 1px solid #d9d0c3;
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.6rem;
}
.metric-label {color:#5b5248; font-size:0.85rem; letter-spacing:0.02em;}
.metric-value {color:#1f2a30; font-size:1.6rem; font-weight:700; margin-top:0.15rem;}
.pass {color:#1f6b3a; font-weight:700;}
.fail {color:#9b2c2c; font-weight:700;}
.warn-box {
  background:#fff6e8; border-left:4px solid #c27a3a; padding:0.8rem 1rem; margin:0.8rem 0;
}
.hero {
  background: radial-gradient(circle at top left, #e7f0f4, #f7f4ef 55%, #efe6d8);
  border: 1px solid #d5cdc0;
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(label: str, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def require_master():
    master = load_csv("data/splits/master_split.csv")
    if master is None:
        st.error("Master split has not been generated yet. Run `python scripts/build_master_split.py`.")
        return None
    return master


def page_overview():
    st.markdown(
        '<div class="hero"><h1>Fingerprint Blood Group Prediction</h1>'
        "<h3>Research Dataset & Leakage Audit</h3>"
        "<p>Leakage-Controlled Dataset Analysis and Master Split</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="warn-box">Fingerprint-to-blood-group prediction is an experimental research task. '
        "This system is not a clinical diagnostic tool and must not replace laboratory blood-group testing.</div>",
        unsafe_allow_html=True,
    )

    summary = load_json("results/splits/split_summary.json")
    validation = load_json("results/splits/split_validation.json")
    master = require_master()
    if master is None or summary is None:
        return

    status = validation.get("overall", "UNKNOWN") if validation else "UNKNOWN"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Images", f"{summary['total_images']:,}")
        metric_card("Train Images", f"{summary['train_count']:,}")
    with c2:
        metric_card("Classes", "8")
        metric_card("Validation Images", f"{summary['validation_count']:,}")
    with c3:
        metric_card("Proxy Groups", f"{summary['total_groups']:,}")
        metric_card("Test Images", f"{summary['test_count']:,}")
    with c4:
        exact = summary.get("group_meta", {}).get("exact_duplicate_groups", "—")
        metric_card("Exact Duplicate Groups", exact)
        color = "pass" if status == "PASS" else "fail"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Leakage Status</div>'
            f'<div class="metric-value {color}">{status}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Dataset Distribution")
        class_counts = pd.Series(summary["class_counts"])
        st.bar_chart(class_counts, color="#2F6F8F")
    with right:
        st.subheader("Split Distribution")
        split_counts = pd.Series(
            {
                "train": summary["train_count"],
                "validation": summary["validation_count"],
                "test": summary["test_count"],
            }
        )
        st.bar_chart(split_counts, color="#6B8F71")

    st.subheader("Methodology")
    st.markdown(
        """
1. Exact MD5 duplicates are forced into the same proxy group.
2. Same-label highly similar pairs (pHash + SSIM) may merge into proxy groups.
3. Whole groups are assigned to train / validation / test (70% / 15% / 15% targets).
4. Because participant IDs are unavailable, groups are **proxy leakage-control units**, not confirmed subjects.
"""
    )


def page_explorer():
    st.header("Dataset Explorer")
    master = require_master()
    audit = load_csv("results/audit/dataset_summary.csv")
    if master is None:
        return

    merged = master.copy()
    if audit is not None:
        keep = ["filepath", "width", "height", "mean_brightness", "std_contrast", "mode", "channels"]
        # audit uses filepath; master uses image_path
        audit2 = audit.rename(columns={"filepath": "image_path"})
        cols = [c for c in keep if c in audit2.columns or c == "image_path"]
        # rebuild keep properly
        audit2 = audit2[
            [c for c in ["image_path", "width", "height", "mean_brightness", "std_contrast", "mode", "channels"] if c in audit2.columns]
        ]
        merged = merged.merge(audit2, on="image_path", how="left")

    f1, f2, f3 = st.columns(3)
    labels = ["All"] + sorted(merged["label"].unique().tolist())
    splits = ["All"] + ["train", "validation", "test"]
    groups = ["All"] + sorted(merged["group_id"].unique().tolist())[:5000]
    with f1:
        label = st.selectbox("Blood Group", labels)
    with f2:
        split = st.selectbox("Split", splits)
    with f3:
        group = st.selectbox("Group", groups)

    view = merged
    if label != "All":
        view = view[view["label"] == label]
    if split != "All":
        view = view[view["split"] == split]
    if group != "All":
        view = view[view["group_id"] == group]

    st.caption(f"Showing {min(len(view), 48)} of {len(view)} matching images")
    sample = view.head(48)
    cols = st.columns(6)
    selected = st.session_state.get("selected_image")
    for i, row in enumerate(sample.itertuples(index=False)):
        with cols[i % 6]:
            thumb = load_thumbnail(row.image_path, (140, 140))
            if thumb is not None:
                st.image(thumb, use_container_width=True)
            if st.button(f"{row.label} | {row.split}", key=f"img_{i}"):
                st.session_state["selected_image"] = row.image_path
                selected = row.image_path
            st.caption(Path(row.image_path).name)

    if selected:
        st.divider()
        st.subheader("Selected image")
        detail = merged[merged["image_path"] == selected].iloc[0]
        c1, c2 = st.columns([1, 1.2])
        with c1:
            big = load_thumbnail(selected, (360, 360))
            if big is not None:
                st.image(big, use_container_width=True)
        with c2:
            st.write(
                {
                    "image_path": detail["image_path"],
                    "label": detail["label"],
                    "group_id": detail["group_id"],
                    "split": detail["split"],
                    "width": detail.get("width"),
                    "height": detail.get("height"),
                    "brightness": detail.get("mean_brightness"),
                    "contrast": detail.get("std_contrast"),
                    "hash": detail.get("md5"),
                }
            )


def page_duplicates():
    st.header("Duplicate Analysis")
    exact = load_csv("results/splits/exact_duplicate_groups.csv")
    master = require_master()
    if exact is None or master is None:
        st.warning("Exact duplicate results not found. Run `python scripts/build_master_split.py`.")
        return

    n_groups = exact["group_id"].nunique() if len(exact) else 0
    n_files = len(exact)
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Exact Duplicate Groups", n_groups)
    with c2:
        metric_card("Files Involved", n_files)

    if exact.empty:
        st.success("No exact duplicate groups detected.")
        return

    rows = []
    for gid, g in exact.groupby("group_id"):
        splits = master.loc[master["group_id"] == gid, "split"].unique().tolist()
        rows.append(
            {
                "group_id": gid,
                "n_images": len(g),
                "labels": ", ".join(sorted(g["label"].unique())),
                "splits": ", ".join(sorted(splits)),
                "contained_in_one_split": len(splits) == 1,
            }
        )
    table = pd.DataFrame(rows).sort_values("n_images", ascending=False)
    st.dataframe(table, use_container_width=True)

    st.subheader("Example duplicate pairs")
    shown = 0
    for gid, g in exact.groupby("group_id"):
        paths = g["image_path"].tolist()
        if len(paths) < 2:
            continue
        st.markdown(f"**{gid}** — MD5 same — labels: {', '.join(sorted(g['label'].unique()))}")
        cols = st.columns(min(len(paths), 4))
        for i, path in enumerate(paths[:4]):
            with cols[i]:
                thumb = load_thumbnail(path, (180, 180))
                if thumb is not None:
                    st.image(thumb, caption=Path(path).name, use_container_width=True)
        shown += 1
        if shown >= 6:
            break


def page_similarity():
    st.header("Similarity Analysis")
    pairs = load_csv("results/splits/similarity_analysis.csv")
    summary = load_json("results/splits/split_summary.json")
    if pairs is None:
        st.warning("Similarity analysis not found. Run `python scripts/build_master_split.py`.")
        return

    methods = ["All"] + sorted(pairs["method"].dropna().unique().tolist())
    method = st.selectbox("Similarity method", methods, index=methods.index("phash_ssim") if "phash_ssim" in methods else 0)
    view = pairs if method == "All" else pairs[pairs["method"] == method]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Candidate pairs", f"{len(view):,}")
    with c2:
        same = float(view["same_label"].mean()) if len(view) else 0.0
        metric_card("Same-label %", f"{100*same:.1f}%")
    with c3:
        metric_card("Different-label %", f"{100*(1-same):.1f}%")
    with c4:
        thr = summary.get("threshold", {}) if summary else {}
        metric_card("Selected method", summary.get("similarity_method", "phash") if summary else "phash")

    st.caption(f"Thresholds: {thr}")
    if "similarity_score" in view.columns and len(view):
        st.subheader("Similarity score distribution")
        st.bar_chart(view["similarity_score"].round(2).value_counts().sort_index())

    st.subheader("Highly similar examples")
    examples_dir = ROOT / "results" / "splits" / "similarity_examples"
    if examples_dir.exists():
        files = sorted(examples_dir.glob("similar_*.png"))[:12]
        cols = st.columns(3)
        for i, path in enumerate(files):
            with cols[i % 3]:
                st.image(str(path), use_container_width=True)
    else:
        top = view.sort_values("similarity_score", ascending=False).head(6)
        for row in top.itertuples(index=False):
            c1, c2 = st.columns(2)
            with c1:
                a = load_thumbnail(row.image_a, (220, 220))
                if a is not None:
                    st.image(a, caption=f"{row.label_a}: {Path(row.image_a).name}")
            with c2:
                b = load_thumbnail(row.image_b, (220, 220))
                if b is not None:
                    st.image(b, caption=f"{row.label_b}: {Path(row.image_b).name}")
            st.caption(f"{row.method} score={row.similarity_score:.3f} same_label={row.same_label}")


def page_master_split():
    st.header("Master Split")
    master = require_master()
    summary = load_json("results/splits/split_summary.json")
    groups = load_csv("data/splits/split_groups.csv")
    if master is None or summary is None:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("TRAIN", f"{summary['train_count']:,} ({100*summary['train_pct']:.1f}%)")
    with c2:
        metric_card("VALIDATION", f"{summary['validation_count']:,} ({100*summary['validation_pct']:.1f}%)")
    with c3:
        metric_card("TEST", f"{summary['test_count']:,} ({100*summary['test_pct']:.1f}%)")

    table = (
        master.groupby(["label", "split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "validation", "test"], fill_value=0)
    )
    table["Total"] = table.sum(axis=1)
    st.subheader("Blood Group | Train | Validation | Test | Total")
    st.dataframe(table, use_container_width=True)

    st.subheader("Per-class stacked distribution")
    st.bar_chart(table[["train", "validation", "test"]])

    if groups is not None:
        st.subheader("Groups per split")
        st.bar_chart(groups["split"].value_counts())


def page_leakage():
    st.header("Leakage Validation")
    st.markdown(
        '<div class="warn-box">This page validates the leakage-controlled proxy split. '
        "Proxy groups are not confirmed human subjects.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Run Split Validation", type="primary"):
        with st.spinner("Validating master split..."):
            result = validate_master_split()
            st.session_state["validation_result"] = result

    result = st.session_state.get("validation_result") or load_json("results/splits/split_validation.json")
    if result is None:
        st.info("No validation result yet. Click **Run Split Validation** or run `python scripts/validate_master_split.py`.")
        return

    overall = result.get("overall", "UNKNOWN")
    color = "pass" if overall == "PASS" else "fail"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">OVERALL STATUS</div>'
        f'<div class="metric-value {color}">{overall}</div></div>',
        unsafe_allow_html=True,
    )

    for check in result.get("checks", []):
        ok = check["status"] == "PASS"
        icon = "PASS" if ok else check["status"]
        st.write(f"- **{check['name']}**: `{icon}` — {check['detail']}")

    if overall != "PASS":
        st.error("LEAKAGE DETECTED OR CRITICAL VALIDATION FAILURE. Do not train models until this is resolved.")


def page_notes():
    st.header("Research Notes")
    st.markdown(
        """
### Why random image-level splitting can be problematic
Multiple fingerprints from the same acquisition context can be highly correlated. A random 80/20 image split may place near-identical or duplicate images into both training and test sets, inflating apparent accuracy.

### Why participant IDs are unavailable
This dataset filenames use `cluster_<class_code>_<image_id>.BMP`. The cluster code maps to blood-group folder labels. Numeric IDs are global sample indices (0–5999), not verified person IDs.

### Why this is called a proxy split
Because true subject identifiers are unavailable, we build **leakage-control proxy groups** from exact duplicates and conservative same-label similarity. These groups attempt to keep obviously related images inside one split. They **cannot** be interpreted as confirmed human subjects.

### Why similarity cannot prove identity
Perceptual hashes and SSIM measure visual similarity, not biometrics identity. Two similar images may share scanner/cropping artifacts without belonging to one person.

### Why the test set must remain untouched
After the master split is fixed, model selection and tuning should use train/validation only. The test split is reserved for final reporting.

### Clinical warning
Fingerprint-to-blood-group prediction is an experimental research task. This system is not a clinical diagnostic tool and must not replace laboratory blood-group testing.
"""
    )
    report = ROOT / "results" / "splits" / "split_validation_report.md"
    if report.exists():
        st.subheader("Validation report")
        st.markdown(report.read_text(encoding="utf-8"))


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Dataset Explorer",
            "Duplicate Analysis",
            "Similarity Analysis",
            "Master Split",
            "Leakage Validation",
            "Research Notes",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Read-only research dashboard")
    st.sidebar.caption("Does not train models")
    st.sidebar.caption("Does not auto-regenerate splits")

    pages = {
        "Overview": page_overview,
        "Dataset Explorer": page_explorer,
        "Duplicate Analysis": page_duplicates,
        "Similarity Analysis": page_similarity,
        "Master Split": page_master_split,
        "Leakage Validation": page_leakage,
        "Research Notes": page_notes,
    }
    pages[page]()


if __name__ == "__main__":
    main()
