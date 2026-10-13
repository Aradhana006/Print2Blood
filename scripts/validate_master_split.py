"""Validate the leakage-controlled master split. Exits non-zero on critical leakage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_config, project_root
from src.validation.split_checks import validate_master_split


def write_report(result: dict, path: Path):
    lines = [
        "# Master Split Validation Report",
        "",
        f"**Overall status: {result['overall']}**",
        "",
        "## Counts",
        "",
    ]
    counts = result.get("counts", {})
    if counts:
        lines.extend(
            [
                f"- Total images: {counts.get('total_images')}",
                f"- Total groups: {counts.get('total_groups')}",
                f"- Train: {counts.get('train')}",
                f"- Validation: {counts.get('validation')}",
                f"- Test: {counts.get('test')}",
                "",
            ]
        )
    lines.extend(["## Checks", ""])
    for check in result.get("checks", []):
        mark = "PASS" if check["status"] == "PASS" else check["status"]
        lines.append(f"- **{check['name']}**: {mark} — {check['detail']}")
    lines.extend(
        [
            "",
            "## Research note",
            "",
            "This is a leakage-controlled **proxy** split. Similarity groups are not confirmed subjects.",
            "",
            "Fingerprint-to-blood-group prediction is an experimental research task. "
            "This system is not a clinical diagnostic tool and must not replace laboratory blood-group testing.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    root = project_root()
    result = validate_master_split(classes=cfg["classes"])

    print("=" * 48)
    print("MASTER SPLIT VALIDATION")
    print("=" * 48)
    counts = result.get("counts", {})
    if counts:
        print(f"Images                 : {counts.get('total_images')}")
        print(f"Train                  : {counts.get('train')}")
        print(f"Validation             : {counts.get('validation')}")
        print(f"Test                   : {counts.get('test')}")
    print(f"Exact duplicate leakage: {result.get('hash_leak')}")
    print(f"Group leakage          : {result.get('group_leak')}")
    overlaps = result.get("overlaps", {})
    print(f"Train/Test overlap     : {overlaps.get('train_test_overlap')}")
    print(f"Train/Validation overlap: {overlaps.get('train_validation_overlap')}")
    print(f"Validation/Test overlap: {overlaps.get('validation_test_overlap')}")
    missing = result.get("missing_per_split", {})
    missing_count = sum(len(v) for v in missing.values()) if missing else 0
    print(f"Missing classes        : {missing_count}")
    print()
    print(f"STATUS: {result['overall']}")
    print("=" * 48)

    report_path = root / cfg["paths"]["validation_report"]
    write_report(result, report_path)
    print(f"Report: {report_path}")

    # Machine-readable sidecar
    with open(root / "results" / "splits" / "split_validation.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    if result["critical_fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
