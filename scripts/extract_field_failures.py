#!/usr/bin/env python3
"""Extract banana/maize field audit failures into a field-failure training set.

This helper selects banana/maize samples from analysis_outputs/field_audit_rows.tsv
that were rejected by the runtime gate and copies them into
datasets/model_ready/field_failures/<crop>/ for retraining.

Usage:
    python -u scripts/extract_field_failures.py
    python -u scripts/extract_field_failures.py --input analysis_outputs/field_audit_rows.tsv \
        --output datasets/model_ready/field_failures --copy

The copied dataset can then be consumed by scripts/train_model.py if the
field_failures directory exists.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "analysis_outputs" / "field_audit_rows.tsv"
DEFAULT_OUTPUT = ROOT / "datasets" / "model_ready" / "field_failures"

AUDIT_FAILURE_TYPES = {
    "G5_other_leaf_winner",
    "G5b_other_leaf_floor",
    "G7a_crop_total",
    "G7a_other_leaf_ratio",
    "G7c_crop_gap",
    "G7c_second_crop",
    "G7c_entropy",
    "G7c_class_ratio",
    "G7d_class_confidence",
    "G7e_healthy_safety",
    "unsupported",
    "uncertain",
}
BANANA_MAIZE_CROPS = {"banana", "maize"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract banana/maize field audit failures for retraining."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to analysis_outputs/field_audit_rows.tsv",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Destination root for field failure images.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy matching image files into the output directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copying images that already exist in the destination.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return (ROOT / p).resolve()
    return p


def main() -> int:
    args = parse_args()
    input_path = normalize_path(args.input)
    output_root = normalize_path(args.output)

    if not input_path.exists():
        print(f"ERROR: input TSV not found: {input_path}", file=sys.stderr)
        return 1

    selected = []
    duplicate_skips = 0
    missing_images = 0
    copied = 0

    with input_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            crop = row.get('expected_crop', '').strip()
            gate = row.get('gate', '').strip()
            file_path = row.get('file', '').strip()
            if not crop or not gate or not file_path:
                continue
            if crop not in BANANA_MAIZE_CROPS:
                continue
            if gate not in AUDIT_FAILURE_TYPES:
                continue
            src_path = normalize_path(file_path)
            selected.append((crop, gate, src_path))

    if not selected:
        print("No banana/maize field failures matched.", flush=True)
        return 0

    print(f"Found {len(selected)} banana/maize field-failure rows.", flush=True)
    if args.copy:
        for crop, gate, src_path in selected:
            if not src_path.exists():
                print(f"WARNING: missing image file: {src_path}", flush=True)
                missing_images += 1
                continue
            dest_dir = output_root / crop
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / src_path.name
            if dest_file.exists():
                duplicate_skips += 1
                if args.skip_existing:
                    continue
            shutil.copy2(src_path, dest_file)
            copied += 1

    print(f"Output root: {output_root}", flush=True)
    print(f"Banana/Maize failures selected: {len(selected)}", flush=True)
    if args.copy:
        print(f"Images copied: {copied}", flush=True)
        if duplicate_skips:
            print(f"Skipped existing files: {duplicate_skips}", flush=True)
        if missing_images:
            print(f"Missing source images: {missing_images}", flush=True)
    else:
        print("Use --copy to actually copy the selected failure images.", flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
