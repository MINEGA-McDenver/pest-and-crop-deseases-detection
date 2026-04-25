"""
Curate other_leaf images without deleting useful hard negatives.

This script audits other_leaf images and assigns an action label:
- REVIEW_REMOVE_SCREENSHOT
- REVIEW_REMOVE_DUPLICATE
- REVIEW_REMOVE_LOW_QUALITY
- KEEP_HARD_NEGATIVE
- KEEP_GENERAL_OTHER_LEAF

By default, it only writes reports. Use --apply-remove to move review-remove
images into a quarantine folder.

Usage examples:
  python -u scripts/curate_other_leaf.py
  python -u scripts/curate_other_leaf.py --split train --apply-remove
  python -u scripts/curate_other_leaf.py --split train --min-focus-total 0.30
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "model_ready"
MODEL_PATH = ROOT / "models" / "best_model.keras"
LABELS_PATH = ROOT / "models" / "labels.txt"
OUT_DIR = ROOT / "analysis_outputs"

IMG_SIZE = (224, 224)
DEFAULT_SPLITS = ("train", "val", "test")


@dataclass
class ImageAuditRow:
    split: str
    file: str
    action: str
    reason: str
    width: int
    height: int
    brightness_std: float
    is_screenshot_like: bool
    is_duplicate: bool
    duplicate_of: str
    top1_label: str
    top1_prob: float
    beans_total: float
    potato_total: float
    other_leaf_prob: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Audit one split or all splits.",
    )
    parser.add_argument(
        "--min-focus-total",
        type=float,
        default=0.35,
        help="If beans_total or potato_total reaches this, keep as hard negative.",
    )
    parser.add_argument(
        "--min-top-focus",
        type=float,
        default=0.20,
        help="If top1 is beans_* or potato_* above this, keep as hard negative.",
    )
    parser.add_argument(
        "--low-detail-std",
        type=float,
        default=12.0,
        help="Brightness std-dev below this is flagged as low detail.",
    )
    parser.add_argument(
        "--tiny-size",
        type=int,
        default=150,
        help="Width or height below this is flagged as tiny.",
    )
    parser.add_argument(
        "--apply-remove",
        action="store_true",
        help="Move REVIEW_REMOVE_* images into datasets/quarantine_other_leaf/.",
    )
    return parser.parse_args()


def choose_splits(split_arg: str) -> List[str]:
    if split_arg == "all":
        return list(DEFAULT_SPLITS)
    return [split_arg]


def load_labels() -> List[str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"labels.txt not found: {LABELS_PATH}")
    labels = [x.strip() for x in LABELS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    if "other_leaf" not in labels:
        raise RuntimeError("labels.txt missing other_leaf")
    return labels


def build_crop_grouping(labels: List[str]) -> Dict[str, str]:
    grouping = {}
    for label in labels:
        if label == "other_leaf":
            continue
        grouping[label] = label.split("_")[0]
    return grouping


def list_other_leaf_images(split: str) -> List[Path]:
    folder = DATASET_ROOT / split / "other_leaf"
    if not folder.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def image_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def screenshot_like(name: str) -> bool:
    n = name.lower()
    return (
        "screenshot" in n
        or n.startswith("screen")
        or "whatsapp" in n
        or "telegram" in n
        or n.endswith(".heic")
    )


def image_quality(path: Path) -> Tuple[int, int, float]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]
        gray = (0.299 * arr[..., 0]) + (0.587 * arr[..., 1]) + (0.114 * arr[..., 2])
        std = float(np.std(gray))
    return w, h, std


def preprocess_for_model(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def classify_image(
    model: tf.keras.Model,
    labels: List[str],
    crop_grouping: Dict[str, str],
    path: Path,
) -> Tuple[str, float, float, float, float]:
    probs = model.predict(preprocess_for_model(path), verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top1 = labels[top_idx]
    top1_prob = float(probs[top_idx])

    beans_total = 0.0
    potato_total = 0.0
    other_leaf_prob = 0.0

    for i, label in enumerate(labels):
        p = float(probs[i])
        if label == "other_leaf":
            other_leaf_prob = p
            continue
        crop = crop_grouping.get(label)
        if crop == "beans":
            beans_total += p
        elif crop == "potato":
            potato_total += p

    return top1, top1_prob, beans_total, potato_total, other_leaf_prob


def decide_action(
    is_dup: bool,
    dup_of: str,
    is_screenshot: bool,
    width: int,
    height: int,
    std: float,
    top1_label: str,
    top1_prob: float,
    beans_total: float,
    potato_total: float,
    min_focus_total: float,
    min_top_focus: float,
    low_detail_std: float,
    tiny_size: int,
) -> Tuple[str, str]:
    if is_screenshot:
        return "REVIEW_REMOVE_SCREENSHOT", "filename pattern indicates screenshot/social export"

    if is_dup:
        return "REVIEW_REMOVE_DUPLICATE", f"exact duplicate of {dup_of}"

    if width < tiny_size or height < tiny_size or std < low_detail_std:
        return "REVIEW_REMOVE_LOW_QUALITY", "tiny resolution or very low detail"

    looks_focus = (
        beans_total >= min_focus_total
        or potato_total >= min_focus_total
        or (top1_label.startswith("beans_") and top1_prob >= min_top_focus)
        or (top1_label.startswith("potato_") and top1_prob >= min_top_focus)
    )
    if looks_focus:
        return "KEEP_HARD_NEGATIVE", "looks like beans/potato boundary example"

    return "KEEP_GENERAL_OTHER_LEAF", "valid negative sample"


def maybe_move_to_quarantine(rows: List[ImageAuditRow]) -> Dict[str, int]:
    qroot = ROOT / "datasets" / "quarantine_other_leaf"
    counts = {"moved": 0, "skipped_missing": 0}
    for row in rows:
        if not row.action.startswith("REVIEW_REMOVE_"):
            continue
        src = Path(row.file)
        if not src.exists():
            counts["skipped_missing"] += 1
            continue
        rel = src.relative_to(DATASET_ROOT)
        dst = qroot / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        counts["moved"] += 1
    return counts


def write_reports(rows: List[ImageAuditRow], split_name: str) -> Tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"other_leaf_curation_{split_name}.json"
    tsv_path = OUT_DIR / f"other_leaf_curation_{split_name}.tsv"

    payload = {
        "split": split_name,
        "total": len(rows),
        "action_counts": {},
        "rows": [r.__dict__ for r in rows],
    }
    for r in rows:
        payload["action_counts"][r.action] = payload["action_counts"].get(r.action, 0) + 1

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    columns = [
        "split",
        "file",
        "action",
        "reason",
        "width",
        "height",
        "brightness_std",
        "is_screenshot_like",
        "is_duplicate",
        "duplicate_of",
        "top1_label",
        "top1_prob",
        "beans_total",
        "potato_total",
        "other_leaf_prob",
    ]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join(str(getattr(r, c)) for c in columns) + "\n")

    return json_path, tsv_path


def main() -> None:
    args = parse_args()
    splits = choose_splits(args.split)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"best_model.keras not found: {MODEL_PATH}")

    labels = load_labels()
    crop_grouping = build_crop_grouping(labels)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    rows: List[ImageAuditRow] = []
    seen_hash: Dict[str, str] = {}

    for split in splits:
        files = list_other_leaf_images(split)
        for path in files:
            fhash = image_hash(path)
            dup_of = seen_hash.get(fhash, "")
            is_dup = bool(dup_of)
            if not is_dup:
                seen_hash[fhash] = str(path)

            w, h, std = image_quality(path)
            top1, top1_prob, beans_total, potato_total, other_leaf_prob = classify_image(
                model=model,
                labels=labels,
                crop_grouping=crop_grouping,
                path=path,
            )
            is_ss = screenshot_like(path.name)

            action, reason = decide_action(
                is_dup=is_dup,
                dup_of=dup_of,
                is_screenshot=is_ss,
                width=w,
                height=h,
                std=std,
                top1_label=top1,
                top1_prob=top1_prob,
                beans_total=beans_total,
                potato_total=potato_total,
                min_focus_total=args.min_focus_total,
                min_top_focus=args.min_top_focus,
                low_detail_std=args.low_detail_std,
                tiny_size=args.tiny_size,
            )

            rows.append(
                ImageAuditRow(
                    split=split,
                    file=str(path),
                    action=action,
                    reason=reason,
                    width=w,
                    height=h,
                    brightness_std=round(std, 4),
                    is_screenshot_like=is_ss,
                    is_duplicate=is_dup,
                    duplicate_of=dup_of,
                    top1_label=top1,
                    top1_prob=round(top1_prob, 6),
                    beans_total=round(beans_total, 6),
                    potato_total=round(potato_total, 6),
                    other_leaf_prob=round(other_leaf_prob, 6),
                )
            )

    split_name = "all" if len(splits) > 1 else splits[0]
    json_path, tsv_path = write_reports(rows, split_name)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {tsv_path}")

    action_counts: Dict[str, int] = {}
    for r in rows:
        action_counts[r.action] = action_counts.get(r.action, 0) + 1

    print("Action counts:")
    for action in sorted(action_counts):
        print(f"  {action}: {action_counts[action]}")

    if args.apply_remove:
        move_counts = maybe_move_to_quarantine(rows)
        print("Quarantine move results:")
        print(f"  moved: {move_counts['moved']}")
        print(f"  skipped_missing: {move_counts['skipped_missing']}")


if __name__ == "__main__":
    main()
