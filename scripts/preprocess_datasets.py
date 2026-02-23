"""
Data Cleaning & Preprocessing Script
======================================
Processes the organized dataset into model-ready train/val/test splits
with image resizing, deduplication, adaptive augmentation, and reporting.

Input:  datasets/processed/{crop}/{class}/  (from organize_datasets.py)
Output: datasets/model_ready/
            ├── train/{class}/
            ├── val/{class}/
            ├── test/{class}/
            ├── labels.txt
            ├── class_index.json
            └── preprocessing_report.json

Usage:
    python scripts/preprocess_datasets.py
"""

import os
import json
import hashlib
import random
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"d:\ALL MY DOCUMENTS\YEAR 4\FINAL YEAR PROJECT\pest-and-crop-deseases-detection")
INPUT_DIR = BASE_DIR / "datasets" / "processed"
OUTPUT_DIR = BASE_DIR / "datasets" / "model_ready"

# Image settings
IMG_SIZE = (224, 224)          # Target size for model input
INTERPOLATION = Image.LANCZOS  # High-quality downsampling

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Augmentation settings (adaptive)
# Classes below this count will be augmented
AUGMENT_THRESHOLD = 1000
# Adaptive target: augment small classes to 30% of the largest class
# Capped to prevent excessive synthetic data
MAX_AUGMENT_TARGET = 3000
# Maximum augmentations per single original image
# Prevents over-reliance on one source image
MAX_AUG_PER_IMAGE = 8

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Crops and classes (must match organize_datasets.py output)
ALL_CLASSES = {
    "cassava": [
        "cassava_bacterial_blight",
        "cassava_brown_streak_disease",
        "cassava_green_mottle",
        "cassava_mosaic_disease",
        "cassava_healthy",
    ],
    "maize": [
        "maize_gray_leaf_spot",
        "maize_common_rust",
        "maize_northern_leaf_blight",
        "maize_healthy",
    ],
    "potato": [
        "potato_early_blight",
        "potato_late_blight",
        "potato_healthy",
    ],
    "beans": [
        "beans_angular_leaf_spot",
        "beans_rust",
        "beans_healthy",
    ],
    "banana": [
        "banana_cordana",
        "banana_pestalotiopsis",
        "banana_sigatoka",
        "banana_healthy",
    ],
}


# ============================================================
# STEP 1: DUPLICATE DETECTION & REMOVAL
# ============================================================

def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_and_remove_duplicates() -> dict:
    """
    Scan all images in the processed directory and remove duplicates.
    Keeps the first occurrence, removes subsequent copies.
    Returns a report of findings.
    """
    print("\n" + "=" * 60)
    print("[STEP 1] Detecting and removing duplicate images...")
    print("=" * 60)

    hash_map = {}
    duplicates = []
    total_scanned = 0
    total_size_saved = 0

    for crop in ALL_CLASSES:
        for cls in ALL_CLASSES[crop]:
            cls_dir = INPUT_DIR / crop / cls
            if not cls_dir.exists():
                continue

            files = sorted([f for f in cls_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])

            for filepath in files:
                total_scanned += 1
                file_hash = compute_file_hash(filepath)

                if file_hash in hash_map:
                    file_size = filepath.stat().st_size
                    total_size_saved += file_size
                    duplicates.append({
                        "removed": str(filepath.name),
                        "class": cls,
                        "duplicate_of": str(hash_map[file_hash].name),
                    })
                    filepath.unlink()
                else:
                    hash_map[file_hash] = filepath

        if total_scanned % 5000 == 0 and total_scanned > 0:
            print(f"  Scanned {total_scanned} images...")

    print(f"\n  Total images scanned:    {total_scanned:,}")
    print(f"  Duplicates found:        {len(duplicates):,}")
    print(f"  Storage saved:           {total_size_saved / (1024*1024):.1f} MB")
    print(f"  Unique images remaining: {total_scanned - len(duplicates):,}")

    if duplicates:
        dup_by_class = defaultdict(int)
        for d in duplicates:
            dup_by_class[d["class"]] += 1
        print(f"\n  Duplicates by class:")
        for cls, count in sorted(dup_by_class.items(), key=lambda x: -x[1]):
            print(f"    {cls:<40} {count:>4} removed")

    return {
        "total_scanned": total_scanned,
        "duplicates_removed": len(duplicates),
        "storage_saved_mb": round(total_size_saved / (1024 * 1024), 2),
        "duplicate_details": duplicates[:50],
    }


# ============================================================
# STEP 2: TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset() -> dict:
    """
    Split each class into train/val/test sets with stratified splitting.
    Copies files into the model_ready directory structure.
    Returns split statistics.
    """
    print("\n" + "=" * 60)
    print("[STEP 2] Splitting dataset into train/val/test...")
    print(f"         Ratios: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print("=" * 60)

    split_stats = {}

    for crop in ALL_CLASSES:
        for cls in ALL_CLASSES[crop]:
            src_dir = INPUT_DIR / crop / cls
            if not src_dir.exists():
                print(f"  WARNING: Missing {src_dir}")
                continue

            images = sorted([f for f in src_dir.iterdir()
                           if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])

            # Shuffle deterministically
            random.shuffle(images)

            total = len(images)
            train_end = int(total * TRAIN_RATIO)
            val_end = train_end + int(total * VAL_RATIO)

            splits = {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:],
            }

            for split_name, split_files in splits.items():
                dst_dir = OUTPUT_DIR / split_name / cls
                dst_dir.mkdir(parents=True, exist_ok=True)

                for filepath in split_files:
                    dst_path = dst_dir / filepath.name
                    shutil.copy2(filepath, dst_path)

            split_stats[cls] = {
                "total": total,
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
            }

            print(f"  {cls:<40} Total:{total:>6}  "
                  f"Train:{len(splits['train']):>5}  "
                  f"Val:{len(splits['val']):>4}  "
                  f"Test:{len(splits['test']):>4}")

    total_train = sum(s["train"] for s in split_stats.values())
    total_val = sum(s["val"] for s in split_stats.values())
    total_test = sum(s["test"] for s in split_stats.values())
    total_all = total_train + total_val + total_test

    print(f"\n  {'SPLIT TOTALS':<40} Total:{total_all:>6}  "
          f"Train:{total_train:>5}  Val:{total_val:>4}  Test:{total_test:>4}")

    return split_stats


# ============================================================
# STEP 3: RESIZE AND NORMALIZE
# ============================================================

def resize_image(img: Image.Image) -> Image.Image:
    """Resize image to target size using high-quality resampling."""
    return img.resize(IMG_SIZE, INTERPOLATION)


def normalize_and_save(img: Image.Image, save_path: Path):
    """
    Save the resized image as JPG in RGB format.
    
    Note on normalization:
    Pixel values are saved as standard uint8 (0-255).
    Runtime normalization to [0, 1] or [-1, 1] is applied during training
    via the data loading pipeline. This preserves image quality on disk
    and allows flexible normalization schemes per model architecture.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(save_path, "JPEG", quality=95)


def resize_all_images() -> dict:
    """
    Resize all images in train/val/test to the target size (224x224).
    Converts all images to RGB JPG format.
    """
    print("\n" + "=" * 60)
    print(f"[STEP 3] Resizing all images to {IMG_SIZE[0]}x{IMG_SIZE[1]}...")
    print("=" * 60)

    stats = {"resized": 0, "errors": 0, "error_files": []}

    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split
        if not split_dir.exists():
            continue

        split_count = 0
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for cls_dir in class_dirs:
            files = [f for f in cls_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]

            for filepath in files:
                try:
                    with Image.open(filepath) as img:
                        img = resize_image(img)
                        new_path = filepath.with_suffix(".jpg")
                        normalize_and_save(img, new_path)
                        # Remove original if extension changed
                        if new_path != filepath and filepath.exists():
                            filepath.unlink()
                        stats["resized"] += 1
                        split_count += 1
                except Exception as e:
                    stats["errors"] += 1
                    stats["error_files"].append(str(filepath))
                    print(f"  ERROR processing {filepath.name}: {e}")

        print(f"  {split:<10} {split_count:>6} images resized")

    print(f"\n  Total resized: {stats['resized']:,}")
    if stats["errors"]:
        print(f"  Errors: {stats['errors']} (see preprocessing_report.json)")

    return stats


# ============================================================
# STEP 4: DATA AUGMENTATION (Training set only, adaptive)
# ============================================================

def augment_image(img: Image.Image, aug_index: int) -> Image.Image:
    """
    Apply a combination of augmentation transforms to an image.
    Different aug_index values produce different augmentation combinations.

    Augmentations applied (biologically realistic for leaf images):
    - Random rotation (-30 to +30 degrees)
    - Random horizontal flip (50% chance)
    - Random vertical flip (10% chance — low, as leaves have natural orientation)
    - Random brightness adjustment (0.7 to 1.3)
    - Random contrast adjustment (0.7 to 1.3)
    - Random saturation adjustment (0.8 to 1.2)
    - Random zoom/crop (85% to 100% of image)
    - Random light Gaussian blur (20% chance)
    """
    rng = random.Random(aug_index * 7 + 13)

    # 1. Random rotation (-30 to +30 degrees)
    angle = rng.uniform(-30, 30)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False,
                     fillcolor=(0, 0, 0))

    # 2. Random horizontal flip (50% chance)
    # Leaves can naturally appear from either side — this is biologically valid
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 3. Random vertical flip (10% chance only)
    # Leaves have natural gravity/stem orientation — excessive flipping is unrealistic
    if rng.random() > 0.9:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 4. Random brightness adjustment
    factor = rng.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)

    # 5. Random contrast adjustment
    factor = rng.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)

    # 6. Random saturation adjustment
    # Simulates different lighting conditions in the field
    factor = rng.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(factor)

    # 7. Random zoom/crop (crop to 85-100% then resize back)
    # Simulates different camera distances
    crop_fraction = rng.uniform(0.85, 1.0)
    w, h = img.size
    new_w = int(w * crop_fraction)
    new_h = int(h * crop_fraction)
    left = rng.randint(0, w - new_w)
    top = rng.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((w, h), Image.BILINEAR)

    # 8. Optional light Gaussian blur (20% chance)
    # Simulates slight camera blur in field conditions
    if rng.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


def augment_training_set() -> dict:
    """
    Augment underrepresented classes in the training set using adaptive targeting.

    Strategy:
    - Find the largest class in the training set
    - Set adaptive target = min(30% of largest class, MAX_AUGMENT_TARGET)
    - Augment all classes below AUGMENT_THRESHOLD up to that target
    - Cap augmentations per source image to MAX_AUG_PER_IMAGE

    This avoids:
    - Over-augmentation (synthetic data dominating real data)
    - Fixed targets that ignore dataset distribution
    - Excessive reuse of the same source image
    """
    print("\n" + "=" * 60)
    print("[STEP 4] Augmenting underrepresented classes (adaptive)...")
    print("=" * 60)

    train_dir = OUTPUT_DIR / "train"
    aug_stats = {}

    if not train_dir.exists():
        print("  ERROR: train directory not found")
        return aug_stats

    # Step A: Count all classes to determine adaptive target
    class_counts = {}
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    for cls_dir in class_dirs:
        images = [f for f in cls_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        class_counts[cls_dir.name] = len(images)

    if not class_counts:
        print("  ERROR: No classes found in training directory")
        return aug_stats

    largest_class_count = max(class_counts.values())
    smallest_class_count = min(class_counts.values())
    adaptive_target = min(MAX_AUGMENT_TARGET, largest_class_count * 30 // 100)

    # Ensure target is at least AUGMENT_THRESHOLD
    adaptive_target = max(adaptive_target, AUGMENT_THRESHOLD)

    print(f"  Largest class:        {largest_class_count:,} images")
    print(f"  Smallest class:       {smallest_class_count:,} images")
    print(f"  Current imbalance:    {largest_class_count / max(1, smallest_class_count):.1f}x")
    print(f"  Adaptive target:      {adaptive_target:,} images")
    print(f"    (30% of largest = {largest_class_count * 30 // 100:,}, "
          f"capped at {MAX_AUGMENT_TARGET:,})")
    print(f"  Augment threshold:    < {AUGMENT_THRESHOLD} images")
    print(f"  Max aug per image:    {MAX_AUG_PER_IMAGE}")
    print()

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        current_count = class_counts[cls_name]

        if current_count >= AUGMENT_THRESHOLD:
            aug_stats[cls_name] = {
                "original": current_count,
                "augmented": 0,
                "final": current_count,
                "action": "no augmentation needed",
            }
            print(f"  {cls_name:<40} {current_count:>5} images — OK (no augmentation)")
            continue

        # Calculate needed augmentations
        needed = adaptive_target - current_count
        if needed <= 0:
            aug_stats[cls_name] = {
                "original": current_count,
                "augmented": 0,
                "final": current_count,
                "action": "above adaptive target",
            }
            continue

        # Cap total augmentations: no more than MAX_AUG_PER_IMAGE per original
        max_possible = current_count * MAX_AUG_PER_IMAGE
        if needed > max_possible:
            print(f"  {cls_name:<40} {current_count:>5} images — "
                  f"augmenting +{max_possible} (capped from +{needed})")
            needed = max_possible
        else:
            print(f"  {cls_name:<40} {current_count:>5} images — augmenting +{needed}...")

        images = sorted([f for f in cls_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])

        augmented = 0
        aug_index = 0
        images_len = len(images)

        while augmented < needed:
            for img_idx, img_path in enumerate(images):
                if augmented >= needed:
                    break

                # Check per-image cap
                current_round = aug_index // max(1, images_len)
                if current_round >= MAX_AUG_PER_IMAGE:
                    break

                try:
                    with Image.open(img_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        aug_img = augment_image(img, aug_index)

                        aug_name = f"aug_{aug_index:04d}_{img_path.stem}.jpg"
                        aug_path = cls_dir / aug_name
                        aug_img.save(aug_path, "JPEG", quality=95)

                        augmented += 1
                        aug_index += 1

                except Exception as e:
                    print(f"    ERROR augmenting {img_path.name}: {e}")
                    aug_index += 1
                    continue

            # Safety: prevent infinite loop
            if aug_index > needed * 2 or current_round >= MAX_AUG_PER_IMAGE:
                break

        final_count = current_count + augmented
        aug_stats[cls_name] = {
            "original": current_count,
            "augmented": augmented,
            "final": final_count,
            "action": f"augmented +{augmented}",
        }
        print(f"    -> Now has {final_count} images "
              f"(+{augmented} augmented, {augmented / max(1, current_count):.1f}x synthetic ratio)")

    # Summary
    total_original = sum(s["original"] for s in aug_stats.values())
    total_augmented = sum(s["augmented"] for s in aug_stats.values())
    total_final = sum(s["final"] for s in aug_stats.values())

    print(f"\n  Augmentation Summary:")
    print(f"    Original training images:   {total_original:,}")
    print(f"    Augmented images created:    {total_augmented:,}")
    print(f"    Final training set size:     {total_final:,}")

    # Post-augmentation balance check
    final_counts = [s["final"] for s in aug_stats.values()]
    if final_counts:
        new_min = min(final_counts)
        new_max = max(final_counts)
        new_ratio = new_max / max(1, new_min)
        print(f"    Post-augmentation imbalance: {new_ratio:.1f}x "
              f"(was {largest_class_count / max(1, smallest_class_count):.1f}x)")

    return aug_stats


# ============================================================
# STEP 5: GENERATE FINAL REPORT
# ============================================================

def count_split_images() -> dict:
    """Count all images in each split after all processing."""
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split
        counts[split] = {}
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = len([f for f in cls_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
                counts[split][cls_dir.name] = n
    return counts


def generate_report(dedup_report, split_stats, resize_stats, aug_stats):
    """Generate a comprehensive preprocessing report."""
    print("\n" + "=" * 60)
    print("[REPORT] Generating preprocessing report...")
    print("=" * 60)

    final_counts = count_split_images()

    report = {
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            "interpolation": "LANCZOS",
            "output_format": "JPEG RGB (quality=95)",
            "split_ratios": {
                "train": TRAIN_RATIO,
                "val": VAL_RATIO,
                "test": TEST_RATIO,
            },
            "augmentation": {
                "strategy": "adaptive",
                "threshold": AUGMENT_THRESHOLD,
                "max_target": MAX_AUGMENT_TARGET,
                "adaptive_rule": "min(30% of largest class, MAX_AUGMENT_TARGET)",
                "max_aug_per_image": MAX_AUG_PER_IMAGE,
                "transforms": [
                    "rotation (±30°)",
                    "horizontal flip (50%)",
                    "vertical flip (10% — biologically conservative)",
                    "brightness (0.7–1.3)",
                    "contrast (0.7–1.3)",
                    "saturation (0.8–1.2)",
                    "random crop/zoom (85–100%)",
                    "Gaussian blur (20%, radius=0.5)",
                ],
            },
            "normalization": (
                "Pixel values saved as uint8 (0-255). "
                "Runtime normalization to [0,1] applied during model training."
            ),
            "random_seed": RANDOM_SEED,
        },
        "deduplication": dedup_report,
        "split_statistics": split_stats,
        "resize": {
            "target_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            "images_resized": resize_stats.get("resized", 0),
            "errors": resize_stats.get("errors", 0),
            "error_files": resize_stats.get("error_files", [])[:20],
        },
        "augmentation": aug_stats,
        "final_counts": final_counts,
    }

    # Calculate totals
    totals = {"train": 0, "val": 0, "test": 0}
    for split in totals:
        totals[split] = sum(final_counts.get(split, {}).values())
    report["totals"] = totals
    report["grand_total"] = sum(totals.values())

    # Save report
    report_path = OUTPUT_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    # ---- Print final summary table ----
    print("\n" + "=" * 75)
    print("FINAL PREPROCESSED DATASET SUMMARY")
    print("=" * 75)
    print(f"\n  {'CLASS':<40} {'TRAIN':>7} {'VAL':>6} {'TEST':>6} {'TOTAL':>7}")
    print(f"  {'─' * 70}")

    all_classes_flat = []
    for crop in ALL_CLASSES:
        all_classes_flat.extend(ALL_CLASSES[crop])

    # Group by crop for readability
    for crop in ALL_CLASSES:
        crop_train = crop_val = crop_test = 0
        for cls in ALL_CLASSES[crop]:
            train_n = final_counts.get("train", {}).get(cls, 0)
            val_n = final_counts.get("val", {}).get(cls, 0)
            test_n = final_counts.get("test", {}).get(cls, 0)
            total_n = train_n + val_n + test_n
            crop_train += train_n
            crop_val += val_n
            crop_test += test_n

            # Mark augmented classes
            aug_marker = ""
            if cls in aug_stats and aug_stats[cls].get("augmented", 0) > 0:
                aug_marker = " *"
            print(f"  {cls:<40} {train_n:>7} {val_n:>6} {test_n:>6} {total_n:>7}{aug_marker}")

        crop_total = crop_train + crop_val + crop_test
        print(f"  {'  ' + crop.upper() + ' subtotal':<40} {crop_train:>7} {crop_val:>6} "
              f"{crop_test:>6} {crop_total:>7}")
        print(f"  {'─' * 70}")

    print(f"  {'GRAND TOTAL':<40} {totals['train']:>7} {totals['val']:>6} "
          f"{totals['test']:>6} {report['grand_total']:>7}")
    print(f"\n  * = augmented class")
    print("=" * 75)

    # ---- Class balance analysis (training set) ----
    train_counts = list(final_counts.get("train", {}).values())
    if train_counts:
        min_c = min(train_counts)
        max_c = max(train_counts)
        avg_c = sum(train_counts) / len(train_counts)
        ratio = max_c / min_c if min_c > 0 else float("inf")

        print(f"\n  TRAINING SET BALANCE (after augmentation)")
        print(f"  {'─' * 55}")
        print(f"    Smallest class:    {min_c:>6} images")
        print(f"    Largest class:     {max_c:>6} images")
        print(f"    Average class:     {avg_c:>6.0f} images")
        print(f"    Imbalance ratio:   {ratio:>6.1f}x")

        if ratio > 10:
            print(f"\n    NOTE: Remaining imbalance of {ratio:.1f}x is expected for")
            print(f"    cassava_mosaic_disease (dominant real-world class).")
            print(f"    Recommendation: Use class-weighted loss during training:")
            print(f"    weights = 1.0 / class_count (normalized)")
        elif ratio > 5:
            print(f"\n    Moderate imbalance. Consider class-weighted loss or")
            print(f"    oversampling during training.")
        else:
            print(f"\n    Class balance is acceptable for training.")

    return report


# ============================================================
# COPY METADATA FILES
# ============================================================

def copy_metadata():
    """Copy labels.txt and class_index.json to model_ready directory."""
    print("\n  Copying metadata files to model_ready/...")
    for filename in ["labels.txt", "class_index.json"]:
        src = INPUT_DIR / filename
        dst = OUTPUT_DIR / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"    Copied: {filename}")
        else:
            print(f"    WARNING: {filename} not found in {INPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  PEST & CROP DISEASE DETECTION")
    print("  Data Cleaning & Preprocessing Script")
    print("=" * 60)
    print(f"\n  Input:          {INPUT_DIR}")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"  Image size:     {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"  Splits:         Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test {TEST_RATIO:.0%}")
    print(f"  Augmentation:   Adaptive (threshold={AUGMENT_THRESHOLD}, max={MAX_AUGMENT_TARGET})")
    print(f"  Max aug/image:  {MAX_AUG_PER_IMAGE}")
    print(f"  Random seed:    {RANDOM_SEED}")

    # Verify input exists
    if not INPUT_DIR.exists():
        print(f"\n  ERROR: Input directory not found: {INPUT_DIR}")
        print("  Run organize_datasets.py first.")
        return

    # Clean output directory if it exists
    if OUTPUT_DIR.exists():
        print(f"\n  Cleaning existing output directory...")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Remove duplicates (operates on processed/ in-place)
    dedup_report = find_and_remove_duplicates()

    # Step 2: Split into train/val/test
    split_stats = split_dataset()

    # Step 3: Resize all images to 224x224
    resize_stats = resize_all_images()

    # Step 4: Augment small classes (training set only, adaptive)
    aug_stats = augment_training_set()

    # Copy metadata files
    copy_metadata()

    # Step 5: Generate final report
    report = generate_report(dedup_report, split_stats, resize_stats, aug_stats)

    print(f"\n  Preprocessing complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Report: {OUTPUT_DIR / 'preprocessing_report.json'}")
    print(f"\n  Next step: Model training using datasets/model_ready/")


if __name__ == "__main__":
    main()