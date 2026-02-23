"""
Dataset Organization Script
============================
Filters, organizes, and labels crop disease images from multiple
open-source datasets into a unified structure for model training.

Includes:
- Image integrity validation (skips corrupted files)
- Class balance analysis
- Dataset provenance documentation

Target crops: Cassava, Maize, Beans, Potato, Banana
Total: 19 classes across 5 crops

Usage:
    python scripts/organize_datasets.py
"""

import os
import shutil
import json
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Try to import PIL for image validation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow not installed. Image integrity checks disabled.")
    print("  Install with: pip install Pillow")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"d:\ALL MY DOCUMENTS\YEAR 4\FINAL YEAR PROJECT\pest-and-crop-deseases-detection")
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = DATASETS_DIR / "processed"

# ============================================================
# SOURCE PATHS (exact paths matching your extracted folders)
# ============================================================

CASSAVA_IMAGES_DIR = DATASETS_DIR / "cassava-leaf-disease-classification" / "train_images"
CASSAVA_CSV = DATASETS_DIR / "cassava-leaf-disease-classification" / "train.csv"

PLANTVILLAGE_COLOR_DIR = DATASETS_DIR / "plantvillage dataset" / "color"

BEANS_TRAIN_DIR = DATASETS_DIR / "bean-leaf-lesions" / "train"
BEANS_VAL_DIR = DATASETS_DIR / "bean-leaf-lesions" / "val"

BANANA_ORIGINAL_DIR = DATASETS_DIR / "bananalsd" / "BananaLSD" / "OriginalSet"
BANANA_AUGMENTED_DIR = DATASETS_DIR / "bananalsd" / "BananaLSD" / "AugmentedSet"

# ============================================================
# CLASS MAPPINGS
# ============================================================

CASSAVA_LABEL_MAP = {
    0: "cassava_bacterial_blight",
    1: "cassava_brown_streak_disease",
    2: "cassava_green_mottle",
    3: "cassava_mosaic_disease",
    4: "cassava_healthy",
}

MAIZE_FOLDER_MAP = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "maize_gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "maize_common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "maize_northern_leaf_blight",
    "Corn_(maize)___healthy": "maize_healthy",
}

POTATO_FOLDER_MAP = {
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy",
}

BEANS_FOLDER_MAP = {
    "angular_leaf_spot": "beans_angular_leaf_spot",
    "bean_rust": "beans_rust",
    "healthy": "beans_healthy",
}

BANANA_FOLDER_MAP = {
    "cordana": "banana_cordana",
    "pestalotiopsis": "banana_pestalotiopsis",
    "sigatoka": "banana_sigatoka",
    "healthy": "banana_healthy",
}

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

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Track corrupted images globally
corrupted_images = []


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def is_image(filepath: Path) -> bool:
    """Check if a file is a valid image based on extension."""
    return filepath.suffix.lower() in VALID_EXTENSIONS


def validate_image(filepath: Path) -> bool:
    """
    Verify image integrity by attempting to open and load it.
    Returns True if the image is valid, False if corrupted.
    """
    if not PIL_AVAILABLE:
        return True  # Skip validation if Pillow not installed

    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify file header
        # Re-open to fully decode pixel data (verify() doesn't do this)
        with Image.open(filepath) as img:
            img.load()
        return True
    except Exception:
        return False


def copy_image(src: Path, dst_dir: Path, prefix: str = "") -> bool:
    """Copy a single image to destination with integrity check."""
    if not src.exists() or not is_image(src):
        return False

    # Validate image integrity
    if not validate_image(src):
        corrupted_images.append(str(src))
        return False

    suffix = src.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"

    dst_name = f"{prefix}_{src.stem}{suffix}" if prefix else f"{src.stem}{suffix}"
    dst_path = dst_dir / dst_name

    counter = 1
    while dst_path.exists():
        dst_name = f"{prefix}_{src.stem}_{counter}{suffix}" if prefix else f"{src.stem}_{counter}{suffix}"
        dst_path = dst_dir / dst_name
        counter += 1

    shutil.copy2(src, dst_path)
    return True


def create_output_dirs():
    """Create the full processed output directory tree."""
    print("\n[SETUP] Creating output directory structure...")
    for crop, classes in ALL_CLASSES.items():
        for cls in classes:
            dir_path = OUTPUT_DIR / crop / cls
            dir_path.mkdir(parents=True, exist_ok=True)
    total_classes = sum(len(v) for v in ALL_CLASSES.values())
    print(f"  Output root: {OUTPUT_DIR}")
    print(f"  Created {total_classes} class directories across {len(ALL_CLASSES)} crops")


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

def process_cassava():
    """Process cassava dataset (CSV-based labels + flat image folder)."""
    print("\n" + "=" * 50)
    print("[CASSAVA] Processing cassava-leaf-disease-classification...")
    print("=" * 50)

    if not CASSAVA_CSV.exists():
        print(f"  ERROR: CSV not found at {CASSAVA_CSV}")
        return
    if not CASSAVA_IMAGES_DIR.exists():
        print(f"  ERROR: Images folder not found at {CASSAVA_IMAGES_DIR}")
        return

    count = defaultdict(int)
    skipped = 0

    with open(CASSAVA_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"  Found {total} entries in train.csv")
    print(f"  Copying and validating images...")

    for i, row in enumerate(rows):
        image_id = row.get("image_id", "").strip()
        label = int(row.get("label", -1))

        if label not in CASSAVA_LABEL_MAP:
            skipped += 1
            continue

        class_name = CASSAVA_LABEL_MAP[label]
        src = CASSAVA_IMAGES_DIR / image_id
        dst_dir = OUTPUT_DIR / "cassava" / class_name

        if copy_image(src, dst_dir, prefix="cass"):
            count[class_name] += 1
        else:
            skipped += 1

        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{total}...")

    print(f"\n  Results:")
    for cls, n in sorted(count.items()):
        print(f"    {cls:<40} {n:>6} images")
    print(f"    {'TOTAL':<40} {sum(count.values()):>6} images")
    if skipped:
        print(f"    Skipped/corrupted: {skipped}")


def process_maize():
    """Process maize images from PlantVillage color dataset."""
    print("\n" + "=" * 50)
    print("[MAIZE] Processing from plantvillage dataset/color/...")
    print("=" * 50)

    if not PLANTVILLAGE_COLOR_DIR.exists():
        print(f"  ERROR: Folder not found at {PLANTVILLAGE_COLOR_DIR}")
        return

    count = defaultdict(int)

    for folder_name, class_name in MAIZE_FOLDER_MAP.items():
        src_dir = PLANTVILLAGE_COLOR_DIR / folder_name
        dst_dir = OUTPUT_DIR / "maize" / class_name

        if not src_dir.exists():
            print(f"  WARNING: Folder not found: {folder_name}")
            continue

        for f in src_dir.iterdir():
            if copy_image(f, dst_dir, prefix="pv"):
                count[class_name] += 1

    print(f"\n  Results:")
    for cls, n in sorted(count.items()):
        print(f"    {cls:<40} {n:>6} images")
    print(f"    {'TOTAL':<40} {sum(count.values()):>6} images")


def process_potato():
    """Process potato images from PlantVillage color dataset."""
    print("\n" + "=" * 50)
    print("[POTATO] Processing from plantvillage dataset/color/...")
    print("=" * 50)

    if not PLANTVILLAGE_COLOR_DIR.exists():
        print(f"  ERROR: Folder not found at {PLANTVILLAGE_COLOR_DIR}")
        return

    count = defaultdict(int)

    for folder_name, class_name in POTATO_FOLDER_MAP.items():
        src_dir = PLANTVILLAGE_COLOR_DIR / folder_name
        dst_dir = OUTPUT_DIR / "potato" / class_name

        if not src_dir.exists():
            print(f"  WARNING: Folder not found: {folder_name}")
            continue

        for f in src_dir.iterdir():
            if copy_image(f, dst_dir, prefix="pv"):
                count[class_name] += 1

    print(f"\n  Results:")
    for cls, n in sorted(count.items()):
        print(f"    {cls:<40} {n:>6} images")
    print(f"    {'TOTAL':<40} {sum(count.values()):>6} images")


def process_beans():
    """Process beans dataset — merge train/ and val/ into single set per class."""
    print("\n" + "=" * 50)
    print("[BEANS] Processing from bean-leaf-lesions/...")
    print("=" * 50)

    count = defaultdict(int)

    for source_dir, source_name in [(BEANS_TRAIN_DIR, "train"), (BEANS_VAL_DIR, "val")]:
        if not source_dir.exists():
            print(f"  WARNING: {source_name} folder not found: {source_dir}")
            continue

        for folder_name, class_name in BEANS_FOLDER_MAP.items():
            src_dir = source_dir / folder_name
            dst_dir = OUTPUT_DIR / "beans" / class_name

            if not src_dir.exists():
                continue

            for f in src_dir.iterdir():
                if copy_image(f, dst_dir, prefix=f"bean_{source_name}"):
                    count[class_name] += 1

    print(f"\n  Results:")
    for cls, n in sorted(count.items()):
        print(f"    {cls:<40} {n:>6} images")
    print(f"    {'TOTAL':<40} {sum(count.values()):>6} images")


def process_banana():
    """Process banana images — OriginalSet + AugmentedSet for small classes."""
    print("\n" + "=" * 50)
    print("[BANANA] Processing from bananalsd/BananaLSD/...")
    print("=" * 50)

    if not BANANA_ORIGINAL_DIR.exists():
        print(f"  ERROR: OriginalSet not found at {BANANA_ORIGINAL_DIR}")
        return

    count = defaultdict(int)
    original_count = defaultdict(int)

    # Step 1: Copy all original images
    print("  Step 1: Copying OriginalSet images...")
    for folder_name, class_name in BANANA_FOLDER_MAP.items():
        src_dir = BANANA_ORIGINAL_DIR / folder_name
        dst_dir = OUTPUT_DIR / "banana" / class_name

        if not src_dir.exists():
            print(f"  WARNING: Folder not found: {folder_name}")
            continue

        for f in src_dir.iterdir():
            if copy_image(f, dst_dir, prefix="bana_orig"):
                count[class_name] += 1
                original_count[class_name] += 1

    print(f"\n  OriginalSet counts:")
    for cls, n in sorted(original_count.items()):
        print(f"    {cls:<40} {n:>6} images")

    # Step 2: Supplement small classes from AugmentedSet
    MIN_THRESHOLD = 300
    print(f"\n  Step 2: Supplementing classes with < {MIN_THRESHOLD} originals...")

    if BANANA_AUGMENTED_DIR.exists():
        for folder_name, class_name in BANANA_FOLDER_MAP.items():
            if original_count[class_name] >= MIN_THRESHOLD:
                print(f"    {class_name}: {original_count[class_name]} originals — skipping")
                continue

            src_dir = BANANA_AUGMENTED_DIR / folder_name
            dst_dir = OUTPUT_DIR / "banana" / class_name

            if not src_dir.exists():
                continue

            added = 0
            for f in src_dir.iterdir():
                if copy_image(f, dst_dir, prefix="bana_aug"):
                    count[class_name] += 1
                    added += 1
            print(f"    {class_name}: added {added} augmented images")
    else:
        print(f"  WARNING: AugmentedSet not found at {BANANA_AUGMENTED_DIR}")

    print(f"\n  Final Results:")
    for cls, n in sorted(count.items()):
        print(f"    {cls:<40} {n:>6} images")
    print(f"    {'TOTAL':<40} {sum(count.values()):>6} images")


# ============================================================
# METADATA GENERATION
# ============================================================

def generate_metadata():
    """Generate labels.txt, class_index.json, dataset_summary.json, and provenance.json."""
    print("\n" + "=" * 50)
    print("[METADATA] Generating reference files...")
    print("=" * 50)

    # --- Count images per class ---
    summary = {}
    total_images = 0
    all_counts = []  # flat list of (class_name, count) for balance analysis

    for crop in ALL_CLASSES:
        crop_dir = OUTPUT_DIR / crop
        summary[crop] = {}
        crop_total = 0

        for cls in ALL_CLASSES[crop]:
            cls_dir = crop_dir / cls
            if cls_dir.exists():
                n = len([f for f in cls_dir.iterdir() if f.is_file() and is_image(f)])
            else:
                n = 0
            summary[crop][cls] = n
            all_counts.append((cls, n))
            crop_total += n

        summary[crop]["_total"] = crop_total
        total_images += crop_total

    # --- dataset_summary.json ---
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # --- labels.txt ---
    labels = []
    for crop in ALL_CLASSES:
        labels.extend(ALL_CLASSES[crop])

    labels_path = OUTPUT_DIR / "labels.txt"
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"  Saved: {labels_path}")

    # --- class_index.json ---
    class_index = {label: idx for idx, label in enumerate(labels)}
    index_path = OUTPUT_DIR / "class_index.json"
    with open(index_path, "w") as f:
        json.dump(class_index, f, indent=2)
    print(f"  Saved: {index_path}")

    # --- provenance.json (Improvement 4: dataset sources & ethics) ---
    provenance = {
        "project": "Pest and Crop Disease Early Detection Using Mobile Computer Vision",
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": [
            {
                "name": "Cassava Leaf Disease Classification",
                "source": "https://www.kaggle.com/c/cassava-leaf-disease-classification",
                "provider": "Makerere AI Lab / Kaggle",
                "license": "Kaggle Competition Data (research use)",
                "images_used": summary.get("cassava", {}).get("_total", 0),
                "crops": ["Cassava"],
                "classes": list(ALL_CLASSES["cassava"]),
                "notes": "Real field images collected in Uganda. Labels via expert annotation.",
            },
            {
                "name": "PlantVillage Dataset",
                "source": "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset",
                "provider": "Penn State University / PlantVillage",
                "license": "CC0: Public Domain",
                "images_used": (
                    summary.get("maize", {}).get("_total", 0)
                    + summary.get("potato", {}).get("_total", 0)
                ),
                "crops": ["Maize (Corn)", "Potato"],
                "classes": list(ALL_CLASSES["maize"]) + list(ALL_CLASSES["potato"]),
                "notes": "Lab-controlled images. Color version used. Source: Mohanty et al. (2016).",
            },
            {
                "name": "Bean Leaf Lesions Classification",
                "source": "https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification",
                "provider": "Makerere AI Lab",
                "license": "Open source (research use)",
                "images_used": summary.get("beans", {}).get("_total", 0),
                "crops": ["Beans"],
                "classes": list(ALL_CLASSES["beans"]),
                "notes": "Collected in East Africa. Also known as the iBean dataset.",
            },
            {
                "name": "BananaLSD (Banana Leaf Spot Disease)",
                "source": "https://www.kaggle.com/datasets/shifatearman/bananalsd",
                "provider": "Shifat E Arman / Kaggle",
                "license": "Open source (research use)",
                "images_used": summary.get("banana", {}).get("_total", 0),
                "crops": ["Banana"],
                "classes": list(ALL_CLASSES["banana"]),
                "notes": "Contains OriginalSet and AugmentedSet. OriginalSet used as primary; AugmentedSet used to supplement small classes.",
            },
        ],
        "ethical_statement": (
            "All datasets used in this project are publicly available open-source datasets "
            "obtained from reputable repositories (Kaggle, GitHub). No personally identifiable "
            "information is contained in the data. Images are used solely for academic research "
            "purposes in partial fulfillment of degree requirements. All dataset sources are "
            "properly cited and acknowledged."
        ),
        "total_images": total_images,
        "total_classes": len(labels),
        "total_crops": len(ALL_CLASSES),
    }

    provenance_path = OUTPUT_DIR / "provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Saved: {provenance_path}")

    # --- CLASS BALANCE REPORT (Improvement 3) ---
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    counts_only = [c for _, c in all_counts if c > 0]
    if counts_only:
        min_count = min(counts_only)
        max_count = max(counts_only)
        avg_count = sum(counts_only) / len(counts_only)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    else:
        min_count = max_count = avg_count = imbalance_ratio = 0

    for crop in ALL_CLASSES:
        crop_total = summary[crop]["_total"]
        print(f"\n  {crop.upper()} ({crop_total:,} images)")
        print(f"  {'─' * 55}")
        for cls in ALL_CLASSES[crop]:
            n = summary[crop][cls]
            bar = "█" * max(1, n // 200)
            print(f"    {cls:<40} {n:>6}  {bar}")

    print(f"\n  {'─' * 55}")
    print(f"  {'GRAND TOTAL':<42} {total_images:>6}")
    print(f"  {'TOTAL CLASSES':<42} {len(labels):>6}")

    print(f"\n  CLASS BALANCE ANALYSIS")
    print(f"  {'─' * 55}")
    print(f"    Smallest class:    {min_count:>6} images")
    print(f"    Largest class:     {max_count:>6} images")
    print(f"    Average class:     {avg_count:>6.0f} images")
    print(f"    Imbalance ratio:   {imbalance_ratio:>6.1f}x")

    if imbalance_ratio > 10:
        print(f"\n    ⚠ HIGH IMBALANCE detected ({imbalance_ratio:.1f}x).")
        print(f"      Recommendation: Apply data augmentation for small classes")
        print(f"      and/or use class-weighted loss during training.")
    elif imbalance_ratio > 5:
        print(f"\n    ⚠ MODERATE IMBALANCE detected ({imbalance_ratio:.1f}x).")
        print(f"      Recommendation: Consider augmentation or weighted sampling.")
    else:
        print(f"\n    Class balance is acceptable.")

    # Identify classes needing augmentation
    augmentation_candidates = [(cls, n) for cls, n in all_counts if 0 < n < 500]
    if augmentation_candidates:
        print(f"\n    Classes recommended for augmentation (< 500 images):")
        for cls, n in sorted(augmentation_candidates, key=lambda x: x[1]):
            print(f"      {cls:<40} {n:>6}")

    print("=" * 60)

    # --- CORRUPTED IMAGES REPORT (Improvement 2) ---
    if corrupted_images:
        print(f"\n  CORRUPTED IMAGES REPORT")
        print(f"  {'─' * 55}")
        print(f"    Total corrupted/skipped: {len(corrupted_images)}")
        corrupted_path = OUTPUT_DIR / "corrupted_images.txt"
        with open(corrupted_path, "w") as f:
            for img_path in corrupted_images:
                f.write(f"{img_path}\n")
        print(f"    Full list saved to: {corrupted_path}")
        # Show first 5
        for img_path in corrupted_images[:5]:
            print(f"      {img_path}")
        if len(corrupted_images) > 5:
            print(f"      ... and {len(corrupted_images) - 5} more")
    else:
        print(f"\n  Image integrity: All images passed validation ✓")

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  PEST & CROP DISEASE DETECTION")
    print("  Dataset Organization Script")
    print("=" * 60)
    print(f"\n  Source:  {DATASETS_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Image validation: {'ENABLED' if PIL_AVAILABLE else 'DISABLED'}")

    # Verify source directories
    print("\n  Checking source directories...")
    sources = {
        "Cassava CSV": CASSAVA_CSV,
        "Cassava Images": CASSAVA_IMAGES_DIR,
        "PlantVillage Color": PLANTVILLAGE_COLOR_DIR,
        "Beans Train": BEANS_TRAIN_DIR,
        "Beans Val": BEANS_VAL_DIR,
        "Banana Original": BANANA_ORIGINAL_DIR,
    }

    all_ok = True
    for name, path in sources.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"    {name:<25} {status}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  WARNING: Some source directories are missing.")
        print("  The script will process what is available.\n")

    # Create output structure
    create_output_dirs()

    # Process each crop
    process_cassava()
    process_maize()
    process_potato()
    process_beans()
    process_banana()

    # Generate all metadata
    generate_metadata()

    print(f"\n  Done! Organized dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()