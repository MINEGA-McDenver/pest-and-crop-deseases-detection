"""
Analyze new images in add_this_to_datasets/
- Check dimensions, file size, format
- Check EXIF data to identify real camera/phone photos
- Classify each image as field/phone vs internet/lab
- Map folder names to model labels
- Report quality issues
"""
import os
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict

BASE = r"D:\ALL MY DOCUMENTS\YEAR 4\FINAL YEAR PROJECT\pest-and-crop-deseases-detection\add_this_to_datasets"

# Model label mapping
FOLDER_TO_LABEL = {
    "banana cardana leaf spot": "banana_cordana",
    "banana health leaf": "banana_healthy",
    "banana pestalotiopsis": "banana_pestalotiopsis",
    "banana sigatoka leaf image": "banana_sigatoka",
    "beans angular leaf spot": "beans_angular_leaf_spot",
    "beans health": "beans_healthy",
    "beans rust": "beans_rust",
    "maize common rust": "maize_common_rust",
    "maize gray leaf spot": "maize_gray_leaf_spot",
    "maize health": "maize_healthy",
    "maize nothern leaf  bright": "maize_northern_leaf_blight",
    "potato early bright": "potato_early_blight",
    "potato health": "potato_healthy",
    "potato late bright": "potato_late_blight",
}

ALL_LABELS = [
    "banana_cordana", "banana_healthy", "banana_pestalotiopsis", "banana_sigatoka",
    "beans_angular_leaf_spot", "beans_healthy", "beans_rust",
    "maize_common_rust", "maize_gray_leaf_spot", "maize_healthy", "maize_northern_leaf_blight",
    "potato_early_blight", "potato_healthy", "potato_late_blight",
]

def get_exif_info(filepath):
    """Extract key EXIF fields if present, without fully loading image."""
    exif_data = {}
    try:
        img = Image.open(filepath)
        raw = img._getexif()
        img.close()
        if raw:
            for tag_id, value in raw.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ('Make', 'Model', 'DateTime', 'Software',
                           'ImageWidth', 'ImageLength', 'ExifImageWidth', 
                           'ExifImageHeight', 'DateTimeOriginal'):
                    exif_data[tag] = str(value)
    except Exception:
        pass
    return exif_data

def classify_image(filepath, w, h, exif):
    """Classify image as field_photo, internet_photo, or thumbnail."""
    size_kb = os.path.getsize(filepath) / 1024
    
    # Thumbnail: very small dimensions or tiny file
    if w < 150 or h < 150 or size_kb < 5:
        return "thumbnail"
    
    # Has camera EXIF = definitely a real photo
    if exif.get('Make') or exif.get('Model'):
        if any(kw in str(exif.get('Make', '')).lower() + str(exif.get('Model', '')).lower()
               for kw in ['samsung', 'apple', 'huawei', 'xiaomi', 'oppo', 'vivo',
                          'tecno', 'infinix', 'itel', 'nokia', 'google', 'pixel',
                          'iphone', 'canon', 'nikon', 'sony', 'lg']):
            return "field_phone"
        return "field_camera"
    
    # Large file + high resolution = likely real photo (phone cameras produce big files)
    if size_kb > 500 and w >= 1000 and h >= 1000:
        return "likely_field"
    
    # Medium resolution, decent size = internet download (could be field or lab)
    if w >= 200 and h >= 200 and size_kb >= 10:
        return "internet"
    
    # Small but not tiny
    if size_kb < 10:
        return "low_quality_web"
    
    return "internet"

def analyze():
    print("=" * 80)
    print("DEEP IMAGE ANALYSIS: add_this_to_datasets")
    print("=" * 80)
    
    folder_stats = {}
    all_issues = []
    corrupt_files = []
    
    for folder_name in sorted(os.listdir(BASE)):
        folder_path = os.path.join(BASE, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        label = FOLDER_TO_LABEL.get(folder_name.lower(), "UNKNOWN")
        
        stats = {
            "label": label,
            "total": 0,
            "field_phone": 0,
            "field_camera": 0,
            "likely_field": 0,
            "internet": 0,
            "low_quality_web": 0,
            "thumbnail": 0,
            "corrupt": 0,
            "widths": [],
            "heights": [],
            "sizes_kb": [],
            "has_exif": 0,
            "exif_devices": set(),
        }
        
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_issues.append(f"  NON-IMAGE: {folder_name}/{fname}")
                continue
            
            stats["total"] += 1
            size_kb = os.path.getsize(fpath) / 1024
            stats["sizes_kb"].append(size_kb)
            
            try:
                img = Image.open(fpath)
                w, h = img.size
                stats["widths"].append(w)
                stats["heights"].append(h)
                img.close()
                
                exif = get_exif_info(fpath)
                if exif:
                    stats["has_exif"] += 1
                    device = exif.get('Make', '') + ' ' + exif.get('Model', '')
                    if device.strip():
                        stats["exif_devices"].add(device.strip())
                
                category = classify_image(fpath, w, h, exif)
                stats[category] += 1
                
                # Flag specific issues
                if w < 100 or h < 100:
                    all_issues.append(f"  TINY ({w}x{h}): {folder_name}/{fname}")
                    
            except Exception as e:
                stats["corrupt"] += 1
                corrupt_files.append(f"  CORRUPT: {folder_name}/{fname} - {e}")
        
        folder_stats[folder_name] = stats
    
    # ── Report per folder ──
    print()
    for folder_name, s in folder_stats.items():
        if s["total"] == 0:
            continue
        
        min_w = min(s["widths"]) if s["widths"] else 0
        max_w = max(s["widths"]) if s["widths"] else 0
        min_h = min(s["heights"]) if s["heights"] else 0
        max_h = max(s["heights"]) if s["heights"] else 0
        avg_kb = sum(s["sizes_kb"]) / len(s["sizes_kb"]) if s["sizes_kb"] else 0
        
        field_total = s["field_phone"] + s["field_camera"] + s["likely_field"]
        web_total = s["internet"] + s["low_quality_web"] + s["thumbnail"]
        
        print(f"\n{'─' * 70}")
        print(f"📁 {folder_name}")
        print(f"   Model label: {s['label']}")
        print(f"   Total images: {s['total']}  |  Corrupt: {s['corrupt']}")
        print(f"   Dimensions: {min_w}x{min_h} to {max_w}x{max_h}")
        print(f"   File sizes: {min(s['sizes_kb']):.1f}KB - {max(s['sizes_kb']):.1f}KB (avg {avg_kb:.1f}KB)")
        print(f"   EXIF present: {s['has_exif']}/{s['total']}")
        if s["exif_devices"]:
            print(f"   Devices: {', '.join(s['exif_devices'])}")
        print(f"   Classification:")
        print(f"     FIELD/PHONE photos:  {field_total:3d} ({s['field_phone']} phone, {s['field_camera']} camera, {s['likely_field']} likely)")
        print(f"     INTERNET downloads:  {s['internet']:3d}")
        print(f"     LOW-QUALITY web:     {s['low_quality_web']:3d}")
        print(f"     THUMBNAILS:          {s['thumbnail']:3d}")
        
        # Quality verdict
        if field_total > 0 and web_total == 0:
            print(f"   ✅ VERDICT: All real field photos - EXCELLENT")
        elif field_total > web_total:
            print(f"   ⚠️  VERDICT: Mixed - {field_total} field + {web_total} web. Remove low-quality ones.")
        elif s["thumbnail"] + s["low_quality_web"] > s["total"] * 0.3:
            print(f"   ❌ VERDICT: Too many thumbnails/low-quality ({s['thumbnail'] + s['low_quality_web']}/{s['total']}). Need better images.")
        else:
            print(f"   🔶 VERDICT: Mostly internet images - usable but collect more field photos")
    
    # ── Missing classes ──
    print(f"\n{'=' * 70}")
    print("MISSING CLASSES (no folder found):")
    found_labels = set(s["label"] for s in folder_stats.values())
    for label in ALL_LABELS:
        if label not in found_labels:
            print(f"  ❌ {label} - NO IMAGES COLLECTED")
    
    # ── Overall summary ──
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY:")
    total_all = sum(s["total"] for s in folder_stats.values())
    total_field = sum(s["field_phone"] + s["field_camera"] + s["likely_field"] for s in folder_stats.values())
    total_internet = sum(s["internet"] for s in folder_stats.values())
    total_lowq = sum(s["low_quality_web"] for s in folder_stats.values())
    total_thumb = sum(s["thumbnail"] for s in folder_stats.values())
    total_corrupt = sum(s["corrupt"] for s in folder_stats.values())
    
    print(f"  Total images: {total_all}")
    print(f"  Real field/phone photos: {total_field} ({total_field*100//total_all}%)")
    print(f"  Internet downloads:      {total_internet} ({total_internet*100//total_all}%)")
    print(f"  Low-quality web:         {total_lowq} ({total_lowq*100//total_all}%)")
    print(f"  Thumbnails (unusable):   {total_thumb} ({total_thumb*100//total_all}%)")
    print(f"  Corrupt files:           {total_corrupt}")
    
    print(f"\n  USABLE images (field + internet): {total_field + total_internet}")
    print(f"  REMOVE (low-quality + thumbnails): {total_lowq + total_thumb}")
    
    # ── Issues ──
    if corrupt_files:
        print(f"\n{'=' * 70}")
        print("CORRUPT FILES (delete these):")
        for c in corrupt_files:
            print(c)
    
    if all_issues:
        print(f"\n{'=' * 70}")
        print("OTHER ISSUES:")
        for issue in all_issues[:30]:
            print(issue)
    
    # ── Per-class needs ──
    print(f"\n{'=' * 70}")
    print("IMAGES NEEDED PER CLASS (target: 150 field photos per class):")
    print(f"{'Label':<30} {'Have':>5} {'Field':>6} {'Usable':>7} {'Need':>6}")
    print("-" * 60)
    for label in ALL_LABELS:
        folder_data = None
        for fn, s in folder_stats.items():
            if s["label"] == label:
                folder_data = s
                break
        if folder_data:
            field = folder_data["field_phone"] + folder_data["field_camera"] + folder_data["likely_field"]
            usable = field + folder_data["internet"]
            need = max(0, 150 - usable)
            print(f"  {label:<28} {folder_data['total']:>5} {field:>6} {usable:>7} {need:>6}")
        else:
            print(f"  {label:<28}     0      0       0    150  ❌ MISSING")

if __name__ == "__main__":
    analyze()
