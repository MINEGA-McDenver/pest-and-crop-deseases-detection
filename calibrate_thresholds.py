"""
Threshold calibration script for the 4-crop disease detection model.

Run this AFTER every retrain to find the optimal cropTotalThreshold and
otherLeafAbsoluteFloor values from your own validation data, rather than
tuning blindly with fixed constants.

Goal: maximise supported-crop recall while keeping the false-positive
      lookalike rate (other_leaf images mis-classified as a real crop) low.

Usage:
    python -u calibrate_thresholds.py

Output:
    models/threshold_calibration.json   — full sweep results
    models/threshold_calibration.png    — recall vs FP-rate trade-off plot
    Prints the recommended threshold at the end.

Requirements:
    pip install tensorflow scikit-learn matplotlib numpy
"""

import os, sys, json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Loading TensorFlow ...", flush=True)
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if os.path.isdir(os.path.join(SCRIPT_DIR, "models")):
    BASE = SCRIPT_DIR
else:
    BASE = PARENT_DIR
MODEL_DIR  = os.path.join(BASE, "models")
DATA_DIR   = os.path.join(BASE, "datasets", "model_ready")
VAL_DIR    = os.path.join(DATA_DIR, "val")
IMG_SIZE   = (224, 224)
BATCH      = 8

# ── These must match classifier_service.dart ──────────────────────────
TEMPERATURE            = 1.8    # temperatureScaling
OL_THRESHOLD           = 0.30   # otherLeafThreshold (direct softmax winner)
OL_RATIO_THRESHOLD     = 0.18   # otherLeafVsCropRatioThreshold
SECOND_CROP_THRESHOLD  = 0.15   # secondCropAmbiguityThreshold (already relaxed)

# Sweep ranges — adjust if your validation results cluster outside these
CROP_TOTAL_SWEEP  = np.arange(0.60, 0.92, 0.02)   # cropTotalThreshold
OL_FLOOR_SWEEP    = np.arange(0.08, 0.22, 0.02)   # otherLeafAbsoluteFloor

# Target: at minimum this fraction of real supported-crop images must
# reach a confident result (not be rejected as unsupported/uncertain).
# Tune this to match your product requirements.
MIN_CROP_RECALL = 0.90

# Crop grouping — must exactly match classifier_service.dart
CROP_GROUPING = {
    'banana_cordana':           'banana',
    'banana_healthy':           'banana',
    'banana_pestalotiopsis':    'banana',
    'banana_sigatoka':          'banana',
    'beans_angular_leaf_spot':  'beans',
    'beans_healthy':            'beans',
    'beans_rust':               'beans',
    'maize_common_rust':        'maize',
    'maize_gray_leaf_spot':     'maize',
    'maize_healthy':            'maize',
    'maize_northern_leaf_blight': 'maize',
    'potato_early_blight':      'potato',
    'potato_healthy':           'potato',
    'potato_late_blight':       'potato',
    # other_leaf intentionally excluded
}


# ── Load model and labels ─────────────────────────────────────────────
print("Loading best model ...", flush=True)
best_model_path = os.path.join(MODEL_DIR, "best_model.keras")
if not os.path.exists(best_model_path):
    print(f"ERROR: {best_model_path} not found. Run train_model.py first.", flush=True)
    sys.exit(1)

model = tf.keras.models.load_model(best_model_path, compile=False)
print("Model loaded.", flush=True)

labels_path = os.path.join(MODEL_DIR, "labels.txt")
if not os.path.exists(labels_path):
    print(f"ERROR: {labels_path} not found.", flush=True)
    sys.exit(1)
with open(labels_path) as f:
    class_names = [l.strip() for l in f if l.strip()]
print(f"Labels ({len(class_names)}): {class_names}", flush=True)

if "other_leaf" not in class_names:
    print("ERROR: other_leaf not in labels.txt.", flush=True)
    print("  Deploy the 15-class model before calibrating.", flush=True)
    sys.exit(1)

OTHER_LEAF_IDX = class_names.index("other_leaf")
print(f"other_leaf index: {OTHER_LEAF_IDX}", flush=True)


# ── Load validation set ───────────────────────────────────────────────
print("Loading validation set ...", flush=True)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=False,
)
val_class_names = val_ds.class_names
if val_class_names != class_names:
    print("WARNING: val_ds class order differs from labels.txt.", flush=True)
    print(f"  val_ds:     {val_class_names}", flush=True)
    print(f"  labels.txt: {class_names}", flush=True)

def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

val_ds = val_ds.map(preprocess, num_parallel_calls=2).prefetch(1)

print("Running inference on validation set ...", flush=True)
all_probs  = []
all_labels = []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    all_probs.append(probs)
    all_labels.extend(labels.numpy())

all_probs  = np.concatenate(all_probs, axis=0)   # (N, num_classes)
all_labels = np.array(all_labels)                 # (N,)
print(f"Collected {len(all_labels)} validation samples.", flush=True)


# ── Temperature scaling ───────────────────────────────────────────────
def temperature_scale(probs, T):
    """Apply temperature scaling in log-space — matches classifier_service.dart."""
    log_p = np.log(np.clip(probs, 1e-10, 1.0)) / T
    log_p -= log_p.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(log_p)
    return e / e.sum(axis=1, keepdims=True)

scaled = temperature_scale(all_probs, TEMPERATURE)
print(f"Temperature scaling applied (T={TEMPERATURE}).", flush=True)


# ── Aggregate crop totals ─────────────────────────────────────────────
def get_crop_totals(probs_row):
    """Sum class probabilities per crop, excluding other_leaf."""
    crop_totals = {}
    for i, name in enumerate(class_names):
        crop = CROP_GROUPING.get(name)
        if crop is not None:
            crop_totals[crop] = crop_totals.get(crop, 0.0) + probs_row[i]
    return crop_totals


# ── Simulate app decision for one sample ─────────────────────────────
def simulate_decision(probs_row, crop_total_thresh, ol_floor_thresh):
    """
    Mirrors the Step 5 → 7a-ii decision path in classifyImage().
    Returns:
        'accepted'    — would reach Step 7b (crop identified)
        'other_leaf'  — rejected as unsupported by other_leaf checks
        'unsupported' — rejected by crop_total or ratio guard
        'uncertain'   — rejected by second-crop ambiguity
    """
    ol_prob = probs_row[OTHER_LEAF_IDX]

    # Step 5: direct other_leaf winner
    if ol_prob >= OL_THRESHOLD:
        return 'other_leaf'

    # Step 5b: absolute floor
    if ol_prob > ol_floor_thresh:
        return 'other_leaf'

    # Step 6: aggregate crops
    crop_totals = get_crop_totals(probs_row)
    if not crop_totals:
        return 'unsupported'

    sorted_crops = sorted(crop_totals.items(), key=lambda x: -x[1])
    best_crop_total  = sorted_crops[0][1]
    second_crop_total = sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0

    # Step 7a: crop dominance
    if best_crop_total < crop_total_thresh:
        return 'unsupported'

    # Step 7a-ii: ratio guard
    ol_ratio = ol_prob / best_crop_total if best_crop_total > 0 else 0.0
    if ol_ratio > OL_RATIO_THRESHOLD:
        return 'other_leaf'

    # Step 7c-v2: second-crop ambiguity (already using relaxed 0.15)
    if second_crop_total > SECOND_CROP_THRESHOLD:
        return 'uncertain'

    return 'accepted'


# ── Masks for supported vs other_leaf samples ─────────────────────────
is_other_leaf = (all_labels == OTHER_LEAF_IDX)
is_supported  = ~is_other_leaf
n_supported   = int(is_supported.sum())
n_other_leaf  = int(is_other_leaf.sum())
print(f"Validation breakdown: {n_supported} supported-crop, {n_other_leaf} other_leaf", flush=True)


# ── Sweep cropTotalThreshold × otherLeafAbsoluteFloor ────────────────
print(f"\nSweeping {len(CROP_TOTAL_SWEEP)} × {len(OL_FLOOR_SWEEP)} threshold combinations ...", flush=True)

results = []
for ct in CROP_TOTAL_SWEEP:
    for ol_floor in OL_FLOOR_SWEEP:
        decisions = np.array([
            simulate_decision(scaled[i], ct, ol_floor)
            for i in range(len(all_labels))
        ])

        # Crop recall: fraction of real supported-crop images that reach 'accepted'
        crop_recall = float(np.mean(decisions[is_supported] == 'accepted')) \
                      if n_supported > 0 else 0.0

        # Other-leaf false-positive rate: fraction of real other_leaf images
        # that were 'accepted' as a real crop (the bug we are trying to fix)
        ol_fp_rate = float(np.mean(decisions[is_other_leaf] == 'accepted')) \
                     if n_other_leaf > 0 else 0.0

        # Other-leaf recall: fraction of other_leaf images correctly rejected
        ol_recall = float(np.mean(decisions[is_other_leaf] != 'accepted')) \
                    if n_other_leaf > 0 else 0.0

        results.append({
            "crop_total_threshold":      round(float(ct), 4),
            "ol_absolute_floor":         round(float(ol_floor), 4),
            "crop_recall":               round(crop_recall, 4),
            "ol_false_positive_rate":    round(ol_fp_rate, 4),
            "ol_recall":                 round(ol_recall, 4),
            "meets_min_crop_recall":     crop_recall >= MIN_CROP_RECALL,
        })

print(f"Sweep complete. {len(results)} combinations evaluated.", flush=True)


# ── Find best combination ─────────────────────────────────────────────
# Among combinations that meet the minimum crop recall target, pick the
# one that minimises the other_leaf false-positive rate.
# Tiebreak: prefer the higher cropTotalThreshold (more conservative).
valid = [r for r in results if r["meets_min_crop_recall"]]
if valid:
    best = min(valid, key=lambda r: (r["ol_false_positive_rate"], -r["crop_total_threshold"]))
else:
    # No combination meets the recall target — relax and take best overall
    print(
        f"WARNING: No combination achieved crop_recall ≥ {MIN_CROP_RECALL:.0%}. "
        "Your other_leaf dataset may be too aggressive or your supported-crop val "
        "set too small. Showing best available.", flush=True
    )
    best = min(results, key=lambda r: r["ol_false_positive_rate"])

print(f"\n{'='*60}", flush=True)
print(f"RECOMMENDED THRESHOLDS", flush=True)
print(f"{'='*60}", flush=True)
print(f"  cropTotalThreshold:       {best['crop_total_threshold']}", flush=True)
print(f"  otherLeafAbsoluteFloor:   {best['ol_absolute_floor']}", flush=True)
print(f"  Supported-crop recall:    {best['crop_recall']:.2%}", flush=True)
print(f"  Other-leaf FP rate:       {best['ol_false_positive_rate']:.2%}", flush=True)
print(f"  Other-leaf recall:        {best['ol_recall']:.2%}", flush=True)
print(f"  Meets recall target (≥{MIN_CROP_RECALL:.0%}): {best['meets_min_crop_recall']}", flush=True)
print(f"{'='*60}", flush=True)
print(f"\nUpdate these two constants in classifier_service.dart:", flush=True)
print(f"  static const double cropTotalThreshold = {best['crop_total_threshold']};", flush=True)
print(f"  static const double otherLeafAbsoluteFloor = {best['ol_absolute_floor']};", flush=True)


# ── Save full results ─────────────────────────────────────────────────
out_path = os.path.join(MODEL_DIR, "threshold_calibration.json")
with open(out_path, 'w') as f:
    json.dump({
        "recommended": best,
        "min_crop_recall_target": MIN_CROP_RECALL,
        "temperature": TEMPERATURE,
        "fixed_thresholds": {
            "otherLeafThreshold":          OL_THRESHOLD,
            "otherLeafVsCropRatioThreshold": OL_RATIO_THRESHOLD,
            "secondCropAmbiguityThreshold": SECOND_CROP_THRESHOLD,
        },
        "sweep": results,
    }, f, indent=2)
print(f"\nFull sweep saved -> {out_path}", flush=True)


# ── Plot recall vs FP-rate trade-off ─────────────────────────────────
try:
    # Fix ol_floor at recommended value, sweep cropTotalThreshold for plot
    best_ol_floor = best['ol_absolute_floor']
    plot_data = [
        r for r in results
        if abs(r['ol_absolute_floor'] - best_ol_floor) < 0.001
    ]
    plot_data.sort(key=lambda r: r['crop_total_threshold'])

    ct_vals      = [r['crop_total_threshold'] for r in plot_data]
    crop_recalls = [r['crop_recall'] for r in plot_data]
    ol_fp_rates  = [r['ol_false_positive_rate'] for r in plot_data]
    ol_recalls   = [r['ol_recall'] for r in plot_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ct_vals, crop_recalls, 'b-o', label='Supported-crop recall')
    ax1.plot(ct_vals, ol_fp_rates,  'r-o', label='Other-leaf FP rate')
    ax1.axhline(y=MIN_CROP_RECALL, color='blue', linestyle='--', alpha=0.5,
                label=f'Min recall target ({MIN_CROP_RECALL:.0%})')
    ax1.axvline(x=best['crop_total_threshold'], color='green', linestyle='--',
                alpha=0.7, label='Recommended threshold')
    ax1.set_xlabel('cropTotalThreshold')
    ax1.set_ylabel('Rate')
    ax1.set_title(f'Recall vs FP rate (ol_floor={best_ol_floor:.2f})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.02, 1.05)

    ax2.plot(ol_fp_rates, crop_recalls, 'g-o')
    ax2.scatter(
        [best['ol_false_positive_rate']], [best['crop_recall']],
        color='red', s=120, zorder=5, label='Recommended'
    )
    ax2.set_xlabel('Other-leaf false-positive rate')
    ax2.set_ylabel('Supported-crop recall')
    ax2.set_title('ROC-style trade-off curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'Threshold Calibration — Best: cropTotal={best["crop_total_threshold"]}, '
        f'olFloor={best["ol_absolute_floor"]}\n'
        f'Crop recall={best["crop_recall"]:.2%}  OL FP={best["ol_false_positive_rate"]:.2%}'
    )
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "threshold_calibration.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Trade-off plot saved -> {plot_path}", flush=True)
except Exception as e:
    print(f"Could not save plot: {e}", flush=True)

print("\nDONE. Update classifier_service.dart with the recommended thresholds above.", flush=True)