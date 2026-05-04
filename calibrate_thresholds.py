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
from collections import Counter
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
TEMPERATURE            = 1.8    # fallback if no learned temperature is available
OL_THRESHOLD           = 0.30   # otherLeafThreshold (direct softmax winner)
OL_RATIO_THRESHOLD     = 0.30   # otherLeafVsCropRatioThreshold
UNCERTAIN_GAP_THRESHOLD = 0.30  # uncertainGapThreshold
SECOND_CROP_THRESHOLD  = 0.15   # secondCropAmbiguityThreshold (already relaxed)

DEFAULT_BEANS_POTATO_CROP_TOTAL_RELAXATION = 0.05
DEFAULT_BEANS_POTATO_OTHER_LEAF_FLOOR_BOOST = 0.03
DEFAULT_BEANS_POTATO_UNCERTAIN_GAP_THRESHOLD = 0.08
DEFAULT_BEANS_POTATO_SECOND_CROP_THRESHOLD = 0.55
DEFAULT_BEANS_POTATO_CLASS_RATIO_THRESHOLD = 0.45
DEFAULT_BEANS_POTATO_CLASS_CONFIDENCE_THRESHOLD = 0.68
NON_FOCUS_CROP_TOTAL_RELAXATION = 0.05
NON_FOCUS_OTHER_LEAF_FLOOR_BOOST = 0.01
NON_FOCUS_MAX_ENTROPY_THRESHOLD = 1.5
MAX_ENTROPY_THRESHOLD = NON_FOCUS_MAX_ENTROPY_THRESHOLD
NON_FOCUS_CLASS_CONFIDENCE_THRESHOLD = 0.60
HEALTHY_MIN_CONFIDENCE = 0.80
POTATO_HEALTHY_MIN_CONFIDENCE_PILOT = 0.72
FORCE_BEANS_POTATO_NEVER_UNSUPPORTED = True
FORCE_BEANS_POTATO_MIN_CROP_TOTAL = 0.001
FORCE_BEANS_POTATO_MIN_TOP_CLASS_PROB = 0.001
SECONDARY_FOCUS_MIN_CROP_TOTAL = 0.001
SECONDARY_FOCUS_MIN_TOP_CLASS_PROB = 0.001
SECONDARY_FOCUS_MAX_GAP_FROM_BEST = 1.00

RUNTIME_THRESHOLDS_PATH = os.path.join(
    BASE,
    "mobile_app",
    "assets",
    "config",
    "thresholds.json",
)


def _load_runtime_thresholds():
    try:
        with open(RUNTIME_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        thresholds = payload.get("thresholds", {}) if isinstance(payload, dict) else {}
        return thresholds if isinstance(thresholds, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

RUNTIME_THRESHOLDS = _load_runtime_thresholds()
OL_THRESHOLD = float(RUNTIME_THRESHOLDS.get("otherLeafThreshold", OL_THRESHOLD))
OL_RATIO_THRESHOLD = float(RUNTIME_THRESHOLDS.get("otherLeafVsCropRatioThreshold", OL_RATIO_THRESHOLD))

TEST_EVAL_PATH = os.path.join(MODEL_DIR, "test_evaluation.json")

# Sweep ranges — adjust if your validation results cluster outside these
CROP_TOTAL_SWEEP  = np.arange(0.60, 0.92, 0.02)   # cropTotalThreshold
OL_FLOOR_SWEEP    = np.arange(0.08, 0.22, 0.02)   # otherLeafAbsoluteFloor

# Target: at minimum this fraction of real supported-crop images must
# reach a confident result (not be rejected as unsupported/uncertain).
# Tune this to match your product requirements.
MIN_CROP_RECALL = 0.90
MIN_FOCUS_RECALL = 0.92
MIN_BANANA_MAIZE_RECALL = 0.90
MAX_OL_FP_RATE = 0.015
FOCUS_CROPS = {'beans', 'potato'}
BANANA_MAIZE_CROPS = {'banana', 'maize'}
FOCUS_WEIGHT = 0.70
BANANA_MAIZE_WEIGHT = 0.20

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


def _load_runtime_thresholds():
    cfg = {
        "beansPotatoCropTotalRelaxation": DEFAULT_BEANS_POTATO_CROP_TOTAL_RELAXATION,
        "beansPotatoOtherLeafFloorBoost": DEFAULT_BEANS_POTATO_OTHER_LEAF_FLOOR_BOOST,
        "beansPotatoUncertainGapThreshold": DEFAULT_BEANS_POTATO_UNCERTAIN_GAP_THRESHOLD,
        "beansPotatoSecondCropThreshold": DEFAULT_BEANS_POTATO_SECOND_CROP_THRESHOLD,
        "beansPotatoClassRatioThreshold": DEFAULT_BEANS_POTATO_CLASS_RATIO_THRESHOLD,
        "beansPotatoClassConfidenceThreshold": DEFAULT_BEANS_POTATO_CLASS_CONFIDENCE_THRESHOLD,
    }
    if not os.path.isfile(RUNTIME_THRESHOLDS_PATH):
        return cfg

    try:
        with open(RUNTIME_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        values = payload.get("thresholds", {})
        if not isinstance(values, dict):
            return cfg

        for key in cfg:
            raw = values.get(key)
            if isinstance(raw, (int, float)):
                cfg[key] = float(raw)
    except Exception:
        return cfg

    return cfg


def _load_temperature_from_artifacts(default_temperature: float) -> float:
    """Prefer learned temperature from latest evaluation artifacts."""
    if not os.path.isfile(TEST_EVAL_PATH):
        return float(default_temperature)

    try:
        with open(TEST_EVAL_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        temp_block = payload.get("temperature_scaling", {})
        learned = temp_block.get("best_temperature")
        if isinstance(learned, (int, float)) and learned > 0:
            return float(learned)
    except Exception:
        return float(default_temperature)

    return float(default_temperature)


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

label_to_crop = {}
for i, name in enumerate(class_names):
    crop = CROP_GROUPING.get(name)
    if crop is None:
        crop = 'other_leaf' if name == 'other_leaf' else 'unknown'
    label_to_crop[i] = crop


# ── Load validation set ───────────────────────────────────────────────
print("Loading validation set ...", flush=True)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=False,
)
val_class_names = val_ds.class_names
if val_class_names != class_names:
    print("ERROR: val_ds class order differs from labels.txt.", flush=True)
    print(f"  val_ds:     {val_class_names}", flush=True)
    print(f"  labels.txt: {class_names}", flush=True)
    print("  Regenerate model_ready splits and labels.txt before calibration.", flush=True)
    sys.exit(1)

runtime_thresholds = _load_runtime_thresholds()
print(f"Loaded runtime thresholds from {RUNTIME_THRESHOLDS_PATH}", flush=True)
effective_temperature = _load_temperature_from_artifacts(TEMPERATURE)
if os.path.isfile(TEST_EVAL_PATH):
    print(
        f"Loaded learned temperature from {TEST_EVAL_PATH}: T={effective_temperature}",
        flush=True,
    )
else:
    print(
        f"Using fallback temperature (no test_evaluation.json found): T={effective_temperature}",
        flush=True,
    )

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

scaled = temperature_scale(all_probs, effective_temperature)
print(f"Temperature scaling applied (T={effective_temperature}).", flush=True)


# ── Aggregate crop totals ─────────────────────────────────────────────
def get_crop_totals(probs_row):
    """Sum class probabilities per crop, excluding other_leaf."""
    crop_totals = {}
    for i, name in enumerate(class_names):
        crop = CROP_GROUPING.get(name)
        if crop is not None:
            crop_totals[crop] = crop_totals.get(crop, 0.0) + probs_row[i]
    return crop_totals


def top_class_prob_for_crop(probs_row, crop_name):
    top_prob = 0.0
    for i, name in enumerate(class_names):
        if CROP_GROUPING.get(name) == crop_name:
            top_prob = max(top_prob, float(probs_row[i]))
    return top_prob


def top_class_label_for_crop(probs_row, crop_name):
    best_idx = -1
    best_prob = 0.0
    for i, name in enumerate(class_names):
        if CROP_GROUPING.get(name) == crop_name:
            p = float(probs_row[i])
            if p > best_prob:
                best_prob = p
                best_idx = i
    if best_idx < 0:
        return None, 0.0
    return class_names[best_idx], best_prob


def entropy_bits(probs_row):
    p = np.clip(probs_row, 1e-10, 1.0)
    return float(-np.sum(p * (np.log(p) / np.log(2.0))))


def is_beans_or_potato(crop_name):
    return crop_name in FOCUS_CROPS


def choose_focus_crop_candidate(
    beans_total,
    potato_total,
    beans_top_class,
    potato_top_class,
    beans_score,
    potato_score,
):
    if potato_score > beans_score:
        return "potato", potato_total
    if beans_score > potato_score:
        return "beans", beans_total
    if potato_total > beans_total:
        return "potato", potato_total
    if beans_total > potato_total:
        return "beans", beans_total
    if potato_top_class > beans_top_class:
        return "potato", potato_total
    if beans_top_class > potato_top_class:
        return "beans", beans_total
    return None


def select_secondary_focus_candidate(crop_totals, probs_row, best_crop_total):
    beans_total = float(crop_totals.get("beans", 0.0))
    potato_total = float(crop_totals.get("potato", 0.0))
    beans_top = top_class_prob_for_crop(probs_row, "beans")
    potato_top = top_class_prob_for_crop(probs_row, "potato")

    has_evidence = (
        beans_total >= SECONDARY_FOCUS_MIN_CROP_TOTAL
        or potato_total >= SECONDARY_FOCUS_MIN_CROP_TOTAL
        or beans_top >= SECONDARY_FOCUS_MIN_TOP_CLASS_PROB
        or potato_top >= SECONDARY_FOCUS_MIN_TOP_CLASS_PROB
    )
    if not has_evidence:
        return None

    candidate = choose_focus_crop_candidate(
        beans_total=beans_total,
        potato_total=potato_total,
        beans_top_class=beans_top,
        potato_top_class=potato_top,
        beans_score=beans_total + (0.40 * beans_top),
        potato_score=potato_total + (0.40 * potato_top),
    )
    if candidate is None:
        return None

    crop_name, crop_total = candidate
    gap_from_best = max(0.0, best_crop_total - crop_total)
    if gap_from_best > SECONDARY_FOCUS_MAX_GAP_FROM_BEST:
        return None
    return crop_name, crop_total


def should_preserve_focus_crop_identity(candidate_crop, candidate_crop_total, probs_row):
    if not is_beans_or_potato(candidate_crop):
        return False
    top_class_prob = top_class_prob_for_crop(probs_row, candidate_crop)
    return candidate_crop_total >= FORCE_BEANS_POTATO_MIN_CROP_TOTAL or top_class_prob >= FORCE_BEANS_POTATO_MIN_TOP_CLASS_PROB


def build_focus_override_result(gate, crop, crop_total, confidence, second_crop_total, other_leaf_prob, probs_row):
    best_class, best_class_prob = top_class_label_for_crop(probs_row, crop)
    if not best_class:
        best_class = f"{crop}_healthy"
        best_class_prob = 0.0
    return {
        "gate": gate,
        "resultType": "healthy" if "healthy" in best_class else "disease",
        "bestCrop": crop,
        "bestClass": best_class,
        "bestClassProb": float(max(confidence, best_class_prob)),
        "bestCropTotal": float(crop_total),
        "secondCropTotal": float(second_crop_total),
        "otherLeaf": float(other_leaf_prob),
        "entropy": entropy_bits(probs_row),
    }


# ── Simulate app decision for one sample ─────────────────────────────
def simulate_decision(probs_row, crop_total_thresh, ol_floor_thresh):
    """Simulate full runtime gate routing and return gate+result payload."""
    ol_prob = probs_row[OTHER_LEAF_IDX]

    crop_totals = get_crop_totals(probs_row)
    sorted_crops = sorted(crop_totals.items(), key=lambda x: -x[1])
    best_candidate_crop = sorted_crops[0][0] if sorted_crops else "unknown"
    best_candidate_crop_total = sorted_crops[0][1] if sorted_crops else 0.0
    secondary_focus_candidate = select_secondary_focus_candidate(
        crop_totals,
        probs_row,
        best_crop_total=best_candidate_crop_total,
    )

    effective_ol_floor_threshold = ol_floor_thresh
    if is_beans_or_potato(best_candidate_crop):
        effective_ol_floor_threshold = min(
            0.95,
            ol_floor_thresh + runtime_thresholds["beansPotatoOtherLeafFloorBoost"],
        )
    else:
        effective_ol_floor_threshold = min(
            0.95,
            ol_floor_thresh + NON_FOCUS_OTHER_LEAF_FLOOR_BOOST,
        )

    if ol_prob >= OL_THRESHOLD:
        preserve_crop = best_candidate_crop
        preserve_crop_total = best_candidate_crop_total
        if secondary_focus_candidate is not None:
            preserve_crop, preserve_crop_total = secondary_focus_candidate
        if should_preserve_focus_crop_identity(preserve_crop, preserve_crop_total, probs_row):
            return build_focus_override_result(
                gate="G5_preserve_focus_crop",
                crop=preserve_crop,
                crop_total=preserve_crop_total,
                confidence=max(preserve_crop_total, top_class_prob_for_crop(probs_row, preserve_crop)),
                second_crop_total=sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
                other_leaf_prob=ol_prob,
                probs_row=probs_row,
            )
        return {
            "gate": "G5_other_leaf_winner",
            "resultType": "other_leaf",
        }

    if ol_prob > effective_ol_floor_threshold:
        preserve_crop = best_candidate_crop
        preserve_crop_total = best_candidate_crop_total
        if secondary_focus_candidate is not None:
            preserve_crop, preserve_crop_total = secondary_focus_candidate
        if should_preserve_focus_crop_identity(preserve_crop, preserve_crop_total, probs_row):
            return build_focus_override_result(
                gate="G5b_preserve_focus_crop",
                crop=preserve_crop,
                crop_total=preserve_crop_total,
                confidence=max(preserve_crop_total, top_class_prob_for_crop(probs_row, preserve_crop)),
                second_crop_total=sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
                other_leaf_prob=ol_prob,
                probs_row=probs_row,
            )
        return {
            "gate": "G5b_other_leaf_floor",
            "resultType": "other_leaf",
        }

    if not sorted_crops:
        return {
            "gate": "G6_no_crop_candidates",
            "resultType": "unsupported",
        }

    best_crop, best_crop_total = sorted_crops[0]
    second_crop_total = sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0
    crop_gap = best_crop_total - second_crop_total

    effective_crop_total_threshold = crop_total_thresh
    effective_gap_threshold = UNCERTAIN_GAP_THRESHOLD
    effective_second_crop_threshold = SECOND_CROP_THRESHOLD
    effective_class_ratio_threshold = 0.60
    effective_class_confidence_threshold = NON_FOCUS_CLASS_CONFIDENCE_THRESHOLD
    effective_entropy_threshold = NON_FOCUS_MAX_ENTROPY_THRESHOLD
    if is_beans_or_potato(best_crop):
        effective_crop_total_threshold = max(
            0.0,
            crop_total_thresh - runtime_thresholds["beansPotatoCropTotalRelaxation"],
        )
        effective_gap_threshold = runtime_thresholds["beansPotatoUncertainGapThreshold"]
        effective_second_crop_threshold = runtime_thresholds["beansPotatoSecondCropThreshold"]
        effective_class_ratio_threshold = runtime_thresholds["beansPotatoClassRatioThreshold"]
        effective_class_confidence_threshold = runtime_thresholds["beansPotatoClassConfidenceThreshold"]
        effective_entropy_threshold = MAX_ENTROPY_THRESHOLD
    else:
        effective_crop_total_threshold = max(
            0.0,
            crop_total_thresh - NON_FOCUS_CROP_TOTAL_RELAXATION,
        )

    if best_crop_total < effective_crop_total_threshold:
        if (
            is_beans_or_potato(best_crop)
            and FORCE_BEANS_POTATO_NEVER_UNSUPPORTED
            and best_crop_total >= FORCE_BEANS_POTATO_MIN_CROP_TOTAL
            and top_class_prob_for_crop(probs_row, best_crop) >= FORCE_BEANS_POTATO_MIN_TOP_CLASS_PROB
        ):
            return build_focus_override_result(
                gate="G7a_crop_total_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=top_class_prob_for_crop(probs_row, best_crop),
                second_crop_total=second_crop_total,
                other_leaf_prob=ol_prob,
                probs_row=probs_row,
            )
        return {
            "gate": "G7a_crop_total",
            "resultType": "unsupported",
        }

    ol_ratio = ol_prob / best_crop_total if best_crop_total > 0 else 0.0
    if ol_ratio > OL_RATIO_THRESHOLD:
        if (
            is_beans_or_potato(best_crop)
            and FORCE_BEANS_POTATO_NEVER_UNSUPPORTED
            and best_crop_total >= FORCE_BEANS_POTATO_MIN_CROP_TOTAL
            and top_class_prob_for_crop(probs_row, best_crop) >= FORCE_BEANS_POTATO_MIN_TOP_CLASS_PROB
        ):
            return build_focus_override_result(
                gate="G7a_other_leaf_ratio_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=top_class_prob_for_crop(probs_row, best_crop),
                second_crop_total=second_crop_total,
                other_leaf_prob=ol_prob,
                probs_row=probs_row,
            )
        return {
            "gate": "G7a_other_leaf_ratio",
            "resultType": "other_leaf",
        }

    best_class, best_class_prob = top_class_label_for_crop(probs_row, best_crop)
    if not best_class:
        return {
            "gate": "G6_no_crop_candidates",
            "resultType": "unsupported",
        }

    if crop_gap < effective_gap_threshold:
        if is_beans_or_potato(best_crop):
            return build_focus_override_result(
                gate="G7c_crop_gap_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=best_class_prob,
                second_crop_total=second_crop_total,
                other_leaf_prob=ol_prob,
                probs_row=probs_row,
            )
        return {
            "gate": "G7c_crop_gap",
            "resultType": "uncertain",
        }

    if second_crop_total > effective_second_crop_threshold:
        if is_beans_or_potato(best_crop):
            return build_focus_override_result(
                gate="G7c_second_crop_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=best_class_prob,
                second_crop_total=second_crop_total,
                other_leaf_prob=ol_prob,
                probs_row=probs_row,
            )
        return {
            "gate": "G7c_second_crop",
            "resultType": "uncertain",
        }

    e = entropy_bits(probs_row)
    if e > effective_entropy_threshold:
        return {
            "gate": "G7c_entropy",
            "resultType": "uncertain",
        }

    class_ratio = best_class_prob / best_crop_total if best_crop_total > 0 else 0.0
    if class_ratio < effective_class_ratio_threshold:
        return {
            "gate": "G7c_class_ratio",
            "resultType": "uncertain",
        }

    if best_class_prob < effective_class_confidence_threshold:
        if is_beans_or_potato(best_crop):
            return build_focus_override_result(
                gate="G7d_class_confidence_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=best_class_prob,
                second_crop_total=second_crop_total,
                other_leaf_prob=ol_prob,
                probs_row=probs_row,
            )
        return {
            "gate": "G7d_class_confidence",
            "resultType": "uncertain",
        }

    if "healthy" in best_class:
        healthy_confidence_threshold = (
            POTATO_HEALTHY_MIN_CONFIDENCE_PILOT
            if best_crop == "potato"
            else HEALTHY_MIN_CONFIDENCE
        )
        if best_class_prob < healthy_confidence_threshold:
            if is_beans_or_potato(best_crop):
                return build_focus_override_result(
                    gate="G7e_healthy_safety_focus_override",
                    crop=best_crop,
                    crop_total=best_crop_total,
                    confidence=best_class_prob,
                    second_crop_total=second_crop_total,
                    other_leaf_prob=ol_prob,
                    probs_row=probs_row,
                )
            return {
                "gate": "G7e_healthy_safety",
                "resultType": "uncertain",
            }

    return {
        "gate": "G7g_confident",
        "resultType": "healthy" if "healthy" in best_class else "disease",
    }


# ── Masks for supported vs other_leaf samples ─────────────────────────
is_other_leaf = (all_labels == OTHER_LEAF_IDX)
is_supported  = ~is_other_leaf
n_supported   = int(is_supported.sum())
n_other_leaf  = int(is_other_leaf.sum())
sample_crops = np.array([label_to_crop[int(i)] for i in all_labels])
is_focus = np.array([c in FOCUS_CROPS for c in sample_crops])
is_focus_supported = is_supported & is_focus
n_focus_supported = int(is_focus_supported.sum())
is_banana_maize = np.array([c in BANANA_MAIZE_CROPS for c in sample_crops])
is_banana_maize_supported = is_supported & is_banana_maize
n_banana_maize_supported = int(is_banana_maize_supported.sum())
print(f"Validation breakdown: {n_supported} supported-crop, {n_other_leaf} other_leaf", flush=True)
print(f"Focus subset ({sorted(FOCUS_CROPS)}): {n_focus_supported} supported-crop samples", flush=True)
print(f"Banana/Maize subset ({sorted(BANANA_MAIZE_CROPS)}): {n_banana_maize_supported} supported-crop samples", flush=True)


# ── Sweep cropTotalThreshold × otherLeafAbsoluteFloor ────────────────
print(f"\nSweeping {len(CROP_TOTAL_SWEEP)} × {len(OL_FLOOR_SWEEP)} threshold combinations ...", flush=True)

results = []
for ct in CROP_TOTAL_SWEEP:
    for ol_floor in OL_FLOOR_SWEEP:
        decisions = [
            simulate_decision(scaled[i], ct, ol_floor)
            for i in range(len(all_labels))
        ]
        result_types = np.array([d["resultType"] for d in decisions])
        gate_counts = Counter(d["gate"] for d in decisions)
        accepted_mask = np.isin(result_types, ["healthy", "disease"])

        # Crop recall: fraction of real supported-crop images that reach 'accepted'
        crop_recall = float(np.mean(accepted_mask[is_supported])) \
                  if n_supported > 0 else 0.0

        # Focus recall: beans/potato supported-crop acceptance rate.
        focus_recall = float(np.mean(accepted_mask[is_focus_supported])) \
                   if n_focus_supported > 0 else 0.0

        # Other-leaf false-positive rate: fraction of real other_leaf images
        # that were 'accepted' as a real crop (the bug we are trying to fix)
        ol_fp_rate = float(np.mean(accepted_mask[is_other_leaf])) \
                     if n_other_leaf > 0 else 0.0

        # Other-leaf recall: fraction of other_leaf images correctly rejected
        ol_recall = float(np.mean(~accepted_mask[is_other_leaf])) \
                    if n_other_leaf > 0 else 0.0

        banana_maize_recall = float(np.mean(accepted_mask[is_banana_maize_supported])) \
                             if n_banana_maize_supported > 0 else 0.0
        meets_banana_maize = (
            True if n_banana_maize_supported == 0 else banana_maize_recall >= MIN_BANANA_MAIZE_RECALL
        )

        results.append({
            "crop_total_threshold":      round(float(ct), 4),
            "ol_absolute_floor":         round(float(ol_floor), 4),
            "crop_recall":               round(crop_recall, 4),
            "focus_recall":              round(focus_recall, 4),
            "banana_maize_recall":       round(banana_maize_recall, 4),
            "ol_false_positive_rate":    round(ol_fp_rate, 4),
            "ol_recall":                 round(ol_recall, 4),
            "meets_min_crop_recall":     crop_recall >= MIN_CROP_RECALL,
            "meets_min_focus_recall":    focus_recall >= MIN_FOCUS_RECALL,
            "meets_min_banana_maize_recall": meets_banana_maize,
            "meets_ol_fp_target":        ol_fp_rate <= MAX_OL_FP_RATE,
            "gate_counts":               dict(gate_counts),
        })

print(f"Sweep complete. {len(results)} combinations evaluated.", flush=True)


# ── Find best combination ─────────────────────────────────────────────
# Balanced objective prioritises beans/potato and banana/maize recall while
# constraining unsupported risk from other_leaf false positives.
def balanced_score(r):
    return (
        FOCUS_WEIGHT * r["focus_recall"] +
        BANANA_MAIZE_WEIGHT * r["banana_maize_recall"] +
        (1.0 - FOCUS_WEIGHT - BANANA_MAIZE_WEIGHT) * r["crop_recall"] -
        2.0 * r["ol_false_positive_rate"]
    )

strict_valid = [
    r for r in results
    if r["meets_min_crop_recall"] and
       r["meets_min_focus_recall"] and
       r["meets_min_banana_maize_recall"] and
       r["meets_ol_fp_target"]
]

if strict_valid:
    best = max(
        strict_valid,
        key=lambda r: (balanced_score(r), r["crop_total_threshold"], r["ol_absolute_floor"]),
    )
else:
    valid = [r for r in results if r["meets_min_crop_recall"]]
    if valid:
        print(
            "WARNING: No combination met all strict balanced constraints. "
            "Falling back to crop-recall-constrained optimum.",
            flush=True,
        )
        best = max(
            valid,
            key=lambda r: (balanced_score(r), r["crop_total_threshold"], r["ol_absolute_floor"]),
        )
    else:
        print(
            f"WARNING: No combination achieved crop_recall ≥ {MIN_CROP_RECALL:.0%}. "
            "Showing best available by balanced score.",
            flush=True,
        )
        best = max(results, key=balanced_score)

print(f"\n{'='*60}", flush=True)
print(f"RECOMMENDED THRESHOLDS", flush=True)
print(f"{'='*60}", flush=True)
print(f"  cropTotalThreshold:       {best['crop_total_threshold']}", flush=True)
print(f"  otherLeafAbsoluteFloor:   {best['ol_absolute_floor']}", flush=True)
print(f"  Supported-crop recall:    {best['crop_recall']:.2%}", flush=True)
print(f"  Beans/Potato recall:      {best['focus_recall']:.2%}", flush=True)
print(f"  Banana/Maize recall:      {best['banana_maize_recall']:.2%}", flush=True)
print(f"  Other-leaf FP rate:       {best['ol_false_positive_rate']:.2%}", flush=True)
print(f"  Other-leaf recall:        {best['ol_recall']:.2%}", flush=True)
print(f"  Meets recall target (≥{MIN_CROP_RECALL:.0%}): {best['meets_min_crop_recall']}", flush=True)
print(f"  Meets focus recall (≥{MIN_FOCUS_RECALL:.0%}): {best['meets_min_focus_recall']}", flush=True)
print(f"  Meets banana/maize target (≥{MIN_BANANA_MAIZE_RECALL:.0%}): {best['meets_min_banana_maize_recall']}", flush=True)
print(f"  Meets other_leaf FP target (≤{MAX_OL_FP_RATE:.2%}): {best['meets_ol_fp_target']}", flush=True)
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
        "min_focus_recall_target": MIN_FOCUS_RECALL,
        "min_banana_maize_recall_target": MIN_BANANA_MAIZE_RECALL,
        "max_other_leaf_fp_target": MAX_OL_FP_RATE,
        "focus_crops": sorted(list(FOCUS_CROPS)),
        "banana_maize_crops": sorted(list(BANANA_MAIZE_CROPS)),
        "focus_weight": FOCUS_WEIGHT,
        "banana_maize_weight": BANANA_MAIZE_WEIGHT,
        "temperature": effective_temperature,
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