# ...existing code for training/evaluation...
"""
Train MobileNetV2 model for 4-crop disease detection (15 classes).
Crops: Banana, Beans, Maize, Potato  +  other_leaf (rejection class)

Changes vs previous version (Fix D — retrain improvements):
  1. FocalLoss replaces sparse_categorical_crossentropy — forces the model
     to focus on hard lookalike cases instead of easy correct ones.
  2. Moderate class weighting: other_leaf remains boosted for rejection,
      and beans/potato receive a smaller focus boost to reduce under-confident
      supported-crop outcomes.
  3. Class-aware augmentation: other_leaf images get heavier augmentation
     (stronger rotation, zoom, contrast, brightness) to improve generalisation
     across unseen plant types that were never in training.
  4. MixUp-style interpolation is optional (enabled by default in this
      config, can be disabled for stability-first retrains).
  5. NEW — Per-class label smoothing for other_leaf inside FocalLoss.
     Instead of pushing other_leaf probability to a hard 1.0 target, we
     use 0.9 (smoothing ε=0.1). This prevents the model from becoming
     overconfident about the specific other_leaf training images, making
     the learned boundary more robust to unseen plant types.

Run: python -u -X utf8 scripts/train_model.py
"""

import os, sys, json, gc, time, csv, random, math
import hashlib
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '42'

print("Loading TensorFlow ...", flush=True)
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from PIL import Image, UnidentifiedImageError

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 8 GB RAM-safe runtime defaults.
DATA_AUTOTUNE = False
PARALLEL_CALLS = 2
PREFETCH_BUFFER = 2
INTER_OP_THREADS = 2
INTRA_OP_THREADS = 4

try:
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
except Exception:
    # Some TensorFlow builds lock threading config after initialization.
    pass

# ── Config ──────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE, "datasets", "model_ready")
MODEL_DIR   = os.path.join(BASE, "models")
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "val")
TEST_DIR    = os.path.join(DATA_DIR, "test")
FIELD_FAILURE_DIR = os.path.join(DATA_DIR, "field_failures")
IMG_SIZE    = (224, 224)
BATCH       = 8
PHASE1_EP   = 15          # frozen backbone
PHASE2_EP   = 35          # fine-tune
FIELD_FAILURE_MAX_SAMPLES = 200
FIELD_FAILURE_SHUFFLE_BUFFER = 2048
INIT_LR     = 1e-3
FINE_LR     = 1e-5
PATIENCE    = 7
MAX_CLASS_WEIGHT = 4.0

# Stability-first defaults: keep the recipe interpretable.
ENABLE_OL_MIXUP = True
ENABLE_OL_LABEL_SMOOTHING = True
ENABLE_FOCUS_CROP_WEIGHTING = True

# Keep other_leaf strong enough for rejection, but avoid over-penalising
# supported crops into uncertain/unsupported outcomes.
OTHER_LEAF_WEIGHT_MULTIPLIER = 3.0

# Boost beans/potato class gradients to reduce under-confident supported
# predictions on focus crops without changing architecture.
FOCUS_CROP_WEIGHT_MULTIPLIER = 1.15

# Prefer unfreezing by depth-from-end over hard layer index.
USE_LAST_N_UNFREEZE = True
UNFREEZE_LAST_BASE_LAYERS = 60
FINE_TUNE_FROM = 80       # fallback when USE_LAST_N_UNFREEZE=False

# Standard augmentation profile (crop disease classes — unchanged)
AUG_ROTATION   = 0.05
AUG_ZOOM       = 0.10
AUG_CONTRAST   = 0.20
AUG_BRIGHTNESS = 0.20

# Heavier augmentation profile for other_leaf only (slightly reduced to
# avoid unrealistic synthetic leaves).
OL_AUG_ROTATION   = 0.15
OL_AUG_ZOOM       = 0.15
OL_AUG_CONTRAST   = 0.40
OL_AUG_BRIGHTNESS = 0.30

# FIX D4 — MixUp parameters for other_leaf
# alpha controls the Beta distribution used to sample the mix ratio λ.
# alpha=0.4 gives a distribution centred near 0.5 (roughly equal mix)
# with enough variance to produce both subtle and strong blends.
# Higher alpha (e.g. 0.6) → blends closer to 50/50 (harder boundary).
# Lower alpha (e.g. 0.2) → blends closer to the originals (weaker effect).
OL_MIXUP_ALPHA = 0.4

# FIX D5 — Label smoothing for other_leaf inside FocalLoss
# Instead of a hard target of 1.0 for other_leaf, we use (1 - ε).
# The remaining ε is spread uniformly across all classes.
# ε=0.1 is the standard value (Szegedy et al., "Rethinking the Inception
# Architecture"). This prevents the model from becoming overconfident on
# the specific other_leaf training images — the learned boundary generalises
# better to unseen lookalike plants.
OL_LABEL_SMOOTHING = 0.10

# Data quality and analysis extras
ENABLE_IMAGE_SANITY_CHECK = True
MAX_HARD_NEGATIVES_EXPORT = 250
ENABLE_GRAD_CAM_EXPORT = False
GRAD_CAM_MAX_IMAGES = 12

# Strict release policy (approved):
# 1) Field supported-confident rate per crop >= 80%
# 2) Beans<->Potato confusion <= 2%
# 3) Supported<->Unsupported confusion <= 2%
STRICT_FIELD_MIN_SUPPORTED_CONFIDENT_RATE = 0.80
STRICT_BEANS_POTATO_CONFUSION_MAX = 0.02
STRICT_SUPPORTED_UNSUPPORTED_CONFUSION_MAX = 0.02

# Fail training early if split leakage is detected. This avoids optimistic
# metrics from duplicate/overlapping samples leaking across train/val/test.
ENFORCE_LEAKAGE_GUARD = True
MAX_ALLOWED_STEM_OVERLAP_TOTAL = 0
MAX_ALLOWED_HASH_OVERLAP_TOTAL = 0

CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]
DEFAULT_APP_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_APP_MARGIN_THRESHOLD = 0.20
DEFAULT_APP_MAX_ENTROPY = 1.50
DEFAULT_APP_CROP_TOTAL_THRESHOLD = 0.82
DEFAULT_HEALTHY_MIN_CONFIDENCE = 0.80
DEFAULT_POTATO_HEALTHY_MIN_CONFIDENCE_PILOT = 0.72
CROP_TOTAL_THRESHOLDS = [0.70, 0.74, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]

THRESHOLDS_CONFIG_PATH = os.path.join(BASE, "mobile_app", "assets", "config", "thresholds.json")
MOBILE_RUNTIME_RECO_PATH = os.path.join(BASE, "mobile_app", "assets", "config", "mobile_runtime_recommendations.json")

DEFAULT_OTHER_LEAF_THRESHOLD = 0.30
DEFAULT_OTHER_LEAF_VS_CROP_RATIO_THRESHOLD = 0.50
DEFAULT_OTHER_LEAF_ABSOLUTE_FLOOR = 0.12
DEFAULT_UNCERTAIN_GAP_THRESHOLD = 0.30
DEFAULT_SECOND_CROP_AMBIGUITY_THRESHOLD = 0.15
DEFAULT_NON_FOCUS_CROP_TOTAL_RELAXATION = 0.05
DEFAULT_NON_FOCUS_OTHER_LEAF_FLOOR_BOOST = 0.01
DEFAULT_NON_FOCUS_CLASS_RATIO_THRESHOLD = 0.60
DEFAULT_BEANS_POTATO_RESCUE_MIN_CROP_TOTAL = 0.18
DEFAULT_BEANS_POTATO_RESCUE_MAX_GAP_FROM_BEST = 0.30
DEFAULT_BEANS_POTATO_RESCUE_MAX_OTHER_LEAF = 0.38
DEFAULT_RESCUE_FOCUS_SWAP_GUARD_CROP_GAP = 0.08
DEFAULT_RESCUE_FOCUS_SWAP_GUARD_TOP_CLASS_MARGIN = 0.05
DEFAULT_FORCE_BEANS_POTATO_NEVER_UNSUPPORTED = True
DEFAULT_FORCE_BEANS_POTATO_MIN_CROP_TOTAL = 0.001
DEFAULT_FORCE_BEANS_POTATO_MIN_TOP_CLASS_PROB = 0.001
DEFAULT_LAST_CHANCE_FOCUS_MIN_CROP_TOTAL = 0.001
DEFAULT_LAST_CHANCE_FOCUS_MIN_TOP_CLASS_PROB = 0.001
DEFAULT_BEANS_POTATO_CROP_TOTAL_RELAXATION = 0.05
DEFAULT_BEANS_POTATO_OTHER_LEAF_FLOOR_BOOST = 0.03
DEFAULT_BEANS_POTATO_UNCERTAIN_GAP_THRESHOLD = 0.08
DEFAULT_BEANS_POTATO_SECOND_CROP_THRESHOLD = 0.55
DEFAULT_BEANS_POTATO_CLASS_RATIO_THRESHOLD = 0.45
DEFAULT_BEANS_POTATO_CLASS_CONFIDENCE_THRESHOLD = 0.68
DEFAULT_SECONDARY_FOCUS_MIN_CROP_TOTAL = 0.001
DEFAULT_SECONDARY_FOCUS_MIN_TOP_CLASS_PROB = 0.001
DEFAULT_SECONDARY_FOCUS_MAX_GAP_FROM_BEST = 1.00
DEFAULT_PREFER_BEANS_POTATO_WHEN_COMPETITIVE = True
DEFAULT_FOCUS_CROP_SWITCH_MAX_GAP = 0.18
DEFAULT_FOCUS_CROP_SWITCH_MIN_TOTAL = 0.20

# Optional real-world OOD validation folder.
REAL_WORLD_OOD_DIR = os.path.join(BASE, "datasets", "real_world_test")

ALLOWED_CROPS = {"banana", "beans", "maize", "potato", "other_leaf"}
MULTI_WORD_PREFIXES = ["other_leaf"]

os.makedirs(MODEL_DIR, exist_ok=True)

MAP_PARALLEL_CALLS = tf.data.AUTOTUNE if DATA_AUTOTUNE else PARALLEL_CALLS
PREFETCH_SIZE = tf.data.AUTOTUNE if DATA_AUTOTUNE else PREFETCH_BUFFER


# ── FIX 1 + FIX D5: Focal Loss with per-class label smoothing ───────
# Focal loss down-weights easy examples and up-weights hard ones.
# gamma=2.0 is the standard value from the original paper.
# alpha=0.25 slightly reduces contribution from dominant easy classes.
#
from tensorflow.keras.utils import register_keras_serializable

# FIX D5 addition: if other_leaf_idx and num_classes are provided, the
# loss applies label smoothing (ε = other_leaf_smoothing) to other_leaf
# samples only. For those samples the effective target is:
#
#   pt_effective = (1-ε)*pt  +  ε/N
#
# where N = num_classes. The (1-pt)^gamma focal weight is computed from
# pt_effective, so the gradient penalises overconfident other_leaf
# predictions less harshly — preventing the model from overfitting to
# specific other_leaf training images.
@register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        gamma=2.0,
        alpha=0.25,
        other_leaf_idx=None,
        num_classes=None,
        other_leaf_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma               = gamma
        self.alpha               = alpha
        self.other_leaf_idx      = other_leaf_idx
        self.num_classes         = num_classes
        self.other_leaf_smoothing = other_leaf_smoothing

    def call(self, y_true, y_pred):
        # y_true: integer class indices, shape (batch,) or (batch,1)
        # y_pred: softmax probabilities, shape (batch, num_classes)
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred     = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Probability assigned to the true class for each sample
        pt = tf.gather(y_pred, y_true_int, batch_dims=1)

        # FIX D5: apply label smoothing to other_leaf samples only
        if (
            self.other_leaf_idx is not None
            and self.num_classes is not None
            and self.other_leaf_smoothing > 0.0
        ):
            eps = self.other_leaf_smoothing
            n   = tf.cast(self.num_classes, tf.float32)

            # Boolean mask: 1.0 for other_leaf samples, 0.0 for everything else
            is_other = tf.cast(
                tf.equal(y_true_int, self.other_leaf_idx), tf.float32
            )

            # Smoothed probability for the true class:
            #   other_leaf sample:  (1-ε)*pt + ε/N
            #   any other sample:   pt  (unchanged)
            pt = pt * (1.0 - is_other * eps) + is_other * (eps / n)

        # Standard cross-entropy for the (possibly smoothed) true-class prob
        # Note: this intentionally uses smoothed pt for other_leaf samples,
        # softening both CE and focal weight for that class.
        ce = -tf.math.log(tf.clip_by_value(pt, 1e-7, 1.0))

        # Focal weight: (1-pt)^gamma
        # Large when the model is wrong, near-zero when already confident
        focal = self.alpha * tf.pow(1.0 - pt, self.gamma) * ce
        return tf.reduce_mean(focal)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma":                self.gamma,
            "alpha":                self.alpha,
            "other_leaf_idx":       self.other_leaf_idx,
            "num_classes":          self.num_classes,
            "other_leaf_smoothing": self.other_leaf_smoothing,
        })
        return config


# ── Helper: extract crop prefix from folder name ────────────────────
def get_crop_prefix(folder_name):
    for prefix in MULTI_WORD_PREFIXES:
        if folder_name.startswith(prefix):
            return prefix
    return folder_name.split("_")[0]


def run_image_sanity_check(split_dir, split_name):
    """
    Detect broken images and report non-RGB channel modes before training.
    This catches data issues that can silently degrade calibration quality.
    """
    broken = []
    non_rgb = []
    total = 0

    for folder in sorted(os.listdir(split_dir)):
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            total += 1
            try:
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    if img.mode != 'RGB':
                        non_rgb.append({"file": fpath, "mode": img.mode})
            except Exception as exc:
                broken.append({"file": fpath, "error": str(exc)})

    print(
        f"Image sanity [{split_name}]: total={total}, broken={len(broken)}, non_rgb={len(non_rgb)}",
        flush=True,
    )

    return {
        "split": split_name,
        "total_images": total,
        "broken_count": len(broken),
        "non_rgb_count": len(non_rgb),
        "broken_examples": broken[:50],
        "non_rgb_examples": non_rgb[:50],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or re-save model with custom loss registration.")
    parser.add_argument("--resave", action="store_true", help="Re-save the model with updated custom loss registration.")
    args = parser.parse_args()

    if args.resave:
        print("[INFO] Loading and re-saving model with registered FocalLoss ...")
        model_path = os.path.join(MODEL_DIR, "best_model.keras")
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, custom_objects={"FocalLoss": FocalLoss})
        model.save(model_path)
        print(f"[INFO] Model re-saved to {model_path}")
        exit(0)
    # Default: run full training pipeline (not invoked by resave)


def temperature_scale_probs(probs, temperature):
    # Convert probabilities back to logits for temperature scaling.
    # Clipping avoids log(0) for near-zero probabilities.
    logits = np.log(np.clip(probs, 1e-10, 1.0)) / max(temperature, 1e-6)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)


def nll_from_probs(probs, labels):
    idx = np.arange(len(labels))
    p_true = np.clip(probs[idx, labels], 1e-10, 1.0)
    return float(-np.mean(np.log(p_true)))


def fit_temperature_from_probs(val_probs, val_labels):
    temps = np.arange(0.8, 3.01, 0.1)
    best_t = 1.0
    best_nll = nll_from_probs(val_probs, val_labels)

    for t in temps:
        scaled = temperature_scale_probs(val_probs, t)
        nll = nll_from_probs(scaled, val_labels)
        if nll < best_nll:
            best_nll = nll
            best_t = float(round(t, 2))

    return best_t, best_nll


def label_to_crop(label_name):
    if label_name == "other_leaf":
        return "other_leaf"
    return label_name.split("_")[0]


def aggregate_supported_crop_totals(probs, class_names):
    supported_crops = sorted(
        {label_to_crop(name) for name in class_names if label_to_crop(name) != "other_leaf"}
    )
    crop_to_idx = {crop: i for i, crop in enumerate(supported_crops)}
    crop_totals = np.zeros((probs.shape[0], len(supported_crops)), dtype=np.float32)

    for class_idx, class_name in enumerate(class_names):
        crop = label_to_crop(class_name)
        if crop == "other_leaf":
            continue
        crop_totals[:, crop_to_idx[crop]] += probs[:, class_idx]

    return supported_crops, crop_totals


def evaluate_crop_total_gate(crop_totals, true_crop_names, supported_crops, threshold):
    row_idx = np.arange(crop_totals.shape[0])
    best_crop_idx = np.argmax(crop_totals, axis=1)
    best_crop_total = crop_totals[row_idx, best_crop_idx]
    best_crop_names = np.array([supported_crops[int(i)] for i in best_crop_idx], dtype=object)

    true_crop_arr = np.array(true_crop_names, dtype=object)
    accepted_mask = best_crop_total >= threshold
    supported_mask = true_crop_arr != "other_leaf"
    unsupported_mask = ~supported_mask

    supported_total = int(np.sum(supported_mask))
    unsupported_total = int(np.sum(unsupported_mask))

    supported_confident = accepted_mask & supported_mask & (best_crop_names == true_crop_arr)
    supported_false_unsupported = supported_mask & ~accepted_mask
    unsupported_rejected = unsupported_mask & ~accepted_mask
    unsupported_accepted = unsupported_mask & accepted_mask

    supported_confident_rate = (
        float(np.mean(supported_confident[supported_mask])) if supported_total > 0 else 0.0
    )
    supported_false_unsupported_rate = (
        float(np.mean(supported_false_unsupported[supported_mask])) if supported_total > 0 else 0.0
    )
    unsupported_reject_rate = (
        float(np.mean(unsupported_rejected[unsupported_mask])) if unsupported_total > 0 else 0.0
    )
    unsupported_accept_rate = (
        float(np.mean(unsupported_accepted[unsupported_mask])) if unsupported_total > 0 else 0.0
    )

    score = 0.5 * supported_confident_rate + 0.5 * unsupported_reject_rate

    return {
        "threshold": round(float(threshold), 4),
        "supported_confident_rate": round(float(supported_confident_rate), 4),
        "supported_false_unsupported_rate": round(float(supported_false_unsupported_rate), 4),
        "unsupported_reject_rate": round(float(unsupported_reject_rate), 4),
        "unsupported_accept_rate": round(float(unsupported_accept_rate), 4),
        "supported_confident_count": int(np.sum(supported_confident)),
        "supported_false_unsupported_count": int(np.sum(supported_false_unsupported)),
        "unsupported_reject_count": int(np.sum(unsupported_rejected)),
        "unsupported_accept_count": int(np.sum(unsupported_accepted)),
        "supported_total": supported_total,
        "unsupported_total": unsupported_total,
        "score": round(float(score), 4),
    }


def load_runtime_gate_config():
    cfg = {
        "cropTotalThreshold": DEFAULT_APP_CROP_TOTAL_THRESHOLD,
        "otherLeafThreshold": DEFAULT_OTHER_LEAF_THRESHOLD,
        "otherLeafVsCropRatioThreshold": DEFAULT_OTHER_LEAF_VS_CROP_RATIO_THRESHOLD,
        "otherLeafAbsoluteFloor": DEFAULT_OTHER_LEAF_ABSOLUTE_FLOOR,
        "uncertainGapThreshold": DEFAULT_UNCERTAIN_GAP_THRESHOLD,
        "secondCropAmbiguityThreshold": DEFAULT_SECOND_CROP_AMBIGUITY_THRESHOLD,
        "maxEntropyThreshold": DEFAULT_APP_MAX_ENTROPY,
        "nonFocusMaxEntropyThreshold": DEFAULT_APP_MAX_ENTROPY,
        "nonFocusClassConfidenceThreshold": DEFAULT_APP_CONFIDENCE_THRESHOLD,
        "nonFocusCropTotalRelaxation": DEFAULT_NON_FOCUS_CROP_TOTAL_RELAXATION,
        "nonFocusOtherLeafFloorBoost": DEFAULT_NON_FOCUS_OTHER_LEAF_FLOOR_BOOST,
        "nonFocusClassRatioThreshold": DEFAULT_NON_FOCUS_CLASS_RATIO_THRESHOLD,
        "beansPotatoRescueMinCropTotal": DEFAULT_BEANS_POTATO_RESCUE_MIN_CROP_TOTAL,
        "beansPotatoRescueMaxGapFromBest": DEFAULT_BEANS_POTATO_RESCUE_MAX_GAP_FROM_BEST,
        "beansPotatoRescueMaxOtherLeaf": DEFAULT_BEANS_POTATO_RESCUE_MAX_OTHER_LEAF,
        "rescueFocusSwapGuardCropGap": DEFAULT_RESCUE_FOCUS_SWAP_GUARD_CROP_GAP,
        "rescueFocusSwapGuardTopClassMargin": DEFAULT_RESCUE_FOCUS_SWAP_GUARD_TOP_CLASS_MARGIN,
        "forceBeansPotatoNeverUnsupported": 1.0 if DEFAULT_FORCE_BEANS_POTATO_NEVER_UNSUPPORTED else 0.0,
        "forceBeansPotatoMinCropTotal": DEFAULT_FORCE_BEANS_POTATO_MIN_CROP_TOTAL,
        "forceBeansPotatoMinTopClassProb": DEFAULT_FORCE_BEANS_POTATO_MIN_TOP_CLASS_PROB,
        "lastChanceFocusMinCropTotal": DEFAULT_LAST_CHANCE_FOCUS_MIN_CROP_TOTAL,
        "lastChanceFocusMinTopClassProb": DEFAULT_LAST_CHANCE_FOCUS_MIN_TOP_CLASS_PROB,
        "beansPotatoCropTotalRelaxation": DEFAULT_BEANS_POTATO_CROP_TOTAL_RELAXATION,
        "beansPotatoOtherLeafFloorBoost": DEFAULT_BEANS_POTATO_OTHER_LEAF_FLOOR_BOOST,
        "beansPotatoUncertainGapThreshold": DEFAULT_BEANS_POTATO_UNCERTAIN_GAP_THRESHOLD,
        "beansPotatoSecondCropThreshold": DEFAULT_BEANS_POTATO_SECOND_CROP_THRESHOLD,
        "beansPotatoClassRatioThreshold": DEFAULT_BEANS_POTATO_CLASS_RATIO_THRESHOLD,
        "beansPotatoClassConfidenceThreshold": DEFAULT_BEANS_POTATO_CLASS_CONFIDENCE_THRESHOLD,
        "secondaryFocusMinCropTotal": DEFAULT_SECONDARY_FOCUS_MIN_CROP_TOTAL,
        "secondaryFocusMinTopClassProb": DEFAULT_SECONDARY_FOCUS_MIN_TOP_CLASS_PROB,
        "secondaryFocusMaxGapFromBest": DEFAULT_SECONDARY_FOCUS_MAX_GAP_FROM_BEST,
        "preferBeansPotatoWhenCompetitive": 1.0 if DEFAULT_PREFER_BEANS_POTATO_WHEN_COMPETITIVE else 0.0,
        "focusCropSwitchMaxGap": DEFAULT_FOCUS_CROP_SWITCH_MAX_GAP,
        "focusCropSwitchMinTotal": DEFAULT_FOCUS_CROP_SWITCH_MIN_TOTAL,
    }

    if os.path.isfile(THRESHOLDS_CONFIG_PATH):
        with open(THRESHOLDS_CONFIG_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        thresholds = payload.get("thresholds", {}) if isinstance(payload, dict) else {}
        for k in [
            "cropTotalThreshold",
            "otherLeafVsCropRatioThreshold",
            "otherLeafAbsoluteFloor",
            "beansPotatoCropTotalRelaxation",
            "beansPotatoOtherLeafFloorBoost",
            "beansPotatoUncertainGapThreshold",
            "beansPotatoSecondCropThreshold",
            "beansPotatoClassRatioThreshold",
            "beansPotatoClassConfidenceThreshold",
            "nonFocusClassRatioThreshold",
            "beansPotatoRescueMinCropTotal",
            "beansPotatoRescueMaxGapFromBest",
            "beansPotatoRescueMaxOtherLeaf",
        ]:
            v = thresholds.get(k)
            if isinstance(v, (int, float)):
                cfg[k] = float(v)

    if os.path.isfile(MOBILE_RUNTIME_RECO_PATH):
        with open(MOBILE_RUNTIME_RECO_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        reco = payload.get("recommendedThresholds", {}) if isinstance(payload, dict) else {}
        for src, dst in [
            ("confidenceThreshold", "nonFocusClassConfidenceThreshold"),
            ("maxEntropyThreshold", "nonFocusMaxEntropyThreshold"),
            ("cropTotalThreshold", "cropTotalThreshold"),
        ]:
            v = reco.get(src)
            if isinstance(v, (int, float)):
                cfg[dst] = float(v)

    return cfg


def simulate_runtime_gates(raw_probs, calibrated_probs, labels, class_names, gate_cfg):
    class_to_crop = {name: label_to_crop(name) for name in class_names}

    def is_beans_or_potato(crop_name):
        return crop_name in ("beans", "potato")

    def top_class_for_crop(crop_name, prob_map):
        best_name = ""
        best_prob = 0.0
        for cname in class_names:
            if class_to_crop[cname] == crop_name:
                p = float(prob_map.get(cname, 0.0))
                if p > best_prob:
                    best_prob = p
                    best_name = cname
        return best_name, best_prob

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

    def select_beans_potato_rescue_candidate(crop_probs, prob_map, best_crop_total):
        beans_total = float(crop_probs.get("beans", 0.0))
        potato_total = float(crop_probs.get("potato", 0.0))
        _, beans_top = top_class_for_crop("beans", prob_map)
        _, potato_top = top_class_for_crop("potato", prob_map)

        candidate = choose_focus_crop_candidate(
            beans_total=beans_total,
            potato_total=potato_total,
            beans_top_class=beans_top,
            potato_top_class=potato_top,
            beans_score=beans_total + (0.35 * beans_top),
            potato_score=potato_total + (0.35 * potato_top),
        )
        if candidate is None:
            return None
        crop_name, crop_total = candidate
        if crop_total <= 0.0:
            return None
        gap_from_best = max(0.0, best_crop_total - crop_total)
        if gap_from_best > gate_cfg["beansPotatoRescueMaxGapFromBest"]:
            return None
        return crop_name, crop_total

    def select_secondary_focus_candidate(crop_probs, prob_map, best_crop_total):
        beans_total = float(crop_probs.get("beans", 0.0))
        potato_total = float(crop_probs.get("potato", 0.0))
        _, beans_top = top_class_for_crop("beans", prob_map)
        _, potato_top = top_class_for_crop("potato", prob_map)

        has_evidence = (
            beans_total >= gate_cfg["secondaryFocusMinCropTotal"]
            or potato_total >= gate_cfg["secondaryFocusMinCropTotal"]
            or beans_top >= gate_cfg["secondaryFocusMinTopClassProb"]
            or potato_top >= gate_cfg["secondaryFocusMinTopClassProb"]
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
        if gap_from_best > gate_cfg["secondaryFocusMaxGapFromBest"]:
            return None
        return crop_name, crop_total

    def select_preferred_beans_potato_crop(
        crop_probs,
        prob_map,
        current_best_crop,
        current_best_crop_total,
    ):
        if gate_cfg.get("preferBeansPotatoWhenCompetitive", 1.0) < 0.5:
            return None
        if is_beans_or_potato(current_best_crop):
            return None

        beans_total = float(crop_probs.get("beans", 0.0))
        potato_total = float(crop_probs.get("potato", 0.0))
        _, beans_top = top_class_for_crop("beans", prob_map)
        _, potato_top = top_class_for_crop("potato", prob_map)

        candidate = choose_focus_crop_candidate(
            beans_total=beans_total,
            potato_total=potato_total,
            beans_top_class=beans_top,
            potato_top_class=potato_top,
            beans_score=beans_total + (0.45 * beans_top),
            potato_score=potato_total + (0.45 * potato_top),
        )
        if candidate is None:
            return None

        crop_name, crop_total = candidate
        gap_from_best = max(0.0, current_best_crop_total - crop_total)
        if crop_total < gate_cfg["focusCropSwitchMinTotal"]:
            return None
        if gap_from_best > gate_cfg["focusCropSwitchMaxGap"]:
            return None
        return crop_name, crop_total

    def second_best_crop_total_excluding(crop_probs, excluded_crop):
        candidates = sorted(
            ((k, v) for k, v in crop_probs.items() if k != excluded_crop),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return float(candidates[0][1]) if candidates else 0.0

    def should_preserve_focus_crop_identity(candidate_crop, candidate_crop_total, prob_map):
        if not is_beans_or_potato(candidate_crop):
            return False
        _, top_class_prob = top_class_for_crop(candidate_crop, prob_map)
        return candidate_crop_total >= 0.001 or top_class_prob >= 0.001

    def build_focus_crop_outcome(gate_name, crop_name, crop_total, prob_map):
        focus_class, focus_class_prob = top_class_for_crop(crop_name, prob_map)
        if not focus_class:
            focus_class = "beans_healthy" if crop_name == "beans" else "potato_healthy"
            focus_class_prob = float(prob_map.get(focus_class, 0.0))
        return {
            "gate": gate_name,
            "resultType": "healthy" if "healthy" in focus_class else "disease",
            "bestCrop": crop_name,
            "bestCropTotal": float(crop_total),
            "bestClass": focus_class,
            "bestClassProb": float(max(crop_total, focus_class_prob)),
        }

    def attempt_focus_rescue(
        fallback_gate,
        crop_probs,
        prob_map,
        candidate_crop,
        candidate_crop_total,
        other_leaf_prob,
    ):
        # Rescue path 1: targeted beans/potato rescue.
        rescue_crop = candidate_crop
        rescue_crop_total = candidate_crop_total

        if not is_beans_or_potato(rescue_crop):
            alternative = select_beans_potato_rescue_candidate(
                crop_probs,
                prob_map,
                best_crop_total=candidate_crop_total,
            )
            if alternative is not None:
                rescue_crop, rescue_crop_total = alternative
            else:
                rescue_crop = None

        if rescue_crop is not None:
            if rescue_crop_total >= gate_cfg["beansPotatoRescueMinCropTotal"]:
                gap_from_best = max(0.0, candidate_crop_total - rescue_crop_total)
                focus_guard_ok = True
                if gap_from_best <= gate_cfg["beansPotatoRescueMaxGapFromBest"] and other_leaf_prob <= gate_cfg["beansPotatoRescueMaxOtherLeaf"]:
                    beans_total = float(crop_probs.get("beans", 0.0))
                    potato_total = float(crop_probs.get("potato", 0.0))
                    if abs(beans_total - potato_total) <= gate_cfg["rescueFocusSwapGuardCropGap"]:
                        _, beans_top = top_class_for_crop("beans", prob_map)
                        _, potato_top = top_class_for_crop("potato", prob_map)
                        top_margin = (beans_top - potato_top) if rescue_crop == "beans" else (potato_top - beans_top)
                        if top_margin < gate_cfg["rescueFocusSwapGuardTopClassMargin"]:
                            focus_guard_ok = False
                    if focus_guard_ok:
                        rescue_best_class, rescue_top = top_class_for_crop(rescue_crop, prob_map)
                        return {
                            "gate": "RESCUE_beans_potato",
                            "resultType": "healthy" if "healthy" in rescue_best_class else "disease",
                            "bestCrop": rescue_crop,
                            "bestCropTotal": float(rescue_crop_total),
                            "bestClass": rescue_best_class,
                            "bestClassProb": float(max(rescue_crop_total, rescue_top)),
                        }

        # Rescue path 2: forced beans/potato fallback.
        if gate_cfg.get("forceBeansPotatoNeverUnsupported", 1.0) >= 0.5:
            beans_total = float(crop_probs.get("beans", 0.0))
            potato_total = float(crop_probs.get("potato", 0.0))
            beans_class, beans_top = top_class_for_crop("beans", prob_map)
            potato_class, potato_top = top_class_for_crop("potato", prob_map)
            candidate = choose_focus_crop_candidate(
                beans_total=beans_total,
                potato_total=potato_total,
                beans_top_class=beans_top,
                potato_top_class=potato_top,
                beans_score=beans_total + (0.50 * beans_top),
                potato_score=potato_total + (0.50 * potato_top),
            )
            if candidate is not None:
                rescue_crop, rescue_total = candidate
                rescue_class = potato_class if rescue_crop == "potato" else beans_class
                rescue_top = potato_top if rescue_crop == "potato" else beans_top
                has_min_evidence = (
                    rescue_total >= gate_cfg["forceBeansPotatoMinCropTotal"]
                    or rescue_top >= gate_cfg["forceBeansPotatoMinTopClassProb"]
                )
                if has_min_evidence:
                    return {
                        "gate": "FORCED_beans_potato_fallback",
                        "resultType": "healthy" if "healthy" in rescue_class else "disease",
                        "bestCrop": rescue_crop,
                        "bestCropTotal": float(rescue_total),
                        "bestClass": rescue_class,
                        "bestClassProb": float(max(rescue_total, rescue_top)),
                    }

        # Rescue path 3: last-chance focus override.
        beans_total = float(crop_probs.get("beans", 0.0))
        potato_total = float(crop_probs.get("potato", 0.0))
        beans_class, beans_top = top_class_for_crop("beans", prob_map)
        potato_class, potato_top = top_class_for_crop("potato", prob_map)
        no_focus_evidence = (
            beans_total < gate_cfg["lastChanceFocusMinCropTotal"]
            and potato_total < gate_cfg["lastChanceFocusMinCropTotal"]
            and beans_top < gate_cfg["lastChanceFocusMinTopClassProb"]
            and potato_top < gate_cfg["lastChanceFocusMinTopClassProb"]
        )
        if no_focus_evidence:
            return None

        candidate = choose_focus_crop_candidate(
            beans_total=beans_total,
            potato_total=potato_total,
            beans_top_class=beans_top,
            potato_top_class=potato_top,
            beans_score=beans_total + (0.50 * beans_top),
            potato_score=potato_total + (0.50 * potato_top),
        )
        if candidate is None:
            return None
        rescue_crop, rescue_total = candidate
        rescue_class = potato_class if rescue_crop == "potato" else beans_class
        rescue_top = potato_top if rescue_crop == "potato" else beans_top
        return {
            "gate": "LAST_CHANCE_focus_override",
            "resultType": "healthy" if "healthy" in rescue_class else "disease",
            "bestCrop": rescue_crop,
            "bestCropTotal": float(rescue_total),
            "bestClass": rescue_class,
            "bestClassProb": float(max(rescue_total, rescue_top)),
        }

    rows = []
    gate_counts = Counter()
    result_counts = Counter()

    for i in range(len(labels)):
        raw_row = raw_probs[i]
        cal_row = calibrated_probs[i]

        raw_map = {class_names[j]: float(raw_row[j]) for j in range(len(class_names))}
        cal_map = {class_names[j]: float(cal_row[j]) for j in range(len(class_names))}

        true_label = class_names[int(labels[i])]
        true_crop = class_to_crop[true_label]

        other_leaf_scaled = cal_map.get("other_leaf", 0.0)
        other_leaf_raw = raw_map.get("other_leaf", 0.0)
        other_leaf = max(other_leaf_scaled, other_leaf_raw)

        crop_probs = {}
        for cname, prob in cal_map.items():
            crop = class_to_crop[cname]
            if crop == "other_leaf":
                continue
            crop_probs[crop] = crop_probs.get(crop, 0.0) + float(prob)

        sorted_crops = sorted(crop_probs.items(), key=lambda kv: kv[1], reverse=True)
        best_candidate_crop = sorted_crops[0][0] if sorted_crops else "unknown"
        best_candidate_crop_total = sorted_crops[0][1] if sorted_crops else 0.0

        # Mirror app Step 6 pre-selection: optional beans/potato substitution
        # before any downstream gate is evaluated.
        best_crop = best_candidate_crop
        best_crop_total = best_candidate_crop_total

        secondary_focus_candidate = select_secondary_focus_candidate(
            crop_probs,
            cal_map,
            best_crop_total=best_candidate_crop_total,
        )
        if secondary_focus_candidate is not None and not is_beans_or_potato(best_crop):
            best_crop, best_crop_total = secondary_focus_candidate

        preferred_focus_crop = select_preferred_beans_potato_crop(
            crop_probs,
            cal_map,
            current_best_crop=best_crop,
            current_best_crop_total=best_crop_total,
        )
        if preferred_focus_crop is not None:
            best_crop, best_crop_total = preferred_focus_crop

        second_crop_total = second_best_crop_total_excluding(crop_probs, best_crop)

        gate = ""
        result_type = ""
        best_class = ""
        best_class_prob = 0.0

        effective_other_leaf_floor = gate_cfg["otherLeafAbsoluteFloor"]
        if best_candidate_crop in ("beans", "potato"):
            effective_other_leaf_floor = min(
                0.95,
                gate_cfg["otherLeafAbsoluteFloor"] + gate_cfg["beansPotatoOtherLeafFloorBoost"],
            )
        else:
            effective_other_leaf_floor = min(
                0.95,
                gate_cfg["otherLeafAbsoluteFloor"] + gate_cfg["nonFocusOtherLeafFloorBoost"],
            )

        if other_leaf >= gate_cfg["otherLeafThreshold"]:
            preserve_crop = best_candidate_crop
            preserve_crop_total = best_candidate_crop_total
            if secondary_focus_candidate is not None:
                preserve_crop, preserve_crop_total = secondary_focus_candidate

            if should_preserve_focus_crop_identity(
                preserve_crop,
                preserve_crop_total,
                cal_map,
            ):
                preserve_outcome = build_focus_crop_outcome(
                    "G5_preserve_focus_crop",
                    preserve_crop,
                    preserve_crop_total,
                    cal_map,
                )
                gate = preserve_outcome["gate"]
                result_type = preserve_outcome["resultType"]
                best_crop = preserve_outcome["bestCrop"]
                best_crop_total = preserve_outcome["bestCropTotal"]
                best_class = preserve_outcome["bestClass"]
                best_class_prob = preserve_outcome["bestClassProb"]
            else:
                rescue_outcome = attempt_focus_rescue(
                    fallback_gate="G5_other_leaf_winner",
                    crop_probs=crop_probs,
                    prob_map=cal_map,
                    candidate_crop=best_candidate_crop,
                    candidate_crop_total=best_candidate_crop_total,
                    other_leaf_prob=other_leaf,
                )
                if rescue_outcome is not None:
                    gate = rescue_outcome["gate"]
                    result_type = rescue_outcome["resultType"]
                    best_crop = rescue_outcome["bestCrop"]
                    best_crop_total = rescue_outcome["bestCropTotal"]
                    best_class = rescue_outcome["bestClass"]
                    best_class_prob = rescue_outcome["bestClassProb"]
                else:
                    gate = "G5_other_leaf_winner"
                    result_type = "other_leaf"
        elif other_leaf > effective_other_leaf_floor:
            preserve_crop = best_candidate_crop
            preserve_crop_total = best_candidate_crop_total
            if secondary_focus_candidate is not None:
                preserve_crop, preserve_crop_total = secondary_focus_candidate

            if should_preserve_focus_crop_identity(
                preserve_crop,
                preserve_crop_total,
                cal_map,
            ):
                preserve_outcome = build_focus_crop_outcome(
                    "G5b_preserve_focus_crop",
                    preserve_crop,
                    preserve_crop_total,
                    cal_map,
                )
                gate = preserve_outcome["gate"]
                result_type = preserve_outcome["resultType"]
                best_crop = preserve_outcome["bestCrop"]
                best_crop_total = preserve_outcome["bestCropTotal"]
                best_class = preserve_outcome["bestClass"]
                best_class_prob = preserve_outcome["bestClassProb"]
            else:
                rescue_outcome = attempt_focus_rescue(
                    fallback_gate="G5b_other_leaf_floor",
                    crop_probs=crop_probs,
                    prob_map=cal_map,
                    candidate_crop=best_candidate_crop,
                    candidate_crop_total=best_candidate_crop_total,
                    other_leaf_prob=other_leaf,
                )
                if rescue_outcome is not None:
                    gate = rescue_outcome["gate"]
                    result_type = rescue_outcome["resultType"]
                    best_crop = rescue_outcome["bestCrop"]
                    best_crop_total = rescue_outcome["bestCropTotal"]
                    best_class = rescue_outcome["bestClass"]
                    best_class_prob = rescue_outcome["bestClassProb"]
                else:
                    gate = "G5b_other_leaf_floor"
                    result_type = "other_leaf"
        elif not sorted_crops:
            rescue_outcome = attempt_focus_rescue(
                fallback_gate="G6_no_crop_candidates",
                crop_probs=crop_probs,
                prob_map=cal_map,
                candidate_crop=best_candidate_crop,
                candidate_crop_total=best_candidate_crop_total,
                other_leaf_prob=other_leaf,
            )
            if rescue_outcome is not None:
                gate = rescue_outcome["gate"]
                result_type = rescue_outcome["resultType"]
                best_crop = rescue_outcome["bestCrop"]
                best_crop_total = rescue_outcome["bestCropTotal"]
                best_class = rescue_outcome["bestClass"]
                best_class_prob = rescue_outcome["bestClassProb"]
            else:
                gate = "G6_no_crop_candidates"
                result_type = "unsupported"
        else:
            effective_crop_total = gate_cfg["cropTotalThreshold"]
            effective_gap_threshold = gate_cfg["uncertainGapThreshold"]
            effective_second_crop_threshold = gate_cfg["secondCropAmbiguityThreshold"]
            effective_entropy_threshold = gate_cfg["nonFocusMaxEntropyThreshold"]
            effective_class_ratio_threshold = gate_cfg["nonFocusClassRatioThreshold"]
            effective_class_confidence_threshold = gate_cfg["nonFocusClassConfidenceThreshold"]

            if best_crop in ("beans", "potato"):
                effective_crop_total = max(
                    0.0,
                    gate_cfg["cropTotalThreshold"] - gate_cfg["beansPotatoCropTotalRelaxation"],
                )
                effective_gap_threshold = gate_cfg["beansPotatoUncertainGapThreshold"]
                effective_second_crop_threshold = gate_cfg["beansPotatoSecondCropThreshold"]
                effective_entropy_threshold = gate_cfg["maxEntropyThreshold"]
                effective_class_ratio_threshold = gate_cfg["beansPotatoClassRatioThreshold"]
                effective_class_confidence_threshold = gate_cfg["beansPotatoClassConfidenceThreshold"]
            else:
                effective_crop_total = max(
                    0.0,
                    gate_cfg["cropTotalThreshold"] - gate_cfg["nonFocusCropTotalRelaxation"],
                )

            if best_crop_total < effective_crop_total:
                rescue_outcome = attempt_focus_rescue(
                    fallback_gate="G7a_crop_total",
                    crop_probs=crop_probs,
                    prob_map=cal_map,
                    candidate_crop=best_crop,
                    candidate_crop_total=best_crop_total,
                    other_leaf_prob=other_leaf,
                )
                if rescue_outcome is not None:
                    gate = rescue_outcome["gate"]
                    result_type = rescue_outcome["resultType"]
                    best_crop = rescue_outcome["bestCrop"]
                    best_crop_total = rescue_outcome["bestCropTotal"]
                    best_class = rescue_outcome["bestClass"]
                    best_class_prob = rescue_outcome["bestClassProb"]
                else:
                    gate = "G7a_crop_total"
                    result_type = "unsupported"
            else:
                other_leaf_ratio = (other_leaf_scaled / best_crop_total) if best_crop_total > 0 else 0.0
                if other_leaf_ratio > gate_cfg["otherLeafVsCropRatioThreshold"]:
                    rescue_outcome = attempt_focus_rescue(
                        fallback_gate="G7a_other_leaf_ratio",
                        crop_probs=crop_probs,
                        prob_map=cal_map,
                        candidate_crop=best_crop,
                        candidate_crop_total=best_crop_total,
                        other_leaf_prob=other_leaf,
                    )
                    if rescue_outcome is not None:
                        gate = rescue_outcome["gate"]
                        result_type = rescue_outcome["resultType"]
                        best_crop = rescue_outcome["bestCrop"]
                        best_crop_total = rescue_outcome["bestCropTotal"]
                        best_class = rescue_outcome["bestClass"]
                        best_class_prob = rescue_outcome["bestClassProb"]
                    else:
                        gate = "G7a_other_leaf_ratio"
                        result_type = "other_leaf"
                else:
                    crop_classes = [
                        (name, cal_map[name])
                        for name in class_names
                        if class_to_crop[name] == best_crop
                    ]
                    crop_classes.sort(key=lambda kv: kv[1], reverse=True)
                    if crop_classes:
                        best_class, best_class_prob = crop_classes[0]

                    crop_gap = best_crop_total - second_crop_total
                    cal_entropy = float(
                        -np.sum(cal_row * (np.log(np.clip(cal_row, 1e-10, 1.0)) / np.log(2.0)))
                    )
                    class_total_ratio = (best_class_prob / best_crop_total) if best_crop_total > 0 else 0.0

                    if crop_gap < effective_gap_threshold:
                        if is_beans_or_potato(best_crop):
                            focus_outcome = build_focus_crop_outcome(
                                "G7c_crop_gap_focus_override",
                                best_crop,
                                best_class_prob,
                                cal_map,
                            )
                            gate = focus_outcome["gate"]
                            result_type = focus_outcome["resultType"]
                            best_crop = focus_outcome["bestCrop"]
                            best_crop_total = focus_outcome["bestCropTotal"]
                            best_class = focus_outcome["bestClass"]
                            best_class_prob = focus_outcome["bestClassProb"]
                        else:
                            gate = "G7c_crop_gap"
                            result_type = "uncertain"
                    elif second_crop_total > effective_second_crop_threshold:
                        if is_beans_or_potato(best_crop):
                            focus_outcome = build_focus_crop_outcome(
                                "G7c_second_crop_focus_override",
                                best_crop,
                                best_class_prob,
                                cal_map,
                            )
                            gate = focus_outcome["gate"]
                            result_type = focus_outcome["resultType"]
                            best_crop = focus_outcome["bestCrop"]
                            best_crop_total = focus_outcome["bestCropTotal"]
                            best_class = focus_outcome["bestClass"]
                            best_class_prob = focus_outcome["bestClassProb"]
                        else:
                            gate = "G7c_second_crop"
                            result_type = "uncertain"
                    elif cal_entropy > effective_entropy_threshold:
                        rescue_outcome = attempt_focus_rescue(
                            fallback_gate="G7c_entropy",
                            crop_probs=crop_probs,
                            prob_map=cal_map,
                            candidate_crop=best_crop,
                            candidate_crop_total=best_crop_total,
                            other_leaf_prob=other_leaf,
                        )
                        if rescue_outcome is not None:
                            gate = rescue_outcome["gate"]
                            result_type = rescue_outcome["resultType"]
                            best_crop = rescue_outcome["bestCrop"]
                            best_crop_total = rescue_outcome["bestCropTotal"]
                            best_class = rescue_outcome["bestClass"]
                            best_class_prob = rescue_outcome["bestClassProb"]
                        else:
                            gate = "G7c_entropy"
                            result_type = "uncertain"
                    elif class_total_ratio < effective_class_ratio_threshold:
                        rescue_outcome = attempt_focus_rescue(
                            fallback_gate="G7c_class_ratio",
                            crop_probs=crop_probs,
                            prob_map=cal_map,
                            candidate_crop=best_crop,
                            candidate_crop_total=best_crop_total,
                            other_leaf_prob=other_leaf,
                        )
                        if rescue_outcome is not None:
                            gate = rescue_outcome["gate"]
                            result_type = rescue_outcome["resultType"]
                            best_crop = rescue_outcome["bestCrop"]
                            best_crop_total = rescue_outcome["bestCropTotal"]
                            best_class = rescue_outcome["bestClass"]
                            best_class_prob = rescue_outcome["bestClassProb"]
                        else:
                            gate = "G7c_class_ratio"
                            result_type = "uncertain"
                    elif best_class_prob < effective_class_confidence_threshold:
                        if is_beans_or_potato(best_crop):
                            focus_outcome = build_focus_crop_outcome(
                                "G7d_class_confidence_focus_override",
                                best_crop,
                                best_class_prob,
                                cal_map,
                            )
                            gate = focus_outcome["gate"]
                            result_type = focus_outcome["resultType"]
                            best_crop = focus_outcome["bestCrop"]
                            best_crop_total = focus_outcome["bestCropTotal"]
                            best_class = focus_outcome["bestClass"]
                            best_class_prob = focus_outcome["bestClassProb"]
                        else:
                            gate = "G7d_class_confidence"
                            result_type = "uncertain"
                    elif "healthy" in best_class:
                        healthy_confidence_threshold = (
                            DEFAULT_POTATO_HEALTHY_MIN_CONFIDENCE_PILOT
                            if best_crop == "potato"
                            else DEFAULT_HEALTHY_MIN_CONFIDENCE
                        )
                        if best_class_prob < healthy_confidence_threshold:
                            if is_beans_or_potato(best_crop):
                                focus_outcome = build_focus_crop_outcome(
                                    "G7e_healthy_safety_focus_override",
                                    best_crop,
                                    best_class_prob,
                                    cal_map,
                                )
                                gate = focus_outcome["gate"]
                                result_type = focus_outcome["resultType"]
                                best_crop = focus_outcome["bestCrop"]
                                best_crop_total = focus_outcome["bestCropTotal"]
                                best_class = focus_outcome["bestClass"]
                                best_class_prob = focus_outcome["bestClassProb"]
                            else:
                                gate = "G7e_healthy_safety"
                                result_type = "uncertain"
                        else:
                            gate = "G7g_confident"
                            result_type = "healthy"
                    else:
                        gate = "G7g_confident"
                        result_type = "disease"

        gate_counts[gate] += 1
        result_counts[result_type] += 1
        rows.append({
            "index": i,
            "trueLabel": true_label,
            "trueCrop": true_crop,
            "gate": gate,
            "resultType": result_type,
            "bestCrop": best_crop,
            "bestClass": best_class,
            "bestClassProb": round(float(best_class_prob), 6),
            "bestCropTotal": round(float(best_crop_total), 6),
            "secondCropTotal": round(float(second_crop_total), 6),
            "otherLeaf": round(float(other_leaf), 6),
            "otherLeafRaw": round(float(other_leaf_raw), 6),
            "otherLeafScaled": round(float(other_leaf_scaled), 6),
        })

    total = len(rows)
    supported_rows = [r for r in rows if r["trueCrop"] != "other_leaf"]
    unsupported_rows = [r for r in rows if r["trueCrop"] == "other_leaf"]

    supported_confident_correct = sum(
        1 for r in supported_rows if r["resultType"] in ("healthy", "disease") and r["bestCrop"] == r["trueCrop"]
    )
    supported_false_unsupported = sum(
        1 for r in supported_rows if r["resultType"] in ("unsupported", "other_leaf")
    )
    unsupported_rejected = sum(
        1 for r in unsupported_rows if r["resultType"] in ("unsupported", "other_leaf")
    )

    summary = {
        "total": total,
        "gate_counts": dict(gate_counts),
        "result_type_counts": dict(result_counts),
        "supported_confident_rate": round(
            float(supported_confident_correct / len(supported_rows)) if supported_rows else 0.0,
            4,
        ),
        "supported_false_unsupported_rate": round(
            float(supported_false_unsupported / len(supported_rows)) if supported_rows else 0.0,
            4,
        ),
        "unsupported_reject_rate": round(
            float(unsupported_rejected / len(unsupported_rows)) if unsupported_rows else 0.0,
            4,
        ),
    }

    return summary, rows


# ── Validate dataset ────────────────────────────────────────────────
print("Validating dataset ...", flush=True)
for split_name in ["train", "val", "test"]:
    split_dir = os.path.join(DATA_DIR, split_name)
    if not os.path.isdir(split_dir):
        print(f"ERROR: Missing required split directory: {split_dir}", flush=True)
        sys.exit(1)
    for folder in sorted(os.listdir(split_dir)):
        if not os.path.isdir(os.path.join(split_dir, folder)):
            continue
        crop_prefix = get_crop_prefix(folder)
        if crop_prefix not in ALLOWED_CROPS:
            print(f"ERROR: Unexpected class '{folder}' in {split_name}/", flush=True)
            print(f"  Crop prefix '{crop_prefix}' is not in {ALLOWED_CROPS}", flush=True)
            print("  Remove it and re-run.", flush=True)
            sys.exit(1)
print(f"Dataset validated: only {ALLOWED_CROPS} crops found.", flush=True)

if ENABLE_IMAGE_SANITY_CHECK:
    print("Running image sanity checks ...", flush=True)
    sanity_report = {
        "train": run_image_sanity_check(TRAIN_DIR, "train"),
        "val": run_image_sanity_check(VAL_DIR, "val"),
        "test": run_image_sanity_check(TEST_DIR, "test"),
    }
    with open(os.path.join(MODEL_DIR, "image_sanity_report.json"), 'w') as f:
        json.dump(sanity_report, f, indent=2)

    total_broken = sum(sanity_report[s]["broken_count"] for s in ["train", "val", "test"])
    if total_broken > 0:
        print("ERROR: Broken/corrupted images found. See models/image_sanity_report.json", flush=True)
        sys.exit(1)


# ── Datasets ────────────────────────────────────────────────────────
print("Loading datasets ...", flush=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=True, seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}", flush=True)

if os.path.isdir(FIELD_FAILURE_DIR):
    field_failure_subdirs = sorted(
        name for name in os.listdir(FIELD_FAILURE_DIR)
        if os.path.isdir(os.path.join(FIELD_FAILURE_DIR, name))
    )
    invalid_classes = [name for name in field_failure_subdirs if name not in class_names]
    if invalid_classes:
        print(
            "ERROR: field failure directory contains classes not present in labels.txt:",
            invalid_classes,
            flush=True,
        )
        print("  Ensure field failures are organized under exact class names from train labels.", flush=True)
        sys.exit(1)

    field_failure_count = 0
    for root, _, files in os.walk(FIELD_FAILURE_DIR):
        field_failure_count += len([f for f in files if os.path.isfile(os.path.join(root, f))])
    if field_failure_count > 0:
        print(f"Loading field-failure hard examples from {FIELD_FAILURE_DIR} ...", flush=True)
        field_ds = tf.keras.utils.image_dataset_from_directory(
            FIELD_FAILURE_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH,
            label_mode='int',
            shuffle=True,
            seed=SEED,
            class_names=class_names,
        )

        max_field_samples = field_failure_count
        if FIELD_FAILURE_MAX_SAMPLES is not None and field_failure_count > FIELD_FAILURE_MAX_SAMPLES:
            print(
                f"Capping field-failure samples to {FIELD_FAILURE_MAX_SAMPLES} of {field_failure_count} total.",
                flush=True,
            )
            max_field_samples = FIELD_FAILURE_MAX_SAMPLES
            field_ds = field_ds.unbatch().shuffle(FIELD_FAILURE_SHUFFLE_BUFFER, seed=SEED)
            field_ds = field_ds.take(max_field_samples).batch(BATCH)
        else:
            field_ds = field_ds.unbatch().batch(BATCH)

        train_ds = train_ds.concatenate(field_ds)
        print(f"Included {min(field_failure_count, max_field_samples)} field-failure samples in training.", flush=True)

print(f"Classes ({NUM_CLASSES}): {class_names}", flush=True)

if "other_leaf" not in class_names:
    print("ERROR: 'other_leaf' class not found in training data.", flush=True)
    print("  Make sure datasets/model_ready/train/other_leaf/ exists and has images.", flush=True)
    sys.exit(1)
OTHER_LEAF_IDX = class_names.index("other_leaf")
print(f"other_leaf class index: {OTHER_LEAF_IDX}", flush=True)


def collect_split_stems(split_dir, cls_name):
    cls_dir = os.path.join(split_dir, cls_name)
    stems = set()
    if not os.path.isdir(cls_dir):
        return stems
    for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname)
        if os.path.isfile(fpath):
            stems.add(os.path.splitext(fname)[0])
    return stems


def collect_split_hashes(split_dir, cls_name):
    cls_dir = os.path.join(split_dir, cls_name)
    hashes = set()
    if not os.path.isdir(cls_dir):
        return hashes
    for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                hashes.add(hashlib.md5(f.read()).hexdigest())
    return hashes


# Save labels
labels_path = os.path.join(MODEL_DIR, "labels.txt")
with open(labels_path, 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print(f"Saved labels -> {labels_path}", flush=True)

class_index = {name: i for i, name in enumerate(class_names)}
with open(os.path.join(MODEL_DIR, "class_index.json"), 'w') as f:
    json.dump(class_index, f, indent=2)


# ── FIX D2: Class weights — other_leaf boosted ×3 ──────────────────
print("Computing class weights ...", flush=True)
class_counts = {}
for cname in class_names:
    folder = os.path.join(TRAIN_DIR, cname)
    count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    class_counts[cname] = count
    print(f"  {cname}: {count}", flush=True)

total = sum(class_counts.values())
print(f"Total training images: {total}", flush=True)

val_total = 0
test_total = 0
val_counts = {}
test_counts = {}
leakage_check = {}
leakage_check_hash = {}
total_train_val_overlap = 0
total_train_test_overlap = 0
total_val_test_overlap = 0
total_train_val_hash_overlap = 0
total_train_test_hash_overlap = 0
total_val_test_hash_overlap = 0
for cname in class_names:
    val_folder  = os.path.join(VAL_DIR, cname)
    test_folder = os.path.join(TEST_DIR, cname)
    val_count  = len([f for f in os.listdir(val_folder)  if os.path.isfile(os.path.join(val_folder, f))])
    test_count = len([f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))])
    val_counts[cname]  = val_count
    test_counts[cname] = test_count
    val_total  += val_count
    test_total += test_count

    train_stems = collect_split_stems(TRAIN_DIR, cname)
    val_stems   = collect_split_stems(VAL_DIR,   cname)
    test_stems  = collect_split_stems(TEST_DIR,  cname)
    train_hashes = collect_split_hashes(TRAIN_DIR, cname)
    val_hashes   = collect_split_hashes(VAL_DIR,   cname)
    test_hashes  = collect_split_hashes(TEST_DIR,  cname)

    overlap_train_val  = len(train_stems.intersection(val_stems))
    overlap_train_test = len(train_stems.intersection(test_stems))
    overlap_val_test   = len(val_stems.intersection(test_stems))
    overlap_train_val_hash  = len(train_hashes.intersection(val_hashes))
    overlap_train_test_hash = len(train_hashes.intersection(test_hashes))
    overlap_val_test_hash   = len(val_hashes.intersection(test_hashes))

    leakage_check[cname] = {
        "train_val_overlap":  overlap_train_val,
        "train_test_overlap": overlap_train_test,
        "val_test_overlap":   overlap_val_test,
    }
    leakage_check_hash[cname] = {
        "train_val_overlap":  overlap_train_val_hash,
        "train_test_overlap": overlap_train_test_hash,
        "val_test_overlap":   overlap_val_test_hash,
    }
    total_train_val_overlap  += overlap_train_val
    total_train_test_overlap += overlap_train_test
    total_val_test_overlap   += overlap_val_test
    total_train_val_hash_overlap  += overlap_train_val_hash
    total_train_test_hash_overlap += overlap_train_test_hash
    total_val_test_hash_overlap   += overlap_val_test_hash

print(f"Total validation images: {val_total}", flush=True)
print(f"Total test images: {test_total}", flush=True)
print("Filename-stem overlap check (possible leakage):", flush=True)
print(f"  train↔val:  {total_train_val_overlap}", flush=True)
print(f"  train↔test: {total_train_test_overlap}", flush=True)
print(f"  val↔test:   {total_val_test_overlap}", flush=True)
print("Byte-hash overlap check (exact duplicate leakage):", flush=True)
print(f"  train↔val:  {total_train_val_hash_overlap}", flush=True)
print(f"  train↔test: {total_train_test_hash_overlap}", flush=True)
print(f"  val↔test:   {total_val_test_hash_overlap}", flush=True)

if ENFORCE_LEAKAGE_GUARD:
    stem_overlap_total = (
        total_train_val_overlap +
        total_train_test_overlap +
        total_val_test_overlap
    )
    hash_overlap_total = (
        total_train_val_hash_overlap +
        total_train_test_hash_overlap +
        total_val_test_hash_overlap
    )
    leakage_guard_report = {
        "enforced": True,
        "max_allowed_stem_overlap_total": MAX_ALLOWED_STEM_OVERLAP_TOTAL,
        "max_allowed_hash_overlap_total": MAX_ALLOWED_HASH_OVERLAP_TOTAL,
        "observed": {
            "stem_overlap_total": stem_overlap_total,
            "hash_overlap_total": hash_overlap_total,
            "train_val_stem_overlap": total_train_val_overlap,
            "train_test_stem_overlap": total_train_test_overlap,
            "val_test_stem_overlap": total_val_test_overlap,
            "train_val_hash_overlap": total_train_val_hash_overlap,
            "train_test_hash_overlap": total_train_test_hash_overlap,
            "val_test_hash_overlap": total_val_test_hash_overlap,
        },
    }
    with open(os.path.join(MODEL_DIR, "leakage_guard_report.json"), 'w') as f:
        json.dump(leakage_guard_report, f, indent=2)

    if (
        stem_overlap_total > MAX_ALLOWED_STEM_OVERLAP_TOTAL or
        hash_overlap_total > MAX_ALLOWED_HASH_OVERLAP_TOTAL
    ):
        print("ERROR: leakage guard failed. Training aborted.", flush=True)
        print(
            "  Clean splits to remove overlapping/duplicate samples across "
            "train/val/test, then retrain.",
            flush=True,
        )
        print(
            f"  See {os.path.join(MODEL_DIR, 'leakage_guard_report.json')} for details.",
            flush=True,
        )
        sys.exit(1)

class_weight = {}
cap_label = f"capped at {MAX_CLASS_WEIGHT}" if MAX_CLASS_WEIGHT is not None else "uncapped"
print(
    f"Class weights ({cap_label}, other_leaf boosted ×{OTHER_LEAF_WEIGHT_MULTIPLIER}, "
    f"focus crops boosted ×{FOCUS_CROP_WEIGHT_MULTIPLIER}):",
    flush=True,
)

for i, cname in enumerate(class_names):
    w = total / (NUM_CLASSES * class_counts[cname])
    if MAX_CLASS_WEIGHT is not None:
        w = min(w, MAX_CLASS_WEIGHT)
    boost_label = ""
    # Cap is applied before boosts, so boosted classes can exceed
    # MAX_CLASS_WEIGHT by design.
    if cname == "other_leaf":
        w *= OTHER_LEAF_WEIGHT_MULTIPLIER
        boost_label = f"  ← BOOSTED ×{OTHER_LEAF_WEIGHT_MULTIPLIER}"
    elif ENABLE_FOCUS_CROP_WEIGHTING and (
        cname.startswith("beans_") or cname.startswith("potato_")
    ):
        w *= FOCUS_CROP_WEIGHT_MULTIPLIER
        boost_label = f"  ← FOCUS BOOST ×{FOCUS_CROP_WEIGHT_MULTIPLIER}"
    class_weight[i] = round(w, 4)
    print(f"  {cname}: {class_weight[i]}{boost_label}", flush=True)

with open(os.path.join(MODEL_DIR, "class_weights.json"), 'w') as f:
    json.dump({class_names[i]: class_weight[i] for i in range(NUM_CLASSES)}, f, indent=2)


# ── Preprocessing ────────────────────────────────────────────────────
def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels


# ── Standard and other_leaf augmentation (unchanged) ─────────────────
standard_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(AUG_ROTATION),
    tf.keras.layers.RandomZoom(AUG_ZOOM),
    tf.keras.layers.RandomContrast(AUG_CONTRAST, value_range=(-1, 1)),
    tf.keras.layers.RandomBrightness(AUG_BRIGHTNESS, value_range=(-1, 1)),
], name="standard_augmentation")

other_leaf_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(OL_AUG_ROTATION),
    tf.keras.layers.RandomZoom(OL_AUG_ZOOM),
    tf.keras.layers.RandomContrast(OL_AUG_CONTRAST, value_range=(-1, 1)),
    tf.keras.layers.RandomBrightness(OL_AUG_BRIGHTNESS, value_range=(-1, 1)),
], name="other_leaf_augmentation")


def augment_class_aware(images, labels):
    """
    Apply standard augmentation to crop images and heavier augmentation
    to other_leaf images within the same batch.
    Uses tf.where to select per-sample without a Python loop so it runs
    inside a tf.data pipeline without breaking graph execution.
    """
    std_aug   = standard_augmentation(images,   training=True)
    other_aug = other_leaf_augmentation(images, training=True)

    is_other = tf.equal(labels, OTHER_LEAF_IDX)
    mask = tf.reshape(is_other, [-1, 1, 1, 1])
    mask = tf.broadcast_to(mask, tf.shape(images))

    augmented = tf.where(mask, other_aug, std_aug)
    return augmented, labels


# ── FIX D4: MixUp for other_leaf images only ─────────────────────────
# For each other_leaf image in the batch, blend it with a randomly chosen
# partner image from the same batch using a mix ratio λ ~ Beta(α, α).
#
# Why MixUp specifically for other_leaf?
#   The crop disease classes have well-defined visual features — specific
#   lesion shapes, colours, and textures that the model can memorise.
#   other_leaf is different: it must cover an infinite variety of unseen
#   plant types. MixUp forces the model to learn "not a supported crop"
#   as a concept that interpolates across the pixel space, not as a fixed
#   set of memorised training images. A blend of other_leaf + maize should
#   still trigger a high other_leaf probability — and MixUp trains exactly
#   this gradient.
#
# Labels are NOT mixed: the other_leaf label is kept intact.
# We only mix the pixel values. The gradient from a blended image still
# teaches the model where the other_leaf region of feature space is.
#
# Implementation note: both augmented pipelines have already been applied
# before MixUp, so MixUp blends two already-augmented images — increasing
# the diversity of the training signal even further.
def mixup_other_leaf(images, labels):
    """
    For other_leaf samples in the batch, blend them with a random partner
    from the same batch. Crop images are returned unchanged.
    Labels are not modified — the mixed image retains its original label.
    """
    is_other = tf.equal(labels, OTHER_LEAF_IDX)

    def _mix():
        batch_size = tf.shape(images)[0]

        # Sample mix ratio λ from Beta(alpha, alpha) via Gamma variables:
        #   g1~Gamma(alpha,1), g2~Gamma(alpha,1), λ=g1/(g1+g2)
        g1 = tf.random.gamma([batch_size], OL_MIXUP_ALPHA)
        g2 = tf.random.gamma([batch_size], OL_MIXUP_ALPHA)
        lam = g1 / (g1 + g2 + 1e-8)
        # Reshape for broadcasting over (H, W, C)
        lam_img = tf.reshape(lam, [-1, 1, 1, 1])

        # Randomly shuffle the batch to get mix partners for each sample
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)

        # Blended images: λ * original + (1-λ) * partner
        mixed = lam_img * images + (1.0 - lam_img) * shuffled_images

        # Only replace other_leaf samples; keep crop samples untouched
        mask = tf.reshape(is_other, [-1, 1, 1, 1])
        mask = tf.broadcast_to(mask, tf.shape(images))
        return tf.where(mask, mixed, images)

    mixed_images = tf.cond(
        tf.reduce_any(is_other),
        true_fn=_mix,
        false_fn=lambda: images,
    )
    return mixed_images, labels


# ── Build tf.data pipeline ────────────────────────────────────────────
# Order: preprocess → class-aware augment → MixUp (other_leaf only)
# Val and test sets: preprocessing only (no augmentation — consistent eval)
train_ds = train_ds.map(preprocess, num_parallel_calls=MAP_PARALLEL_CALLS)
train_ds = train_ds.map(augment_class_aware, num_parallel_calls=MAP_PARALLEL_CALLS)
if ENABLE_OL_MIXUP:
    train_ds = train_ds.map(mixup_other_leaf, num_parallel_calls=MAP_PARALLEL_CALLS)
train_ds = train_ds.prefetch(PREFETCH_SIZE)
val_ds  = val_ds.map(preprocess,  num_parallel_calls=MAP_PARALLEL_CALLS).prefetch(PREFETCH_SIZE)
test_ds = test_ds.map(preprocess, num_parallel_calls=MAP_PARALLEL_CALLS).prefetch(PREFETCH_SIZE)


# ── Build model ──────────────────────────────────────────────────────
print("Building model ...", flush=True)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dropout(0.2)(x)
x       = tf.keras.layers.Dense(256, activation='relu')(x)
x       = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

summary_lines = []
model.summary(print_fn=lambda line: summary_lines.append(line))
with open(os.path.join(MODEL_DIR, "model_summary.txt"), 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))
print(f"Model params: {model.count_params():,}", flush=True)


# ── Callbacks ────────────────────────────────────────────────────────
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")
HISTORY_PATH     = os.path.join(MODEL_DIR, "training_history.csv")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True,
    mode='max', verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=PATIENCE,
    restore_best_weights=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3,
    min_lr=1e-7, verbose=1
)


class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.phase_start = None

    def on_train_begin(self, logs=None):
        self.phase_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = (time.time() - self.phase_start) / 60
        acc     = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        loss    = logs.get('loss', 0)
        val_loss= logs.get('val_loss', 0)
        lr      = logs.get('lr', float(self.model.optimizer.learning_rate))
        print(
            f"  Epoch {epoch+1}: acc={acc:.4f} val_acc={val_acc:.4f} "
            f"loss={loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e} "
            f"[{elapsed:.1f} min]",
            flush=True,
        )


progress = ProgressCallback()


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, total_epochs, min_lr=1e-7):
        super().__init__()
        self.initial_lr   = initial_lr
        self.total_epochs = total_epochs
        self.min_lr       = min_lr
        self.current_lr   = float(initial_lr)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + math.cos(math.pi * epoch / self.total_epochs))
        self.current_lr = float(lr)
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
        # Log the LR used during this epoch to avoid off-by-one ambiguity.
        logs['lr'] = float(self.current_lr)


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        # Always log effective epoch LR for CSV/reporting consistency.
        logs['lr'] = float(self.model.optimizer.learning_rate)


class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def append(self, history, phase):
        for i, acc in enumerate(history.history.get('accuracy', [])):
            self.rows.append({
                'phase':        phase,
                'epoch':        i + 1,
                'accuracy':     acc,
                'val_accuracy': history.history['val_accuracy'][i],
                'loss':         history.history['loss'][i],
                'val_loss':     history.history['val_loss'][i],
                'lr':           history.history.get('lr', [0])[
                                    min(i, len(history.history.get('lr', [0])) - 1)
                                ],
            })

    def save(self):
        with open(self.path, 'w', newline='') as f:
            w = csv.DictWriter(
                f,
                fieldnames=['phase','epoch','accuracy','val_accuracy','loss','val_loss','lr'],
            )
            w.writeheader()
            w.writerows(self.rows)


csv_log = CSVLogger(HISTORY_PATH)
lr_logger = LearningRateLogger()

# FIX D5: FocalLoss now receives other_leaf_idx and num_classes so it
# can apply label smoothing (ε=OL_LABEL_SMOOTHING) to other_leaf samples.
# Created AFTER dataset loading so OTHER_LEAF_IDX and NUM_CLASSES are known.
focal_loss = FocalLoss(
    gamma=2.0,
    alpha=0.25,
    other_leaf_idx=OTHER_LEAF_IDX,
    num_classes=NUM_CLASSES,
    other_leaf_smoothing=OL_LABEL_SMOOTHING if ENABLE_OL_LABEL_SMOOTHING else 0.0,
    name="focal_loss",
)


# ══════════════════ PHASE 1: Frozen backbone ════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"PHASE 1: Training top layers ({PHASE1_EP} epochs max)", flush=True)
print(f"  Loss:                  FocalLoss(gamma=2.0, alpha=0.25)", flush=True)
print(f"  other_leaf boost:      ×{OTHER_LEAF_WEIGHT_MULTIPLIER}", flush=True)
print(
    f"  other_leaf smoothing:  {'enabled' if ENABLE_OL_LABEL_SMOOTHING else 'disabled'}"
    f" (ε={OL_LABEL_SMOOTHING})",
    flush=True,
)
print(
    f"  MixUp:                 {'enabled' if ENABLE_OL_MIXUP else 'disabled'}",
    flush=True,
)
if ENABLE_OL_MIXUP:
    print(f"  MixUp alpha:           {OL_MIXUP_ALPHA}", flush=True)
print(f"  Class weight cap:      {MAX_CLASS_WEIGHT}", flush=True)
print(f"{'='*60}", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    loss=focal_loss,
    metrics=[
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc'),
    ]
)

t1 = time.time()
h1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE1_EP,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop, reduce_lr, lr_logger, progress],
    verbose=0,
)
t1_elapsed = (time.time() - t1) / 60
best_val_1 = max(h1.history['val_accuracy'])
csv_log.append(h1, 'phase1')
print(
    f"\nPhase 1 done: {len(h1.history['accuracy'])} epochs, "
    f"{t1_elapsed:.1f} min, best val_acc={best_val_1:.4f}",
    flush=True,
)

gc.collect()


# ══════════════════ PHASE 2: Fine-tune ══════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"PHASE 2: Fine-tuning from layer {FINE_TUNE_FROM} ({PHASE2_EP} epochs max)", flush=True)
print(f"  Loss:        FocalLoss(gamma=2.0, alpha=0.25, smoothing={OL_LABEL_SMOOTHING})", flush=True)
print(f"  LR schedule: Cosine annealing from {FINE_LR}", flush=True)
print(f"{'='*60}", flush=True)

base_model.trainable = True
if USE_LAST_N_UNFREEZE:
    freeze_until = max(0, len(base_model.layers) - UNFREEZE_LAST_BASE_LAYERS)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
else:
    for layer in base_model.layers[:FINE_TUNE_FROM]:
        layer.trainable = False

# Keep BatchNorm frozen when fine-tuning with small batches.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable}", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_LR),
    loss=focal_loss,
    metrics=[
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc'),
    ]
)

cosine_lr = CosineAnnealingScheduler(
    initial_lr=FINE_LR,
    total_epochs=PHASE2_EP,
    min_lr=1e-7,
)
early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=PATIENCE,
    restore_best_weights=True, verbose=1,
)

t2 = time.time()
h2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE2_EP,
    class_weight=class_weight,
    # Reuse the same checkpoint callback to track the global best across phases.
    callbacks=[checkpoint, early_stop_2, cosine_lr, lr_logger, progress],
    verbose=0,
)
t2_elapsed = (time.time() - t2) / 60
best_val_2 = max(h2.history['val_accuracy'])
csv_log.append(h2, 'phase2')
print(
    f"\nPhase 2 done: {len(h2.history['accuracy'])} epochs, "
    f"{t2_elapsed:.1f} min, best val_acc={best_val_2:.4f}",
    flush=True,
)


# ── Save final model ─────────────────────────────────────────────────
print(f"\nSaving final model ...", flush=True)
model.save(FINAL_MODEL_PATH)
print(f"  best_model.keras  -> best validation accuracy checkpoint", flush=True)
print(f"  final_model.keras -> model state at end of training", flush=True)

csv_log.save()
print(f"Saved history -> {HISTORY_PATH}", flush=True)

total_time = t1_elapsed + t2_elapsed
best_val   = max(best_val_1, best_val_2)

config = {
    "model":                  "MobileNetV2",
    "num_classes":            NUM_CLASSES,
    "class_names":            class_names,
    "crops":                  list(ALLOWED_CROPS),
    "image_size":             list(IMG_SIZE),
    "batch_size":             BATCH,
    "data_pipeline": {
        "data_autotune": DATA_AUTOTUNE,
        "parallel_calls": PARALLEL_CALLS,
        "prefetch": PREFETCH_BUFFER,
        "inter_op_threads": INTER_OP_THREADS,
        "intra_op_threads": INTRA_OP_THREADS,
    },
    "seed":                   SEED,
    "loss":                   f"FocalLoss(gamma=2.0, alpha=0.25, ol_smoothing={OL_LABEL_SMOOTHING})",
    "other_leaf_weight_multiplier": OTHER_LEAF_WEIGHT_MULTIPLIER,
    "focus_crop_weight_multiplier": FOCUS_CROP_WEIGHT_MULTIPLIER,
    "enable_focus_crop_weighting": ENABLE_FOCUS_CROP_WEIGHTING,
    "other_leaf_label_smoothing":   OL_LABEL_SMOOTHING,
    "other_leaf_mixup_alpha":       OL_MIXUP_ALPHA,
    "enable_ol_mixup":              ENABLE_OL_MIXUP,
    "enable_ol_label_smoothing":    ENABLE_OL_LABEL_SMOOTHING,
    "max_class_weight":       MAX_CLASS_WEIGHT,
    "use_last_n_unfreeze":    USE_LAST_N_UNFREEZE,
    "unfreeze_last_base_layers": UNFREEZE_LAST_BASE_LAYERS,
    "fine_tune_from_layer":   FINE_TUNE_FROM,
    "phase1_epochs":          len(h1.history['accuracy']),
    "phase2_epochs":          len(h2.history['accuracy']),
    "total_epochs":           len(h1.history['accuracy']) + len(h2.history['accuracy']),
    "phase1_time_min":        round(t1_elapsed, 1),
    "phase2_time_min":        round(t2_elapsed, 1),
    "total_time_min":         round(total_time, 1),
    "best_val_accuracy":      round(float(best_val), 4),
    "phase1_best_val":        round(float(best_val_1), 4),
    "phase2_best_val":        round(float(best_val_2), 4),
    "best_model_path":        "best_model.keras",
    "final_model_path":       "final_model.keras",
    "phase2_lr_schedule":     "cosine_annealing",
    "calibration_defaults": {
        "min_confidence": DEFAULT_APP_CONFIDENCE_THRESHOLD,
        "min_margin": DEFAULT_APP_MARGIN_THRESHOLD,
        "max_entropy": DEFAULT_APP_MAX_ENTROPY,
        "temperature": 1.0,
    },
    "augmentation": {
        "crop_classes": {
            "flip":       "horizontal",
            "rotation":   AUG_ROTATION,
            "zoom":       AUG_ZOOM,
            "contrast":   AUG_CONTRAST,
            "brightness": AUG_BRIGHTNESS,
        },
        "other_leaf": {
            "flip":       "horizontal_and_vertical",
            "rotation":   OL_AUG_ROTATION,
            "zoom":       OL_AUG_ZOOM,
            "contrast":   OL_AUG_CONTRAST,
            "brightness": OL_AUG_BRIGHTNESS,
            "mixup_alpha": OL_MIXUP_ALPHA,
            "label_smoothing": OL_LABEL_SMOOTHING,
        },
    },
}
with open(os.path.join(MODEL_DIR, "training_config.json"), 'w') as f:
    json.dump(config, f, indent=2)

field_failure_counts = {}
if os.path.isdir(FIELD_FAILURE_DIR):
    for cname in class_names:
        folder = os.path.join(FIELD_FAILURE_DIR, cname)
        if os.path.isdir(folder):
            field_failure_counts[cname] = len([
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            ])

dataset_stats = {
    "train_counts": class_counts,
    "field_failure_counts": field_failure_counts,
    "val_counts":   val_counts,
    "test_counts":  test_counts,
    "totals": {
        "train": total,
        "val":   val_total,
        "test":  test_total,
    },
    "leakage_check": {
        "by_class_stem": leakage_check,
        "by_class_hash": leakage_check_hash,
        "totals": {
            "train_val_overlap":  total_train_val_overlap,
            "train_test_overlap": total_train_test_overlap,
            "val_test_overlap":   total_val_test_overlap,
            "train_val_hash_overlap":  total_train_val_hash_overlap,
            "train_test_hash_overlap": total_train_test_hash_overlap,
            "val_test_hash_overlap":   total_val_test_hash_overlap,
        },
    },
}
with open(os.path.join(MODEL_DIR, "dataset_stats.json"), 'w') as f:
    json.dump(dataset_stats, f, indent=2)


# ── Final evaluation on independent test set ─────────────────────────
print("\nEvaluating best model on independent test set ...", flush=True)

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH,
    custom_objects={"FocalLoss": FocalLoss},
)

test_loss, test_acc, test_top3 = best_model.evaluate(test_ds, verbose=0)
print(f"Test metrics: loss={test_loss:.4f} acc={test_acc:.4f} top3={test_top3:.4f}", flush=True)

all_test_labels = []
all_test_preds  = []
all_test_probs  = []
for images, labels in test_ds:
    probs = best_model.predict(images, verbose=0)
    all_test_probs.append(probs)
    all_test_preds.extend(np.argmax(probs, axis=1))
    all_test_labels.extend(labels.numpy())

all_test_labels = np.array(all_test_labels)
all_test_preds  = np.array(all_test_preds)
all_test_probs  = np.concatenate(all_test_probs, axis=0)

all_val_labels = []
all_val_probs = []
for images, labels in val_ds:
    probs = best_model.predict(images, verbose=0)
    all_val_probs.append(probs)
    all_val_labels.extend(labels.numpy())
all_val_labels = np.array(all_val_labels)
all_val_probs = np.concatenate(all_val_probs, axis=0)

best_temperature, calibrated_val_nll = fit_temperature_from_probs(
    all_val_probs,
    all_val_labels,
)
raw_val_nll = nll_from_probs(all_val_probs, all_val_labels)
if best_temperature == 0.8:
    print(
        "WARNING: best temperature is at lower search boundary (0.8); "
        "consider extending the sweep range.",
        flush=True,
    )

print(
    f"Temperature scaling: raw_val_nll={raw_val_nll:.4f}, "
    f"calibrated_val_nll={calibrated_val_nll:.4f}, T={best_temperature:.2f}",
    flush=True,
)

calibrated_test_probs = temperature_scale_probs(all_test_probs, best_temperature)

max_test_probs  = np.max(all_test_probs, axis=1)
sorted_test_probs = np.sort(all_test_probs, axis=1)
top2_test_probs = sorted_test_probs[:, -2] if all_test_probs.shape[1] > 1 else np.zeros_like(max_test_probs)
test_margins = max_test_probs - top2_test_probs
test_entropy = -np.sum(
    all_test_probs * (np.log(np.clip(all_test_probs, 1e-10, 1.0)) / np.log(2.0)),
    axis=1,
)

precision_macro,    recall_macro,    f1_macro,    _ = precision_recall_fscore_support(
    all_test_labels, all_test_preds, average='macro',    zero_division=0)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_test_labels, all_test_preds, average='weighted', zero_division=0)

report = classification_report(
    all_test_labels, all_test_preds,
    target_names=class_names, digits=4, zero_division=0,
)

sorted_calibrated = np.sort(calibrated_test_probs, axis=1)
cal_top1 = np.max(calibrated_test_probs, axis=1)
cal_top2 = sorted_calibrated[:, -2] if calibrated_test_probs.shape[1] > 1 else np.zeros_like(cal_top1)
cal_margin = cal_top1 - cal_top2
cal_entropy = -np.sum(
    calibrated_test_probs * (np.log(np.clip(calibrated_test_probs, 1e-10, 1.0)) / np.log(2.0)),
    axis=1,
)

true_labels_names = [class_names[int(i)] for i in all_test_labels]
true_crops = [label_to_crop(x) for x in true_labels_names]

supported_crops, raw_crop_totals = aggregate_supported_crop_totals(all_test_probs, class_names)
_, calibrated_crop_totals = aggregate_supported_crop_totals(calibrated_test_probs, class_names)

crop_total_analysis = {}
recommended_crop_total_threshold = DEFAULT_APP_CROP_TOTAL_THRESHOLD
best_crop_total_score = -1.0

for threshold in CROP_TOTAL_THRESHOLDS:
    raw_metrics = evaluate_crop_total_gate(
        raw_crop_totals,
        true_crops,
        supported_crops,
        threshold,
    )
    calibrated_metrics = evaluate_crop_total_gate(
        calibrated_crop_totals,
        true_crops,
        supported_crops,
        threshold,
    )

    crop_total_analysis[str(threshold)] = {
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }

    if calibrated_metrics["score"] > best_crop_total_score:
        best_crop_total_score = calibrated_metrics["score"]
        recommended_crop_total_threshold = float(threshold)

runtime_gate_cfg = load_runtime_gate_config()
runtime_gate_summary, runtime_gate_rows = simulate_runtime_gates(
    raw_probs=all_test_probs,
    calibrated_probs=calibrated_test_probs,
    labels=all_test_labels,
    class_names=class_names,
    gate_cfg=runtime_gate_cfg,
)

runtime_gate_validation_path = os.path.join(MODEL_DIR, "runtime_gate_validation.json")
runtime_gate_rows_path = os.path.join(MODEL_DIR, "runtime_gate_validation_rows.tsv")

with open(runtime_gate_validation_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "gate_config": runtime_gate_cfg,
            "summary": runtime_gate_summary,
        },
        f,
        indent=2,
    )

with open(runtime_gate_rows_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "index",
            "trueLabel",
            "trueCrop",
            "gate",
            "resultType",
            "bestCrop",
            "bestClass",
            "bestClassProb",
            "bestCropTotal",
            "secondCropTotal",
            "otherLeaf",
            "otherLeafRaw",
            "otherLeafScaled",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(runtime_gate_rows)

with open(os.path.join(MODEL_DIR, "test_classification_report.txt"), 'w') as f:
    f.write("Independent Test Set Report\n")
    f.write(f"Model: {BEST_MODEL_PATH}\n")
    f.write(f"Loss: FocalLoss(gamma=2.0, alpha=0.25, ol_smoothing={OL_LABEL_SMOOTHING})\n")
    f.write(f"other_leaf weight multiplier: ×{OTHER_LEAF_WEIGHT_MULTIPLIER}\n")
    f.write(f"other_leaf MixUp alpha: {OL_MIXUP_ALPHA}\n\n")
    f.write(f"Test accuracy:          {test_acc:.4f}\n")
    f.write(f"Test top-3 accuracy:    {test_top3:.4f}\n")
    f.write(f"Macro precision:        {precision_macro:.4f}\n")
    f.write(f"Macro recall:           {recall_macro:.4f}\n")
    f.write(f"Macro F1:               {f1_macro:.4f}\n")
    f.write(f"Weighted precision:     {precision_weighted:.4f}\n")
    f.write(f"Weighted recall:        {recall_weighted:.4f}\n")
    f.write(f"Weighted F1:            {f1_weighted:.4f}\n\n")
    f.write(report)

# Confidence threshold analysis
confidence_thresholds = {}
for threshold in CONFIDENCE_THRESHOLDS:
    mask  = max_test_probs >= threshold
    count = int(np.sum(mask))
    if count > 0:
        acc      = float(np.mean(all_test_preds[mask] == all_test_labels[mask]))
        coverage = float(count / len(all_test_labels))
        confidence_thresholds[str(threshold)] = {
            "accuracy": round(acc, 4),
            "coverage": round(coverage, 4),
            "count":    count,
        }

# Margin and entropy calibration sweeps for safer app-side acceptance.
margin_thresholds = [0.10, 0.15, 0.20, 0.25]
margin_analysis = {}
for threshold in margin_thresholds:
    mask = test_margins >= threshold
    count = int(np.sum(mask))
    if count > 0:
        acc = float(np.mean(all_test_preds[mask] == all_test_labels[mask]))
        coverage = float(count / len(all_test_labels))
        margin_analysis[str(threshold)] = {
            "accuracy": round(acc, 4),
            "coverage": round(coverage, 4),
            "count": count,
        }

entropy_thresholds = [1.0, 1.2, 1.4, 1.5, 1.7]
entropy_analysis = {}
for threshold in entropy_thresholds:
    mask = test_entropy <= threshold
    count = int(np.sum(mask))
    if count > 0:
        acc = float(np.mean(all_test_preds[mask] == all_test_labels[mask]))
        coverage = float(count / len(all_test_labels))
        entropy_analysis[str(threshold)] = {
            "accuracy": round(acc, 4),
            "coverage": round(coverage, 4),
            "count": count,
        }

combined_mask = (
    (max_test_probs >= DEFAULT_APP_CONFIDENCE_THRESHOLD)
    & (test_margins >= DEFAULT_APP_MARGIN_THRESHOLD)
    & (test_entropy <= DEFAULT_APP_MAX_ENTROPY)
)
combined_count = int(np.sum(combined_mask))
combined_gate_metrics = {
    "confidence_threshold": DEFAULT_APP_CONFIDENCE_THRESHOLD,
    "margin_threshold": DEFAULT_APP_MARGIN_THRESHOLD,
    "max_entropy_threshold": DEFAULT_APP_MAX_ENTROPY,
    "coverage": round(float(combined_count / len(all_test_labels)), 4),
    "count": combined_count,
}
if combined_count > 0:
    combined_gate_metrics["accuracy"] = round(
        float(np.mean(all_test_preds[combined_mask] == all_test_labels[combined_mask])),
        4,
    )
else:
    combined_gate_metrics["accuracy"] = 0.0

combined_calibrated_mask = (
    (cal_top1 >= DEFAULT_APP_CONFIDENCE_THRESHOLD)
    & (cal_margin >= DEFAULT_APP_MARGIN_THRESHOLD)
    & (cal_entropy <= DEFAULT_APP_MAX_ENTROPY)
)
combined_calibrated_count = int(np.sum(combined_calibrated_mask))
combined_gate_calibrated = {
    "temperature": best_temperature,
    "coverage": round(float(combined_calibrated_count / len(all_test_labels)), 4),
    "count": combined_calibrated_count,
}
if combined_calibrated_count > 0:
    combined_gate_calibrated["accuracy"] = round(
        float(np.mean(all_test_preds[combined_calibrated_mask] == all_test_labels[combined_calibrated_mask])),
        4,
    )
else:
    combined_gate_calibrated["accuracy"] = 0.0

# Hard negative mining export from test split.
test_file_paths = []
for cname in class_names:
    cdir = os.path.join(TEST_DIR, cname)
    if not os.path.isdir(cdir):
        continue
    for fname in sorted(os.listdir(cdir)):
        fpath = os.path.join(cdir, fname)
        if os.path.isfile(fpath):
            test_file_paths.append(fpath)

assert len(test_file_paths) == len(all_test_labels), (
    f"Hard-negative path alignment failed: paths={len(test_file_paths)} "
    f"labels={len(all_test_labels)}"
)

alignment_mismatches = []
for i, fpath in enumerate(test_file_paths):
    folder_name = os.path.basename(os.path.dirname(fpath))
    expected_class = class_names[int(all_test_labels[i])]
    if folder_name != expected_class:
        alignment_mismatches.append({
            "index": i,
            "path_class": folder_name,
            "label_class": expected_class,
            "file": fpath,
        })
        if len(alignment_mismatches) >= 5:
            break

assert len(alignment_mismatches) == 0, (
    "Hard-negative path/class alignment mismatch. "
    f"Examples: {alignment_mismatches}"
)

hard_negatives = []
for i in range(len(all_test_labels)):
    if all_test_preds[i] == all_test_labels[i]:
        continue
    probs = all_test_probs[i]
    top_idx = np.argsort(probs)[::-1][:3]
    hard_negatives.append({
        "file": test_file_paths[i],
        "true_label": class_names[int(all_test_labels[i])],
        "pred_label": class_names[int(all_test_preds[i])],
        "pred_confidence": round(float(probs[int(all_test_preds[i])]), 6),
        "margin": round(float(test_margins[i]), 6),
        "entropy": round(float(test_entropy[i]), 6),
        "top3": [
            {"label": class_names[int(j)], "prob": round(float(probs[int(j)]), 6)}
            for j in top_idx
        ],
    })
hard_negatives.sort(key=lambda x: x["pred_confidence"], reverse=True)

hard_negatives_path = os.path.join(MODEL_DIR, "hard_negatives_test.json")
with open(hard_negatives_path, 'w') as f:
    json.dump(hard_negatives[:MAX_HARD_NEGATIVES_EXPORT], f, indent=2)
print(f"Saved hard negatives -> {hard_negatives_path}", flush=True)

# Crop-pair and supported-vs-unsupported confusion diagnostics.
true_labels_names = [class_names[int(i)] for i in all_test_labels]
pred_labels_names = [class_names[int(i)] for i in all_test_preds]
pred_crops = [label_to_crop(x) for x in pred_labels_names]

unique_crops = sorted(set(true_crops + pred_crops))
crop_pair_counts = {c: {k: 0 for k in unique_crops} for c in unique_crops}
for t_crop, p_crop in zip(true_crops, pred_crops):
    crop_pair_counts[t_crop][p_crop] += 1

crop_pair_rates = {}
for t_crop in unique_crops:
    total_crop = sum(crop_pair_counts[t_crop].values())
    crop_pair_rates[t_crop] = {}
    for p_crop in unique_crops:
        crop_pair_rates[t_crop][p_crop] = round(
            (crop_pair_counts[t_crop][p_crop] / total_crop) if total_crop > 0 else 0.0,
            6,
        )

confusion_by_crop_pair = {
    "crops": unique_crops,
    "counts": crop_pair_counts,
    "rates": crop_pair_rates,
}
with open(os.path.join(MODEL_DIR, "confusion_by_crop_pair.json"), 'w') as f:
    json.dump(confusion_by_crop_pair, f, indent=2)

beans_potato_total = 0
beans_potato_confusions = 0
supported_total = 0
supported_to_unsupported = 0
unsupported_total = 0
unsupported_to_supported = 0

for t_crop, p_crop in zip(true_crops, pred_crops):
    if t_crop in ("beans", "potato"):
        beans_potato_total += 1
        if (t_crop == "beans" and p_crop == "potato") or (t_crop == "potato" and p_crop == "beans"):
            beans_potato_confusions += 1

    if t_crop != "other_leaf":
        supported_total += 1
        if p_crop == "other_leaf":
            supported_to_unsupported += 1
    else:
        unsupported_total += 1
        if p_crop != "other_leaf":
            unsupported_to_supported += 1

beans_potato_confusion_rate = (
    beans_potato_confusions / beans_potato_total if beans_potato_total > 0 else 0.0
)
supported_to_unsupported_rate = (
    supported_to_unsupported / supported_total if supported_total > 0 else 0.0
)
unsupported_to_supported_rate = (
    unsupported_to_supported / unsupported_total if unsupported_total > 0 else 0.0
)

supported_vs_unsupported_confusion = {
    "beans_potato": {
        "confusions": int(beans_potato_confusions),
        "total": int(beans_potato_total),
        "rate": round(float(beans_potato_confusion_rate), 6),
    },
    "supported_to_unsupported": {
        "confusions": int(supported_to_unsupported),
        "total": int(supported_total),
        "rate": round(float(supported_to_unsupported_rate), 6),
    },
    "unsupported_to_supported": {
        "confusions": int(unsupported_to_supported),
        "total": int(unsupported_total),
        "rate": round(float(unsupported_to_supported_rate), 6),
    },
}
with open(os.path.join(MODEL_DIR, "supported_vs_unsupported_confusion.json"), 'w') as f:
    json.dump(supported_vs_unsupported_confusion, f, indent=2)

# ── other_leaf specific metrics ──────────────────────────────────────
ol_mask_true = (all_test_labels == OTHER_LEAF_IDX)
ol_mask_pred = (all_test_preds  == OTHER_LEAF_IDX)
ol_tp = int(np.sum(ol_mask_true  & ol_mask_pred))
ol_fp = int(np.sum(~ol_mask_true & ol_mask_pred))
ol_fn = int(np.sum(ol_mask_true  & ~ol_mask_pred))
ol_precision = ol_tp / (ol_tp + ol_fp) if (ol_tp + ol_fp) > 0 else 0.0
ol_recall    = ol_tp / (ol_tp + ol_fn) if (ol_tp + ol_fn) > 0 else 0.0
ol_f1        = (2 * ol_precision * ol_recall / (ol_precision + ol_recall)
                if (ol_precision + ol_recall) > 0 else 0.0)

print(f"\nother_leaf class metrics:", flush=True)
print(f"  Precision: {ol_precision:.4f}  (of images predicted other_leaf, how many really were)", flush=True)
print(f"  Recall:    {ol_recall:.4f}  (of real other_leaf images, how many were caught)", flush=True)
print(f"  F1:        {ol_f1:.4f}", flush=True)
print(f"  TP={ol_tp}  FP={ol_fp}  FN={ol_fn}", flush=True)

# Confusion matrix
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm      = confusion_matrix(all_test_labels, all_test_preds)
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    short_names = [
        n.replace('banana_', 'B:').replace('beans_', 'Be:')
         .replace('maize_', 'M:').replace('potato_', 'P:')
         .replace('other_leaf', 'OL')
        for n in class_names
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12))

    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Test Confusion Matrix (Counts)')
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(short_names, fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Test Confusion Matrix (Normalized)')
    ax2.set_xticks(range(NUM_CLASSES))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(short_names, fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.suptitle(
        f'Independent Test Results — Accuracy: {test_acc:.2%} | '
        f'Macro F1: {f1_macro:.4f} | OL Recall: {ol_recall:.2%}'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "test_confusion_matrix.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Could not save confusion matrix: {e}", flush=True)

# Export TFLite (float16)
print("Exporting float16 TFLite model from best checkpoint ...", flush=True)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
tflite_path  = os.path.join(MODEL_DIR, "crop_disease_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite -> {tflite_path}", flush=True)

test_eval = {
    "test_loss":            round(float(test_loss), 4),
    "test_accuracy":        round(float(test_acc), 4),
    "test_top3_accuracy":   round(float(test_top3), 4),
    "macro_precision":      round(float(precision_macro), 4),
    "macro_recall":         round(float(recall_macro), 4),
    "macro_f1":             round(float(f1_macro), 4),
    "weighted_precision":   round(float(precision_weighted), 4),
    "weighted_recall":      round(float(recall_weighted), 4),
    "weighted_f1":          round(float(f1_weighted), 4),
    "other_leaf": {
        "precision": round(ol_precision, 4),
        "recall":    round(ol_recall, 4),
        "f1":        round(ol_f1, 4),
        "tp": ol_tp, "fp": ol_fp, "fn": ol_fn,
    },
    "confidence_thresholds":      confidence_thresholds,
    "margin_thresholds":          margin_analysis,
    "entropy_thresholds":         entropy_analysis,
    "crop_total_thresholds":      crop_total_analysis,
    "combined_gate":              combined_gate_metrics,
    "combined_gate_calibrated":   combined_gate_calibrated,
    "runtime_gate_validation": {
        "summary": runtime_gate_summary,
        "artifact_path": runtime_gate_validation_path,
        "rows_path": runtime_gate_rows_path,
    },
    "temperature_scaling": {
        "best_temperature": round(float(best_temperature), 4),
        "raw_val_nll": round(float(raw_val_nll), 4),
        "calibrated_val_nll": round(float(calibrated_val_nll), 4),
    },
    "hard_negative_mining": {
        "export_path": hard_negatives_path,
        "exported_count": int(min(len(hard_negatives), MAX_HARD_NEGATIVES_EXPORT)),
        "total_misclassified": int(len(hard_negatives)),
        "path_alignment_ok": bool(len(test_file_paths) == len(all_test_labels)),
    },
    "confusion_diagnostics": {
        "crop_pair_path": os.path.join(MODEL_DIR, "confusion_by_crop_pair.json"),
        "supported_vs_unsupported_path": os.path.join(MODEL_DIR, "supported_vs_unsupported_confusion.json"),
        "beans_potato_confusion_rate": round(float(beans_potato_confusion_rate), 6),
        "supported_to_unsupported_rate": round(float(supported_to_unsupported_rate), 6),
        "unsupported_to_supported_rate": round(float(unsupported_to_supported_rate), 6),
    },
    "strict_release_policy": {
        "field_min_supported_confident_rate": STRICT_FIELD_MIN_SUPPORTED_CONFIDENT_RATE,
        "beans_potato_confusion_max": STRICT_BEANS_POTATO_CONFUSION_MAX,
        "supported_unsupported_confusion_max": STRICT_SUPPORTED_UNSUPPORTED_CONFUSION_MAX,
    },
    "recommended_app_threshold":  DEFAULT_APP_CONFIDENCE_THRESHOLD,
    "recommended_crop_total_threshold": round(float(recommended_crop_total_threshold), 4),
}

release_blockers = []
if beans_potato_confusion_rate > STRICT_BEANS_POTATO_CONFUSION_MAX:
    release_blockers.append(
        f"test_beans_potato_confusion_rate={beans_potato_confusion_rate:.4f} exceeds {STRICT_BEANS_POTATO_CONFUSION_MAX:.4f}"
    )
if supported_to_unsupported_rate > STRICT_SUPPORTED_UNSUPPORTED_CONFUSION_MAX:
    release_blockers.append(
        f"test_supported_to_unsupported_rate={supported_to_unsupported_rate:.4f} exceeds {STRICT_SUPPORTED_UNSUPPORTED_CONFUSION_MAX:.4f}"
    )
if unsupported_to_supported_rate > STRICT_SUPPORTED_UNSUPPORTED_CONFUSION_MAX:
    release_blockers.append(
        f"test_unsupported_to_supported_rate={unsupported_to_supported_rate:.4f} exceeds {STRICT_SUPPORTED_UNSUPPORTED_CONFUSION_MAX:.4f}"
    )

test_eval["release_gate"] = {
    "test_metrics_pass": len(release_blockers) == 0,
    "field_evidence_required": True,
    "field_readiness_validator": "scripts/validate_release_readiness.py",
    "release_blockers": release_blockers,
    "release_verdict": "blocked" if len(release_blockers) > 0 else "pending_field_validation",
}

if os.path.isdir(REAL_WORLD_OOD_DIR):
    print("Running optional real-world OOD check ...", flush=True)
    try:
        ood_ds = tf.keras.utils.image_dataset_from_directory(
            REAL_WORLD_OOD_DIR,
            image_size=IMG_SIZE,
            batch_size=BATCH,
            label_mode='int',
            shuffle=False,
        )
        print(
            f"OOD directory classes discovered (labels ignored for metric): {ood_ds.class_names}",
            flush=True,
        )
        ood_ds = ood_ds.map(preprocess, num_parallel_calls=MAP_PARALLEL_CALLS).prefetch(PREFETCH_SIZE)

        ood_probs = []
        for images, _ in ood_ds:
            probs = best_model.predict(images, verbose=0)
            ood_probs.append(probs)

        if len(ood_probs) > 0:
            ood_probs = np.concatenate(ood_probs, axis=0)
            ood_top1 = np.max(ood_probs, axis=1)
            ood_sorted = np.sort(ood_probs, axis=1)
            ood_top2 = ood_sorted[:, -2] if ood_probs.shape[1] > 1 else np.zeros_like(ood_top1)
            ood_margin = ood_top1 - ood_top2
            ood_entropy = -np.sum(
                ood_probs * (np.log(np.clip(ood_probs, 1e-10, 1.0)) / np.log(2.0)),
                axis=1,
            )

            ood_reject_mask = (
                (ood_top1 < DEFAULT_APP_CONFIDENCE_THRESHOLD)
                | (ood_margin < DEFAULT_APP_MARGIN_THRESHOLD)
                | (ood_entropy > DEFAULT_APP_MAX_ENTROPY)
            )
            reject_rate = float(np.mean(ood_reject_mask))
            test_eval["real_world_ood"] = {
                "path": REAL_WORLD_OOD_DIR,
                "samples": int(len(ood_top1)),
                "reject_rate": round(reject_rate, 4),
                "accept_rate": round(1.0 - reject_rate, 4),
            }
        else:
            print("WARNING: OOD directory loaded but yielded zero valid images.", flush=True)
            test_eval["real_world_ood"] = {
                "path": REAL_WORLD_OOD_DIR,
                "samples": 0,
                "status": "no_valid_images",
            }
    except Exception as exc:
        print(f"WARNING: OOD evaluation skipped due to error: {exc}", flush=True)
        test_eval["real_world_ood"] = {
            "path": REAL_WORLD_OOD_DIR,
            "samples": 0,
            "status": "error",
            "error": str(exc),
        }
with open(os.path.join(MODEL_DIR, "test_evaluation.json"), 'w') as f:
    json.dump(test_eval, f, indent=2)

# Export mobile-runtime recommendations so app-side temperature and gates can
# be synchronized with the trained checkpoint calibration.
mobile_runtime_recommendations = {
    "temperatureScaling": round(float(best_temperature), 4),
    "recommendedThresholds": {
        "confidenceThreshold": DEFAULT_APP_CONFIDENCE_THRESHOLD,
        "marginThreshold": DEFAULT_APP_MARGIN_THRESHOLD,
        "maxEntropyThreshold": DEFAULT_APP_MAX_ENTROPY,
        "cropTotalThreshold": round(float(recommended_crop_total_threshold), 4),
    },
    "notes": [
        "Update mobile_app runtime config to use this temperatureScaling.",
        "Crop-total threshold is calibrated from crop-group probability sums on test data.",
        "Keep app gate thresholds aligned with these values after retrain.",
    ],
}
runtime_reco_path = os.path.join(MODEL_DIR, "mobile_runtime_recommendations.json")
with open(runtime_reco_path, 'w') as f:
    json.dump(mobile_runtime_recommendations, f, indent=2)

if ENABLE_GRAD_CAM_EXPORT and len(hard_negatives) > 0:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Pick last convolutional feature layer for Grad-CAM. For wrapped
        # backbones, recurse into nested Model layers.
        def find_last_conv_layer(model_obj):
            for layer in reversed(model_obj.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return model_obj, layer
                if isinstance(layer, tf.keras.Model):
                    nested_model, nested_layer = find_last_conv_layer(layer)
                    if nested_layer is not None:
                        return nested_model, nested_layer
            return None, None

        conv_parent_model, conv_layer = find_last_conv_layer(best_model)
        if conv_layer is not None and conv_parent_model is not None:
            grad_model = tf.keras.models.Model(
                [best_model.inputs],
                [conv_layer.output, best_model.output],
            )
            explain_dir = os.path.join(MODEL_DIR, "explainability")
            os.makedirs(explain_dir, exist_ok=True)

            for i, sample in enumerate(hard_negatives[:GRAD_CAM_MAX_IMAGES]):
                img_path = sample["file"]
                raw_img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                raw_arr = np.array(raw_img).astype(np.float32)
                inp = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(raw_arr, axis=0))

                with tf.GradientTape() as tape:
                    conv_out, preds = grad_model(inp, training=False)
                    pred_idx = tf.argmax(preds[0])
                    loss = preds[:, pred_idx]

                grads = tape.gradient(loss, conv_out)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_out = conv_out[0]
                heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
                heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
                heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()

                plt.figure(figsize=(6, 6))
                plt.imshow(raw_arr.astype(np.uint8))
                plt.imshow(heatmap, cmap='jet', alpha=0.35)
                plt.axis('off')
                plt.title(f"true={sample['true_label']} pred={sample['pred_label']}")
                out_path = os.path.join(explain_dir, f"gradcam_hard_negative_{i+1:02d}.png")
                plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()
            print(f"Saved Grad-CAM overlays -> {explain_dir}", flush=True)
        else:
            print("Could not find a Conv2D layer for Grad-CAM export.", flush=True)
    except Exception as e:
        print(f"Could not save Grad-CAM overlays: {e}", flush=True)

del best_model
gc.collect()

# Training curves
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_acc     = h1.history['accuracy']     + h2.history['accuracy']
    all_val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    all_loss    = h1.history['loss']         + h2.history['loss']
    all_val_loss= h1.history['val_loss']     + h2.history['val_loss']
    epochs      = range(1, len(all_acc) + 1)
    phase1_end  = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, all_acc,     'b-', label='Train')
    ax1.plot(epochs, all_val_acc, 'r-', label='Validation')
    ax1.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, all_loss,     'b-', label='Train')
    ax2.plot(epochs, all_val_loss, 'r-', label='Validation')
    ax2.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
    ax2.set_title('Loss (Focal)')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'4-Crop Model Training (Best Val Acc: {best_val:.2%})')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("Saved training curves plot", flush=True)
except Exception as e:
    print(f"Could not save plot: {e}", flush=True)


# ── Final summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"TRAINING COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
print(f"Crops: Banana, Beans, Maize, Potato", flush=True)
print(f"Classes: {NUM_CLASSES}", flush=True)
print(f"Loss: FocalLoss(gamma=2.0, alpha=0.25, ol_smoothing={OL_LABEL_SMOOTHING})", flush=True)
print(f"other_leaf weight boost:    ×{OTHER_LEAF_WEIGHT_MULTIPLIER}", flush=True)
print(f"other_leaf MixUp alpha:     {OL_MIXUP_ALPHA}", flush=True)
print(f"other_leaf label smoothing: ε={OL_LABEL_SMOOTHING}", flush=True)
print(f"Training images: {total}", flush=True)
print(f"Validation images: {val_total}", flush=True)
print(f"Test images: {test_total}", flush=True)
print(f"Fine-tune from layer: {FINE_TUNE_FROM}", flush=True)
print(f"Phase 1: {len(h1.history['accuracy'])} epochs, {t1_elapsed:.1f} min, best val={best_val_1:.4f}", flush=True)
print(f"Phase 2: {len(h2.history['accuracy'])} epochs, {t2_elapsed:.1f} min, best val={best_val_2:.4f}", flush=True)
print(f"Total: {total_time:.1f} min ({total_time/60:.1f} hrs)", flush=True)
print(f"Best val accuracy: {best_val:.2%}", flush=True)
print(f"Independent test accuracy: {test_acc:.2%}", flush=True)
print(f"Independent test macro F1: {f1_macro:.4f}", flush=True)
print(f"other_leaf recall: {ol_recall:.2%}  precision: {ol_precision:.2%}  F1: {ol_f1:.4f}", flush=True)
print(f"Recommended app confidence threshold: {DEFAULT_APP_CONFIDENCE_THRESHOLD:.2f}", flush=True)
print(f"Models saved:", flush=True)
print(f"  {BEST_MODEL_PATH}", flush=True)
print(f"  {FINAL_MODEL_PATH}", flush=True)
print(f"  {tflite_path}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'test_evaluation.json')}", flush=True)
print(f"  {runtime_reco_path}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'test_classification_report.txt')}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'test_confusion_matrix.png')}", flush=True)
print("DONE!", flush=True)