"""
Run a one-pass field audit with gate-level tracing for balanced deployment.

Usage:
  python -u scripts/audit_field_photos.py --images-dir "C:\\Users\\Mbakenge\\Downloads\\test_app"

Outputs:
  analysis_outputs/field_audit_summary.json
  analysis_outputs/field_audit_rows.tsv
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "crop_disease_model.tflite"
LABELS_PATH = ROOT / "models" / "labels.txt"
THRESHOLDS_PATH = ROOT / "mobile_app" / "assets" / "config" / "thresholds.json"
MODELS_RUNTIME_RECO_PATH = ROOT / "models" / "mobile_runtime_recommendations.json"
APP_RUNTIME_RECO_PATH = ROOT / "mobile_app" / "assets" / "config" / "mobile_runtime_recommendations.json"
OUT_DIR = ROOT / "analysis_outputs"

INPUT_SIZE = 224
DEFAULT_TEMPERATURE_SCALING = 1.8

DEFAULT_CROP_TOTAL_THRESHOLD = 0.82
DEFAULT_OTHER_LEAF_ABSOLUTE_FLOOR = 0.12
OTHER_LEAF_THRESHOLD = 0.40
DEFAULT_OTHER_LEAF_VS_CROP_RATIO_THRESHOLD = 0.30
OTHER_LEAF_VS_CROP_RATIO_THRESHOLD = DEFAULT_OTHER_LEAF_VS_CROP_RATIO_THRESHOLD
UNCERTAIN_GAP_THRESHOLD = 0.30
SECOND_CROP_AMBIGUITY_THRESHOLD = 0.15
MAX_ENTROPY_THRESHOLD = 1.5
HEALTHY_MIN_CONFIDENCE = 0.80
POTATO_HEALTHY_MIN_CONFIDENCE_PILOT = 0.72

DEFAULT_BEANS_POTATO_CROP_TOTAL_RELAXATION = 0.05
DEFAULT_BEANS_POTATO_OTHER_LEAF_FLOOR_BOOST = 0.03
NON_FOCUS_CROP_TOTAL_RELAXATION = 0.05
NON_FOCUS_OTHER_LEAF_FLOOR_BOOST = 0.01
NON_FOCUS_MAX_ENTROPY_THRESHOLD = 1.5
NON_FOCUS_CLASS_CONFIDENCE_THRESHOLD = 0.55
RESCUE_FOCUS_SWAP_GUARD_CROP_GAP = 0.08
RESCUE_FOCUS_SWAP_GUARD_TOP_CLASS_MARGIN = 0.05
DEFAULT_BEANS_POTATO_GAP_THRESHOLD = 0.08
DEFAULT_BEANS_POTATO_SECOND_CROP_THRESHOLD = 0.55
DEFAULT_BEANS_POTATO_CLASS_RATIO_THRESHOLD = 0.45
DEFAULT_BEANS_POTATO_CLASS_CONFIDENCE_THRESHOLD = 0.68
SECONDARY_FOCUS_MIN_CROP_TOTAL = 0.001
SECONDARY_FOCUS_MIN_TOP_CLASS_PROB = 0.001
SECONDARY_FOCUS_MAX_GAP_FROM_BEST = 1.00

CROP_GROUPING = {
    "banana_cordana": "banana",
    "banana_healthy": "banana",
    "banana_pestalotiopsis": "banana",
    "banana_sigatoka": "banana",
    "beans_angular_leaf_spot": "beans",
    "beans_healthy": "beans",
    "beans_rust": "beans",
    "maize_common_rust": "maize",
    "maize_gray_leaf_spot": "maize",
    "maize_healthy": "maize",
    "maize_northern_leaf_blight": "maize",
    "potato_early_blight": "potato",
    "potato_healthy": "potato",
    "potato_late_blight": "potato",
}

HEALTHY_LABELS = {
    "banana": "banana_healthy",
    "beans": "beans_healthy",
    "maize": "maize_healthy",
    "potato": "potato_healthy",
}


def is_beans_or_potato(crop: str) -> bool:
    return crop in ("beans", "potato")


def infer_expected_crop(path: Path) -> str:
    text = str(path).lower()
    if "bean" in text or "beans" in text:
        return "beans"
    if "potato" in text:
        return "potato"
    if "banana" in text or "bana" in text:
        return "banana"
    if "maize" in text or "corn" in text:
        return "maize"
    return "unknown"


def infer_source_type(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("aug_"):
        return "augmented-like"
    if "screenshot" in name or name.startswith("2026"):
        return "screenshot-like"
    return "raw_photo-like"


def is_real_field_photo(source_type: str) -> bool:
    return source_type == "raw_photo-like"


def preprocess(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    lp = np.log(np.maximum(probs, 1e-10)) / t
    exps = np.exp(lp - np.max(lp))
    return exps / np.sum(exps)


def entropy(probs: np.ndarray) -> float:
    p = np.maximum(probs, 1e-10)
    return float(-np.sum(p * (np.log(p) / np.log(2.0))))


def aggregate_by_crop(prob_map):
    out = defaultdict(float)
    for label, p in prob_map.items():
        crop = CROP_GROUPING.get(label)
        if crop:
            out[crop] += float(p)
    return dict(out)


def classes_for_crop(crop: str, prob_map):
    return {k: float(v) for k, v in prob_map.items() if CROP_GROUPING.get(k) == crop}


def load_thresholds():
    cfg = {
        "cropTotalThreshold": DEFAULT_CROP_TOTAL_THRESHOLD,
        "otherLeafAbsoluteFloor": DEFAULT_OTHER_LEAF_ABSOLUTE_FLOOR,
        "beansPotatoCropTotalRelaxation": DEFAULT_BEANS_POTATO_CROP_TOTAL_RELAXATION,
        "beansPotatoOtherLeafFloorBoost": DEFAULT_BEANS_POTATO_OTHER_LEAF_FLOOR_BOOST,
        "beansPotatoUncertainGapThreshold": DEFAULT_BEANS_POTATO_GAP_THRESHOLD,
        "beansPotatoSecondCropThreshold": DEFAULT_BEANS_POTATO_SECOND_CROP_THRESHOLD,
        "beansPotatoClassRatioThreshold": DEFAULT_BEANS_POTATO_CLASS_RATIO_THRESHOLD,
        "beansPotatoClassConfidenceThreshold": DEFAULT_BEANS_POTATO_CLASS_CONFIDENCE_THRESHOLD,
        "otherLeafVsCropRatioThreshold": DEFAULT_OTHER_LEAF_VS_CROP_RATIO_THRESHOLD,
        "temperatureScaling": DEFAULT_TEMPERATURE_SCALING,
        "nonFocusClassConfidenceThreshold": NON_FOCUS_CLASS_CONFIDENCE_THRESHOLD,
        "nonFocusMaxEntropyThreshold": NON_FOCUS_MAX_ENTROPY_THRESHOLD,
    }
    if THRESHOLDS_PATH.exists():
        data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        for k, v in data.get("thresholds", {}).items():
            if k in cfg and isinstance(v, (int, float)):
                cfg[k] = float(v)

    # Prefer model-side runtime recommendations (fresh after retrain), then app-side.
    runtime_payload = None
    for path in (MODELS_RUNTIME_RECO_PATH, APP_RUNTIME_RECO_PATH):
        if not path.exists():
            continue
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                runtime_payload = candidate
                break
        except Exception:
            continue

    if isinstance(runtime_payload, dict):
        temperature = runtime_payload.get("temperatureScaling")
        if isinstance(temperature, (int, float)) and float(temperature) > 0:
            cfg["temperatureScaling"] = float(temperature)

        reco = runtime_payload.get("recommendedThresholds")
        if isinstance(reco, dict):
            crop_total = reco.get("cropTotalThreshold")
            if isinstance(crop_total, (int, float)) and 0 < float(crop_total) < 1:
                cfg["cropTotalThreshold"] = float(crop_total)

            conf = reco.get("confidenceThreshold")
            if isinstance(conf, (int, float)) and 0 < float(conf) < 1:
                cfg["nonFocusClassConfidenceThreshold"] = float(conf)

            max_entropy = reco.get("maxEntropyThreshold")
            if isinstance(max_entropy, (int, float)) and 0 < float(max_entropy) < 10:
                cfg["nonFocusMaxEntropyThreshold"] = float(max_entropy)

    return cfg


def maybe_rescue(candidate_crop: str, candidate_crop_total: float, other_leaf_prob: float) -> bool:
    if not is_beans_or_potato(candidate_crop):
        return False
    if candidate_crop_total < 0.04:
        return False
    if other_leaf_prob > 0.96:
        return False
    return True


def top_class_prob_for_crop(crop: str, prob_map: dict) -> float:
    crop_classes = classes_for_crop(crop, prob_map)
    if not crop_classes:
        return 0.0
    return max(float(v) for v in crop_classes.values())


def top_class_label_for_crop(crop: str, prob_map: dict):
    crop_classes = classes_for_crop(crop, prob_map)
    if not crop_classes:
        return None, 0.0
    best_label, best_prob = max(crop_classes.items(), key=lambda kv: kv[1])
    return best_label, float(best_prob)


def choose_focus_crop_candidate(
    beans_total: float,
    potato_total: float,
    beans_top_class: float,
    potato_top_class: float,
    beans_score: float,
    potato_score: float,
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


def select_secondary_focus_candidate(crop_probs: dict, prob_map: dict, best_crop_total: float):
    beans_total = float(crop_probs.get("beans", 0.0))
    potato_total = float(crop_probs.get("potato", 0.0))
    beans_top = top_class_prob_for_crop("beans", prob_map)
    potato_top = top_class_prob_for_crop("potato", prob_map)

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


def should_preserve_focus_crop_identity(candidate_crop: str, candidate_crop_total: float, prob_map: dict) -> bool:
    if not is_beans_or_potato(candidate_crop):
        return False
    top_class_prob = top_class_prob_for_crop(candidate_crop, prob_map)
    return candidate_crop_total >= 0.001 or top_class_prob >= 0.001


def build_focus_override_result(
    gate: str,
    crop: str,
    crop_total: float,
    confidence: float,
    second_crop_total: float,
    other_leaf: float,
    probs: np.ndarray,
    prob_map: dict,
    top: list,
):
    best_class, best_class_prob = top_class_label_for_crop(crop, prob_map)
    if not best_class:
        best_class = HEALTHY_LABELS.get(crop, f"{crop}_healthy")
        best_class_prob = float(prob_map.get(best_class, 0.0))

    return {
        "gate": gate,
        "resultType": "healthy" if "healthy" in best_class else "disease",
        "bestCrop": crop,
        "bestClass": best_class,
        "bestClassProb": float(max(confidence, best_class_prob)),
        "bestCropTotal": float(crop_total),
        "secondCropTotal": float(second_crop_total),
        "otherLeaf": float(other_leaf),
        "entropy": entropy(probs),
        "top5": top[:5],
    }


def rescue_swap_guard(candidate_crop: str, crop_probs: dict, prob_map: dict) -> bool:
    if not is_beans_or_potato(candidate_crop):
        return True

    beans_total = float(crop_probs.get("beans", 0.0))
    potato_total = float(crop_probs.get("potato", 0.0))
    if abs(beans_total - potato_total) > RESCUE_FOCUS_SWAP_GUARD_CROP_GAP:
        return True

    beans_top = top_class_prob_for_crop("beans", prob_map)
    potato_top = top_class_prob_for_crop("potato", prob_map)
    top_margin = (beans_top - potato_top) if candidate_crop == "beans" else (potato_top - beans_top)
    return top_margin >= RESCUE_FOCUS_SWAP_GUARD_TOP_CLASS_MARGIN


def build_focus_rescue_result(
    gate: str,
    crop: str,
    crop_total: float,
    second_crop_total: float,
    other_leaf: float,
    probs: np.ndarray,
    prob_map: dict,
    top: list,
):
    crop_classes = classes_for_crop(crop, prob_map)
    sorted_classes = sorted(crop_classes.items(), key=lambda kv: kv[1], reverse=True)
    if sorted_classes:
        best_class, best_class_prob = sorted_classes[0]
    else:
        best_class = HEALTHY_LABELS.get(crop, f"{crop}_healthy")
        best_class_prob = 0.0

    return {
        "gate": gate,
        "resultType": "healthy" if "healthy" in best_class else "disease",
        "bestCrop": crop,
        "bestClass": best_class,
        "bestClassProb": best_class_prob,
        "bestCropTotal": crop_total,
        "secondCropTotal": second_crop_total,
        "otherLeaf": other_leaf,
        "entropy": entropy(probs),
        "top5": top[:5],
    }


def classify_with_gates(path: Path, interpreter, in_idx, out_idx, labels, cfg):
    x = preprocess(path).astype(np.float32)
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_idx)[0].astype(np.float64)

    probs = temperature_scale(raw, cfg.get("temperatureScaling", DEFAULT_TEMPERATURE_SCALING))
    prob_map = {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}
    top = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)

    other_leaf = prob_map.get("other_leaf", 0.0)
    crop_probs = aggregate_by_crop(prob_map)
    sorted_crops = sorted(crop_probs.items(), key=lambda kv: kv[1], reverse=True)

    best_candidate_crop = sorted_crops[0][0] if sorted_crops else "unknown"
    best_candidate_crop_total = sorted_crops[0][1] if sorted_crops else 0.0
    secondary_focus_candidate = select_secondary_focus_candidate(
        crop_probs,
        prob_map,
        best_crop_total=best_candidate_crop_total,
    )

    effective_other_leaf_floor = cfg["otherLeafAbsoluteFloor"]
    if is_beans_or_potato(best_candidate_crop):
        effective_other_leaf_floor = min(
            0.95,
            cfg["otherLeafAbsoluteFloor"] + cfg["beansPotatoOtherLeafFloorBoost"],
        )
    else:
        effective_other_leaf_floor = min(
            0.95,
            cfg["otherLeafAbsoluteFloor"] + NON_FOCUS_OTHER_LEAF_FLOOR_BOOST,
        )

    if other_leaf >= OTHER_LEAF_THRESHOLD:
        preserve_crop = best_candidate_crop
        preserve_crop_total = best_candidate_crop_total
        if secondary_focus_candidate is not None:
            preserve_crop, preserve_crop_total = secondary_focus_candidate

        if should_preserve_focus_crop_identity(preserve_crop, preserve_crop_total, prob_map):
            return build_focus_override_result(
                gate="G5_preserve_focus_crop",
                crop=preserve_crop,
                crop_total=preserve_crop_total,
                confidence=max(preserve_crop_total, top_class_prob_for_crop(preserve_crop, prob_map)),
                second_crop_total=sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
                other_leaf=other_leaf,
                probs=probs,
                prob_map=prob_map,
                top=top,
            )

        return {
            "gate": "G5_other_leaf_winner",
            "resultType": "other_leaf",
            "bestCrop": best_candidate_crop,
            "bestCropTotal": best_candidate_crop_total,
            "secondCropTotal": sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
            "otherLeaf": other_leaf,
            "entropy": entropy(probs),
            "top5": top[:5],
        }

    if other_leaf > effective_other_leaf_floor:
        preserve_crop = best_candidate_crop
        preserve_crop_total = best_candidate_crop_total
        if secondary_focus_candidate is not None:
            preserve_crop, preserve_crop_total = secondary_focus_candidate

        if should_preserve_focus_crop_identity(preserve_crop, preserve_crop_total, prob_map):
            return build_focus_override_result(
                gate="G5b_preserve_focus_crop",
                crop=preserve_crop,
                crop_total=preserve_crop_total,
                confidence=max(preserve_crop_total, top_class_prob_for_crop(preserve_crop, prob_map)),
                second_crop_total=sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
                other_leaf=other_leaf,
                probs=probs,
                prob_map=prob_map,
                top=top,
            )

        return {
            "gate": "G5b_other_leaf_floor",
            "resultType": "other_leaf",
            "bestCrop": best_candidate_crop,
            "bestCropTotal": best_candidate_crop_total,
            "secondCropTotal": sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
            "otherLeaf": other_leaf,
            "entropy": entropy(probs),
            "top5": top[:5],
        }

    if not sorted_crops:
        return {
            "gate": "G6_no_crop_candidates",
            "resultType": "unsupported",
            "bestCrop": "unknown",
            "bestCropTotal": 0.0,
            "secondCropTotal": 0.0,
            "otherLeaf": other_leaf,
            "entropy": entropy(probs),
            "top5": top[:5],
        }

    best_crop, best_crop_total = sorted_crops[0]
    second_crop_total = sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0
    crop_gap = best_crop_total - second_crop_total

    effective_crop_total = cfg["cropTotalThreshold"]
    effective_gap = UNCERTAIN_GAP_THRESHOLD
    effective_second_crop = SECOND_CROP_AMBIGUITY_THRESHOLD
    effective_class_ratio = 0.60
    effective_class_confidence = float(cfg.get("nonFocusClassConfidenceThreshold", NON_FOCUS_CLASS_CONFIDENCE_THRESHOLD))
    effective_entropy_threshold = MAX_ENTROPY_THRESHOLD
    if is_beans_or_potato(best_crop):
        effective_crop_total = max(0.0, cfg["cropTotalThreshold"] - cfg["beansPotatoCropTotalRelaxation"])
        effective_gap = cfg["beansPotatoUncertainGapThreshold"]
        effective_second_crop = cfg["beansPotatoSecondCropThreshold"]
        effective_class_ratio = cfg["beansPotatoClassRatioThreshold"]
        effective_class_confidence = cfg["beansPotatoClassConfidenceThreshold"]
    else:
        effective_crop_total = max(0.0, cfg["cropTotalThreshold"] - NON_FOCUS_CROP_TOTAL_RELAXATION)
        effective_class_confidence = float(cfg.get("nonFocusClassConfidenceThreshold", NON_FOCUS_CLASS_CONFIDENCE_THRESHOLD))
        effective_entropy_threshold = float(cfg.get("nonFocusMaxEntropyThreshold", NON_FOCUS_MAX_ENTROPY_THRESHOLD))

    if best_crop_total < effective_crop_total:
        if (
            maybe_rescue(best_crop, best_crop_total, other_leaf)
            and rescue_swap_guard(best_crop, crop_probs, prob_map)
        ):
            return build_focus_rescue_result(
                gate="G7a_crop_total_focus_rescue",
                crop=best_crop,
                crop_total=best_crop_total,
                second_crop_total=second_crop_total,
                other_leaf=other_leaf,
                probs=probs,
                prob_map=prob_map,
                top=top,
            )
        return {
            "gate": "G7a_crop_total",
            "resultType": "unsupported",
            "bestCrop": best_crop,
            "bestCropTotal": best_crop_total,
            "secondCropTotal": second_crop_total,
            "otherLeaf": other_leaf,
            "entropy": entropy(probs),
            "top5": top[:5],
        }

    ratio = other_leaf / best_crop_total if best_crop_total > 0 else 0.0
    if ratio > cfg["otherLeafVsCropRatioThreshold"]:
        if (
            maybe_rescue(best_crop, best_crop_total, other_leaf)
            and rescue_swap_guard(best_crop, crop_probs, prob_map)
        ):
            return build_focus_rescue_result(
                gate="G7a_other_leaf_ratio_focus_rescue",
                crop=best_crop,
                crop_total=best_crop_total,
                second_crop_total=second_crop_total,
                other_leaf=other_leaf,
                probs=probs,
                prob_map=prob_map,
                top=top,
            )
        return {
            "gate": "G7a_other_leaf_ratio",
            "resultType": "other_leaf",
            "bestCrop": best_crop,
            "bestCropTotal": best_crop_total,
            "secondCropTotal": second_crop_total,
            "otherLeaf": other_leaf,
            "entropy": entropy(probs),
            "top5": top[:5],
        }

    crop_classes = classes_for_crop(best_crop, prob_map)
    sorted_classes = sorted(crop_classes.items(), key=lambda kv: kv[1], reverse=True)
    best_class, best_class_prob = sorted_classes[0]

    if crop_gap < effective_gap:
        if is_beans_or_potato(best_crop):
            return build_focus_override_result(
                gate="G7c_crop_gap_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=best_class_prob,
                second_crop_total=second_crop_total,
                other_leaf=other_leaf,
                probs=probs,
                prob_map=prob_map,
                top=top,
            )
        return {
            "gate": "G7c_crop_gap",
            "resultType": "uncertain",
            "bestCrop": best_crop,
            "bestClass": best_class,
            "bestClassProb": best_class_prob,
            "bestCropTotal": best_crop_total,
            "secondCropTotal": second_crop_total,
            "otherLeaf": other_leaf,
            "entropy": entropy(probs),
            "top5": top[:5],
        }

    if second_crop_total > effective_second_crop:
        if is_beans_or_potato(best_crop):
            return build_focus_override_result(
                gate="G7c_second_crop_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=best_class_prob,
                second_crop_total=second_crop_total,
                other_leaf=other_leaf,
                probs=probs,
                prob_map=prob_map,
                top=top,
            )
        return {
            "gate": "G7c_second_crop",
            "resultType": "uncertain",
            "bestCrop": best_crop,
            "bestClass": best_class,
            "bestClassProb": best_class_prob,
            "bestCropTotal": best_crop_total,
            "secondCropTotal": second_crop_total,
            "otherLeaf": other_leaf,
            "entropy": entropy(probs),
            "top5": top[:5],
        }

    e = entropy(probs)
    if e > effective_entropy_threshold:
        return {
            "gate": "G7c_entropy",
            "resultType": "uncertain",
            "bestCrop": best_crop,
            "bestClass": best_class,
            "bestClassProb": best_class_prob,
            "bestCropTotal": best_crop_total,
            "secondCropTotal": second_crop_total,
            "otherLeaf": other_leaf,
            "entropy": e,
            "top5": top[:5],
        }

    class_ratio = best_class_prob / best_crop_total if best_crop_total > 0 else 0.0
    if class_ratio < effective_class_ratio:
        return {
            "gate": "G7c_class_ratio",
            "resultType": "uncertain",
            "bestCrop": best_crop,
            "bestClass": best_class,
            "bestClassProb": best_class_prob,
            "bestCropTotal": best_crop_total,
            "secondCropTotal": second_crop_total,
            "otherLeaf": other_leaf,
            "entropy": e,
            "top5": top[:5],
        }

    if best_class_prob < effective_class_confidence:
        if is_beans_or_potato(best_crop):
            return build_focus_override_result(
                gate="G7d_class_confidence_focus_override",
                crop=best_crop,
                crop_total=best_crop_total,
                confidence=best_class_prob,
                second_crop_total=second_crop_total,
                other_leaf=other_leaf,
                probs=probs,
                prob_map=prob_map,
                top=top,
            )
        healthy_label = HEALTHY_LABELS.get(best_crop)
        healthy_prob = crop_classes.get(healthy_label, 0.0)
        if healthy_prob > 0 and abs(healthy_prob - best_class_prob) < 1e-12:
            gate = "G7d_class_confidence_healthy_candidate"
        else:
            gate = "G7d_class_confidence"
        return {
            "gate": gate,
            "resultType": "uncertain",
            "bestCrop": best_crop,
            "bestClass": best_class,
            "bestClassProb": best_class_prob,
            "bestCropTotal": best_crop_total,
            "secondCropTotal": second_crop_total,
            "otherLeaf": other_leaf,
            "entropy": e,
            "top5": top[:5],
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
                    other_leaf=other_leaf,
                    probs=probs,
                    prob_map=prob_map,
                    top=top,
                )

            return {
                "gate": "G7e_healthy_safety",
                "resultType": "uncertain",
                "bestCrop": best_crop,
                "bestClass": best_class,
                "bestClassProb": best_class_prob,
                "bestCropTotal": best_crop_total,
                "secondCropTotal": second_crop_total,
                "otherLeaf": other_leaf,
                "entropy": e,
                "top5": top[:5],
            }

    return {
        "gate": "G7g_confident",
        "resultType": "healthy" if "healthy" in best_class else "disease",
        "bestCrop": best_crop,
        "bestClass": best_class,
        "bestClassProb": best_class_prob,
        "bestCropTotal": best_crop_total,
        "secondCropTotal": second_crop_total,
        "otherLeaf": other_leaf,
        "entropy": e,
        "top5": top[:5],
    }


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def safe_div(a, b):
    return float(a / b) if b else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Folder containing field photos")
    parser.add_argument("--parity-gap-max", type=float, default=0.12)
    parser.add_argument(
        "--strict-real-photos",
        action="store_true",
        help="Only audit raw_photo-like files; skips screenshot-like and augmented-like inputs.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")

    labels = [x.strip() for x in LABELS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    if "other_leaf" not in labels:
        raise RuntimeError("labels.txt missing other_leaf")

    cfg = load_thresholds()

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    files = list_images(images_dir)
    rows = []

    for p in files:
        expected_crop = infer_expected_crop(p)
        source_type = infer_source_type(p)
        if args.strict_real_photos and not is_real_field_photo(source_type):
            continue
        try:
            pred = classify_with_gates(p, interpreter, in_det["index"], out_det["index"], labels, cfg)
            top5 = pred.get("top5", [])
            top1 = top5[0][0] if len(top5) > 0 else ""
            top1_p = top5[0][1] if len(top5) > 0 else 0.0
            top2 = top5[1][0] if len(top5) > 1 else ""
            top2_p = top5[1][1] if len(top5) > 1 else 0.0
            gate = pred["gate"]
            uncertainty_gates = {
                "G7c_crop_gap",
                "G7c_second_crop",
                "G7c_entropy",
                "G7c_class_ratio",
                "G7e_healthy_safety",
            }
            rows.append({
                "file": str(p),
                "expected_crop": expected_crop,
                "source_type": source_type,
                "resultType": pred["resultType"],
                "gate": gate,
                "is_real_field_photo": is_real_field_photo(source_type),
                "trigger_G5": gate in {"G5_other_leaf_winner", "G5_preserve_focus_crop"},
                "trigger_G5b": gate in {"G5b_other_leaf_floor", "G5b_preserve_focus_crop"},
                "trigger_G7a": gate in {
                    "G7a_crop_total",
                    "G7a_other_leaf_ratio",
                    "G7a_crop_total_focus_rescue",
                    "G7a_other_leaf_ratio_focus_rescue",
                },
                "trigger_uncertain": gate in uncertainty_gates,
                "trigger_uncertain_gap": gate == "G7c_crop_gap",
                "trigger_uncertain_second_crop": gate == "G7c_second_crop",
                "trigger_uncertain_entropy": gate == "G7c_entropy",
                "trigger_uncertain_class_ratio": gate == "G7c_class_ratio",
                "bestCrop": pred.get("bestCrop", ""),
                "bestClass": pred.get("bestClass", ""),
                "bestClassProb": pred.get("bestClassProb", 0.0),
                "bestCropTotal": pred.get("bestCropTotal", 0.0),
                "secondCropTotal": pred.get("secondCropTotal", 0.0),
                "otherLeaf": pred.get("otherLeaf", 0.0),
                "entropy": pred.get("entropy", 0.0),
                "top1": top1,
                "top1_p": top1_p,
                "top2": top2,
                "top2_p": top2_p,
                "top3": top5[2][0] if len(top5) > 2 else "",
                "top3_p": top5[2][1] if len(top5) > 2 else 0.0,
            })
        except Exception as exc:
            rows.append({
                "file": str(p),
                "expected_crop": expected_crop,
                "source_type": source_type,
                "resultType": "error",
                "gate": "ERROR",
                "is_real_field_photo": is_real_field_photo(source_type),
                "trigger_G5": False,
                "trigger_G5b": False,
                "trigger_G7a": False,
                "trigger_uncertain": False,
                "trigger_uncertain_gap": False,
                "trigger_uncertain_second_crop": False,
                "trigger_uncertain_entropy": False,
                "trigger_uncertain_class_ratio": False,
                "bestCrop": "",
                "bestClass": "",
                "bestClassProb": 0.0,
                "bestCropTotal": 0.0,
                "secondCropTotal": 0.0,
                "otherLeaf": 0.0,
                "entropy": 0.0,
                "top1": "",
                "top1_p": 0.0,
                "top2": "",
                "top2_p": 0.0,
                "top3": str(exc),
                "top3_p": 0.0,
            })

    known_rows = [r for r in rows if r["expected_crop"] != "unknown" and r["resultType"] != "error"]
    per_crop = {}
    for crop in ["banana", "beans", "maize", "potato"]:
        crows = [r for r in known_rows if r["expected_crop"] == crop]
        total = len(crows)
        supported_conf = sum(1 for r in crows if r["resultType"] in ("healthy", "disease") and r["bestCrop"] == crop)
        uncertain = sum(1 for r in crows if r["resultType"] == "uncertain")
        unsupported = sum(1 for r in crows if r["resultType"] in ("unsupported", "other_leaf"))
        wrong_confident = sum(1 for r in crows if r["resultType"] in ("healthy", "disease") and r["bestCrop"] != crop)
        per_crop[crop] = {
            "total": total,
            "supported_confident": supported_conf,
            "uncertain": uncertain,
            "unsupported_or_other_leaf": unsupported,
            "wrong_confident": wrong_confident,
            "supported_confident_rate": round(safe_div(supported_conf, total), 4),
            "uncertain_rate": round(safe_div(uncertain, total), 4),
            "unsupported_rate": round(safe_div(unsupported, total), 4),
        }

    gate_counts = Counter(r["gate"] for r in rows)
    result_counts = Counter(r["resultType"] for r in rows)
    source_counts = Counter(r["source_type"] for r in rows)

    unsupported_like = [r for r in rows if r["resultType"] in ("unsupported", "other_leaf")]
    unsupported_pairs = Counter((r["top1"], r["top2"]) for r in unsupported_like)

    pattern_counts = {
        "top1_other_leaf_top2_beans": sum(1 for r in unsupported_like if r["top1"] == "other_leaf" and str(r["top2"]).startswith("beans_")),
        "top1_other_leaf_top2_potato": sum(1 for r in unsupported_like if r["top1"] == "other_leaf" and str(r["top2"]).startswith("potato_")),
        "expected_crop_appears_top2_unsupported_like": sum(
            1
            for r in unsupported_like
            if (r["expected_crop"] == "beans" and str(r["top2"]).startswith("beans_"))
            or (r["expected_crop"] == "potato" and str(r["top2"]).startswith("potato_"))
            or (r["expected_crop"] == "banana" and str(r["top2"]).startswith("banana_"))
            or (r["expected_crop"] == "maize" and str(r["top2"]).startswith("maize_"))
        ),
    }

    # Acceptance parity target: beans/potato should not lag banana/maize
    # by more than parity-gap-max in supported_confident_rate.
    ref_rates = [per_crop[c]["supported_confident_rate"] for c in ("banana", "maize") if per_crop[c]["total"] > 0]
    ref_rate = sum(ref_rates) / len(ref_rates) if ref_rates else 0.0
    beans_ok = per_crop["beans"]["supported_confident_rate"] + args.parity_gap_max >= ref_rate
    potato_ok = per_crop["potato"]["supported_confident_rate"] + args.parity_gap_max >= ref_rate

    acceptance = {
        "reference_rate_banana_maize": round(ref_rate, 4),
        "parity_gap_max": args.parity_gap_max,
        "beans_pass": beans_ok,
        "potato_pass": potato_ok,
        "overall_pass": bool(beans_ok and potato_ok),
    }

    summary = {
        "images_dir": str(images_dir),
        "strict_real_photos": bool(args.strict_real_photos),
        "total_images": len(rows),
        "known_label_images": len(known_rows),
        "unknown_label_images": sum(1 for r in rows if r["expected_crop"] == "unknown"),
        "thresholds_used": cfg,
        "result_counts": dict(result_counts),
        "gate_counts": dict(gate_counts),
        "source_counts": dict(source_counts),
        "per_crop_reliability": per_crop,
        "unsupported_top1_top2": [
            {"top1": a, "top2": b, "count": c}
            for (a, b), c in unsupported_pairs.most_common(20)
        ],
        "pattern_counts": pattern_counts,
        "acceptance": acceptance,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "field_audit_summary.json"
    tsv_path = OUT_DIR / "field_audit_rows.tsv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    columns = [
        "expected_crop", "source_type", "file", "resultType", "gate", "bestCrop", "bestClass",
        "is_real_field_photo",
        "trigger_G5", "trigger_G5b", "trigger_G7a", "trigger_uncertain",
        "trigger_uncertain_gap", "trigger_uncertain_second_crop",
        "trigger_uncertain_entropy", "trigger_uncertain_class_ratio",
        "bestClassProb", "bestCropTotal", "secondCropTotal", "otherLeaf", "entropy",
        "top1", "top1_p", "top2", "top2_p", "top3", "top3_p",
    ]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            vals = [str(r.get(c, "")) for c in columns]
            f.write("\t".join(vals) + "\n")

    print("Wrote:", json_path)
    print("Wrote:", tsv_path)
    print("Acceptance:", acceptance)


if __name__ == "__main__":
    main()
