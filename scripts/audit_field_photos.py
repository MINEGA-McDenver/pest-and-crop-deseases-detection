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
OUT_DIR = ROOT / "analysis_outputs"

INPUT_SIZE = 224
TEMPERATURE = 1.8

CONFIDENT_CLASS_THRESHOLD = 0.80
DEFAULT_CROP_TOTAL_THRESHOLD = 0.84
DEFAULT_OTHER_LEAF_ABSOLUTE_FLOOR = 0.12
OTHER_LEAF_THRESHOLD = 0.30
OTHER_LEAF_VS_CROP_RATIO_THRESHOLD = 0.24
UNCERTAIN_GAP_THRESHOLD = 0.30
SECOND_CROP_AMBIGUITY_THRESHOLD = 0.15
MAX_ENTROPY_THRESHOLD = 1.5

DEFAULT_BEANS_POTATO_CROP_TOTAL_RELAXATION = 0.05
DEFAULT_BEANS_POTATO_OTHER_LEAF_FLOOR_BOOST = 0.03
DEFAULT_BEANS_POTATO_GAP_THRESHOLD = 0.22
DEFAULT_BEANS_POTATO_SECOND_CROP_THRESHOLD = 0.20
DEFAULT_BEANS_POTATO_CLASS_RATIO_THRESHOLD = 0.52
DEFAULT_BEANS_POTATO_CLASS_CONFIDENCE_THRESHOLD = 0.74

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
    name = path.name.lower()
    if "bean" in name:
        return "beans"
    if "potato" in name:
        return "potato"
    if "banana" in name:
        return "banana"
    if "maize" in name or "corn" in name:
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


def temperature_scale(probs: np.ndarray) -> np.ndarray:
    lp = np.log(np.maximum(probs, 1e-10)) / TEMPERATURE
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
    }
    if THRESHOLDS_PATH.exists():
        data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        for k, v in data.get("thresholds", {}).items():
            if k in cfg and isinstance(v, (int, float)):
                cfg[k] = float(v)
    return cfg


def maybe_rescue(candidate_crop: str, candidate_crop_total: float, other_leaf_prob: float) -> bool:
    if not is_beans_or_potato(candidate_crop):
        return False
    if candidate_crop_total < 0.04:
        return False
    if other_leaf_prob > 0.96:
        return False
    return True


def classify_with_gates(path: Path, interpreter, in_idx, out_idx, labels, cfg):
    x = preprocess(path).astype(np.float32)
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_idx)[0].astype(np.float64)

    probs = temperature_scale(raw)
    prob_map = {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}
    top = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)

    other_leaf = prob_map.get("other_leaf", 0.0)
    crop_probs = aggregate_by_crop(prob_map)
    sorted_crops = sorted(crop_probs.items(), key=lambda kv: kv[1], reverse=True)

    best_candidate_crop = sorted_crops[0][0] if sorted_crops else "unknown"
    best_candidate_crop_total = sorted_crops[0][1] if sorted_crops else 0.0

    effective_other_leaf_floor = cfg["otherLeafAbsoluteFloor"]
    if is_beans_or_potato(best_candidate_crop):
        effective_other_leaf_floor = min(
            0.95,
            cfg["otherLeafAbsoluteFloor"] + cfg["beansPotatoOtherLeafFloorBoost"],
        )

    if other_leaf >= OTHER_LEAF_THRESHOLD:
        if maybe_rescue(best_candidate_crop, best_candidate_crop_total, other_leaf):
            return {
                "gate": "RESCUE_beans_potato_from_G5",
                "resultType": "uncertain",
                "bestCrop": best_candidate_crop,
                "bestCropTotal": best_candidate_crop_total,
                "secondCropTotal": sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
                "otherLeaf": other_leaf,
                "entropy": entropy(probs),
                "top5": top[:5],
            }
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
        if maybe_rescue(best_candidate_crop, best_candidate_crop_total, other_leaf):
            return {
                "gate": "RESCUE_beans_potato_from_G5b",
                "resultType": "uncertain",
                "bestCrop": best_candidate_crop,
                "bestCropTotal": best_candidate_crop_total,
                "secondCropTotal": sorted_crops[1][1] if len(sorted_crops) > 1 else 0.0,
                "otherLeaf": other_leaf,
                "entropy": entropy(probs),
                "top5": top[:5],
            }
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
    effective_class_confidence = CONFIDENT_CLASS_THRESHOLD
    if is_beans_or_potato(best_crop):
        effective_crop_total = max(0.0, cfg["cropTotalThreshold"] - cfg["beansPotatoCropTotalRelaxation"])
        effective_gap = cfg["beansPotatoUncertainGapThreshold"]
        effective_second_crop = cfg["beansPotatoSecondCropThreshold"]
        effective_class_ratio = cfg["beansPotatoClassRatioThreshold"]
        effective_class_confidence = cfg["beansPotatoClassConfidenceThreshold"]

    if best_crop_total < effective_crop_total:
        if maybe_rescue(best_crop, best_crop_total, other_leaf):
            return {
                "gate": "RESCUE_beans_potato_from_G7a",
                "resultType": "uncertain",
                "bestCrop": best_crop,
                "bestCropTotal": best_crop_total,
                "secondCropTotal": second_crop_total,
                "otherLeaf": other_leaf,
                "entropy": entropy(probs),
                "top5": top[:5],
            }
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
    if ratio > OTHER_LEAF_VS_CROP_RATIO_THRESHOLD:
        if maybe_rescue(best_crop, best_crop_total, other_leaf):
            return {
                "gate": "RESCUE_beans_potato_from_G7a_ratio",
                "resultType": "uncertain",
                "bestCrop": best_crop,
                "bestCropTotal": best_crop_total,
                "secondCropTotal": second_crop_total,
                "otherLeaf": other_leaf,
                "entropy": entropy(probs),
                "top5": top[:5],
            }
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
    if e > MAX_ENTROPY_THRESHOLD:
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
            }
            rows.append({
                "file": str(p),
                "expected_crop": expected_crop,
                "source_type": source_type,
                "resultType": pred["resultType"],
                "gate": gate,
                "is_real_field_photo": is_real_field_photo(source_type),
                "trigger_G5": gate in {"G5_other_leaf_winner", "RESCUE_beans_potato_from_G5"},
                "trigger_G5b": gate in {"G5b_other_leaf_floor", "RESCUE_beans_potato_from_G5b"},
                "trigger_G7a": gate in {"G7a_crop_total", "RESCUE_beans_potato_from_G7a"},
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
