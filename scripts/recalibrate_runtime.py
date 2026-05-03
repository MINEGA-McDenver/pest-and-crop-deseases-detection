"""Recalibrate mobile runtime recommendations (no retraining).

What it does:
  1) Loads models/best_model.keras
  2) Runs inference on datasets/model_ready/{val,test}
  3) Fits temperature scaling (sweep includes T < 1.0)
  4) Recomputes recommended cropTotalThreshold from calibrated test probs
  5) Updates:
       - models/mobile_runtime_recommendations.json
       - mobile_app/assets/config/mobile_runtime_recommendations.json (unless --models-only)

Run:
  python -u scripts/recalibrate_runtime.py

Notes:
  - Uses log-prob temperature scaling to mirror mobile runtime behavior.
  - Does not change any other gate thresholds (confidence/margin/entropy).
"""

import argparse
import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("Loading TensorFlow ...", flush=True)
import tensorflow as tf


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-10, 1.0)) / max(float(temperature), 1e-6)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)


def nll_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    idx = np.arange(len(labels))
    p_true = np.clip(probs[idx, labels], 1e-10, 1.0)
    return float(-np.mean(np.log(p_true)))


def fit_temperature_from_probs(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    coarse_step: float,
    fine_step: float,
    fine_radius: float,
) -> tuple[float, float]:
    if t_min <= 0 or t_max <= 0 or t_max <= t_min:
        raise ValueError(f"Invalid temperature sweep range: min={t_min}, max={t_max}")

    temps = np.arange(float(t_min), float(t_max) + 1e-9, float(coarse_step))
    best_t = 1.0
    best_nll = nll_from_probs(val_probs, val_labels)

    for t in temps:
        scaled = temperature_scale_probs(val_probs, float(t))
        nll = nll_from_probs(scaled, val_labels)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    refine_min = max(float(t_min), best_t - float(fine_radius))
    refine_max = min(float(t_max), best_t + float(fine_radius))
    fine_temps = np.arange(refine_min, refine_max + 1e-9, float(fine_step))
    for t in fine_temps:
        scaled = temperature_scale_probs(val_probs, float(t))
        nll = nll_from_probs(scaled, val_labels)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return float(round(best_t, 4)), float(best_nll)


def label_to_crop(label_name: str) -> str:
    if label_name == "other_leaf":
        return "other_leaf"
    return label_name.split("_")[0]


def aggregate_supported_crop_totals(probs: np.ndarray, class_names: list[str]):
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


def evaluate_crop_total_gate(
    crop_totals: np.ndarray,
    true_crop_names: list[str],
    supported_crops: list[str],
    threshold: float,
):
    row_idx = np.arange(crop_totals.shape[0])
    best_crop_idx = np.argmax(crop_totals, axis=1)
    best_crop_total = crop_totals[row_idx, best_crop_idx]
    best_crop_names = np.array([supported_crops[int(i)] for i in best_crop_idx], dtype=object)

    true_crop_arr = np.array(true_crop_names, dtype=object)
    accepted_mask = best_crop_total >= float(threshold)
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


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _update_runtime_reco(path: str, *, temperature: float, crop_total_threshold: float, dry_run: bool):
    payload = _load_json(path)
    payload["temperatureScaling"] = round(float(temperature), 4)

    thresholds = payload.get("recommendedThresholds")
    if not isinstance(thresholds, dict):
        thresholds = {}
        payload["recommendedThresholds"] = thresholds

    thresholds["cropTotalThreshold"] = round(float(crop_total_threshold), 4)

    if dry_run:
        print(f"[dry-run] Would update: {path}", flush=True)
        return

    _write_json(path, payload)
    print(f"Updated: {path}", flush=True)


def _build_dataset(split_dir: str, *, img_size: tuple[int, int], batch: int):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=img_size,
        batch_size=int(batch),
        label_mode="int",
        shuffle=False,
    )

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

    return ds.map(preprocess, num_parallel_calls=2).prefetch(1)


def _collect_probs(model, ds):
    probs_chunks = []
    labels_all = []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        probs_chunks.append(probs)
        labels_all.extend(labels.numpy())
    return np.concatenate(probs_chunks, axis=0), np.array(labels_all)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recalibrate temperature + cropTotal threshold without retraining")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write any files")
    parser.add_argument("--models-only", action="store_true", help="Only update models/mobile_runtime_recommendations.json")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)

    parser.add_argument("--t-min", type=float, default=0.20)
    parser.add_argument("--t-max", type=float, default=3.00)
    parser.add_argument("--coarse-step", type=float, default=0.05)
    parser.add_argument("--fine-step", type=float, default=0.01)
    parser.add_argument("--fine-radius", type=float, default=0.10)

    args = parser.parse_args(argv)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base, "models")
    data_dir = os.path.join(base, "datasets", "model_ready")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    model_path = os.path.join(model_dir, "best_model.keras")
    labels_path = os.path.join(model_dir, "labels.txt")

    if not os.path.isfile(model_path):
        print(f"ERROR: Missing model: {model_path}", flush=True)
        return 2
    if not os.path.isfile(labels_path):
        print(f"ERROR: Missing labels: {labels_path}", flush=True)
        return 2

    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    img_size = (int(args.img_size), int(args.img_size))

    print("Loading model ...", flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Loading datasets ...", flush=True)
    # Validate class order matches labels.txt
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=int(args.batch), label_mode="int", shuffle=False
    )
    if raw_val_ds.class_names != class_names:
        print("ERROR: Validation class order differs from models/labels.txt", flush=True)
        print(f"  val_ds:     {raw_val_ds.class_names}")
        print(f"  labels.txt: {class_names}")
        return 3

    raw_test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=int(args.batch), label_mode="int", shuffle=False
    )
    if raw_test_ds.class_names != class_names:
        print("ERROR: Test class order differs from models/labels.txt", flush=True)
        print(f"  test_ds:    {raw_test_ds.class_names}")
        print(f"  labels.txt: {class_names}")
        return 3

    # Preprocess datasets
    val_ds = raw_val_ds.map(
        lambda images, labels: (
            tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(images, tf.float32)),
            labels,
        ),
        num_parallel_calls=2,
    ).prefetch(1)

    test_ds = raw_test_ds.map(
        lambda images, labels: (
            tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(images, tf.float32)),
            labels,
        ),
        num_parallel_calls=2,
    ).prefetch(1)

    print("Running inference on val split ...", flush=True)
    val_probs, val_labels = _collect_probs(model, val_ds)

    raw_val_nll = nll_from_probs(val_probs, val_labels)
    best_t, calibrated_val_nll = fit_temperature_from_probs(
        val_probs,
        val_labels,
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        coarse_step=float(args.coarse_step),
        fine_step=float(args.fine_step),
        fine_radius=float(args.fine_radius),
    )

    boundary_hit = False
    if abs(best_t - float(args.t_min)) < 1e-6:
        boundary_hit = True
        print(
            f"WARNING: best temperature is at lower search boundary ({args.t_min}); "
            "consider extending the sweep range.",
            flush=True,
        )
    if abs(best_t - float(args.t_max)) < 1e-6:
        boundary_hit = True
        print(
            f"WARNING: best temperature is at upper search boundary ({args.t_max}); "
            "consider extending the sweep range.",
            flush=True,
        )

    print(
        f"Temperature scaling: raw_val_nll={raw_val_nll:.4f}, calibrated_val_nll={calibrated_val_nll:.4f}, T={best_t:.4f}",
        flush=True,
    )

    print("Running inference on test split ...", flush=True)
    test_probs, test_labels = _collect_probs(model, test_ds)

    calibrated_test_probs = temperature_scale_probs(test_probs, best_t)

    supported_crops, _raw_crop_totals = aggregate_supported_crop_totals(test_probs, class_names)
    _, calibrated_crop_totals = aggregate_supported_crop_totals(calibrated_test_probs, class_names)

    true_label_names = [class_names[int(i)] for i in test_labels]
    true_crops = [label_to_crop(x) for x in true_label_names]

    crop_total_thresholds = [0.70, 0.74, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
    best_crop_total_score = -1.0
    recommended_crop_total_threshold = float(crop_total_thresholds[0])
    best_metrics = None

    for threshold in crop_total_thresholds:
        metrics = evaluate_crop_total_gate(calibrated_crop_totals, true_crops, supported_crops, threshold)
        if metrics["score"] > best_crop_total_score:
            best_crop_total_score = float(metrics["score"])
            recommended_crop_total_threshold = float(threshold)
            best_metrics = metrics

    print(
        "Recommended cropTotalThreshold="
        f"{recommended_crop_total_threshold:.2f} (score={best_crop_total_score:.4f}, "
        f"supported_confident_rate={best_metrics['supported_confident_rate']:.4f}, "
        f"unsupported_reject_rate={best_metrics['unsupported_reject_rate']:.4f})",
        flush=True,
    )

    models_reco_path = os.path.join(model_dir, "mobile_runtime_recommendations.json")
    mobile_reco_path = os.path.join(
        base, "mobile_app", "assets", "config", "mobile_runtime_recommendations.json"
    )

    _update_runtime_reco(
        models_reco_path,
        temperature=best_t,
        crop_total_threshold=recommended_crop_total_threshold,
        dry_run=bool(args.dry_run),
    )

    if not args.models_only:
        _update_runtime_reco(
            mobile_reco_path,
            temperature=best_t,
            crop_total_threshold=recommended_crop_total_threshold,
            dry_run=bool(args.dry_run),
        )

    if boundary_hit:
        print("NOTE: Temperature hit sweep boundary; consider widening --t-min/--t-max.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
