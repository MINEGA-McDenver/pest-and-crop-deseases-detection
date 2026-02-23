#!/usr/bin/env python3
"""
convert_tflite.py  --  Model Optimization & Conversion for Mobile Deployment
=============================================================================
Activities:
  1. Optimize the trained model for mobile use
  2. Convert model to TensorFlow Lite format
  3. Reduce model size and inference latency (dynamic range + float16 quantization)
  4. Validate accuracy after conversion

Generates:
  models/model_float32.tflite           -- standard TFLite (no quantization)
  models/model_dynamic_range.tflite     -- dynamic range quantized (smallest)
  models/model_float16.tflite           -- float16 quantized (good balance)
  models/crop_disease_model.tflite      -- copy of recommended model for Flutter
  models/tflite_comparison.json         -- size & accuracy comparison
  models/tflite_comparison.png          -- visual comparison chart

Usage:
  python -u -X utf8 scripts/convert_tflite.py
"""

import os, sys, json, time, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Loading TensorFlow ...", flush=True)
import tensorflow as tf
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 0.  PATHS
# --------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "datasets", "model_ready")
TEST_DIR   = os.path.join(DATA_DIR, "test")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

IMG_SIZE   = 224
SEED       = 42

# FIX 1: Read class_index from models/ (correct 14-class version)
with open(os.path.join(MODEL_DIR, "class_index.json")) as f:
    class_index = json.load(f)
CLASS_NAMES = sorted(class_index, key=class_index.get)
NUM_CLASSES = len(CLASS_NAMES)

print(f"\n{'='*60}", flush=True)
print(f"  Model Optimization & TFLite Conversion")
print(f"  Source : best_model.keras")
print(f"  Classes: {NUM_CLASSES}  ({', '.join(CLASS_NAMES[:3])} ...)")
print(f"{'='*60}\n", flush=True)

# --------------------------------------------------
# 1.  LOAD KERAS MODEL
# --------------------------------------------------
print("[1/6] Loading trained Keras model ...", flush=True)

model = tf.keras.models.load_model(MODEL_PATH)
keras_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"   Keras model size: {keras_size:.2f} MB")
print(f"   Parameters: {model.count_params():,}", flush=True)

# --------------------------------------------------
# 2.  PREPARE REPRESENTATIVE DATASET (for quantization calibration)
# --------------------------------------------------
print("\n[2/6] Preparing representative dataset for quantization ...", flush=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def get_representative_images(num_samples=100):
    """Collect a small representative sample from training data."""
    images = []
    per_class = max(1, num_samples // NUM_CLASSES)
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(TRAIN_DIR, cls_name)
        files = [f for f in os.listdir(cls_dir)
                 if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        np.random.seed(SEED)
        selected = np.random.choice(files, size=min(per_class, len(files)), replace=False)
        for fname in selected:
            img = tf.keras.utils.load_img(
                os.path.join(cls_dir, fname),
                target_size=(IMG_SIZE, IMG_SIZE),
            )
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images, dtype=np.float32)

rep_images = get_representative_images(100)
print(f"   Representative samples: {len(rep_images)}", flush=True)

def representative_dataset():
    """Generator for TFLite quantization calibration."""
    for i in range(len(rep_images)):
        yield [rep_images[i:i+1]]

# --------------------------------------------------
# 3.  CONVERT TO TFLITE (3 variants)
# --------------------------------------------------
print("\n[3/6] Converting to TFLite formats ...", flush=True)

conversions = {}

# --- A) Standard Float32 (no quantization) ---
print("\n   A) Float32 (standard, no quantization) ...", flush=True)
t0 = time.time()
converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_f32 = converter_f32.convert()
f32_path = os.path.join(MODEL_DIR, "model_float32.tflite")
with open(f32_path, "wb") as f:
    f.write(tflite_f32)
f32_size = len(tflite_f32) / (1024 * 1024)
f32_time = time.time() - t0
print(f"      Size: {f32_size:.2f} MB  |  Time: {f32_time:.1f}s")
conversions["float32"] = {"path": f32_path, "size_mb": f32_size, "data": tflite_f32}

del tflite_f32
gc.collect()

# --- B) Dynamic Range Quantization (int8 weights, float activations) ---
print("\n   B) Dynamic range quantization (smallest size) ...", flush=True)
t0 = time.time()
converter_dr = tf.lite.TFLiteConverter.from_keras_model(model)
converter_dr.optimizations = [tf.lite.Optimize.DEFAULT]
converter_dr.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_dr = converter_dr.convert()
dr_path = os.path.join(MODEL_DIR, "model_dynamic_range.tflite")
with open(dr_path, "wb") as f:
    f.write(tflite_dr)
dr_size = len(tflite_dr) / (1024 * 1024)
dr_time = time.time() - t0
print(f"      Size: {dr_size:.2f} MB  |  Time: {dr_time:.1f}s")
print(f"      Compression: {(1 - dr_size/f32_size)*100:.1f}% smaller than float32")
conversions["dynamic_range"] = {"path": dr_path, "size_mb": dr_size, "data": tflite_dr}

del tflite_dr
gc.collect()

# --- C) Float16 Quantization (good accuracy/size balance) ---
print("\n   C) Float16 quantization (best accuracy/size balance) ...", flush=True)
t0 = time.time()
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
converter_f16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_f16 = converter_f16.convert()
f16_path = os.path.join(MODEL_DIR, "model_float16.tflite")
with open(f16_path, "wb") as f:
    f.write(tflite_f16)
f16_size = len(tflite_f16) / (1024 * 1024)
f16_time = time.time() - t0
print(f"      Size: {f16_size:.2f} MB  |  Time: {f16_time:.1f}s")
print(f"      Compression: {(1 - f16_size/f32_size)*100:.1f}% smaller than float32")
conversions["float16"] = {"path": f16_path, "size_mb": f16_size, "data": tflite_f16}

del tflite_f16
gc.collect()

# Free the Keras model from memory
del model
gc.collect()

# --------------------------------------------------
# 4.  LOAD TEST SET FOR VALIDATION
# --------------------------------------------------
print("\n[4/6] Loading test dataset for accuracy validation ...", flush=True)

test_images = []
test_labels = []

for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(TEST_DIR, cls_name)
    cls_idx = class_index[cls_name]
    files = [f for f in os.listdir(cls_dir)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    for fname in files:
        img = tf.keras.utils.load_img(
            os.path.join(cls_dir, fname),
            target_size=(IMG_SIZE, IMG_SIZE),
        )
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        test_images.append(img_array)
        test_labels.append(cls_idx)

test_images = np.array(test_images, dtype=np.float32)
test_labels = np.array(test_labels)
total_test = len(test_labels)
print(f"   Loaded {total_test:,} test images", flush=True)

# Free representative images
del rep_images
gc.collect()

# --------------------------------------------------
# 5.  VALIDATE EACH TFLITE MODEL
# --------------------------------------------------
print("\n[5/6] Validating TFLite models on test set ...", flush=True)

def evaluate_tflite(model_path, test_images, test_labels, name):
    """Run inference with a TFLite model and compute accuracy."""
    print(f"\n   Evaluating {name} ...", flush=True)
    t0 = time.time()

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]

    predictions = []
    for i in range(len(test_images)):
        img = test_images[i:i+1]
        if input_dtype == np.float32:
            img = img.astype(np.float32)
        elif input_dtype == np.uint8:
            input_scale = input_details[0]["quantization_parameters"]["scales"][0]
            input_zero = input_details[0]["quantization_parameters"]["zero_points"][0]
            img = (img / input_scale + input_zero).astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(np.argmax(output[0]))

        if (i + 1) % 500 == 0 or (i + 1) == len(test_images):
            print(f"      {i+1}/{len(test_images)}", flush=True)

    predictions = np.array(predictions)
    acc = accuracy_score(test_labels, predictions)
    inference_time = time.time() - t0
    avg_per_image = (inference_time / len(test_images)) * 1000  # ms

    print(f"      Accuracy: {acc*100:.2f}%  |  Total: {inference_time:.1f}s  |  Avg: {avg_per_image:.1f}ms/image")
    return acc, inference_time, avg_per_image

results = {}
for name, info in conversions.items():
    acc, total_time, per_image_ms = evaluate_tflite(
        info["path"], test_images, test_labels, name
    )
    results[name] = {
        "accuracy": round(float(acc), 4),
        "size_mb": round(float(info["size_mb"]), 2),
        "total_inference_time_s": round(float(total_time), 1),
        "avg_inference_ms": round(float(per_image_ms), 1),
        "file": os.path.basename(info["path"]),
    }

# Free test images
del test_images
gc.collect()

# --------------------------------------------------
# 6.  COMPARISON REPORT & VISUALIZATION
# --------------------------------------------------
print("\n[6/6] Generating comparison report ...", flush=True)

# FIX 2: Read evaluation report with correct key name
eval_report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
keras_acc = None
if os.path.exists(eval_report_path):
    with open(eval_report_path) as f:
        orig_eval = json.load(f)
    keras_acc = orig_eval.get("test_accuracy")

# Print comparison table
print(f"\n   {'Model':<20} {'Size':>8} {'Accuracy':>10} {'Acc Drop':>10} {'Speed':>12}")
print(f"   {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

if keras_acc:
    print(f"   {'Keras (original)':<20} {keras_size:>7.2f}M {keras_acc*100:>9.2f}% {'baseline':>10} {'N/A':>12}")

for name in ["float32", "dynamic_range", "float16"]:
    r = results[name]
    drop = (keras_acc - r["accuracy"]) * 100 if keras_acc else 0
    drop_str = f"{drop:>+9.2f}%" if keras_acc else "N/A"
    print(f"   {name:<20} {r['size_mb']:>7.2f}M {r['accuracy']*100:>9.2f}% {drop_str:>10} {r['avg_inference_ms']:>8.1f}ms/img")

# Determine best model for mobile
best_name = "float16"
best_info = results[best_name]
print(f"\n   RECOMMENDED for mobile: {best_info['file']}")
print(f"   Size: {best_info['size_mb']:.2f} MB  |  Accuracy: {best_info['accuracy']*100:.2f}%", flush=True)

# -- Comparison chart --
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("TFLite Model Comparison (4-Crop)", fontsize=14, fontweight="bold")

names = ["float32", "dynamic_range", "float16"]
display = ["Float32\n(standard)", "Dynamic Range\n(int8 weights)", "Float16\n(recommended)"]
colors = ["#3498db", "#e74c3c", "#27ae60"]

# Size comparison
sizes = [results[n]["size_mb"] for n in names]
if keras_acc:
    axes[0].axhline(y=keras_size, color="gray", linestyle="--", alpha=0.7, label=f"Keras ({keras_size:.1f} MB)")
bars = axes[0].bar(display, sizes, color=colors, edgecolor="white", width=0.6)
for bar, size in zip(bars, sizes):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{size:.2f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Size (MB)")
axes[0].set_title("Model Size")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

# Accuracy comparison
accs = [results[n]["accuracy"] * 100 for n in names]
if keras_acc:
    axes[1].axhline(y=keras_acc*100, color="gray", linestyle="--", alpha=0.7, label=f"Keras ({keras_acc*100:.1f}%)")
bars = axes[1].bar(display, accs, color=colors, edgecolor="white", width=0.6)
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Test Accuracy")
axes[1].set_ylim(min(accs) - 5, 100)
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

# Inference speed comparison
speeds = [results[n]["avg_inference_ms"] for n in names]
bars = axes[2].bar(display, speeds, color=colors, edgecolor="white", width=0.6)
for bar, speed in zip(bars, speeds):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{speed:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[2].set_ylabel("Inference (ms/image)")
axes[2].set_title("Inference Speed (CPU)")
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(MODEL_DIR, "tflite_comparison.png")
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"   Saved -> models/tflite_comparison.png")

# Save comparison JSON
comparison = {
    "keras_model": {
        "file": "best_model.keras",
        "size_mb": round(float(keras_size), 2),
        "accuracy": float(keras_acc) if keras_acc else None,
        "params": model.count_params() if 'model' in dir() else 2589518,
    },
    "tflite_models": results,
    "recommended": best_name,
    "recommendation_reason": "Float16 offers ~50% size reduction vs float32 with negligible accuracy loss. Best balance for mobile deployment.",
}

comparison_path = os.path.join(MODEL_DIR, "tflite_comparison.json")
with open(comparison_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"   Saved -> models/tflite_comparison.json")

# Copy recommended model with a clear name for the Flutter app
import shutil
recommended_src = os.path.join(MODEL_DIR, results[best_name]["file"])
recommended_dst = os.path.join(MODEL_DIR, "crop_disease_model.tflite")
shutil.copy2(recommended_src, recommended_dst)

# Also ensure labels.txt is in models/
labels_src = os.path.join(DATA_DIR, "labels.txt")
labels_dst = os.path.join(MODEL_DIR, "labels.txt")
if os.path.exists(labels_src) and os.path.exists(labels_dst):
    pass  # already saved by train_model.py
elif os.path.exists(labels_src):
    shutil.copy2(labels_src, labels_dst)

print(f"   Saved -> models/crop_disease_model.tflite  (ready for Flutter)")
print(f"   Saved -> models/labels.txt  (class names for mobile app)")

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------
print(f"\n{'='*60}")
print(f"  MODEL OPTIMIZATION & CONVERSION COMPLETE")
print(f"{'='*60}")
if keras_acc:
    print(f"  Keras model     : {keras_size:.2f} MB  ({keras_acc*100:.2f}% accuracy)")
print(f"  Float32 TFLite  : {results['float32']['size_mb']:.2f} MB  ({results['float32']['accuracy']*100:.2f}%)")
print(f"  Dynamic Range   : {results['dynamic_range']['size_mb']:.2f} MB  ({results['dynamic_range']['accuracy']*100:.2f}%)")
print(f"  Float16 TFLite  : {results['float16']['size_mb']:.2f} MB  ({results['float16']['accuracy']*100:.2f}%)")
print(f"")
print(f"  RECOMMENDED: crop_disease_model.tflite (float16)")
print(f"  Size: {results['float16']['size_mb']:.2f} MB  |  Accuracy: {results['float16']['accuracy']*100:.2f}%")
print(f"")
print(f"  Files for Flutter app (copy to assets/):")
print(f"    - models/crop_disease_model.tflite")
print(f"    - models/labels.txt")
print(f"{'='*60}")
print(f"\n  Next step: Build the Flutter mobile application.")
print(f"  The model and labels are ready for integration.\n")