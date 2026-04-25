"""
Evaluate 4-crop model on test set with confidence analysis and TTA.
Run: python -u -X utf8 scripts/evaluate_model.py
"""

import os, json, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Loading TensorFlow ...", flush=True)
import numpy as np
import tensorflow as tf

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE, "models")
TEST_DIR   = os.path.join(BASE, "datasets", "model_ready", "test")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
IMG_SIZE   = (224, 224)
BATCH      = 8
TTA_AUGMENTS = 5   # NEW: number of augmented copies per image

# ── Load model ──────────────────────────────────────────────────────
print("Loading model ...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ── Load test set ───────────────────────────────────────────────────
print("Loading test set ...", flush=True)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=False
)

with open(os.path.join(MODEL_DIR, "class_index.json")) as f:
    _class_index = json.load(f)
class_names = sorted(_class_index, key=_class_index.get)
NUM_CLASSES = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}", flush=True)

if test_ds.class_names != class_names:
    print("ERROR: test dataset class order differs from models/class_index.json.", flush=True)
    print(f"  test_ds:     {test_ds.class_names}", flush=True)
    print(f"  class_index: {class_names}", flush=True)
    print("  Rebuild dataset splits and class_index before evaluation.", flush=True)
    raise SystemExit(1)

# Preprocess
def preprocess(images, labels):
    # MobileNetV2 expects inputs normalized to [-1, 1].
    images = tf.cast(images, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

test_ds_processed = test_ds.map(preprocess, num_parallel_calls=2).prefetch(1)

# ── Collect predictions (standard) ──────────────────────────────────
print("Predicting (standard) ...", flush=True)
all_labels = []
all_preds  = []
all_probs  = []

for images, labels in test_ds_processed:
    probs = model.predict(images, verbose=0)
    all_probs.append(probs)
    all_preds.extend(np.argmax(probs, axis=1))
    all_labels.extend(labels.numpy())

all_probs  = np.concatenate(all_probs, axis=0)
all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)

gc.collect()

# ── TTA predictions ─────────────────────────────────────────────────  # NEW
print(f"Predicting with TTA ({TTA_AUGMENTS} augmentations) ...", flush=True)

tta_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

# Start with original probabilities
tta_probs_sum = all_probs.copy()

for aug_i in range(TTA_AUGMENTS):
    print(f"  TTA pass {aug_i + 1}/{TTA_AUGMENTS} ...", flush=True)
    aug_probs = []
    for images, labels in test_ds:
        # Apply augmentation then preprocess
        aug_images = tta_augmentation(images, training=True)
        aug_images = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.cast(aug_images, tf.float32)
        )
        probs = model.predict(aug_images, verbose=0)
        aug_probs.append(probs)
    aug_probs = np.concatenate(aug_probs, axis=0)
    tta_probs_sum += aug_probs

# Average over original + augmented
tta_probs = tta_probs_sum / (1 + TTA_AUGMENTS)
tta_preds = np.argmax(tta_probs, axis=1)
tta_accuracy = np.mean(tta_preds == all_labels)

gc.collect()

# ── Overall metrics ─────────────────────────────────────────────────
accuracy = np.mean(all_preds == all_labels)

# Top-3 accuracy
top3 = np.argsort(all_probs, axis=1)[:, -3:]
top3_correct = np.array([all_labels[i] in top3[i] for i in range(len(all_labels))])
top3_acc = np.mean(top3_correct)

# Top-5 accuracy
top5 = np.argsort(all_probs, axis=1)[:, -5:]
top5_correct = np.array([all_labels[i] in top5[i] for i in range(len(all_labels))])
top5_acc = np.mean(top5_correct)

# TTA top-3/5
tta_top3 = np.argsort(tta_probs, axis=1)[:, -3:]
tta_top3_acc = np.mean([all_labels[i] in tta_top3[i] for i in range(len(all_labels))])
tta_top5 = np.argsort(tta_probs, axis=1)[:, -5:]
tta_top5_acc = np.mean([all_labels[i] in tta_top5[i] for i in range(len(all_labels))])

print(f"\n{'='*50}", flush=True)
print(f"TEST RESULTS (4-Crop Model)", flush=True)
print(f"{'='*50}", flush=True)
print(f"Overall Accuracy:     {accuracy:.2%}", flush=True)
print(f"Overall Accuracy TTA: {tta_accuracy:.2%}  ({'+' if tta_accuracy > accuracy else ''}{(tta_accuracy-accuracy)*100:.2f}pp)", flush=True)
print(f"Top-3 Accuracy:       {top3_acc:.2%}  (TTA: {tta_top3_acc:.2%})", flush=True)
print(f"Top-5 Accuracy:       {top5_acc:.2%}  (TTA: {tta_top5_acc:.2%})", flush=True)

# ── Per-class accuracy ──────────────────────────────────────────────
print(f"\nPer-Class Accuracy:", flush=True)
print(f"  {'Class':<35} {'Standard':>9} {'TTA':>9}  {'Count':>6}", flush=True)
print(f"  {'-'*65}", flush=True)
per_class = {}
per_class_tta = {}
for i, cname in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() > 0:
        acc = np.mean(all_preds[mask] == i)
        acc_tta = np.mean(tta_preds[mask] == i)
        per_class[cname] = {"accuracy": round(float(acc), 4), "count": int(mask.sum())}
        per_class_tta[cname] = {"accuracy": round(float(acc_tta), 4), "count": int(mask.sum())}
        delta = acc_tta - acc
        arrow = "+" if delta > 0 else ""
        print(f"  {cname:<35} {acc:8.2%}  {acc_tta:8.2%}  ({mask.sum():>4})  {arrow}{delta*100:.1f}pp", flush=True)

# ── Per-crop accuracy ───────────────────────────────────────────────
print(f"\nPer-Crop Accuracy:", flush=True)
crops = {}
crops_tta = {}
for cname in class_names:
    crop = cname.split("_")[0].capitalize()
    if crop not in crops:
        crops[crop] = {"correct": 0, "total": 0}
        crops_tta[crop] = {"correct": 0, "total": 0}

for i, cname in enumerate(class_names):
    crop = cname.split("_")[0].capitalize()
    mask = all_labels == i
    crops[crop]["total"] += int(mask.sum())
    crops[crop]["correct"] += int(np.sum(all_preds[mask] == i))
    crops_tta[crop]["total"] += int(mask.sum())
    crops_tta[crop]["correct"] += int(np.sum(tta_preds[mask] == i))

per_crop = {}
per_crop_tta = {}
for crop in sorted(crops.keys()):
    acc = crops[crop]["correct"] / crops[crop]["total"] if crops[crop]["total"] > 0 else 0
    acc_tta = crops_tta[crop]["correct"] / crops_tta[crop]["total"] if crops_tta[crop]["total"] > 0 else 0
    per_crop[crop] = {"accuracy": round(acc, 4), "count": crops[crop]["total"]}
    per_crop_tta[crop] = {"accuracy": round(acc_tta, 4), "count": crops_tta[crop]["total"]}
    print(f"  {crop:<15} {acc:6.2%}  (TTA: {acc_tta:6.2%})  ({crops[crop]['total']} images)", flush=True)

# ── Confidence analysis (using TTA probs) ───────────────────────────
print(f"\nConfidence Analysis (TTA):", flush=True)
max_probs = np.max(tta_probs, axis=1)
avg_confidence = float(np.mean(max_probs))
print(f"  Average confidence: {avg_confidence:.2%}", flush=True)

confidence_thresholds = {}
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    above = max_probs >= threshold
    count_above = int(np.sum(above))
    if count_above > 0:
        acc_above = float(np.mean(tta_preds[above] == all_labels[above]))
        coverage = count_above / len(all_labels)
        confidence_thresholds[str(threshold)] = {
            "accuracy": round(acc_above, 4),
            "coverage": round(float(coverage), 4),
            "count": count_above
        }
        print(f"  Threshold >= {threshold:.0%}: "
              f"accuracy={acc_above:.2%}, "
              f"coverage={coverage:.2%} ({count_above}/{len(all_labels)} images)", flush=True)

# Recommended threshold for Flutter app
best_thresh = 0.7
above_best = max_probs >= best_thresh
if np.sum(above_best) > 0:
    acc_best = float(np.mean(tta_preds[above_best] == all_labels[above_best]))
    cov_best = float(np.sum(above_best) / len(all_labels))
    print(f"\n  >> Recommended app threshold: {best_thresh:.0%} "
          f"-> {acc_best:.2%} accuracy on {cov_best:.2%} of predictions", flush=True)
    print(f"  >> Below {best_thresh:.0%}: show 'Uncertain - please retake photo'", flush=True)

# ── Classification report (using TTA) ──────────────────────────────
print(f"\nFull Classification Report (TTA):", flush=True)
from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(all_labels, tta_preds, target_names=class_names)
print(report, flush=True)

# Also show standard report for comparison
report_std = classification_report(all_labels, all_preds, target_names=class_names)

with open(os.path.join(MODEL_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"4-Crop Model Test Evaluation\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Accuracy: {accuracy:.2%}\n")
    f.write(f"Test Accuracy (TTA): {tta_accuracy:.2%}\n")
    f.write(f"Top-3 Accuracy: {top3_acc:.2%}\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.2%}\n")
    f.write(f"Recommended Confidence Threshold: {best_thresh:.0%}\n\n")
    f.write(f"=== Standard (no TTA) ===\n{report_std}\n")
    f.write(f"=== With TTA ({TTA_AUGMENTS} augmentations) ===\n{report}\n")

# ── Confusion matrix plot (TTA) ────────────────────────────────────
# Define short_names here so both plot blocks can use it safely
short_names = [n.replace('banana_', 'B:').replace('beans_', 'Be:')
                .replace('maize_', 'M:').replace('potato_', 'P:')
               for n in class_names]

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_labels, tta_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 12))

    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix (Counts) — TTA')
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(short_names, fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Normalized) — TTA')
    ax2.set_xticks(range(NUM_CLASSES))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(short_names, fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.suptitle(f'4-Crop Model — Test Accuracy: {accuracy:.2%} | TTA: {tta_accuracy:.2%}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved confusion matrix plot", flush=True)
except Exception as e:
    print(f"Could not save confusion matrix plot: {e}", flush=True)

# ── Per-class accuracy bar chart ────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(14, 6))
    accs = [per_class_tta[c]["accuracy"] * 100 for c in class_names]
    accs_std = [per_class[c]["accuracy"] * 100 for c in class_names]
    colors = []
    for c in class_names:
        if c.startswith("banana"):   colors.append('#FFD700')
        elif c.startswith("beans"):  colors.append('#228B22')
        elif c.startswith("maize"):  colors.append('#FF8C00')
        elif c.startswith("potato"): colors.append('#8B4513')

    x = np.arange(NUM_CLASSES)
    bars = ax.bar(x - 0.15, accs_std, 0.3, color=colors, edgecolor='gray', alpha=0.5, label='Standard')
    bars2 = ax.bar(x + 0.15, accs, 0.3, color=colors, edgecolor='black', alpha=0.85, label='TTA')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Per-Class Test Accuracy (Standard: {accuracy:.2%} | TTA: {tta_accuracy:.2%})')
    ax.set_ylim(0, 105)
    ax.axhline(y=tta_accuracy * 100, color='red', linestyle='--', alpha=0.7, label=f'TTA Overall: {tta_accuracy:.2%}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars2, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "per_class_accuracy.png"), dpi=150)
    plt.close()
    print("Saved per-class accuracy plot", flush=True)
except Exception as e:
    print(f"Could not save per-class plot: {e}", flush=True)

# ── Save evaluation report ──────────────────────────────────────────
eval_report = {
    "model": "MobileNetV2 (4-crop)",
    "model_path": "best_model.keras",
    "test_accuracy": round(float(accuracy), 4),
    "test_accuracy_tta": round(float(tta_accuracy), 4),
    "tta_augments": TTA_AUGMENTS,
    "top3_accuracy": round(float(top3_acc), 4),
    "top5_accuracy": round(float(top5_acc), 4),
    "top3_accuracy_tta": round(float(tta_top3_acc), 4),
    "top5_accuracy_tta": round(float(tta_top5_acc), 4),
    "num_classes": NUM_CLASSES,
    "crops": ["banana", "beans", "maize", "potato"],
    "total_test_images": int(len(all_labels)),
    "average_confidence": round(avg_confidence, 4),
    "recommended_threshold": best_thresh,
    "confidence_thresholds": confidence_thresholds,
    "per_class": per_class,
    "per_class_tta": per_class_tta,
    "per_crop": per_crop,
    "per_crop_tta": per_crop_tta,
}
with open(os.path.join(MODEL_DIR, "evaluation_report.json"), 'w') as f:
    json.dump(eval_report, f, indent=2)

print(f"\nSaved -> models/evaluation_report.json", flush=True)
print(f"Saved -> models/classification_report.txt", flush=True)
print("DONE!", flush=True)