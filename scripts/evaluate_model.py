"""
Evaluate 4-crop model on test set with confidence analysis.
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

# ── Load model ──────────────────────────────────────────────────────
print("Loading model ...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)

# ── Load test set ───────────────────────────────────────────────────
print("Loading test set ...", flush=True)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=False
)

class_names = sorted([d for d in os.listdir(TEST_DIR)
                      if os.path.isdir(os.path.join(TEST_DIR, d))])
NUM_CLASSES = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}", flush=True)

# Preprocess
def preprocess(images, labels):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

test_ds = test_ds.map(preprocess, num_parallel_calls=2).prefetch(1)

# ── Collect predictions ─────────────────────────────────────────────
print("Predicting ...", flush=True)
all_labels = []
all_preds  = []
all_probs  = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    all_probs.append(probs)
    all_preds.extend(np.argmax(probs, axis=1))
    all_labels.extend(labels.numpy())

all_probs  = np.concatenate(all_probs, axis=0)
all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)

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

print(f"\n{'='*50}", flush=True)
print(f"TEST RESULTS (4-Crop Model)", flush=True)
print(f"{'='*50}", flush=True)
print(f"Overall Accuracy: {accuracy:.2%}", flush=True)
print(f"Top-3 Accuracy:   {top3_acc:.2%}", flush=True)
print(f"Top-5 Accuracy:   {top5_acc:.2%}", flush=True)

# ── Per-class accuracy ──────────────────────────────────────────────
print(f"\nPer-Class Accuracy:", flush=True)
per_class = {}
for i, cname in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() > 0:
        acc = np.mean(all_preds[mask] == i)
        per_class[cname] = {"accuracy": round(float(acc), 4), "count": int(mask.sum())}
        print(f"  {cname:<35} {acc:6.2%}  ({mask.sum()} images)", flush=True)

# ── Per-crop accuracy ───────────────────────────────────────────────
print(f"\nPer-Crop Accuracy:", flush=True)
crops = {}
for cname in class_names:
    crop = cname.split("_")[0].capitalize()
    if crop not in crops:
        crops[crop] = {"correct": 0, "total": 0}

for i, cname in enumerate(class_names):
    crop = cname.split("_")[0].capitalize()
    mask = all_labels == i
    crops[crop]["total"] += int(mask.sum())
    crops[crop]["correct"] += int(np.sum(all_preds[mask] == i))

per_crop = {}
for crop in sorted(crops.keys()):
    acc = crops[crop]["correct"] / crops[crop]["total"] if crops[crop]["total"] > 0 else 0
    per_crop[crop] = {"accuracy": round(acc, 4), "count": crops[crop]["total"]}
    print(f"  {crop:<15} {acc:6.2%}  ({crops[crop]['total']} images)", flush=True)

# ── Confidence analysis ─────────────────────────────────────────────
print(f"\nConfidence Analysis:", flush=True)
max_probs = np.max(all_probs, axis=1)
avg_confidence = float(np.mean(max_probs))
print(f"  Average confidence: {avg_confidence:.2%}", flush=True)

confidence_thresholds = {}
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    above = max_probs >= threshold
    count_above = int(np.sum(above))
    if count_above > 0:
        acc_above = float(np.mean(all_preds[above] == all_labels[above]))
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
    acc_best = float(np.mean(all_preds[above_best] == all_labels[above_best]))
    cov_best = float(np.sum(above_best) / len(all_labels))
    print(f"\n  >> Recommended app threshold: {best_thresh:.0%} "
          f"-> {acc_best:.2%} accuracy on {cov_best:.2%} of predictions", flush=True)
    print(f"  >> Below {best_thresh:.0%}: show 'Uncertain - please retake photo'", flush=True)

# ── Classification report ───────────────────────────────────────────
print(f"\nFull Classification Report:", flush=True)
from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report, flush=True)

with open(os.path.join(MODEL_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"4-Crop Model Test Evaluation\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Accuracy: {accuracy:.2%}\n")
    f.write(f"Top-3 Accuracy: {top3_acc:.2%}\n")
    f.write(f"Top-5 Accuracy: {top5_acc:.2%}\n")
    f.write(f"Recommended Confidence Threshold: {best_thresh:.0%}\n\n")
    f.write(report)

# ── Confusion matrix plot ───────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # Raw counts
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    short_names = [n.replace('banana_', 'B:').replace('beans_', 'Be:')
                    .replace('maize_', 'M:').replace('potato_', 'P:')
                   for n in class_names]
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(short_names, fontsize=8)
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Normalized
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xticks(range(NUM_CLASSES))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(short_names, fontsize=8)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.suptitle(f'4-Crop Model — Test Accuracy: {accuracy:.2%}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved confusion matrix plot", flush=True)
except Exception as e:
    print(f"Could not save confusion matrix plot: {e}", flush=True)

# ── Per-class accuracy bar chart ────────────────────────────────────
try:
    fig, ax = plt.subplots(figsize=(14, 6))
    accs = [per_class[c]["accuracy"] * 100 for c in class_names]
    colors = []
    for c in class_names:
        if c.startswith("banana"):   colors.append('#FFD700')
        elif c.startswith("beans"):  colors.append('#228B22')
        elif c.startswith("maize"):  colors.append('#FF8C00')
        elif c.startswith("potato"): colors.append('#8B4513')

    bars = ax.bar(range(NUM_CLASSES), accs, color=colors, edgecolor='gray', alpha=0.85)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Per-Class Test Accuracy (Overall: {accuracy:.2%})')
    ax.set_ylim(0, 105)
    ax.axhline(y=accuracy * 100, color='red', linestyle='--', alpha=0.7, label=f'Overall: {accuracy:.2%}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accs):
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
    "top3_accuracy": round(float(top3_acc), 4),
    "top5_accuracy": round(float(top5_acc), 4),
    "num_classes": NUM_CLASSES,
    "crops": ["banana", "beans", "maize", "potato"],
    "total_test_images": int(len(all_labels)),
    "average_confidence": round(avg_confidence, 4),
    "recommended_threshold": best_thresh,
    "confidence_thresholds": confidence_thresholds,
    "per_class": per_class,
    "per_crop": per_crop,
}
with open(os.path.join(MODEL_DIR, "evaluation_report.json"), 'w') as f:
    json.dump(eval_report, f, indent=2)

print(f"\nSaved -> models/evaluation_report.json", flush=True)
print(f"Saved -> models/classification_report.txt", flush=True)
print("DONE!", flush=True)