"""
Train MobileNetV2 model for 4-crop disease detection (14 classes).
Crops: Banana, Beans, Maize, Potato
Run: python -u -X utf8 scripts/train_model.py
"""

import os, sys, json, gc, time, csv, random, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '42'

print("Loading TensorFlow ...", flush=True)
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config ──────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE, "datasets", "model_ready")
MODEL_DIR   = os.path.join(BASE, "models")
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "val")
TEST_DIR    = os.path.join(DATA_DIR, "test")
IMG_SIZE    = (224, 224)
BATCH       = 8
PHASE1_EP   = 15          # frozen backbone
PHASE2_EP   = 35          # CHANGED: 20 -> 35 (model was still learning at epoch 20)
INIT_LR     = 1e-3
FINE_LR     = 1e-5
PATIENCE    = 7            # CHANGED: 5 -> 7 (more room with longer Phase 2)
MAX_CLASS_WEIGHT = None    # None = uncapped class weights (better minority recall)
FINE_TUNE_FROM   = 80      # MobileNetV2: unfreeze from layer 80 for fine-tuning

# Light-but-realistic field augmentation profile
AUG_ROTATION   = 0.05
AUG_ZOOM       = 0.10
AUG_CONTRAST   = 0.20
AUG_BRIGHTNESS = 0.20

CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]
DEFAULT_APP_CONFIDENCE_THRESHOLD = 0.60

ALLOWED_CROPS = {"banana", "beans", "maize", "potato"}

os.makedirs(MODEL_DIR, exist_ok=True)

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
        crop_prefix = folder.split("_")[0]
        if crop_prefix not in ALLOWED_CROPS:
            print(f"ERROR: Unexpected class '{folder}' in {split_name}/", flush=True)
            print(f"  Crop prefix '{crop_prefix}' is not in {ALLOWED_CROPS}", flush=True)
            print("  Remove it and re-run.", flush=True)
            sys.exit(1)
print(f"Dataset validated: only {ALLOWED_CROPS} crops found.", flush=True)

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

def collect_split_stems(split_dir, cls_name):
    cls_dir = os.path.join(split_dir, cls_name)
    stems = set()
    for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname)
        if os.path.isfile(fpath):
            stems.add(os.path.splitext(fname)[0])
    return stems

# Save labels
labels_path = os.path.join(MODEL_DIR, "labels.txt")
with open(labels_path, 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print(f"Saved labels -> {labels_path}", flush=True)

# Class index mapping
class_index = {name: i for i, name in enumerate(class_names)}
with open(os.path.join(MODEL_DIR, "class_index.json"), 'w') as f:
    json.dump(class_index, f, indent=2)

# ── Class weights (with cap) ────────────────────────────────────────
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
total_train_val_overlap = 0
total_train_test_overlap = 0
total_val_test_overlap = 0
for cname in class_names:
    val_folder = os.path.join(VAL_DIR, cname)
    test_folder = os.path.join(TEST_DIR, cname)
    val_count = len([f for f in os.listdir(val_folder) if os.path.isfile(os.path.join(val_folder, f))])
    test_count = len([f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))])
    val_counts[cname] = val_count
    test_counts[cname] = test_count
    val_total += val_count
    test_total += test_count

    train_stems = collect_split_stems(TRAIN_DIR, cname)
    val_stems = collect_split_stems(VAL_DIR, cname)
    test_stems = collect_split_stems(TEST_DIR, cname)

    overlap_train_val = len(train_stems.intersection(val_stems))
    overlap_train_test = len(train_stems.intersection(test_stems))
    overlap_val_test = len(val_stems.intersection(test_stems))

    leakage_check[cname] = {
        "train_val_overlap": overlap_train_val,
        "train_test_overlap": overlap_train_test,
        "val_test_overlap": overlap_val_test,
    }
    total_train_val_overlap += overlap_train_val
    total_train_test_overlap += overlap_train_test
    total_val_test_overlap += overlap_val_test

print(f"Total validation images: {val_total}", flush=True)
print(f"Total test images: {test_total}", flush=True)
print("Filename-stem overlap check (possible leakage):", flush=True)
print(f"  train↔val:  {total_train_val_overlap}", flush=True)
print(f"  train↔test: {total_train_test_overlap}", flush=True)
print(f"  val↔test:   {total_val_test_overlap}", flush=True)

class_weight = {}
for i, cname in enumerate(class_names):
    w = total / (NUM_CLASSES * class_counts[cname])
    if MAX_CLASS_WEIGHT is not None:
        w = min(w, MAX_CLASS_WEIGHT)
    class_weight[i] = round(w, 4)

# Report capped weights
cap_label = f"capped at {MAX_CLASS_WEIGHT}" if MAX_CLASS_WEIGHT is not None else "uncapped"
print(f"Class weights ({cap_label}):", flush=True)
for i, cname in enumerate(class_names):
    raw_w = total / (NUM_CLASSES * class_counts[cname])
    capped = " (CAPPED)" if (MAX_CLASS_WEIGHT is not None and raw_w > MAX_CLASS_WEIGHT) else ""
    print(f"  {cname}: {class_weight[i]}{capped}", flush=True)

with open(os.path.join(MODEL_DIR, "class_weights.json"), 'w') as f:
    json.dump({class_names[i]: class_weight[i] for i in range(NUM_CLASSES)}, f, indent=2)

# ── Performance tuning ──────────────────────────────────────────────
def preprocess(images, labels):
    # MobileNetV2 expects inputs normalized to [-1, 1].
    images = tf.cast(images, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

train_ds = train_ds.map(preprocess, num_parallel_calls=2).prefetch(1)
val_ds   = val_ds.map(preprocess, num_parallel_calls=2).prefetch(1)
test_ds  = test_ds.map(preprocess, num_parallel_calls=2).prefetch(1)

# Light online augmentation for field robustness while avoiding over-distortion.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(AUG_ROTATION),
    tf.keras.layers.RandomZoom(AUG_ZOOM),
    tf.keras.layers.RandomContrast(AUG_CONTRAST, value_range=(-1, 1)),
    tf.keras.layers.RandomBrightness(AUG_BRIGHTNESS, value_range=(-1, 1)),
], name="light_field_augmentation")

# ── Build model ─────────────────────────────────────────────────────
print("Building model ...", flush=True)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Save summary
summary_lines = []
model.summary(print_fn=lambda line: summary_lines.append(line))
with open(os.path.join(MODEL_DIR, "model_summary.txt"), 'w') as f:
    f.write('\n'.join(summary_lines))
print(f"Model params: {model.count_params():,}", flush=True)

# ── Callbacks ───────────────────────────────────────────────────────
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
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        lr = float(self.model.optimizer.learning_rate)
        print(f"  Epoch {epoch+1}: acc={acc:.4f} val_acc={val_acc:.4f} "
              f"loss={loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e} "
              f"[{elapsed:.1f} min]", flush=True)

progress = ProgressCallback()

# ── Cosine annealing LR for Phase 2 ────────────────────────────────  # NEW
class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, total_epochs, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + math.cos(math.pi * epoch / self.total_epochs))
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
        logs['lr'] = float(self.model.optimizer.learning_rate)

# ── CSV logger ──────────────────────────────────────────────────────
class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []
    def append(self, history, phase):
        for i, acc in enumerate(history.history.get('accuracy', [])):
            self.rows.append({
                'phase': phase,
                'epoch': i + 1,
                'accuracy': acc,
                'val_accuracy': history.history['val_accuracy'][i],
                'loss': history.history['loss'][i],
                'val_loss': history.history['val_loss'][i],
                'lr': history.history.get('lr', [0])[min(i, len(history.history.get('lr', [0]))-1)]
            })
    def save(self):
        with open(self.path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['phase','epoch','accuracy','val_accuracy','loss','val_loss','lr'])
            w.writeheader()
            w.writerows(self.rows)

csv_log = CSVLogger(HISTORY_PATH)

# ══════════════════ PHASE 1: Frozen backbone ════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"PHASE 1: Training top layers ({PHASE1_EP} epochs max)", flush=True)
print(f"  Class weight cap: {MAX_CLASS_WEIGHT}", flush=True)
print(f"{'='*60}", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

t1 = time.time()
h1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE1_EP,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop, reduce_lr, progress],
    verbose=0
)
t1_elapsed = (time.time() - t1) / 60
best_val_1 = max(h1.history['val_accuracy'])
csv_log.append(h1, 'phase1')
print(f"\nPhase 1 done: {len(h1.history['accuracy'])} epochs, "
      f"{t1_elapsed:.1f} min, best val_acc={best_val_1:.4f}", flush=True)

gc.collect()

# ══════════════════ PHASE 2: Fine-tune ══════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"PHASE 2: Fine-tuning from layer {FINE_TUNE_FROM} ({PHASE2_EP} epochs max)", flush=True)
print(f"  LR schedule: Cosine annealing from {FINE_LR}", flush=True)
print(f"{'='*60}", flush=True)

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_FROM]:    # CHANGED: use config variable
    layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable}", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_LR),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

# NEW: Cosine annealing replaces ReduceLROnPlateau for Phase 2
cosine_lr = CosineAnnealingScheduler(
    initial_lr=FINE_LR,
    total_epochs=PHASE2_EP,
    min_lr=1e-7
)

early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=PATIENCE,
    restore_best_weights=True, verbose=1
)

t2 = time.time()
h2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE2_EP,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop_2, cosine_lr, progress],  # CHANGED: cosine_lr replaces reduce_lr_2
    verbose=0
)
t2_elapsed = (time.time() - t2) / 60
best_val_2 = max(h2.history['val_accuracy'])
csv_log.append(h2, 'phase2')
print(f"\nPhase 2 done: {len(h2.history['accuracy'])} epochs, "
      f"{t2_elapsed:.1f} min, best val_acc={best_val_2:.4f}", flush=True)

# ── Save final model explicitly ─────────────────────────────────────
print(f"\nSaving final model ...", flush=True)
model.save(FINAL_MODEL_PATH)
print(f"  best_model.keras  -> best validation accuracy checkpoint", flush=True)
print(f"  final_model.keras -> model state at end of training", flush=True)

# ── Save artifacts ──────────────────────────────────────────────────
csv_log.save()
print(f"Saved history -> {HISTORY_PATH}", flush=True)

total_time = t1_elapsed + t2_elapsed
best_val = max(best_val_1, best_val_2)

config = {
    "model": "MobileNetV2",
    "num_classes": NUM_CLASSES,
    "class_names": class_names,
    "crops": list(ALLOWED_CROPS),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH,
    "seed": SEED,
    "max_class_weight": MAX_CLASS_WEIGHT,
    "fine_tune_from_layer": FINE_TUNE_FROM,
    "phase1_epochs": len(h1.history['accuracy']),
    "phase2_epochs": len(h2.history['accuracy']),
    "total_epochs": len(h1.history['accuracy']) + len(h2.history['accuracy']),
    "phase1_time_min": round(t1_elapsed, 1),
    "phase2_time_min": round(t2_elapsed, 1),
    "total_time_min": round(total_time, 1),
    "best_val_accuracy": round(float(best_val), 4),
    "phase1_best_val": round(float(best_val_1), 4),
    "phase2_best_val": round(float(best_val_2), 4),
    "best_model_path": "best_model.keras",
    "final_model_path": "final_model.keras",
    "phase2_lr_schedule": "cosine_annealing",
}
with open(os.path.join(MODEL_DIR, "training_config.json"), 'w') as f:
    json.dump(config, f, indent=2)

dataset_stats = {
    "train_counts": class_counts,
    "val_counts": val_counts,
    "test_counts": test_counts,
    "totals": {
        "train": total,
        "val": val_total,
        "test": test_total,
    },
    "leakage_check": {
        "by_class": leakage_check,
        "totals": {
            "train_val_overlap": total_train_val_overlap,
            "train_test_overlap": total_train_test_overlap,
            "val_test_overlap": total_val_test_overlap,
        },
    },
    "augmentation_profile": {
        "random_flip": True,
        "random_rotation": AUG_ROTATION,
        "random_zoom": AUG_ZOOM,
        "random_contrast": AUG_CONTRAST,
        "random_brightness": AUG_BRIGHTNESS,
    },
}
with open(os.path.join(MODEL_DIR, "dataset_stats.json"), 'w') as f:
    json.dump(dataset_stats, f, indent=2)

# ── Final evaluation on independent test set ───────────────────────
print("\nEvaluating best model on independent test set ...", flush=True)
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

test_loss, test_acc, test_top3 = best_model.evaluate(test_ds, verbose=0)
print(f"Test metrics: loss={test_loss:.4f} acc={test_acc:.4f} top3={test_top3:.4f}", flush=True)

all_test_labels = []
all_test_preds = []
all_test_probs = []
for images, labels in test_ds:
    probs = best_model.predict(images, verbose=0)
    all_test_probs.append(probs)
    all_test_preds.extend(np.argmax(probs, axis=1))
    all_test_labels.extend(labels.numpy())

all_test_labels = np.array(all_test_labels)
all_test_preds = np.array(all_test_preds)
all_test_probs = np.concatenate(all_test_probs, axis=0)
max_test_probs = np.max(all_test_probs, axis=1)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    all_test_labels, all_test_preds, average='macro', zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_test_labels, all_test_preds, average='weighted', zero_division=0
)

report = classification_report(
    all_test_labels,
    all_test_preds,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

with open(os.path.join(MODEL_DIR, "test_classification_report.txt"), 'w') as f:
    f.write("Independent Test Set Report\n")
    f.write(f"Model: {BEST_MODEL_PATH}\n")
    f.write(f"Test accuracy: {test_acc:.4f}\n")
    f.write(f"Test top-3 accuracy: {test_top3:.4f}\n")
    f.write(f"Macro precision: {precision_macro:.4f}\n")
    f.write(f"Macro recall: {recall_macro:.4f}\n")
    f.write(f"Macro F1: {f1_macro:.4f}\n")
    f.write(f"Weighted precision: {precision_weighted:.4f}\n")
    f.write(f"Weighted recall: {recall_weighted:.4f}\n")
    f.write(f"Weighted F1: {f1_weighted:.4f}\n\n")
    f.write(report)

# Confidence threshold analysis for deployment safety
confidence_thresholds = {}
for threshold in CONFIDENCE_THRESHOLDS:
    mask = max_test_probs >= threshold
    count = int(np.sum(mask))
    if count > 0:
        acc = float(np.mean(all_test_preds[mask] == all_test_labels[mask]))
        coverage = float(count / len(all_test_labels))
        confidence_thresholds[str(threshold)] = {
            "accuracy": round(acc, 4),
            "coverage": round(coverage, 4),
            "count": count,
        }

# Confusion matrix visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_test_labels, all_test_preds)
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    short_names = [n.replace('banana_', 'B:').replace('beans_', 'Be:')
                   .replace('maize_', 'M:').replace('potato_', 'P:')
                   for n in class_names]

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

    plt.suptitle(f'Independent Test Results — Accuracy: {test_acc:.2%} | Macro F1: {f1_macro:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "test_confusion_matrix.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Could not save test confusion matrix: {e}", flush=True)

# Export recommended mobile model (float16 TFLite)
print("Exporting float16 TFLite model from best checkpoint ...", flush=True)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
tflite_path = os.path.join(MODEL_DIR, "crop_disease_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite -> {tflite_path}", flush=True)

test_eval = {
    "test_loss": round(float(test_loss), 4),
    "test_accuracy": round(float(test_acc), 4),
    "test_top3_accuracy": round(float(test_top3), 4),
    "macro_precision": round(float(precision_macro), 4),
    "macro_recall": round(float(recall_macro), 4),
    "macro_f1": round(float(f1_macro), 4),
    "weighted_precision": round(float(precision_weighted), 4),
    "weighted_recall": round(float(recall_weighted), 4),
    "weighted_f1": round(float(f1_weighted), 4),
    "confidence_thresholds": confidence_thresholds,
    "recommended_app_threshold": DEFAULT_APP_CONFIDENCE_THRESHOLD,
}
with open(os.path.join(MODEL_DIR, "test_evaluation.json"), 'w') as f:
    json.dump(test_eval, f, indent=2)

del best_model
gc.collect()

# ── Training curves plot ────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_acc = h1.history['accuracy'] + h2.history['accuracy']
    all_val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    all_loss = h1.history['loss'] + h2.history['loss']
    all_val_loss = h1.history['val_loss'] + h2.history['val_loss']
    epochs = range(1, len(all_acc) + 1)
    phase1_end = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, all_acc, 'b-', label='Train')
    ax1.plot(epochs, all_val_acc, 'r-', label='Validation')
    ax1.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, all_loss, 'b-', label='Train')
    ax2.plot(epochs, all_val_loss, 'r-', label='Validation')
    ax2.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
    ax2.set_title('Loss')
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

# ── Final summary ───────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"TRAINING COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
print(f"Crops: Banana, Beans, Maize, Potato", flush=True)
print(f"Classes: {NUM_CLASSES}", flush=True)
print(f"Training images: {total}", flush=True)
print(f"Validation images: {val_total}", flush=True)
print(f"Test images: {test_total}", flush=True)
print(f"Random seed: {SEED}", flush=True)
print(f"Class weight cap: {MAX_CLASS_WEIGHT if MAX_CLASS_WEIGHT is not None else 'None (uncapped)'}", flush=True)
print(f"Fine-tune from layer: {FINE_TUNE_FROM}", flush=True)
print(f"Phase 2 LR: Cosine annealing", flush=True)
print(f"Phase 1: {len(h1.history['accuracy'])} epochs, {t1_elapsed:.1f} min, best val={best_val_1:.4f}", flush=True)
print(f"Phase 2: {len(h2.history['accuracy'])} epochs, {t2_elapsed:.1f} min, best val={best_val_2:.4f}", flush=True)
print(f"Total: {total_time:.1f} min ({total_time/60:.1f} hrs)", flush=True)
print(f"Best val accuracy: {best_val:.2%}", flush=True)
print(f"Independent test accuracy: {test_acc:.2%}", flush=True)
print(f"Independent test macro F1: {f1_macro:.4f}", flush=True)
print(f"Recommended app confidence threshold: {DEFAULT_APP_CONFIDENCE_THRESHOLD:.2f}", flush=True)
print(f"Models saved:", flush=True)
print(f"  {BEST_MODEL_PATH}", flush=True)
print(f"  {FINAL_MODEL_PATH}", flush=True)
print(f"  {tflite_path}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'test_evaluation.json')}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'test_classification_report.txt')}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'test_confusion_matrix.png')}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'dataset_stats.json')}", flush=True)
print("DONE!", flush=True)