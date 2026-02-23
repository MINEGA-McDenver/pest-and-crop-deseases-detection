"""
Train MobileNetV2 model for 4-crop disease detection (14 classes).
Crops: Banana, Beans, Maize, Potato
Run: python -u -X utf8 scripts/train_model.py
"""

import os, sys, json, gc, time, csv, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '42'                # NEW: reproducible hashing

print("Loading TensorFlow ...", flush=True)
import numpy as np
import tensorflow as tf

# NEW: Set all random seeds for reproducibility
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
IMG_SIZE    = (224, 224)
BATCH       = 8
PHASE1_EP   = 15          # frozen backbone
PHASE2_EP   = 20          # fine-tune from layer 100
INIT_LR     = 1e-3
FINE_LR     = 1e-5
PATIENCE    = 5

# NEW: Allowed crops — any folder not matching these prefixes is rejected
ALLOWED_CROPS = {"banana", "beans", "maize", "potato"}

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Validate dataset (no cassava or unexpected classes) ─────────────  # NEW
print("Validating dataset ...", flush=True)
for split_name in ["train", "val"]:
    split_dir = os.path.join(DATA_DIR, split_name)
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
    label_mode='int', shuffle=True, seed=SEED               # NEW: use SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='int', shuffle=False
)

class_names = sorted(os.listdir(TRAIN_DIR))
class_names = [d for d in class_names if os.path.isdir(os.path.join(TRAIN_DIR, d))]
class_names.sort()
NUM_CLASSES = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}", flush=True)

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

# ── Class weights ───────────────────────────────────────────────────
print("Computing class weights ...", flush=True)
class_counts = {}
for cname in class_names:
    folder = os.path.join(TRAIN_DIR, cname)
    count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    class_counts[cname] = count
    print(f"  {cname}: {count}", flush=True)

total = sum(class_counts.values())
print(f"Total training images: {total}", flush=True)

class_weight = {}
for i, cname in enumerate(class_names):
    w = total / (NUM_CLASSES * class_counts[cname])
    class_weight[i] = round(w, 4)

with open(os.path.join(MODEL_DIR, "class_weights.json"), 'w') as f:
    json.dump({class_names[i]: class_weight[i] for i in range(NUM_CLASSES)}, f, indent=2)

# ── Performance tuning ──────────────────────────────────────────────
def preprocess(images, labels):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

train_ds = train_ds.map(preprocess, num_parallel_calls=2).prefetch(1)
val_ds   = val_ds.map(preprocess, num_parallel_calls=2).prefetch(1)

# ── Data augmentation layer ─────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")

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
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.3)(x)
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
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")    # NEW
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
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"  Epoch {epoch+1}: acc={acc:.4f} val_acc={val_acc:.4f} "
              f"loss={loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e} "
              f"[{elapsed:.1f} min]", flush=True)

progress = ProgressCallback()

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
print(f"{'='*60}", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
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
print(f"PHASE 2: Fine-tuning from layer 100 ({PHASE2_EP} epochs max)", flush=True)
print(f"{'='*60}", flush=True)

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

trainable = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers: {trainable}", flush=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=PATIENCE,
    restore_best_weights=True, verbose=1
)
reduce_lr_2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3,
    min_lr=1e-7, verbose=1
)

t2 = time.time()
h2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=PHASE2_EP,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stop_2, reduce_lr_2, progress],
    verbose=0
)
t2_elapsed = (time.time() - t2) / 60
best_val_2 = max(h2.history['val_accuracy'])
csv_log.append(h2, 'phase2')
print(f"\nPhase 2 done: {len(h2.history['accuracy'])} epochs, "
      f"{t2_elapsed:.1f} min, best val_acc={best_val_2:.4f}", flush=True)

# ── Save final model explicitly ─────────────────────────────────────  # NEW
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
    "seed": SEED,                                              # NEW
    "phase1_epochs": len(h1.history['accuracy']),
    "phase2_epochs": len(h2.history['accuracy']),
    "total_epochs": len(h1.history['accuracy']) + len(h2.history['accuracy']),
    "phase1_time_min": round(t1_elapsed, 1),
    "phase2_time_min": round(t2_elapsed, 1),
    "total_time_min": round(total_time, 1),
    "best_val_accuracy": round(float(best_val), 4),
    "phase1_best_val": round(float(best_val_1), 4),
    "phase2_best_val": round(float(best_val_2), 4),
    "best_model_path": "best_model.keras",                     # NEW
    "final_model_path": "final_model.keras",                   # NEW
}
with open(os.path.join(MODEL_DIR, "training_config.json"), 'w') as f:
    json.dump(config, f, indent=2)

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
print(f"Random seed: {SEED}", flush=True)
print(f"Phase 1: {len(h1.history['accuracy'])} epochs, {t1_elapsed:.1f} min, best val={best_val_1:.4f}", flush=True)
print(f"Phase 2: {len(h2.history['accuracy'])} epochs, {t2_elapsed:.1f} min, best val={best_val_2:.4f}", flush=True)
print(f"Total: {total_time:.1f} min ({total_time/60:.1f} hrs)", flush=True)
print(f"Best val accuracy: {best_val:.2%}", flush=True)
print(f"Models saved:", flush=True)
print(f"  {BEST_MODEL_PATH}", flush=True)
print(f"  {FINAL_MODEL_PATH}", flush=True)
print("DONE!", flush=True)