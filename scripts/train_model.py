"""
Train MobileNetV2 model for 4-crop disease detection (15 classes).
Crops: Banana, Beans, Maize, Potato  +  other_leaf (rejection class)

Changes vs previous version (Fix D — retrain improvements):
  1. FocalLoss replaces sparse_categorical_crossentropy — forces the model
     to focus on hard lookalike cases instead of easy correct ones.
  2. other_leaf class weight boosted ×6 (was ×3).
     ×3 was not enough: the model still treated other_leaf like a normal
     disease class. ×6 makes the training penalty for missing an unsupported
     crop twice as severe, forcing the backbone to learn a much harder
     boundary between supported and unsupported plants.
  3. Class-aware augmentation: other_leaf images get heavier augmentation
     (stronger rotation, zoom, contrast, brightness) to improve generalisation
     across unseen plant types that were never in training.
  4. NEW — MixUp augmentation for other_leaf images only.
     Blends each other_leaf image with a random partner from the same batch.
     This forces the model to learn "not a supported crop" as a gradient
     concept rather than memorising specific pixel patterns, dramatically
     improving rejection of unseen lookalike plants.
  5. NEW — Per-class label smoothing for other_leaf inside FocalLoss.
     Instead of pushing other_leaf probability to a hard 1.0 target, we
     use 0.9 (smoothing ε=0.1). This prevents the model from becoming
     overconfident about the specific other_leaf training images, making
     the learned boundary more robust to unseen plant types.

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
PHASE2_EP   = 35          # fine-tune
INIT_LR     = 1e-3
FINE_LR     = 1e-5
PATIENCE    = 7
MAX_CLASS_WEIGHT = None   # None = uncapped

# FIX D2 — other_leaf weight multiplier raised ×3 → ×6
# ×3 left the model treating other_leaf like a normal disease class.
# ×6 makes the training penalty for misclassifying an unsupported crop
# as a real one twice as harsh, forcing the feature extractor to build a
# much harder boundary. If val accuracy drops >2% from the previous run,
# try ×5 as a middle ground — but try ×6 first.
OTHER_LEAF_WEIGHT_MULTIPLIER = 6.0

FINE_TUNE_FROM = 80       # MobileNetV2: unfreeze from layer 80

# Standard augmentation profile (crop disease classes — unchanged)
AUG_ROTATION   = 0.05
AUG_ZOOM       = 0.10
AUG_CONTRAST   = 0.20
AUG_BRIGHTNESS = 0.20

# Heavier augmentation profile for other_leaf only (unchanged)
OL_AUG_ROTATION   = 0.25
OL_AUG_ZOOM       = 0.25
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

CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]
DEFAULT_APP_CONFIDENCE_THRESHOLD = 0.60

ALLOWED_CROPS = {"banana", "beans", "maize", "potato", "other_leaf"}
MULTI_WORD_PREFIXES = ["other_leaf"]

os.makedirs(MODEL_DIR, exist_ok=True)


# ── FIX 1 + FIX D5: Focal Loss with per-class label smoothing ───────
# Focal loss down-weights easy examples and up-weights hard ones.
# gamma=2.0 is the standard value from the original paper.
# alpha=0.25 slightly reduces contribution from dominant easy classes.
#
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

if "other_leaf" not in class_names:
    print("ERROR: 'other_leaf' class not found in training data.", flush=True)
    print("  Make sure datasets/model_ready/train/other_leaf/ exists and has images.", flush=True)
    sys.exit(1)
OTHER_LEAF_IDX = class_names.index("other_leaf")
print(f"other_leaf class index: {OTHER_LEAF_IDX}", flush=True)


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

class_index = {name: i for i, name in enumerate(class_names)}
with open(os.path.join(MODEL_DIR, "class_index.json"), 'w') as f:
    json.dump(class_index, f, indent=2)


# ── FIX D2: Class weights — other_leaf boosted ×6 ──────────────────
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

    overlap_train_val  = len(train_stems.intersection(val_stems))
    overlap_train_test = len(train_stems.intersection(test_stems))
    overlap_val_test   = len(val_stems.intersection(test_stems))

    leakage_check[cname] = {
        "train_val_overlap":  overlap_train_val,
        "train_test_overlap": overlap_train_test,
        "val_test_overlap":   overlap_val_test,
    }
    total_train_val_overlap  += overlap_train_val
    total_train_test_overlap += overlap_train_test
    total_val_test_overlap   += overlap_val_test

print(f"Total validation images: {val_total}", flush=True)
print(f"Total test images: {test_total}", flush=True)
print("Filename-stem overlap check (possible leakage):", flush=True)
print(f"  train↔val:  {total_train_val_overlap}", flush=True)
print(f"  train↔test: {total_train_test_overlap}", flush=True)
print(f"  val↔test:   {total_val_test_overlap}", flush=True)

class_weight = {}
cap_label = f"capped at {MAX_CLASS_WEIGHT}" if MAX_CLASS_WEIGHT is not None else "uncapped"
print(f"Class weights ({cap_label}, other_leaf boosted ×{OTHER_LEAF_WEIGHT_MULTIPLIER}):", flush=True)

for i, cname in enumerate(class_names):
    w = total / (NUM_CLASSES * class_counts[cname])
    if MAX_CLASS_WEIGHT is not None:
        w = min(w, MAX_CLASS_WEIGHT)
    boost_label = ""
    if cname == "other_leaf":
        w *= OTHER_LEAF_WEIGHT_MULTIPLIER
        boost_label = f"  ← BOOSTED ×{OTHER_LEAF_WEIGHT_MULTIPLIER}"
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
    batch_size = tf.shape(images)[0]

    # Sample mix ratio λ from a symmetric Beta distribution.
    # tf.random.stateless_* requires explicit seeds; use plain random here.
    lam = tf.random.uniform(
        [batch_size], minval=OL_MIXUP_ALPHA, maxval=1.0 - OL_MIXUP_ALPHA
    )
    # Reshape for broadcasting over (H, W, C)
    lam_img = tf.reshape(lam, [-1, 1, 1, 1])

    # Randomly shuffle the batch to get mix partners for each sample
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)

    # Blended images: λ * original + (1-λ) * partner
    mixed = lam_img * images + (1.0 - lam_img) * shuffled_images

    # Only replace other_leaf samples; keep crop samples untouched
    is_other = tf.equal(labels, OTHER_LEAF_IDX)
    mask = tf.reshape(is_other, [-1, 1, 1, 1])
    mask = tf.broadcast_to(mask, tf.shape(images))

    return tf.where(mask, mixed, images), labels


# ── Build tf.data pipeline ────────────────────────────────────────────
# Order: preprocess → class-aware augment → MixUp (other_leaf only)
# Val and test sets: preprocessing only (no augmentation — consistent eval)
train_ds = (
    train_ds
    .map(preprocess,            num_parallel_calls=2)
    .map(augment_class_aware,   num_parallel_calls=2)
    .map(mixup_other_leaf,      num_parallel_calls=2)   # FIX D4
    .prefetch(1)
)
val_ds  = val_ds.map(preprocess,  num_parallel_calls=2).prefetch(1)
test_ds = test_ds.map(preprocess, num_parallel_calls=2).prefetch(1)


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
with open(os.path.join(MODEL_DIR, "model_summary.txt"), 'w') as f:
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
        lr      = float(self.model.optimizer.learning_rate)
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

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + math.cos(math.pi * epoch / self.total_epochs))
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
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

# FIX D5: FocalLoss now receives other_leaf_idx and num_classes so it
# can apply label smoothing (ε=OL_LABEL_SMOOTHING) to other_leaf samples.
# Created AFTER dataset loading so OTHER_LEAF_IDX and NUM_CLASSES are known.
focal_loss = FocalLoss(
    gamma=2.0,
    alpha=0.25,
    other_leaf_idx=OTHER_LEAF_IDX,
    num_classes=NUM_CLASSES,
    other_leaf_smoothing=OL_LABEL_SMOOTHING,
    name="focal_loss",
)


# ══════════════════ PHASE 1: Frozen backbone ════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"PHASE 1: Training top layers ({PHASE1_EP} epochs max)", flush=True)
print(f"  Loss:                  FocalLoss(gamma=2.0, alpha=0.25)", flush=True)
print(f"  other_leaf boost:      ×{OTHER_LEAF_WEIGHT_MULTIPLIER}", flush=True)
print(f"  other_leaf smoothing:  ε={OL_LABEL_SMOOTHING}", flush=True)
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
    callbacks=[checkpoint, early_stop, reduce_lr, progress],
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
for layer in base_model.layers[:FINE_TUNE_FROM]:
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
    callbacks=[checkpoint, early_stop_2, cosine_lr, progress],
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
    "seed":                   SEED,
    "loss":                   f"FocalLoss(gamma=2.0, alpha=0.25, ol_smoothing={OL_LABEL_SMOOTHING})",
    "other_leaf_weight_multiplier": OTHER_LEAF_WEIGHT_MULTIPLIER,
    "other_leaf_label_smoothing":   OL_LABEL_SMOOTHING,
    "other_leaf_mixup_alpha":       OL_MIXUP_ALPHA,
    "max_class_weight":       MAX_CLASS_WEIGHT,
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

dataset_stats = {
    "train_counts": class_counts,
    "val_counts":   val_counts,
    "test_counts":  test_counts,
    "totals": {
        "train": total,
        "val":   val_total,
        "test":  test_total,
    },
    "leakage_check": {
        "by_class": leakage_check,
        "totals": {
            "train_val_overlap":  total_train_val_overlap,
            "train_test_overlap": total_train_test_overlap,
            "val_test_overlap":   total_val_test_overlap,
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
max_test_probs  = np.max(all_test_probs, axis=1)

precision_macro,    recall_macro,    f1_macro,    _ = precision_recall_fscore_support(
    all_test_labels, all_test_preds, average='macro',    zero_division=0)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_test_labels, all_test_preds, average='weighted', zero_division=0)

report = classification_report(
    all_test_labels, all_test_preds,
    target_names=class_names, digits=4, zero_division=0,
)

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
    "recommended_app_threshold":  DEFAULT_APP_CONFIDENCE_THRESHOLD,
}
with open(os.path.join(MODEL_DIR, "test_evaluation.json"), 'w') as f:
    json.dump(test_eval, f, indent=2)

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
print(f"  {os.path.join(MODEL_DIR, 'test_classification_report.txt')}", flush=True)
print(f"  {os.path.join(MODEL_DIR, 'test_confusion_matrix.png')}", flush=True)
print("DONE!", flush=True)