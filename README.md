# Crop Doctor — Offline Crop Disease Classifier (Flutter + TFLite)

End-to-end ML + Flutter system for early crop disease support in field conditions.
The app is designed for offline use on Android devices and targets banana, beans,
maize, and potato. A 15th class (`other_leaf`) is included to represent unsupported
leaves / out-of-distribution inputs.

## App Workflow (User Interaction)

This is the end-to-end flow a user follows inside the **Crop Doctor** mobile app.

1) **Open and unlock the app**
  - On launch, the app asks the user to unlock using the phone’s device authentication
    (biometric / screen-lock depending on what the device supports).

2) **Choose a language (first screen)**
  - Select **English** or **Kinyarwanda**, then continue to the Home screen.
  - Language can be changed later from Home (translate icon).

3) **Scan a leaf photo (offline)**
  - From Home, the user taps **Take Photo** (camera) or **Upload from Gallery**.
  - The model runs fully **on-device** (no internet required).
  - The scan is stored locally so it appears in **Scan History**.

4) **Review the result and guidance**
  - If the model is confident, the app shows the detected **crop** and either a
    **disease** label or **healthy**, plus short guidance.
  - If the input is not reliable/supported, the app shows one of:
    **Poor image quality**, **Unsupported / other leaf**, **Uncertain**, or
    **Unknown condition** (crop detected, but disease not in the supported list),
    and prompts the user to retake a clearer photo.

5) **Save and revisit past scans**
  - The user can bookmark/save a useful result (bookmark icon on the Result screen).
  - From Home, **Scan History** (history icon) shows **Recent** vs **Saved** scans.
  - Opening Scan History requires re-authentication.

6) **Resume behavior (security)**
  - If the app is backgrounded, it prompts the user to re-authenticate on return.
  - If backgrounded for more than ~30 seconds, it returns to the unlock + language gate.

## Current Status (Release Gate)

Release readiness is **PASS** based on the latest saved gate artifacts:

- Verdict: `models/release_readiness.json` -> `PASS` (`timestamp_utc`: 2026-05-05T06:56:39.639431+00:00)
- Lab/test evaluation (15-class): `models/test_evaluation.json`
  - `test_accuracy`: **0.9626**
  - `macro_f1`: **0.9505**
  - `test_top3_accuracy`: **0.9969**
  - `other_leaf` recall: **0.9875**
- Confusion safety (from `models/release_readiness.json`):
  - beans↔potato confusion rate: **0.007472** (policy max 0.02)
  - supported→unsupported rate: **0.016939** (policy max 0.02)
  - unsupported→supported rate: **0.016667** (policy max 0.02)
- Field acceptance policy (from `models/release_readiness.json`):
  - per-crop supported_confident_rate must be ≥ **0.30**
- Field audit evidence (from `analysis_outputs/field_audit_summary.json`, n=637):
  - banana supported_confident_rate: **0.7807**
  - beans supported_confident_rate: **0.9792**
  - maize supported_confident_rate: **0.9667**
  - potato supported_confident_rate: **0.9877**

Important:

- The policy thresholds used for PASS come from `models/test_evaluation.json` ->
  `strict_release_policy` and are copied into `models/release_readiness.json`.
- Some docstrings in scripts may describe older policy defaults; treat the JSON
  artifacts above as the canonical record.
- If you change any runtime thresholds or retrain, regenerate the audit + release
  readiness artifacts before deploying.

## Canonical “Truth” Artifacts

If you only read a few files, start here:

- `models/test_evaluation.json` — test metrics, temperature scaling, confusion diagnostics
- `models/runtime_gate_validation.json` — which runtime gate config was simulated on test
- `analysis_outputs/field_audit_summary.json` — per-image gate outcomes aggregated on holdout
- `models/release_readiness.json` — go/no-go verdict derived from the above
- `mobile_app/lib/services/classifier_service.dart` — *actual on-device gate logic*

## Model Specification

### Input / Output

- Input tensor: **224×224×3** RGB
- Preprocess (train + mobile): MobileNetV2 preprocess, equivalent to:

```text
x_norm = (x / 127.5) - 1.0
```

- Output: 15-way softmax over labels in `models/labels.txt`:

```text
banana_cordana
banana_healthy
banana_pestalotiopsis
banana_sigatoka
beans_angular_leaf_spot
beans_healthy
beans_rust
maize_common_rust
maize_gray_leaf_spot
maize_healthy
maize_northern_leaf_blight
other_leaf
potato_early_blight
potato_healthy
potato_late_blight
```

### Architecture (Keras)

From `models/model_summary.txt`:

- Base: MobileNetV2 backbone (frozen then partially unfrozen)
- Head: GlobalAveragePooling → Dense(256) → Dense(15)
- Params: **2,589,775** total (trainable ≈ **331,791**)

## Dataset Snapshot

Directory convention (Keras `image_dataset_from_directory`):

```text
datasets/model_ready/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

Split sizes from `models/dataset_stats.json`:

- Train: **43,183**
- Val: **1,958**
- Test: **1,952**
- Leakage guard: **0 overlaps** across splits (by stem + hash)

Class imbalance note (train split examples):

- `potato_healthy`: **1,372** images (lowest)
- `potato_late_blight`: **3,469** images
- `other_leaf`: **3,399** images

## Training Recipe (What Was Actually Used)

The recorded training configuration is in `models/training_config.json`.

Key points:

- Backbone: MobileNetV2
- Two-phase training (frozen base then fine-tuning last layers)
- Loss: focal loss with `gamma=2.0`, `alpha=0.25`, plus `other_leaf` label smoothing
- Augmentation: crop-specific rotation/zoom/contrast/brightness
- MixUp for `other_leaf` (`alpha=0.4`)
- Beans↔potato CutMix (prob 0.3) to reduce mutual confusion

### Focal Loss (math)

For a single sample with true-class probability $p_t$:

```text
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
gamma = 2.0
alpha = 0.25
```

Label smoothing is applied to `other_leaf` during training (see `models/training_config.json`).

### MixUp (math)

```text
lambda ~ Beta(alpha, alpha)
x' = lambda * x_i + (1 - lambda) * x_j
y' = lambda * y_i + (1 - lambda) * y_j
```

## Evaluation Summary

From `models/test_evaluation.json`:

- Accuracy: **0.9626**
- Macro F1: **0.9505** (macro precision **0.9532**, macro recall **0.9492**)
- Weighted F1: **0.9627**

### Confidence / Margin / Entropy Gates (offline diagnostics)

The evaluation exports acceptance-vs-accuracy tradeoffs for three simple gates:

- confidence threshold (top-1 prob)
- margin threshold (top1 − top2)
- entropy threshold

Example (confidence gate): threshold 0.60 gives accuracy 0.9802 at coverage 0.9595.

### Runtime gate simulation (test split)

The training/eval pipeline also simulates the **full mobile-style gate stack** on the
test set and writes a summary to `models/runtime_gate_validation.json`.

From `models/test_evaluation.json` → `runtime_gate_validation.summary`:

- supported_confident_rate: **0.8096**
- supported_false_unsupported_rate: **0.1846**
- unsupported_reject_rate: **0.8583**

### Temperature Scaling (math + learned value)

Mobile runtime applies temperature scaling to the predicted distribution.
The repository uses probability-space scaling (equivalent to log-prob scaling):

```text
p'_i = p_i^(1/T) / sum_j p_j^(1/T)
```

Learned temperature from `models/test_evaluation.json`:

- best_temperature: **0.8**
- raw_val_nll: **0.2301** → calibrated_val_nll: **0.1934**

### Entropy (math)

```text
H(p) = -sum_i p_i * log(p_i)
```

## Mobile Runtime Inference (On-Device)

The classifier is intentionally conservative. It does not always return a disease label;
it can return `unsupported`, `other_leaf`, or `uncertain` depending on safety gates.

### Crop totals (core idea)

Classes are grouped by crop (banana/beans/maize/potato). For crop $c$:

```text
P(c) = sum_{i in classes(c)} p_i
```

The best crop is `argmax_c P(c)`.

Ambiguity metrics used by later gates:

```text
gap = P(bestCrop) - P(secondCrop)
other_leaf_ratio = p_other_leaf / P(bestCrop)
class_ratio = p_top_class_in_bestCrop / P(bestCrop)
```

### Gate sequence (high level)

Implemented in `mobile_app/lib/services/classifier_service.dart`:

1. **G1 image quality** — brightness/contrast/green-ratio checks
2. **G5 other_leaf winner** — if `other_leaf` dominates, reject (unless focus-crop rescue applies)
3. **G5b other_leaf floor** — reject if `other_leaf` exceeds an absolute floor (with crop-aware boosts)
4. **G7a crop-total** — reject if best crop total is too small
5. **G7a other_leaf ratio** — reject if `other_leaf / bestCropTotal` is too high
6. **G7c ambiguity** — reject if crop gap, second-crop ambiguity, entropy, or within-crop class ratio fail
7. **G7d class confidence** — reject if top class probability is too low
8. **G7e healthy safety** — stricter rule for returning “healthy”
9. **G7g confident** — return disease/healthy

### Key runtime thresholds (mobile app)

Runtime configuration comes from:

- `mobile_app/assets/config/thresholds.json` (balanced-mode thresholds)
- `mobile_app/assets/config/mobile_runtime_recommendations.json` (temperature + some gates)

Current app config values:

- Temperature: **T=0.8**
- `cropTotalThreshold`: **0.70** (runtime recommendations override `thresholds.json` when present)
- `otherLeafAbsoluteFloor`: **0.14**
- `otherLeafThreshold` (winner gate): **0.30**
- `otherLeafVsCropRatioThreshold`: **0.65**
- `maxEntropyThreshold`: **1.5**
- `nonFocusClassConfidenceThreshold`: **0.60**
- `nonFocusClassRatioThreshold`: **0.45**
- `uncertainGapThreshold`: **0.30**
- `secondCropAmbiguityThreshold`: **0.15**
- Beans/potato relaxation: `beansPotatoCropTotalRelaxation` **0.08**
- Non-focus relaxation: `nonFocusCropTotalRelaxation` **0.08**

Healthy safety (hard-coded guardrails in the app):

- “healthy” requires confidence ≥ **0.80** (potato pilot: ≥ **0.72**)

### Configuration drift warning

This repo intentionally stores multiple threshold “snapshots”:

- **Shipped app** thresholds: `mobile_app/assets/config/*.json`
- **Simulated/evaluated** gate config: `models/runtime_gate_validation.json` +
  `analysis_outputs/field_audit_summary.json` (`thresholds_used`)

The saved PASS field audit was generated with (see `analysis_outputs/field_audit_summary.json`):

- `cropTotalThreshold`: 0.9
- `otherLeafAbsoluteFloor`: 0.14
- `otherLeafVsCropRatioThreshold`: 0.65
- `nonFocusClassConfidenceThreshold`: 0.55
- `temperatureScaling`: 0.8

Before deploying, ensure the *same* threshold config is used in:

- on-device assets
- `scripts/audit_field_photos.py` output
- `scripts/validate_release_readiness.py` output

## TFLite Export and Quantization

From `models/tflite_comparison.json`:

| Format | Size (MB) | Accuracy | Avg inference (ms) | File |
|---|---:|---:|---:|---|
| float32 | 9.72 | 0.9518 | 25.3 | `models/model_float32.tflite` |
| float16 | 4.89 | 0.9518 | 19.8 | `models/model_float16.tflite` |
| dynamic range | 2.71 | 0.9442 | 177.4 | `models/model_dynamic_range.tflite` |

Recommended deployment format: **float16**.

Note: These comparison artifacts are legacy benchmarks and are not regenerated by the trimmed workflow.

## Threshold Calibration (Sweep)

`calibrate_thresholds.py` sweeps `cropTotalThreshold × otherLeafAbsoluteFloor` on
the validation split and picks the best combination under constraints.

Constraints and objective (as implemented):

```text
crop_recall >= 0.90
focus_recall (beans/potato) >= 0.92
banana_maize_recall >= 0.90
other_leaf_false_positive_rate <= 0.015

balanced_score =
  0.70 * focus_recall +
  0.20 * banana_maize_recall +
  0.10 * crop_recall -
  2.00 * other_leaf_false_positive_rate
```

Outputs:

- `models/threshold_calibration.json` (full sweep + recommendation)
- `models/threshold_calibration.png` (recall vs FP-rate trade-off plot)

Latest recorded recommendation (from `models/threshold_calibration.json`):

- `crop_total_threshold`: **0.90**
- `ol_absolute_floor`: **0.14**

Note: the same artifact shows `meets_ol_fp_target=false` for the recommendation,
meaning it was selected via the script’s fallback logic (use it as a diagnostic,
not a guaranteed production setting).

## Release Workflow (Recommended)

0) (Optional, when retraining) Train the model:

```bash
scripts\run_py.cmd scripts\train_model.py
```

1) Produce/refresh runtime recommendations (temperature + crop-total):

```bash
scripts\run_py.cmd scripts\recalibrate_runtime.py
```

2) Calibrate thresholds (optional sweep for cropTotal/otherLeaf floor):

```bash
scripts\run_py.cmd calibrate_thresholds.py
```

3) Run a field audit on real photos:

```bash
scripts\run_py.cmd scripts\audit_field_photos.py --images-dir "datasets\holdout_farmer_style"
```

4) Validate strict release readiness (writes `models/release_readiness.json`):

```bash
scripts\run_py.cmd scripts\validate_release_readiness.py
```

5) Enforce a hard pre-build gate (CI/deploy step):

```bash
scripts\run_py.cmd scripts\check_release_gate.py
```

## Mobile Build

```bash
cd mobile_app
flutter clean
flutter pub get
flutter build apk --release
```

## Python Environment (minimal)

This repo does not currently ship a pinned `requirements.txt`. The core scripts
require at least:

```bash
pip install tensorflow numpy pillow scikit-learn matplotlib
```

On Windows, prefer `scripts\run_py.cmd` to force UTF-8 output.

## Quickstart — Run & Build

- **Prepare Python environment**: create a venv and install core deps.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow numpy pillow scikit-learn matplotlib
```

- **Run audit / calibration scripts** (use `run_py.cmd` on Windows):

```bash
scripts\run_py.cmd scripts\recalibrate_runtime.py
scripts\run_py.cmd calibrate_thresholds.py
scripts\run_py.cmd scripts\audit_field_photos.py --images-dir "datasets\holdout_farmer_style"
```

- **Build mobile app (Flutter)**:

```bash
cd mobile_app
flutter pub get
flutter build apk --release
```

## Where key files live

- Model artifacts: [models](models)
- Mobile assets (deployed model + thresholds): [mobile_app/assets/models](mobile_app/assets/models) and [mobile_app/assets/config](mobile_app/assets/config)
- Runtime gate simulator & evaluation: [models/test_evaluation.json](models/test_evaluation.json) and [models/runtime_gate_validation.json](models/runtime_gate_validation.json)
- Field audit outputs: [analysis_outputs/field_audit_summary.json](analysis_outputs/field_audit_summary.json)

## Updating runtime thresholds used by the app

- Edit `mobile_app/assets/config/thresholds.json` (or regenerate via `scripts/recalibrate_runtime.py`).
- After updating assets, rebuild the Flutter app so the new thresholds are embedded in `assets/`.

---

If you want, I can also: (a) add a minimal `requirements.txt`, (b) update `DEPLOY_CHECKLIST.md` with these build steps, or (c) draft short user-facing text for the app store listing. Which should I do next?


