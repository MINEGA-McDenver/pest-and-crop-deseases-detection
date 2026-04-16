# Deployment Checklist — Crop Disease Model

Run this checklist after **every retrain** before shipping a new model to the app.
Do not skip steps. A partial deployment (e.g. new `.tflite` but old `labels.txt`)
silently disables all other_leaf rejection logic and causes false positives in the field.

---

## Go/No-Go Gate (Field Deployment)

Release decision must be made before Step 1.

Current hardening status:

- [x] Release signing config supports env-based keystore values
- [x] Deprecated broad storage permissions removed
- [x] Crash logging and error guardrails added
- [x] Scan delete removes both database row and image file
- [x] Duplicate result-save flow fixed
- [x] Kinyarwanda baseline localization added for critical screens
- [x] Threshold calibration completed for the current model
- [x] Runtime thresholds updated from calibration output
- [x] Signed release APK built (`mobile_app/build/app/outputs/flutter-apk/app-release.apk`, 75.3MB)
- [ ] Release APK installed on a test device
- [ ] Offline smoke tests passed on target phones

Decision rule:

- NO-GO if any unchecked item remains.
- GO only after all items are checked and Step 6 deployment record is filled.

---

## Step 1 — Verify the training run produced a 15-class model

Open the terminal training log or `models/training_config.json` and confirm:

- [ ] `"num_classes": 15` in `training_config.json`
- [ ] Training log line reads: `Classes (15): [...]`
- [ ] `other_leaf` appears in the class list printed during training

**If num_classes ≠ 15:** the training run used an old dataset split.
Check that `datasets/model_ready/train/other_leaf/` exists and has images, then rerun.

---

## Step 2 — Verify model output files

Check each file in the `models/` directory:

- [ ] `labels.txt` contains exactly 15 lines
- [ ] `labels.txt` contains the line `other_leaf`
- [ ] `class_index.json` contains the key `"other_leaf"`
- [ ] `test_evaluation.json` contains an `"other_leaf"` block with `precision`, `recall`, `f1`
- [ ] `other_leaf` recall in `test_evaluation.json` is **≥ 0.80** (adjust target to your requirement)
- [ ] `best_model.keras` and `final_model.keras` have modification timestamps from this run
- [ ] `crop_disease_model.tflite` has a modification timestamp from this run

Quick terminal check (run from project root):
```bash
echo "=== Label count ===" && wc -l models/labels.txt
echo "=== other_leaf in labels ===" && grep other_leaf models/labels.txt
echo "=== other_leaf in class_index ===" && python -c "import json; d=json.load(open('models/class_index.json')); print('other_leaf index:', d.get('other_leaf', 'MISSING'))"
echo "=== other_leaf metrics ===" && python -c "import json; d=json.load(open('models/test_evaluation.json')); print(json.dumps(d.get('other_leaf', 'MISSING'), indent=2))"
```

---

## Step 3 — Run threshold calibration

After every retrain, thresholds must be re-derived from the new model's validation
output. Fixed constants in the app go stale as the model changes.

```bash
python -u calibrate_thresholds.py
```

- [ ] Script completes without errors
- [ ] `models/threshold_calibration.json` is written
- [ ] `models/threshold_calibration.png` is written — review the trade-off plot
- [ ] Supported-crop recall ≥ 0.90 (or your product target)
- [ ] Other-leaf false-positive rate is acceptably low for your field conditions

Update `classifier_service.dart` with the recommended values printed at the end:
```
static const double cropTotalThreshold     = <value from script>;
static const double otherLeafAbsoluteFloor = <value from script>;
```

- [x] `classifier_service.dart` defaults and runtime config updated to calibrated values

Current calibration result (2026-04-07):

- [x] `cropTotalThreshold = 0.84` (pilot runtime hotfix truth source)
- [x] `otherLeafAbsoluteFloor = 0.12` (pilot runtime hotfix truth source)
- [x] Ambiguity policy = **Option A (strict uncertain)** for mixed-scene crop ambiguity gates (`cropGap`, `secondCropTotal`) to avoid false-safe disease/healthy claims
- [x] Supported-crop recall = 90.58%
- [x] Other-leaf false-positive rate = 0.50%

---

## Step 4 — Copy model assets into the app

```bash
cp models/crop_disease_model.tflite  mobile_app/assets/models/crop_disease_model.tflite
cp models/labels.txt                 mobile_app/assets/models/labels.txt
```

Adjust the paths above to match your project layout.

- [ ] `mobile_app/assets/models/crop_disease_model.tflite` replaced
- [ ] `mobile_app/assets/models/labels.txt` replaced
- [ ] Both files have the current timestamp (confirm with `ls -lh mobile_app/assets/models/`)

**Never replace only one of these two files.** A `.tflite` from a 15-class run
paired with a 14-class `labels.txt` (or vice versa) causes a silent label mismatch.
The app's startup sanity check in `initialize()` will throw an exception if
`other_leaf` is missing from `labels.txt`, surfacing this before any scan is attempted.

---

## Step 5 — Rebuild and smoke-test the app

- [x] Rebuild the app (`flutter build apk` or your CI pipeline)
- [ ] Install on a test device
- [ ] App launches without a classifier initialization error
- [ ] Scan a **supported crop** image → result is a disease or healthy classification (not "Unsupported Crop")
- [ ] Scan an **other_leaf** image → result is "Unsupported Crop" (not a false disease prediction)
- [ ] Scan a **clearly unsupported plant** (e.g. grass, tree leaf) → result is "Unsupported Crop"

Latest build evidence:

- [x] `flutter build apk --release` completed on 2026-04-08
- [x] Output artifact: `mobile_app/build/app/outputs/flutter-apk/app-release.apk` (75.3MB)

---

## Step 6 — Record the deployment

Fill in the table below after each successful deployment.

| Date | Model file | num_classes | other_leaf recall | ol FP rate | cropTotalThreshold | olAbsoluteFloor | Deployed by |
|------|-----------|-------------|------------------|------------|-------------------|----------------|-------------|
| 2026-04-08 | best_model.keras + app-release.apk built | 15 | 99.50% | 0.50% | 0.90 | 0.10 | Pending device install + smoke tests |
|      |           |             |                  |            |                   |                |             |

---

## Step 7 — Balanced Mode Acceptance And Drift Monitoring

Before shipping beans/potato balanced mode, run:

```bash
python -u scripts/audit_field_photos.py --images-dir "C:\\Users\\Mbakenge\\Downloads\\test_app"
```

- [ ] `analysis_outputs/field_audit_summary.json` generated
- [ ] `analysis_outputs/field_audit_rows.tsv` generated
- [ ] Beans supported_confident_rate is within 0.12 of banana/maize reference
- [ ] Potato supported_confident_rate is within 0.12 of banana/maize reference
- [ ] Beans unsupported_or_other_leaf rate <= 0.20
- [ ] Potato unsupported_or_other_leaf rate <= 0.20
- [ ] Gate distribution reviewed: G5b/G7a regressions do not spike vs previous release

Post-release monitoring (first 7 days):

- [ ] Keep decision logging enabled for gate-level analysis
- [ ] Track daily counts of G5, G5b, G7a, and uncertain gates by crop
- [ ] If beans/potato unsupported spikes by >25% versus pilot baseline, open one controlled hotfix window
- [ ] Hotfix policy: one threshold-only update, then freeze and re-evaluate with fresh holdout

Strict release gate (mandatory before shipping):

```bash
python -u scripts/validate_release_readiness.py
```

- [ ] `models/release_readiness.json` generated
- [ ] `models/release_readiness.json` verdict is `PASS`
- [ ] No blocker remains for strict policy:
- [ ] Field supported_confident_rate >= 0.80 for banana, beans, maize, potato
- [ ] Beans<->Potato confusion rate <= 0.02
- [ ] Supported<->Unsupported confusion rates <= 0.02 in both directions
- [ ] Do not release when verdict is `BLOCKED`

---

## Quick failure guide

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| App throws "labels.txt is missing other_leaf" at startup | Old 14-class `labels.txt` deployed | Re-copy both `.tflite` + `labels.txt` and rebuild |
| Lookalike plants classified as real crops | Model trained without `other_leaf`, or thresholds too loose | Retrain with 15 classes; re-run calibration script |
| Real crop images always return "Unsupported Crop" | `cropTotalThreshold` too high after calibration | Lower it by 0.04 and re-test, or collect more supported-crop val images |
| other_leaf recall < 0.80 | `other_leaf` training set too small or not diverse enough | Expand other_leaf dataset and retrain |
