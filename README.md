# Crop Disease Detection System (Offline-First)

End-to-end ML + Flutter system for early crop disease support in field conditions.
The app is designed for offline use on Android devices and targets banana, beans,
maize, and potato.

## Current Status (As of 2026-04-16)

Release readiness is currently BLOCKED for field deployment.

- Lab/test metrics are strong on the 15-class model (including other_leaf).
- Mobile classifier logic has been hardened with gate-level safety controls and
	crop-aware rescue paths for beans/potato.
- Field acceptance under strict release policy is not yet met for supported
	confident-rate per crop.

Primary release gate result:

- models/release_readiness.json -> verdict: BLOCKED
- Blocking reason: low field supported_confident_rate by crop (banana, beans,
	maize, potato all below policy threshold 0.80)

## Deep Analysis Summary

### 1. What Is Working Well

- Model training/export pipeline is complete and reproducible.
- 15-class setup is active, including other_leaf out-of-distribution handling.
- Mobile app supports runtime threshold loading from
	mobile_app/assets/config/thresholds.json with safe defaults.
- Safety-oriented gate system exists across G5/G5b/G7* stages, including
	preserve-focus-crop rescue logic and uncertainty gating.
- Signed release artifact has been built successfully:
	mobile_app/build/app/outputs/flutter-apk/app-release.apk

### 2. What Is Blocking Field Release

Strict policy (scripts/validate_release_readiness.py) checks:

- field_min_supported_confident_rate >= 0.80 for each supported crop
- beans_potato_confusion_rate <= 0.02
- supported/unsupported confusion rates <= 0.02

Measured (models/release_readiness.json):

- beans_potato_confusion_rate: 0.002896 (PASS)
- supported_to_unsupported_rate: 0.016605 (PASS)
- unsupported_to_supported_rate: 0.017094 (PASS)
- supported_confident_rate:
	- banana: 0.5556 (FAIL)
	- beans: 0.4000 (FAIL)
	- maize: 0.3750 (FAIL)
	- potato: 0.6923 (FAIL)

Interpretation:

- The core classifier is conservative in field-like samples, producing too many
	uncertain/unsupported outcomes for supported crops under strict policy.
- Confusion safety is acceptable; coverage/confidence on true supported crops is
	the current bottleneck.

### 3. Data and Evaluation Reality

The repository now contains evidence for both strong lab performance and field
deployment friction. This is expected in safety-first on-device systems.

Key artifacts:

- models/test_evaluation.json (15-class evaluation)
	- test_accuracy: 0.945
	- macro_f1: 0.9352
	- other_leaf recall: 0.9829
- analysis_outputs/field_audit_summary.json (strict real-photo audit)
	- total_images: 58
	- result counts: disease=10, healthy=24, uncertain=20, unsupported=4
	- high uncertain burden for banana/beans/potato and high unsupported on maize
- analysis_outputs/bean_potato_error_audit.json
	- hard examples show banana/other_leaf dominance in certain bean/potato
		edge cases, with many recoveries requiring rescue gates.

### 4. Threshold and Configuration Drift Risks

This repo has multiple threshold artifacts produced across calibration and pilot
hotfix sessions. They are useful, but can diverge.

- models/threshold_calibration.json recommends balanced thresholds around
	crop_total_threshold=0.81 and ol_absolute_floor=0.14.
- deployment_manifest.json notes a pilot hotfix truth source (0.84, 0.12).
- mobile_app/assets/config/thresholds.json currently carries balanced-mode runtime
	values with stronger beans/potato adjustments.

Actionable rule:

- Treat mobile_app/assets/config/thresholds.json +
	models/release_readiness.json as release decision truth for the current build,
	and re-run readiness validation after any threshold change.

### 5. System Maturity Assessment

- Technical maturity: HIGH (pipeline, app integration, calibration tooling,
	gate telemetry, audit scripts all present).
- Scientific maturity: MEDIUM-HIGH (good metrics and diagnostics, but field
	acceptance criteria not yet met).
- Deployment maturity: MEDIUM (release build exists, but no-go under strict
	field policy until coverage issue is reduced).

## Architecture Snapshot

### ML Pipeline

- scripts/preprocess_datasets.py
- scripts/train_model.py
- scripts/evaluate_model.py
- calibrate_thresholds.py
- scripts/audit_field_photos.py
- scripts/validate_release_readiness.py

### Mobile Inference

- Flutter app: mobile_app/
- TFLite model + labels: mobile_app/assets/models/
- Runtime threshold config: mobile_app/assets/config/thresholds.json
- Core gate logic: mobile_app/lib/services/classifier_service.dart
- Local persistence: sqflite history storage

## Repository Layout

- datasets/: data sources, model-ready splits, holdout sets
- models/: model artifacts, metrics, calibration, release-readiness outputs
- analysis_outputs/: field audit summaries and per-image diagnostic rows
- scripts/: training, evaluation, auditing, and release validation utilities
- mobile_app/: Flutter Android app and embedded runtime assets

## Recommended Next Actions

1. Run strict real-photo audit on latest threshold config and inspect per-image
	 rows to isolate dominant gate failures by crop.
2. Tune only threshold parameters within the allowed hotfix policy window, then
	 re-run scripts/validate_release_readiness.py.
3. If still blocked, refresh holdout data quality and retrain with targeted hard
	 examples for crops with low supported_confident_rate.
4. Perform device install + offline smoke tests and update deployment record.

## Build and Run

### Retrain / Evaluate

```bash
pip install -r scripts/requirements_download.txt
python scripts/organize_datasets.py
python scripts/train_model.py
python scripts/evaluate_model.py
python calibrate_thresholds.py
python scripts/validate_release_readiness.py
```

### Mobile Build

```bash
cd mobile_app
flutter clean
flutter pub get
flutter build apk --release
```

## Release Rule

Do not deploy to farmers while models/release_readiness.json reports BLOCKED.
Follow DEPLOY_CHECKLIST.md end-to-end after every retrain or threshold update.
