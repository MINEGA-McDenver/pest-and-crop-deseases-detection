# Pest and Crop Disease Early Detection System

An end-to-end offline-first machine learning solution and mobile application designed to assist farmers and agricultural extension officers in identifying crop diseases early. 

## System Overview And Analysis

This project was built following a comprehensive system analysis framework:

### 1. Problem Definition
Pests and crop diseases critically impact agricultural yields and farmer livelihoods in Rwanda. The primary challenge is the lack of accessible, in-field expertise for early detection. This system provides an offline-capable, automated diagnostic tool directly on farmers' smartphones.

### 2. Stakeholder Analysis
- **Primary:** Smallholder farmers needing immediate, offline diagnostics.
- **Researchers:** Can use the underlying ML models and datasets for further agricultural studies.

### 3. Functional Requirements
- **Offline Image Classification:** Real-time, on-device disease detection using a smartphone camera or gallery.
- **Advanced Rejection Logic:** An 8-gate rejection system (e.g., Image Quality Assurance, `other_leaf` out-of-distribution rejection, entropy checks) to strictly avoid false positives.
- **Local Persistence:** Scanning history is saved securely via local SQLite.

### 4. Non-Functional Requirements
- **Performance:** High inference speed capable of running on low-resource and older mobile devices.
- **Storage:** Minimal application footprint (approximately ~8MB APK overall) achieved through model quantization.
- **Usability:** Intuitive, localized interface requiring minimal digital literacy.

### 5. System Boundaries
The application operates entirely within the boundaries of the local un-networked device post-installation. No cloud API or persistent internet connection is required for inference or history logging.

### 6. Data Flow Analysis
1. User captures or selects an image via the Mobile UI.
2. Image strictly passes through Image Quality Assurance (IQA).
3. The image is preprocessed (resizing, normalization) and fed into the `.tflite` model (MobileNetV2).
4. The output probabilities are evaluated against aggressive confidence thresholds.
5. The diagnostic result is returned to the user and securely persisted to the local `sqflite` database.

### 7. Constraints
- **Hardware Limitations:** Must run efficiently on devices with limited RAM and processing power.
- **Connectivity:** Strict offline-first requirement dictates all models and logic are bundled locally.

### 8. Feasibility Analysis
The system leverages a proven MobileNetV2 architecture, compressed using TFLite's float16 and dynamic range quantization. This minimizes the size without aggressively deteriorating multi-class (15 classes across 4 crops) accuracy, making the system highly feasible for real-world deployment.

---

## 📂 Project Structure

- `/datasets/`: Contains the raw, augmented, and model-ready image datasets.
- `/models/`: Contains the trained Keras models (`.keras`) alongside the quantized TFlite models (`.tflite`) for mobile deployment, class indices, and weights.
- `/mobile_app/`: The Flutter source code for the Androi application (`tflite_flutter`, `sqflite`).
- `/scripts/`: Python scripts covering the entire ML pipeline: data augmentation, model training, evaluation, TFLite conversion, and explainability mapping (Grad-CAM).

## Getting Started

### 1. Machine Learning Pipeline
To explore or retrain the models:
```bash
# Navigate to the project root and install requirements
pip install -r scripts/requirements_download.txt

# Run preprocessing and training
python scripts/organize_datasets.py
python scripts/train_model.py
```

### 2. Mobile Application
To run the Flutter application in debug mode:
```bash
cd mobile_app
flutter clean
flutter pub get
flutter build apk --debug

# Install on an attached device
adb install build/app/outputs/flutter-apk/app-debug.apk
```

For farmer field deployment, build a signed release APK:
```bash
cd mobile_app
flutter clean
flutter pub get
flutter build apk --release

# Install release build on an attached device
adb install build/app/outputs/flutter-apk/app-release.apk
```

Before installing on farmers' phones, complete all release gates in DEPLOY_CHECKLIST.md.
