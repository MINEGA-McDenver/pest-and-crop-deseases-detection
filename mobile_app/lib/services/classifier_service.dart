import 'dart:io';
import 'dart:math';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../models/scan_result.dart';
import '../data/disease_info.dart';
import '../l10n/app_strings.dart';

class ClassifierService {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isInitialized = false;

  static const int inputSize = 224;
  static const double defaultTemperatureScaling = 1.8;

  // ─── Thresholds ─────────────────────────────────────────────
  // Keep fallback aligned with training-time simulator defaults.
  static const double defaultCropTotalThreshold = 0.84;

  static const double uncertainGapThreshold = 0.30;
  static const double maxEntropyThreshold = 1.5;
  static const double imageQualityMinStdDev = 12.0;

  // Healthy predictions need higher confidence because a missed disease
  // costs the farmer their harvest.
  static const double healthyMinConfidence = 0.80;
  static const double potatoHealthyMinConfidencePilot = 0.72;

  // Field pilot observability: logs every rejection/uncertain decision gate.
  static const bool enableDecisionLogging = true;

  // Ratio guard: if other_leaf probability is this large relative to the
  // winning crop total the image is suspicious.
  // Raised from 0.30 to 0.65: OL must be much closer to the crop total
  // before triggering rejection, giving crops more chance.
  static const double otherLeafVsCropRatioThreshold = 0.65;

  // Absolute other_leaf floor: reject if other_leaf exceeds this value
  // anywhere in the pipeline, even if it did not win the softmax.
  // Raised from 0.22 to 0.25 to let weak OL signals not override crops.
  static const double defaultOtherLeafAbsoluteFloor = 0.25;

  // Balanced mode (beans/potato): reduce false unsupported while preserving
  // strong unknown-crop protection through other_leaf winner + ratio checks.
  static const double defaultBeansPotatoCropTotalRelaxation = 0.05;
  static const double defaultBeansPotatoOtherLeafFloorBoost = 0.03;
  static const double defaultBeansPotatoUncertainGapThreshold = 0.08;
  static const double defaultBeansPotatoSecondCropThreshold = 0.55;
  static const double defaultBeansPotatoClassRatioThreshold = 0.45;
  static const double defaultBeansPotatoClassConfidenceThreshold = 0.68;
  static const double nonFocusCropTotalRelaxation = 0.05;
  static const double nonFocusOtherLeafFloorBoost = 0.01;
  static const double nonFocusClassRatioThreshold = 0.60;
  static const double nonFocusMaxEntropyThreshold = 1.5;
  static const double nonFocusClassConfidenceThreshold = 0.55;
  static const double defaultBeansPotatoRescueMinCropTotal = 0.18;
  static const double defaultBeansPotatoRescueMaxGapFromBest = 0.30;
  static const double defaultBeansPotatoRescueMaxOtherLeaf = 0.38;
  static const double rescueFocusSwapGuardCropGap = 0.08;
  static const double rescueFocusSwapGuardTopClassMargin = 0.05;
  static const bool forceBeansPotatoNeverUnsupported = true;
  static const double forceBeansPotatoMinCropTotal = 0.001;
  static const double forceBeansPotatoMinTopClassProb = 0.001;
  static const double lastChanceFocusMinCropTotal = 0.001;
  static const double lastChanceFocusMinTopClassProb = 0.001;
  static const double secondaryFocusMinCropTotal = 0.001;
  static const double secondaryFocusMinTopClassProb = 0.001;
  static const double secondaryFocusMaxGapFromBest = 1.00;
  static const bool preferBeansPotatoWhenCompetitive = true;
  static const double focusCropSwitchMaxGap = 0.18;
  static const double focusCropSwitchMinTotal = 0.20;

  // RELAXED 0.10 → 0.15 (Fix from external analysis):
  // A second crop at 9% is clear dominance by the first crop and should not
  // trigger uncertainty. The old 0.10 threshold was rejecting real supported
  // crop images where a small amount of probability legitimately leaked into
  // a second crop. 0.15 still catches genuine multi-crop ambiguity while
  // stopping false uncertainty calls on clean single-crop images.
  static const double secondCropAmbiguityThreshold = 0.15;

  // Crop grouping: maps each label to its crop.
  // 'other_leaf' is intentionally excluded — handled separately.
  static const Map<String, String> cropGrouping = {
    'banana_cordana': 'banana',
    'banana_healthy': 'banana',
    'banana_pestalotiopsis': 'banana',
    'banana_sigatoka': 'banana',
    'beans_angular_leaf_spot': 'beans',
    'beans_healthy': 'beans',
    'beans_rust': 'beans',
    'maize_common_rust': 'maize',
    'maize_gray_leaf_spot': 'maize',
    'maize_healthy': 'maize',
    'maize_northern_leaf_blight': 'maize',
    'potato_early_blight': 'potato',
    'potato_healthy': 'potato',
    'potato_late_blight': 'potato',
  };

  static const Map<String, String> healthyLabels = {
    'banana': 'banana_healthy',
    'beans': 'beans_healthy',
    'maize': 'maize_healthy',
    'potato': 'potato_healthy',
  };

  // Raised 0.30 → 0.40: the model now only rejects when other_leaf is
  // genuinely strong, giving crops more chance to be accepted.
  static const double otherLeafThreshold = 0.40;

  double _cropTotalThreshold = defaultCropTotalThreshold;
  double _temperatureScaling = defaultTemperatureScaling;
  double _otherLeafAbsoluteFloor = defaultOtherLeafAbsoluteFloor;
  double _otherLeafVsCropRatioThreshold = otherLeafVsCropRatioThreshold;
  double _nonFocusClassRatioThreshold = nonFocusClassRatioThreshold;
  double _nonFocusMaxEntropyThreshold = nonFocusMaxEntropyThreshold;
  double _nonFocusClassConfidenceThreshold = nonFocusClassConfidenceThreshold;
  double _beansPotatoCropTotalRelaxation =
      defaultBeansPotatoCropTotalRelaxation;
  double _beansPotatoOtherLeafFloorBoost =
      defaultBeansPotatoOtherLeafFloorBoost;
  double _beansPotatoUncertainGapThreshold =
      defaultBeansPotatoUncertainGapThreshold;
  double _beansPotatoSecondCropThreshold =
      defaultBeansPotatoSecondCropThreshold;
  double _beansPotatoClassRatioThreshold =
      defaultBeansPotatoClassRatioThreshold;
  double _beansPotatoClassConfidenceThreshold =
      defaultBeansPotatoClassConfidenceThreshold;
  double _beansPotatoRescueMinCropTotal = defaultBeansPotatoRescueMinCropTotal;
  double _beansPotatoRescueMaxGapFromBest =
      defaultBeansPotatoRescueMaxGapFromBest;
  double _beansPotatoRescueMaxOtherLeaf = defaultBeansPotatoRescueMaxOtherLeaf;

  Future<Interpreter> _loadInterpreter() async {
    const modelAssetPath = 'assets/models/crop_disease_model.tflite';

    try {
      return await Interpreter.fromAsset(modelAssetPath);
    } catch (_) {
      // Some Android builds/devices fail to memory-map .tflite assets.
      // Fallback: copy model bytes to a temp file and open from file path.
      final modelData = await rootBundle.load(modelAssetPath);
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/crop_disease_model.tflite');
      await tempFile.writeAsBytes(modelData.buffer.asUint8List(), flush: true);
      return Interpreter.fromFile(tempFile);
    }
  }

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      _interpreter = await _loadInterpreter();

      final labelData = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelData
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();

      // ── DEPLOYMENT SANITY CHECK ──────────────────────────────
      // If the deployed labels.txt is from an old 14-class run, every
      // other_leaf guard in classifyImage() silently resolves to 0.0 and
      // the entire false-positive defence is inactive. Surface this at
      // startup so the bug is caught immediately rather than discovered
      // in the field when lookalike plants produce confident wrong results.
      //
      // Fix: retrain with the updated train_model.py (which enforces
      // other_leaf presence), copy the new .tflite + labels.txt into
      // assets/models/, and rebuild the app.
      if (!_labels.contains('other_leaf')) {
        throw Exception(
          'Model deployment error: labels.txt is missing the other_leaf class.\n'
          'The deployed model is a ${_labels.length}-class build from a previous '
          'training run. All unsupported-crop rejection logic is currently inactive.\n'
          'Action required: deploy the 15-class model (retrain with '
          'train_model.py, copy new .tflite + labels.txt to assets/models/, '
          'and rebuild the app) before enabling scanning.',
        );
      }

      await _loadThresholdsFromAssets();

      _isInitialized = true;
    } catch (e) {
      throw Exception('Failed to initialize classifier: $e');
    }
  }

  Future<void> _loadThresholdsFromAssets() async {
    try {
      final jsonText = await rootBundle.loadString(
        'assets/config/thresholds.json',
      );
      final decoded = jsonDecode(jsonText) as Map<String, dynamic>;
      final thresholds =
          (decoded['thresholds'] as Map<String, dynamic>?) ??
          <String, dynamic>{};

      final runtimeRecommendations = await _loadRuntimeRecommendations();
      final runtimeRecoTemperature =
          runtimeRecommendations['temperatureScaling'];
      final runtimeRecoCropTotalThreshold =
          runtimeRecommendations['cropTotalThreshold'];
      final runtimeRecoConfidenceThreshold =
          runtimeRecommendations['confidenceThreshold'];
      final runtimeRecoMaxEntropyThreshold =
          runtimeRecommendations['maxEntropyThreshold'];
      final topLevelTemperature = (decoded['temperatureScaling'] as num?)
          ?.toDouble();
      final nestedTemperature = (thresholds['temperatureScaling'] as num?)
          ?.toDouble();
      final configuredTemperature =
          runtimeRecoTemperature ?? topLevelTemperature ?? nestedTemperature;

      if (runtimeRecoConfidenceThreshold != null &&
          runtimeRecoConfidenceThreshold > 0 &&
          runtimeRecoConfidenceThreshold < 1) {
        _nonFocusClassConfidenceThreshold = runtimeRecoConfidenceThreshold;
      }
      if (runtimeRecoMaxEntropyThreshold != null &&
          runtimeRecoMaxEntropyThreshold > 0 &&
          runtimeRecoMaxEntropyThreshold < 5) {
        _nonFocusMaxEntropyThreshold = runtimeRecoMaxEntropyThreshold;
      }

      final cropTotal = (thresholds['cropTotalThreshold'] as num?)?.toDouble();
      final configuredCropTotal = runtimeRecoCropTotalThreshold ?? cropTotal;
      final otherLeafFloor = (thresholds['otherLeafAbsoluteFloor'] as num?)
          ?.toDouble();
      final nonFocusClassRatioThreshold =
          (thresholds['nonFocusClassRatioThreshold'] as num?)?.toDouble();
      final beansPotatoCropTotalRelaxation =
          (thresholds['beansPotatoCropTotalRelaxation'] as num?)?.toDouble();
      final beansPotatoOtherLeafFloorBoost =
          (thresholds['beansPotatoOtherLeafFloorBoost'] as num?)?.toDouble();
      final beansPotatoUncertainGapThreshold =
          (thresholds['beansPotatoUncertainGapThreshold'] as num?)?.toDouble();
      final beansPotatoSecondCropThreshold =
          (thresholds['beansPotatoSecondCropThreshold'] as num?)?.toDouble();
      final beansPotatoClassRatioThreshold =
          (thresholds['beansPotatoClassRatioThreshold'] as num?)?.toDouble();
      final beansPotatoClassConfidenceThreshold =
          (thresholds['beansPotatoClassConfidenceThreshold'] as num?)
              ?.toDouble();
      final beansPotatoRescueMinCropTotal =
          (thresholds['beansPotatoRescueMinCropTotal'] as num?)?.toDouble();
      final beansPotatoRescueMaxGapFromBest =
          (thresholds['beansPotatoRescueMaxGapFromBest'] as num?)?.toDouble();
      final beansPotatoRescueMaxOtherLeaf =
          (thresholds['beansPotatoRescueMaxOtherLeaf'] as num?)?.toDouble();
      final otherLeafVsCropRatioThreshold =
          (thresholds['otherLeafVsCropRatioThreshold'] as num?)?.toDouble();

      if (otherLeafVsCropRatioThreshold != null &&
          otherLeafVsCropRatioThreshold > 0 &&
          otherLeafVsCropRatioThreshold < 1) {
        _otherLeafVsCropRatioThreshold = otherLeafVsCropRatioThreshold;
      }

      if (configuredTemperature != null &&
          configuredTemperature > 0 &&
          configuredTemperature <= 5) {
        _temperatureScaling = configuredTemperature;
      }

      if (configuredCropTotal != null &&
          configuredCropTotal > 0 &&
          configuredCropTotal < 1) {
        _cropTotalThreshold = configuredCropTotal;
      }
      if (otherLeafFloor != null && otherLeafFloor >= 0 && otherLeafFloor < 1) {
        _otherLeafAbsoluteFloor = otherLeafFloor;
      }
      if (nonFocusClassRatioThreshold != null &&
          nonFocusClassRatioThreshold > 0 &&
          nonFocusClassRatioThreshold < 1) {
        _nonFocusClassRatioThreshold = nonFocusClassRatioThreshold;
      }
      if (beansPotatoCropTotalRelaxation != null &&
          beansPotatoCropTotalRelaxation >= 0 &&
          beansPotatoCropTotalRelaxation < 0.2) {
        _beansPotatoCropTotalRelaxation = beansPotatoCropTotalRelaxation;
      }
      if (beansPotatoOtherLeafFloorBoost != null &&
          beansPotatoOtherLeafFloorBoost >= 0 &&
          beansPotatoOtherLeafFloorBoost < 0.5) {
        _beansPotatoOtherLeafFloorBoost = beansPotatoOtherLeafFloorBoost;
      }
      if (beansPotatoUncertainGapThreshold != null &&
          beansPotatoUncertainGapThreshold > 0 &&
          beansPotatoUncertainGapThreshold < 1) {
        _beansPotatoUncertainGapThreshold = beansPotatoUncertainGapThreshold;
      }
      if (beansPotatoSecondCropThreshold != null &&
          beansPotatoSecondCropThreshold > 0 &&
          beansPotatoSecondCropThreshold < 1) {
        _beansPotatoSecondCropThreshold = beansPotatoSecondCropThreshold;
      }
      if (beansPotatoClassRatioThreshold != null &&
          beansPotatoClassRatioThreshold > 0 &&
          beansPotatoClassRatioThreshold < 1) {
        _beansPotatoClassRatioThreshold = beansPotatoClassRatioThreshold;
      }
      if (beansPotatoClassConfidenceThreshold != null &&
          beansPotatoClassConfidenceThreshold > 0 &&
          beansPotatoClassConfidenceThreshold < 1) {
        _beansPotatoClassConfidenceThreshold =
            beansPotatoClassConfidenceThreshold;
      }
      if (beansPotatoRescueMinCropTotal != null &&
          beansPotatoRescueMinCropTotal > 0 &&
          beansPotatoRescueMinCropTotal < 1) {
        _beansPotatoRescueMinCropTotal = beansPotatoRescueMinCropTotal;
      }
      if (beansPotatoRescueMaxGapFromBest != null &&
          beansPotatoRescueMaxGapFromBest >= 0 &&
          beansPotatoRescueMaxGapFromBest < 1) {
        _beansPotatoRescueMaxGapFromBest = beansPotatoRescueMaxGapFromBest;
      }
      if (beansPotatoRescueMaxOtherLeaf != null &&
          beansPotatoRescueMaxOtherLeaf > 0 &&
          beansPotatoRescueMaxOtherLeaf < 1) {
        _beansPotatoRescueMaxOtherLeaf = beansPotatoRescueMaxOtherLeaf;
      }
    } catch (_) {
      // Use compiled defaults when threshold config is absent or invalid.
      _cropTotalThreshold = defaultCropTotalThreshold;
      _temperatureScaling = defaultTemperatureScaling;
      _otherLeafAbsoluteFloor = defaultOtherLeafAbsoluteFloor;
      _nonFocusClassRatioThreshold = nonFocusClassRatioThreshold;
      _nonFocusMaxEntropyThreshold = nonFocusMaxEntropyThreshold;
      _nonFocusClassConfidenceThreshold = nonFocusClassConfidenceThreshold;
      _beansPotatoCropTotalRelaxation = defaultBeansPotatoCropTotalRelaxation;
      _beansPotatoOtherLeafFloorBoost = defaultBeansPotatoOtherLeafFloorBoost;
      _beansPotatoUncertainGapThreshold =
          defaultBeansPotatoUncertainGapThreshold;
      _beansPotatoSecondCropThreshold = defaultBeansPotatoSecondCropThreshold;
      _beansPotatoClassRatioThreshold = defaultBeansPotatoClassRatioThreshold;
      _beansPotatoClassConfidenceThreshold =
          defaultBeansPotatoClassConfidenceThreshold;
      _beansPotatoRescueMinCropTotal = defaultBeansPotatoRescueMinCropTotal;
      _beansPotatoRescueMaxGapFromBest = defaultBeansPotatoRescueMaxGapFromBest;
      _beansPotatoRescueMaxOtherLeaf = defaultBeansPotatoRescueMaxOtherLeaf;
    }
  }

  Future<Map<String, double>> _loadRuntimeRecommendations() async {
    final runtimeRecommendations = <String, double>{};
    try {
      final jsonText = await rootBundle.loadString(
        'assets/config/mobile_runtime_recommendations.json',
      );
      final decoded = jsonDecode(jsonText) as Map<String, dynamic>;
      final recommendedThresholds =
          decoded['recommendedThresholds'] as Map<String, dynamic>?;

      final temperatureScaling = (decoded['temperatureScaling'] as num?)
          ?.toDouble();
      final confidenceThreshold =
          (recommendedThresholds?['confidenceThreshold'] as num?)?.toDouble();
      final cropTotalThreshold =
          (recommendedThresholds?['cropTotalThreshold'] as num?)?.toDouble();
      final maxEntropyThreshold =
          (recommendedThresholds?['maxEntropyThreshold'] as num?)?.toDouble();

      if (temperatureScaling != null) {
        runtimeRecommendations['temperatureScaling'] = temperatureScaling;
      }
      if (confidenceThreshold != null) {
        runtimeRecommendations['confidenceThreshold'] = confidenceThreshold;
      }
      if (cropTotalThreshold != null) {
        runtimeRecommendations['cropTotalThreshold'] = cropTotalThreshold;
      }
      if (maxEntropyThreshold != null) {
        runtimeRecommendations['maxEntropyThreshold'] = maxEntropyThreshold;
      }
    } catch (_) {
      // Ignore: defaults and thresholds.json continue to apply.
    }
    return runtimeRecommendations;
  }

  bool get isInitialized => _isInitialized;

  // ─── Image Quality Check ────────────────────────────────────
  ImageQualityResult _checkImageQuality(img.Image image) {
    double totalBrightness = 0;
    double totalVariance = 0;
    int pixelCount = 0;
    List<double> brightnessValues = [];
    int greenPixels = 0;

    int stepX = max(1, image.width ~/ 50);
    int stepY = max(1, image.height ~/ 50);

    for (int y = 0; y < image.height; y += stepY) {
      for (int x = 0; x < image.width; x += stepX) {
        final pixel = image.getPixel(x, y);
        double brightness =
            (pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114);
        totalBrightness += brightness;
        brightnessValues.add(brightness);
        pixelCount++;

        if (pixel.g > pixel.r * 1.1 &&
            pixel.g > pixel.b * 1.1 &&
            pixel.g > 50) {
          greenPixels++;
        }
      }
    }

    double meanBrightness = totalBrightness / pixelCount;

    for (double b in brightnessValues) {
      totalVariance += (b - meanBrightness) * (b - meanBrightness);
    }
    double stdDev = sqrt(totalVariance / pixelCount);
    double greenRatio = greenPixels / pixelCount;

    List<String> issues = [];
    bool isAcceptable = true;

    if (meanBrightness < 30) {
      issues.add(AppStrings.trCode('qualityTooDark'));
      isAcceptable = false;
    } else if (meanBrightness > 240) {
      issues.add(AppStrings.trCode('qualityTooBright'));
      isAcceptable = false;
    }

    if (stdDev < imageQualityMinStdDev) {
      issues.add(AppStrings.trCode('qualityLowDetail'));
      isAcceptable = false;
    }

    if (greenRatio < 0.03) {
      issues.add(AppStrings.trCode('qualityNoLeaf'));
      isAcceptable = false;
    }

    return ImageQualityResult(
      isAcceptable: isAcceptable,
      meanBrightness: meanBrightness,
      stdDev: stdDev,
      greenRatio: greenRatio,
      issues: issues,
    );
  }

  // ─── Preprocessing ──────────────────────────────────────────
  Float32List _preprocessImage(img.Image image) {
    final resized = img.copyResize(image, width: inputSize, height: inputSize);
    final input = Float32List(1 * inputSize * inputSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        // MobileNetV2 expects normalized input in [-1, 1].
        input[pixelIndex++] = (pixel.r / 127.5) - 1.0;
        input[pixelIndex++] = (pixel.g / 127.5) - 1.0;
        input[pixelIndex++] = (pixel.b / 127.5) - 1.0;
      }
    }
    return input;
  }

  // ─── Softmax with Temperature ───────────────────────────────
  // Applies temperature scaling in log-space:
  //   scaled[i] = log(p[i]) / T  →  softmax
  // This is the mathematically correct way to temperature-scale a model
  // that already outputs softmax probabilities. T=1.8 (>1) reduces
  // overconfidence without collapsing the distribution.
  List<double> _applySoftmaxWithTemperature(
    List<double> probs,
    double temperature,
  ) {
    final logProbs = probs
        .map((p) => log(max(p, 1e-10)) / temperature)
        .toList();
    final maxVal = logProbs.reduce(max);
    final exps = logProbs.map((l) => exp(l - maxVal)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExps).toList();
  }

  // ─── Aggregate by Crop ──────────────────────────────────────
  // 'other_leaf' is excluded from cropGrouping so it never accumulates
  // into any crop total.
  Map<String, double> _aggregateByCrop(List<double> probabilities) {
    Map<String, double> cropProbs = {};
    for (int i = 0; i < _labels.length && i < probabilities.length; i++) {
      final label = _labels[i];
      final crop = cropGrouping[label];
      if (crop != null) {
        cropProbs[crop] = (cropProbs[crop] ?? 0) + probabilities[i];
      }
    }
    return cropProbs;
  }

  // ─── Get classes for a specific crop ────────────────────────
  Map<String, double> _getClassesForCrop(
    String crop,
    List<double> probabilities,
  ) {
    Map<String, double> classes = {};
    for (int i = 0; i < _labels.length && i < probabilities.length; i++) {
      if (cropGrouping[_labels[i]] == crop) {
        classes[_labels[i]] = probabilities[i];
      }
    }
    return classes;
  }

  // ─── Shannon Entropy ────────────────────────────────────────
  // High entropy = spread probabilities = uncertain/noisy image.
  // Real crop leaf: ~0.8–1.2. Grass/non-target: ~2.0+.
  double _calculateEntropy(List<double> probabilities) {
    double entropy = 0.0;
    for (double p in probabilities) {
      if (p > 1e-10) {
        entropy -= p * log(p) / log(2.0);
      }
    }
    return entropy;
  }

  // ─── Main Classification ────────────────────────────────────
  Future<ScanResult> classifyImage(String imagePath) async {
    if (!_isInitialized) await initialize();

    final imageFile = File(imagePath);
    if (!await imageFile.exists()) {
      throw Exception(AppStrings.trCode('selectedImageMissing'));
    }

    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) {
      throw Exception(AppStrings.trCode('couldNotDecodeImage'));
    }

    // ── Step 1: Check image quality ──
    final quality = _checkImageQuality(image);
    if (!quality.isAcceptable) {
      _logDecision(
        gate: 'G1_quality',
        resultType: 'poor_quality',
        allProbs: const {},
        cropName: 'unknown',
        confidence: 0.0,
        note: quality.issues.join(' | '),
      );
      return ScanResult(
        imagePath: imagePath,
        cropName: 'unknown',
        diseaseName: 'poorImageQuality',
        confidence: 0.0,
        resultType: 'poor_quality',
        allProbabilities: {},
        dateTime: DateTime.now().toIso8601String(),
        qualityIssues: quality.issues,
      );
    }

    // ── Step 2: Run model inference ──
    final input = _preprocessImage(image);
    final inputTensor = input.reshape([1, inputSize, inputSize, 3]);
    final output = List.filled(
      1 * _labels.length,
      0.0,
    ).reshape([1, _labels.length]);

    _interpreter!.run(inputTensor, output);
    final rawOutputs = List<double>.from(output[0]);

    // Keep a pre-temperature view for other_leaf safety gates.
    // This avoids missing strong other_leaf signals that get deflated by T>1.
    Map<String, double> allRawProbs = {};
    for (int i = 0; i < _labels.length && i < rawOutputs.length; i++) {
      allRawProbs[_labels[i]] = rawOutputs[i];
    }

    // ── Step 3: Apply temperature scaling ──
    final probabilities = _applySoftmaxWithTemperature(
      rawOutputs,
      _temperatureScaling,
    );

    // ── Step 4: Build probability map ──
    Map<String, double> allProbs = {};
    for (int i = 0; i < _labels.length && i < probabilities.length; i++) {
      allProbs[_labels[i]] = probabilities[i];
    }

    final otherLeafProbScaled = allProbs['other_leaf'] ?? 0.0;
    final otherLeafProbRaw = allRawProbs['other_leaf'] ?? 0.0;
    final otherLeafProb = max(otherLeafProbScaled, otherLeafProbRaw);

    final cropProbs = _aggregateByCrop(probabilities);
    final sortedCrops = cropProbs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final bestCandidateCrop = sortedCrops.isNotEmpty
        ? sortedCrops[0].key
        : 'unknown';
    final bestCandidateCropTotal = sortedCrops.isNotEmpty
        ? sortedCrops[0].value
        : 0.0;
    final secondaryFocusCandidate = _selectSecondaryFocusCandidate(
      cropProbs,
      allProbs,
      bestCropTotal: bestCandidateCropTotal,
    );

    // ── Step 5: Early exit — other_leaf direct softmax winner ──
    // Threshold raised 0.30 → 0.40 to give crops more chance; the model
    // now only rejects when other_leaf is genuinely strong.
    if (otherLeafProb >= otherLeafThreshold) {
      final preserveCrop = secondaryFocusCandidate?.key ?? bestCandidateCrop;
      final preserveCropTotal =
          secondaryFocusCandidate?.value ?? bestCandidateCropTotal;
      if (_shouldPreserveFocusCropIdentity(
        candidateCrop: preserveCrop,
        candidateCropTotal: preserveCropTotal,
        allProbs: allProbs,
      )) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: preserveCrop,
          confidence: max(
            preserveCropTotal,
            _topClassProbForCrop(preserveCrop, allProbs),
          ),
          gate: 'G5_preserve_focus_crop',
          reasonKey: 'focusCropRetake',
          note:
              'other_leaf_gate=${otherLeafProb.toStringAsFixed(3)} raw=${otherLeafProbRaw.toStringAsFixed(3)} scaled=${otherLeafProbScaled.toStringAsFixed(3)} candidateCrop=$preserveCrop candidateCropTotal=${preserveCropTotal.toStringAsFixed(3)}',
        );
      }

      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCrop: bestCandidateCrop,
        candidateCropTotal: bestCandidateCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowLight',
      );
      if (rescue != null) return rescue;

      final forcedFallback = _buildForcedBeansPotatoFallback(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCandidateCropTotal,
        otherLeafProb: otherLeafProb,
        fallbackGate: 'G5_other_leaf_winner',
      );
      if (forcedFallback != null) return forcedFallback;

      final lastChance = _buildLastChanceFocusOverride(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCandidateCropTotal,
        fallbackGate: 'G5_other_leaf_winner',
        reasonKey: 'focusCropPriority',
      );
      if (lastChance != null) return lastChance;

      _logDecision(
        gate: 'G5_other_leaf_winner',
        resultType: 'other_leaf',
        allProbs: allProbs,
        cropName: 'unknown',
        confidence: otherLeafProb,
        note:
            'other_leaf_gate=${otherLeafProb.toStringAsFixed(3)} raw=${otherLeafProbRaw.toStringAsFixed(3)} scaled=${otherLeafProbScaled.toStringAsFixed(3)}',
      );

      return ScanResult(
        imagePath: imagePath,
        cropName: 'unknown',
        diseaseName: 'unsupportedCrop',
        confidence: otherLeafProb,
        resultType: 'other_leaf',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // ── Step 5b: Absolute other_leaf floor ──────────────────────
    // Even when other_leaf did not win the softmax, any probability above
    // otherLeafAbsoluteFloor is a red flag. A genuine supported-crop image
    // almost never gives other_leaf more than ~10%.
    final effectiveOtherLeafFloor = _effectiveOtherLeafAbsoluteFloor(
      bestCandidateCrop,
    );

    if (otherLeafProb > effectiveOtherLeafFloor) {
      final preserveCrop = secondaryFocusCandidate?.key ?? bestCandidateCrop;
      final preserveCropTotal =
          secondaryFocusCandidate?.value ?? bestCandidateCropTotal;
      if (_shouldPreserveFocusCropIdentity(
        candidateCrop: preserveCrop,
        candidateCropTotal: preserveCropTotal,
        allProbs: allProbs,
      )) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: preserveCrop,
          confidence: max(
            preserveCropTotal,
            _topClassProbForCrop(preserveCrop, allProbs),
          ),
          gate: 'G5b_preserve_focus_crop',
          reasonKey: 'focusCropRetake',
          note:
              'other_leaf_gate=${otherLeafProb.toStringAsFixed(3)} raw=${otherLeafProbRaw.toStringAsFixed(3)} scaled=${otherLeafProbScaled.toStringAsFixed(3)} floor=${effectiveOtherLeafFloor.toStringAsFixed(3)} candidateCrop=$preserveCrop',
        );
      }

      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCrop: bestCandidateCrop,
        candidateCropTotal: bestCandidateCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescuePossibleCropRetake',
      );
      if (rescue != null) return rescue;

      final forcedFallback = _buildForcedBeansPotatoFallback(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCandidateCropTotal,
        otherLeafProb: otherLeafProb,
        fallbackGate: 'G5b_other_leaf_floor',
      );
      if (forcedFallback != null) return forcedFallback;

      final lastChance = _buildLastChanceFocusOverride(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCandidateCropTotal,
        fallbackGate: 'G5b_other_leaf_floor',
        reasonKey: 'focusCropPriority',
      );
      if (lastChance != null) return lastChance;

      _logDecision(
        gate: 'G5b_other_leaf_floor',
        resultType: 'other_leaf',
        allProbs: allProbs,
        cropName: 'unknown',
        confidence: otherLeafProb,
        note:
            'other_leaf_gate=${otherLeafProb.toStringAsFixed(3)} raw=${otherLeafProbRaw.toStringAsFixed(3)} scaled=${otherLeafProbScaled.toStringAsFixed(3)} floor=${effectiveOtherLeafFloor.toStringAsFixed(3)} candidateCrop=$bestCandidateCrop',
      );

      return ScanResult(
        imagePath: imagePath,
        cropName: 'unknown',
        diseaseName: 'unsupportedCrop',
        confidence: otherLeafProb,
        resultType: 'other_leaf',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // ── Step 6: Aggregate by crop ──
    if (sortedCrops.isEmpty) {
      final lastChance = _buildLastChanceFocusOverride(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: 0.0,
        fallbackGate: 'G6_no_crop_candidates',
        reasonKey: 'focusCropPriority',
      );
      if (lastChance != null) return lastChance;

      _logDecision(
        gate: 'G6_no_crop_candidates',
        resultType: 'unsupported',
        allProbs: allProbs,
        cropName: 'unknown',
        confidence: 0.0,
        note: 'cropProbs empty',
      );
      return ScanResult(
        imagePath: imagePath,
        cropName: 'unknown',
        diseaseName: 'classificationError',
        confidence: 0.0,
        resultType: 'unsupported',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    var bestCrop = sortedCrops[0].key;
    var bestCropTotal = sortedCrops[0].value;

    if (secondaryFocusCandidate != null && !_isBeansOrPotato(bestCrop)) {
      bestCrop = secondaryFocusCandidate.key;
      bestCropTotal = secondaryFocusCandidate.value;
    }

    final preferredFocusCrop = _selectPreferredBeansPotatoCrop(
      cropProbs,
      allProbs,
      currentBestCrop: bestCrop,
      currentBestCropTotal: bestCropTotal,
    );
    if (preferredFocusCrop != null) {
      bestCrop = preferredFocusCrop.key;
      bestCropTotal = preferredFocusCrop.value;
    }

    final secondCropTotal = _secondBestCropTotalExcluding(cropProbs, bestCrop);
    final cropGap = bestCropTotal - secondCropTotal;

    // ── Step 7: DECISION LOGIC ──

    // 7a: Is this a supported crop?
    // Calibrated 0.78 → 0.90 (calibrate_thresholds.py, 2314 val samples).
    final effectiveCropTotalThreshold = _effectiveCropTotalThreshold(bestCrop);

    if (bestCropTotal < effectiveCropTotalThreshold) {
      if (_shouldPreserveFocusCropIdentity(
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        allProbs: allProbs,
      )) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: max(
            bestCropTotal,
            _topClassProbForCrop(bestCrop, allProbs),
          ),
          gate: 'G7a_preserve_focus_crop',
          reasonKey: 'focusCropLowConfidence',
          note:
              'bestCropTotal=${bestCropTotal.toStringAsFixed(3)} threshold=${effectiveCropTotalThreshold.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowConfidence',
      );
      if (rescue != null) return rescue;

      final forcedFallback = _buildForcedBeansPotatoFallback(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        fallbackGate: 'G7a_crop_total',
      );
      if (forcedFallback != null) return forcedFallback;

      final lastChance = _buildLastChanceFocusOverride(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCropTotal,
        fallbackGate: 'G7a_crop_total',
        reasonKey: 'focusCropLowConfidence',
      );
      if (lastChance != null) return lastChance;

      _logDecision(
        gate: 'G7a_crop_total',
        resultType: 'unsupported',
        allProbs: allProbs,
        cropName: _formatCropName(bestCrop),
        confidence: bestCropTotal,
        note:
            'bestCropTotal=${bestCropTotal.toStringAsFixed(3)} threshold=${effectiveCropTotalThreshold.toStringAsFixed(3)} crop=$bestCrop',
      );

      return ScanResult(
        imagePath: imagePath,
        cropName: 'unknown',
        diseaseName: 'unsupportedCrop',
        confidence: bestCropTotal,
        resultType: 'unsupported',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7a-ii: other_leaf ratio guard ──────────────────────────────
    // Even a high crop_total is suspicious if other_leaf holds a meaningful
    // share relative to the winner. Threshold tightened 0.25 → 0.18.
    final otherLeafVsCropRatio = bestCropTotal > 0
        ? otherLeafProbScaled / bestCropTotal
        : 0.0;
    if (otherLeafVsCropRatio > _otherLeafVsCropRatioThreshold) {
      if (_shouldPreserveFocusCropIdentity(
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        allProbs: allProbs,
      )) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: max(
            bestCropTotal,
            _topClassProbForCrop(bestCrop, allProbs),
          ),
          gate: 'G7a_ratio_preserve_focus_crop',
          reasonKey: 'focusCropRetake',
          note:
              'ratio=${otherLeafVsCropRatio.toStringAsFixed(3)} threshold=${_otherLeafVsCropRatioThreshold.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescuePossibleCropRetake',
      );
      if (rescue != null) return rescue;

      final forcedFallback = _buildForcedBeansPotatoFallback(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        fallbackGate: 'G7a_other_leaf_ratio',
      );
      if (forcedFallback != null) return forcedFallback;

      final lastChance = _buildLastChanceFocusOverride(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCropTotal: bestCropTotal,
        fallbackGate: 'G7a_other_leaf_ratio',
        reasonKey: 'focusCropPriority',
      );
      if (lastChance != null) return lastChance;

      _logDecision(
        gate: 'G7a_other_leaf_ratio',
        resultType: 'other_leaf',
        allProbs: allProbs,
        cropName: _formatCropName(bestCrop),
        confidence: otherLeafProb,
        note:
            'ratio=${otherLeafVsCropRatio.toStringAsFixed(3)} threshold=${_otherLeafVsCropRatioThreshold.toStringAsFixed(3)}',
      );

      return ScanResult(
        imagePath: imagePath,
        cropName: 'unknown',
        diseaseName: 'unsupportedCrop',
        confidence: otherLeafProb,
        resultType: 'other_leaf',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7b: Crop identified — find best class within the crop.
    final cropClasses = _getClassesForCrop(bestCrop, probabilities);
    final sortedClasses = cropClasses.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final bestClass = sortedClasses[0].key;
    final bestClassProb = sortedClasses[0].value;
    final cropDisplayName = _formatCropName(bestCrop);

    // 7c: Are two crops too close to each other? (gap check)
    // Policy A (field safety): keep strict uncertain for mixed-scene ambiguity.
    final effectiveGapThreshold = _effectiveUncertainGapThreshold(bestCrop);

    if (cropGap < effectiveGapThreshold) {
      if (_isBeansOrPotato(bestCrop)) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: bestClassProb,
          gate: 'G7c_crop_gap_focus_override',
          reasonKey: 'focusCropPriority',
          note:
              'gap=${cropGap.toStringAsFixed(3)} threshold=${effectiveGapThreshold.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      _logDecision(
        gate: 'G7c_crop_gap',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'gap=${cropGap.toStringAsFixed(3)} threshold=${effectiveGapThreshold.toStringAsFixed(3)} crop=$bestCrop',
      );
      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'uncertain',
        confidence: bestClassProb,
        resultType: 'uncertain',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7c-v2: Is the second crop also strong? (multi-crop ambiguity)
    // Policy A (field safety): keep strict uncertain for mixed-scene ambiguity.
    // RELAXED 0.10 → 0.15: a second crop at 9% is clear dominance by the
    // first and should not trigger uncertainty. The old 0.10 threshold was
    // rejecting real supported-crop images where a small amount of probability
    // legitimately leaked into a second crop.
    final effectiveSecondCropThreshold = _effectiveSecondCropAmbiguityThreshold(
      bestCrop,
    );

    if (secondCropTotal > effectiveSecondCropThreshold) {
      if (_isBeansOrPotato(bestCrop)) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: bestClassProb,
          gate: 'G7c_second_crop_focus_override',
          reasonKey: 'focusCropPriority',
          note:
              'secondCrop=${secondCropTotal.toStringAsFixed(3)} threshold=${effectiveSecondCropThreshold.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      _logDecision(
        gate: 'G7c_second_crop',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'secondCrop=${secondCropTotal.toStringAsFixed(3)} threshold=${effectiveSecondCropThreshold.toStringAsFixed(3)} crop=$bestCrop',
      );
      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'uncertain',
        confidence: bestClassProb,
        resultType: 'uncertain',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7c-v3: Is the model internally uncertain? (entropy check)
    final entropy = _calculateEntropy(probabilities);
    final effectiveMaxEntropy = _effectiveMaxEntropyThreshold(bestCrop);
    if (entropy > effectiveMaxEntropy) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowConfidence',
      );
      if (rescue != null) return rescue;

      if (_isBeansOrPotato(bestCrop)) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: bestClassProb,
          gate: 'G7c_entropy_focus_override',
          reasonKey: 'focusCropPriority',
          note:
              'entropy=${entropy.toStringAsFixed(3)} threshold=${effectiveMaxEntropy.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      _logDecision(
        gate: 'G7c_entropy',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'entropy=${entropy.toStringAsFixed(3)} threshold=${effectiveMaxEntropy.toStringAsFixed(3)}',
      );

      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'uncertain',
        confidence: bestClassProb,
        resultType: 'uncertain',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7c-plus: Does the best class strongly dominate within the crop?
    final classTotalRatio = bestClassProb / bestCropTotal;
    final effectiveClassRatioThreshold = _effectiveClassRatioThreshold(
      bestCrop,
    );
    if (classTotalRatio < effectiveClassRatioThreshold) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        cropProbs: cropProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowConfidence',
      );
      if (rescue != null) return rescue;

      if (_isBeansOrPotato(bestCrop)) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: bestClassProb,
          gate: 'G7c_class_ratio_focus_override',
          reasonKey: 'focusCropPriority',
          note:
              'classRatio=${classTotalRatio.toStringAsFixed(3)} threshold=${effectiveClassRatioThreshold.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      _logDecision(
        gate: 'G7c_class_ratio',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'classRatio=${classTotalRatio.toStringAsFixed(3)} threshold=${effectiveClassRatioThreshold.toStringAsFixed(3)}',
      );

      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'uncertain',
        confidence: bestClassProb,
        resultType: 'uncertain',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7d: Is the specific class confident enough?
    final effectiveClassConfidenceThreshold =
        _effectiveClassConfidenceThreshold(bestCrop);
    if (bestClassProb < effectiveClassConfidenceThreshold) {
      if (_isBeansOrPotato(bestCrop)) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: bestClassProb,
          gate: 'G7d_class_confidence_focus_override',
          reasonKey: 'focusCropPriority',
          note:
              'bestClassProb=${bestClassProb.toStringAsFixed(3)} threshold=${effectiveClassConfidenceThreshold.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      final healthyLabel = healthyLabels[bestCrop];
      final healthyProb = cropClasses[healthyLabel] ?? 0.0;

      if (healthyProb > 0 && healthyProb == bestClassProb) {
        _logDecision(
          gate: 'G7d_class_confidence_healthy_candidate',
          resultType: 'uncertain',
          allProbs: allProbs,
          cropName: cropDisplayName,
          confidence: bestClassProb,
          note:
              'bestClassProb=${bestClassProb.toStringAsFixed(3)} threshold=${effectiveClassConfidenceThreshold.toStringAsFixed(3)}',
        );
        return ScanResult(
          imagePath: imagePath,
          cropName: cropDisplayName,
          diseaseName: 'likelyHealthy',
          confidence: bestClassProb,
          resultType: 'uncertain',
          allProbabilities: allProbs,
          dateTime: DateTime.now().toIso8601String(),
        );
      }

      _logDecision(
        gate: 'G7d_class_confidence',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'bestClassProb=${bestClassProb.toStringAsFixed(3)} threshold=${effectiveClassConfidenceThreshold.toStringAsFixed(3)}',
      );

      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'unidentifiedCondition',
        confidence: bestClassProb,
        resultType: 'uncertain',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7e: SAFETY GATE — healthy predictions need 80% minimum confidence.
    final isHealthy = bestClass.contains('healthy');
    final healthyConfidenceThreshold = bestCrop == 'potato'
        ? potatoHealthyMinConfidencePilot
        : healthyMinConfidence;

    if (isHealthy && bestClassProb < healthyConfidenceThreshold) {
      if (_isBeansOrPotato(bestCrop)) {
        return _buildFocusCropUncertainResult(
          imagePath: imagePath,
          allProbs: allProbs,
          crop: bestCrop,
          confidence: bestClassProb,
          gate: 'G7e_healthy_safety_focus_override',
          reasonKey: 'focusCropPriority',
          note:
              'bestClassProb=${bestClassProb.toStringAsFixed(3)} threshold=${healthyConfidenceThreshold.toStringAsFixed(3)} crop=$bestCrop',
        );
      }

      _logDecision(
        gate: 'G7e_healthy_safety',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'bestClassProb=${bestClassProb.toStringAsFixed(3)} threshold=${healthyConfidenceThreshold.toStringAsFixed(3)}',
      );
      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'likelyHealthyVerify',
        confidence: bestClassProb,
        resultType: 'uncertain',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7f: Confident prediction — check disease database.
    final diseaseKey = bestClass;
    final diseaseExists = DiseaseInfo.all.containsKey(diseaseKey);
    final diseaseName = diseaseKey;

    if (!isHealthy && !diseaseExists) {
      _logDecision(
        gate: 'G7f_unknown_disease',
        resultType: 'unknown_disease',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note: 'label=$bestClass',
      );
      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: diseaseName,
        confidence: bestClassProb,
        resultType: 'unknown_disease',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 7g: Everything checks out — confident result.
    _logDecision(
      gate: 'G7g_confident',
      resultType: isHealthy ? 'healthy' : 'disease',
      allProbs: allProbs,
      cropName: cropDisplayName,
      confidence: bestClassProb,
      note: 'label=$bestClass',
    );

    return ScanResult(
      imagePath: imagePath,
      cropName: cropDisplayName,
      diseaseName: diseaseName,
      confidence: bestClassProb,
      resultType: isHealthy ? 'healthy' : 'disease',
      allProbabilities: allProbs,
      dateTime: DateTime.now().toIso8601String(),
    );
  }

  bool _isBeansOrPotato(String crop) => crop == 'beans' || crop == 'potato';

  // FIX G2: Include maize as a focus crop for identity preservation.
  // Maize gets lighter protection than beans/potato — only preserve-identity,
  // not the forced-fallback or last-chance rescue paths.
  bool _isFocusCrop(String crop) =>
      crop == 'beans' || crop == 'potato' || crop == 'maize';

  double _effectiveCropTotalThreshold(String crop) {
    if (_isBeansOrPotato(crop)) {
      return max(0.0, _cropTotalThreshold - _beansPotatoCropTotalRelaxation);
    }
    return max(0.0, _cropTotalThreshold - nonFocusCropTotalRelaxation);
  }

  double _effectiveOtherLeafAbsoluteFloor(String crop) {
    if (_isBeansOrPotato(crop)) {
      return min(
        0.95,
        _otherLeafAbsoluteFloor + _beansPotatoOtherLeafFloorBoost,
      );
    }
    return min(0.95, _otherLeafAbsoluteFloor + nonFocusOtherLeafFloorBoost);
  }

  double _effectiveUncertainGapThreshold(String crop) {
    if (_isBeansOrPotato(crop)) {
      return _beansPotatoUncertainGapThreshold;
    }
    return uncertainGapThreshold;
  }

  double _effectiveSecondCropAmbiguityThreshold(String crop) {
    if (_isBeansOrPotato(crop)) {
      return _beansPotatoSecondCropThreshold;
    }
    return secondCropAmbiguityThreshold;
  }

  double _effectiveClassRatioThreshold(String crop) {
    if (_isBeansOrPotato(crop)) {
      return _beansPotatoClassRatioThreshold;
    }
    return _nonFocusClassRatioThreshold;
  }

  double _effectiveClassConfidenceThreshold(String crop) {
    if (_isBeansOrPotato(crop)) {
      return _beansPotatoClassConfidenceThreshold;
    }
    return _nonFocusClassConfidenceThreshold;
  }

  double _effectiveMaxEntropyThreshold(String crop) {
    if (_isBeansOrPotato(crop)) {
      return maxEntropyThreshold;
    }
    return _nonFocusMaxEntropyThreshold;
  }

  bool _shouldPreserveFocusCropIdentity({
    required String candidateCrop,
    required double candidateCropTotal,
    required Map<String, double> allProbs,
  }) {
    // FIX G2: Extend to maize with a slightly higher evidence bar
    // to avoid false-accepting non-maize plants.
    if (_isBeansOrPotato(candidateCrop)) {
      final topClassProb = _topClassProbForCrop(candidateCrop, allProbs);
      return candidateCropTotal >= 0.001 || topClassProb >= 0.001;
    }
    if (candidateCrop == 'maize') {
      final topClassProb = _topClassProbForCrop('maize', allProbs);
      // Require stronger maize signal before preserving identity:
      // at least 10% crop total or 8% top-class probability.
      return candidateCropTotal >= 0.10 || topClassProb >= 0.08;
    }
    return false;
  }

  ScanResult _buildFocusCropUncertainResult({
    required String imagePath,
    required Map<String, double> allProbs,
    required String crop,
    required double confidence,
    required String gate,
    required String reasonKey,
    required String note,
  }) {
    final cropName = _formatCropName(crop);
    final bestClassLabel =
        _topClassLabelForCrop(crop, allProbs) ??
        (crop == 'beans'
            ? 'beans_healthy'
            : crop == 'maize'
                ? 'maize_healthy'
                : 'potato_healthy');
    final bestClassProb = allProbs[bestClassLabel] ?? 0.0;
    final isHealthy = bestClassLabel.contains('healthy');
    final diseaseExists = DiseaseInfo.all.containsKey(bestClassLabel);
    final forcedResultType = isHealthy
        ? 'healthy'
        : (diseaseExists ? 'disease' : 'unknown_disease');
    final forcedConfidence = max(confidence, bestClassProb);

    _logDecision(
      gate: gate,
      resultType: forcedResultType,
      allProbs: allProbs,
      cropName: cropName,
      confidence: forcedConfidence,
      note: '$note reason=$reasonKey forcedClass=$bestClassLabel',
    );

    return ScanResult(
      imagePath: imagePath,
      cropName: cropName,
      diseaseName: bestClassLabel,
      confidence: forcedConfidence,
      resultType: forcedResultType,
      allProbabilities: allProbs,
      dateTime: DateTime.now().toIso8601String(),
    );
  }

  ScanResult? _buildBeansPotatoRescue({
    required String imagePath,
    required Map<String, double> allProbs,
    required Map<String, double> cropProbs,
    required String candidateCrop,
    required double candidateCropTotal,
    required double otherLeafProb,
    required String reasonKey,
  }) {
    // Rescue is crop-aware and limited to beans/potato.
    // If the top crop is not beans/potato, allow an alternative beans/potato
    // candidate only when it is still competitively strong.
    var rescueCrop = candidateCrop;
    var rescueCropTotal = candidateCropTotal;

    if (!_isBeansOrPotato(rescueCrop)) {
      final alternative = _selectBeansPotatoRescueCandidate(
        cropProbs,
        allProbs,
        bestCropTotal: candidateCropTotal,
      );
      if (alternative == null) return null;
      rescueCrop = alternative.key;
      rescueCropTotal = alternative.value;
    }

    // Require a meaningful beans/potato signal before rescue.
    if (rescueCropTotal < _beansPotatoRescueMinCropTotal) return null;

    // Keep unsupported when beans/potato evidence is far behind the winner.
    final gapFromBest = max(0.0, candidateCropTotal - rescueCropTotal);
    if (gapFromBest > _beansPotatoRescueMaxGapFromBest) return null;

    // Keep unsupported when other_leaf is too strong.
    if (otherLeafProb > _beansPotatoRescueMaxOtherLeaf) return null;

    // Rescue-only anti-swap guard: if beans and potato are nearly tied,
    // require a clear top-class margin before forcing one focus crop.
    final beansTotal = cropProbs['beans'] ?? 0.0;
    final potatoTotal = cropProbs['potato'] ?? 0.0;
    final focusGap = (beansTotal - potatoTotal).abs();
    if (focusGap <= rescueFocusSwapGuardCropGap) {
      final beansTopClass = _topClassProbForCrop('beans', allProbs);
      final potatoTopClass = _topClassProbForCrop('potato', allProbs);
      final topMargin = rescueCrop == 'beans'
          ? (beansTopClass - potatoTopClass)
          : (potatoTopClass - beansTopClass);
      if (topMargin < rescueFocusSwapGuardTopClassMargin) return null;
    }

    return _buildFocusCropUncertainResult(
      imagePath: imagePath,
      allProbs: allProbs,
      crop: rescueCrop,
      confidence: rescueCropTotal,
      gate: 'RESCUE_beans_potato',
      reasonKey: reasonKey,
      note:
          'rescueCrop=$rescueCrop rescueCropTotal=${rescueCropTotal.toStringAsFixed(3)} otherLeaf=${otherLeafProb.toStringAsFixed(3)}',
    );
  }

  ScanResult? _buildForcedBeansPotatoFallback({
    required String imagePath,
    required Map<String, double> allProbs,
    required Map<String, double> cropProbs,
    required double candidateCropTotal,
    required double otherLeafProb,
    required String fallbackGate,
  }) {
    if (!forceBeansPotatoNeverUnsupported) return null;

    final beansTotal = cropProbs['beans'] ?? 0.0;
    final potatoTotal = cropProbs['potato'] ?? 0.0;
    final beansTopClass = _topClassProbForCrop('beans', allProbs);
    final potatoTopClass = _topClassProbForCrop('potato', allProbs);

    final beansScore = beansTotal + (0.50 * beansTopClass);
    final potatoScore = potatoTotal + (0.50 * potatoTopClass);

    final candidate = _chooseFocusCropCandidate(
      beansTotal: beansTotal,
      potatoTotal: potatoTotal,
      beansTopClass: beansTopClass,
      potatoTopClass: potatoTopClass,
      beansScore: beansScore,
      potatoScore: potatoScore,
    );
    if (candidate == null) return null;

    final rescueCrop = candidate.key;
    final rescueCropTotal = candidate.value;
    final rescueTopClass = rescueCrop == 'potato'
        ? potatoTopClass
        : beansTopClass;

    final hasMinimumEvidence =
        rescueCropTotal >= forceBeansPotatoMinCropTotal ||
        rescueTopClass >= forceBeansPotatoMinTopClassProb;
    if (!hasMinimumEvidence) return null;

    final gapFromBest = max(0.0, candidateCropTotal - rescueCropTotal);

    return _buildFocusCropUncertainResult(
      imagePath: imagePath,
      allProbs: allProbs,
      crop: rescueCrop,
      confidence: max(rescueCropTotal, rescueTopClass),
      gate: 'FORCED_beans_potato_fallback',
      reasonKey: 'focusCropPriority',
      note:
          'from=$fallbackGate rescueCrop=$rescueCrop rescueCropTotal=${rescueCropTotal.toStringAsFixed(3)} topClass=${rescueTopClass.toStringAsFixed(3)} gap=${gapFromBest.toStringAsFixed(3)} otherLeaf=${otherLeafProb.toStringAsFixed(3)}',
    );
  }

  ScanResult? _buildLastChanceFocusOverride({
    required String imagePath,
    required Map<String, double> allProbs,
    required Map<String, double> cropProbs,
    required double candidateCropTotal,
    required String fallbackGate,
    required String reasonKey,
  }) {
    final beansTotal = cropProbs['beans'] ?? 0.0;
    final potatoTotal = cropProbs['potato'] ?? 0.0;
    final beansTopClass = _topClassProbForCrop('beans', allProbs);
    final potatoTopClass = _topClassProbForCrop('potato', allProbs);

    if (beansTotal < lastChanceFocusMinCropTotal &&
        potatoTotal < lastChanceFocusMinCropTotal &&
        beansTopClass < lastChanceFocusMinTopClassProb &&
        potatoTopClass < lastChanceFocusMinTopClassProb) {
      return null;
    }

    final candidate = _chooseFocusCropCandidate(
      beansTotal: beansTotal,
      potatoTotal: potatoTotal,
      beansTopClass: beansTopClass,
      potatoTopClass: potatoTopClass,
      beansScore: beansTotal + (0.50 * beansTopClass),
      potatoScore: potatoTotal + (0.50 * potatoTopClass),
    );
    if (candidate == null) return null;

    final rescueCrop = candidate.key;
    final rescueTotal = candidate.value;
    final rescueTop = rescueCrop == 'potato' ? potatoTopClass : beansTopClass;
    final gapFromBest = max(0.0, candidateCropTotal - rescueTotal);

    return _buildFocusCropUncertainResult(
      imagePath: imagePath,
      allProbs: allProbs,
      crop: rescueCrop,
      confidence: max(rescueTotal, rescueTop),
      gate: 'LAST_CHANCE_focus_override',
      reasonKey: reasonKey,
      note:
          'from=$fallbackGate rescueCrop=$rescueCrop rescueCropTotal=${rescueTotal.toStringAsFixed(3)} topClass=${rescueTop.toStringAsFixed(3)} gap=${gapFromBest.toStringAsFixed(3)}',
    );
  }

  MapEntry<String, double>? _selectBeansPotatoRescueCandidate(
    Map<String, double> cropProbs,
    Map<String, double> allProbs, {
    required double bestCropTotal,
  }) {
    final beansTotal = cropProbs['beans'] ?? 0.0;
    final potatoTotal = cropProbs['potato'] ?? 0.0;
    final beansTopClass = _topClassProbForCrop('beans', allProbs);
    final potatoTopClass = _topClassProbForCrop('potato', allProbs);

    final beansScore = beansTotal + (0.35 * beansTopClass);
    final potatoScore = potatoTotal + (0.35 * potatoTopClass);

    final candidate = _chooseFocusCropCandidate(
      beansTotal: beansTotal,
      potatoTotal: potatoTotal,
      beansTopClass: beansTopClass,
      potatoTopClass: potatoTopClass,
      beansScore: beansScore,
      potatoScore: potatoScore,
    );

    if (candidate == null) return null;

    if (candidate.value <= 0.0) return null;
    final gapFromBest = max(0.0, bestCropTotal - candidate.value);
    if (gapFromBest > _beansPotatoRescueMaxGapFromBest) return null;
    return candidate;
  }

  MapEntry<String, double>? _selectPreferredBeansPotatoCrop(
    Map<String, double> cropProbs,
    Map<String, double> allProbs, {
    required String currentBestCrop,
    required double currentBestCropTotal,
  }) {
    if (!preferBeansPotatoWhenCompetitive) return null;
    if (_isBeansOrPotato(currentBestCrop)) return null;

    final beansTotal = cropProbs['beans'] ?? 0.0;
    final potatoTotal = cropProbs['potato'] ?? 0.0;
    final beansTopClass = _topClassProbForCrop('beans', allProbs);
    final potatoTopClass = _topClassProbForCrop('potato', allProbs);

    final beansScore = beansTotal + (0.45 * beansTopClass);
    final potatoScore = potatoTotal + (0.45 * potatoTopClass);

    final candidate = _chooseFocusCropCandidate(
      beansTotal: beansTotal,
      potatoTotal: potatoTotal,
      beansTopClass: beansTopClass,
      potatoTopClass: potatoTopClass,
      beansScore: beansScore,
      potatoScore: potatoScore,
    );

    if (candidate == null) return null;

    final gapFromBest = max(0.0, currentBestCropTotal - candidate.value);
    if (candidate.value < focusCropSwitchMinTotal) return null;
    if (gapFromBest > focusCropSwitchMaxGap) return null;
    return candidate;
  }

  MapEntry<String, double>? _selectSecondaryFocusCandidate(
    Map<String, double> cropProbs,
    Map<String, double> allProbs, {
    required double bestCropTotal,
  }) {
    final beansTotal = cropProbs['beans'] ?? 0.0;
    final potatoTotal = cropProbs['potato'] ?? 0.0;
    final beansTopClass = _topClassProbForCrop('beans', allProbs);
    final potatoTopClass = _topClassProbForCrop('potato', allProbs);

    final hasEvidence =
        beansTotal >= secondaryFocusMinCropTotal ||
        potatoTotal >= secondaryFocusMinCropTotal ||
        beansTopClass >= secondaryFocusMinTopClassProb ||
        potatoTopClass >= secondaryFocusMinTopClassProb;
    if (!hasEvidence) return null;

    final candidate = _chooseFocusCropCandidate(
      beansTotal: beansTotal,
      potatoTotal: potatoTotal,
      beansTopClass: beansTopClass,
      potatoTopClass: potatoTopClass,
      beansScore: beansTotal + (0.40 * beansTopClass),
      potatoScore: potatoTotal + (0.40 * potatoTopClass),
    );
    if (candidate == null) return null;

    final gapFromBest = max(0.0, bestCropTotal - candidate.value);
    if (gapFromBest > secondaryFocusMaxGapFromBest) return null;
    return candidate;
  }

  double _secondBestCropTotalExcluding(
    Map<String, double> cropProbs,
    String excludedCrop,
  ) {
    final candidates =
        cropProbs.entries.where((e) => e.key != excludedCrop).toList()
          ..sort((a, b) => b.value.compareTo(a.value));
    if (candidates.isEmpty) return 0.0;
    return candidates[0].value;
  }

  MapEntry<String, double>? _chooseFocusCropCandidate({
    required double beansTotal,
    required double potatoTotal,
    required double beansTopClass,
    required double potatoTopClass,
    required double beansScore,
    required double potatoScore,
  }) {
    if (potatoScore > beansScore) {
      return MapEntry('potato', potatoTotal);
    }
    if (beansScore > potatoScore) {
      return MapEntry('beans', beansTotal);
    }

    if (potatoTotal > beansTotal) {
      return MapEntry('potato', potatoTotal);
    }
    if (beansTotal > potatoTotal) {
      return MapEntry('beans', beansTotal);
    }

    if (potatoTopClass > beansTopClass) {
      return MapEntry('potato', potatoTotal);
    }
    if (beansTopClass > potatoTopClass) {
      return MapEntry('beans', beansTotal);
    }

    // True tie: avoid arbitrary beans bias.
    return null;
  }

  double _topClassProbForCrop(String crop, Map<String, double> allProbs) {
    double best = 0.0;
    allProbs.forEach((label, prob) {
      if (cropGrouping[label] == crop && prob > best) {
        best = prob;
      }
    });
    return best;
  }

  String? _topClassLabelForCrop(String crop, Map<String, double> allProbs) {
    String? bestLabel;
    double bestProb = -1.0;
    allProbs.forEach((label, prob) {
      if (cropGrouping[label] == crop && prob > bestProb) {
        bestLabel = label;
        bestProb = prob;
      }
    });
    return bestLabel;
  }

  String _formatCropName(String crop) {
    return crop[0].toUpperCase() + crop.substring(1);
  }

  void _logDecision({
    required String gate,
    required String resultType,
    required Map<String, double> allProbs,
    required String cropName,
    required double confidence,
    String? note,
  }) {
    if (!enableDecisionLogging) return;

    final top = allProbs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final top3 = top
        .take(3)
        .map((e) => '${e.key}:${e.value.toStringAsFixed(3)}')
        .join(', ');

    debugPrint(
      '[Classifier][$gate] result=$resultType crop=$cropName conf=${confidence.toStringAsFixed(3)} top3=[$top3] note=${note ?? '-'}',
    );
  }

  void dispose() {
    _interpreter?.close();
    _isInitialized = false;
  }
}

class ImageQualityResult {
  final bool isAcceptable;
  final double meanBrightness;
  final double stdDev;
  final double greenRatio;
  final List<String> issues;

  ImageQualityResult({
    required this.isAcceptable,
    required this.meanBrightness,
    required this.stdDev,
    required this.greenRatio,
    required this.issues,
  });
}
