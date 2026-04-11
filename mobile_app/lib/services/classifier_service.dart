import 'dart:io';
import 'dart:typed_data';
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
  static const double temperatureScaling = 1.8;

  // ─── Thresholds ─────────────────────────────────────────────
  static const double confidentClassThreshold = 0.80;

  // Field hotfix: relaxed from 0.90 to reduce false unsupported rejections.
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
  // Field hotfix: relaxed from 0.18 to reduce false unsupported rejections.
  static const double otherLeafVsCropRatioThreshold = 0.24;

  // Absolute other_leaf floor: reject if other_leaf exceeds this value
  // anywhere in the pipeline, even if it did not win the softmax.
  // Field hotfix: aligned with runtime thresholds config.
  static const double defaultOtherLeafAbsoluteFloor = 0.12;

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

  // Lowered 0.50 → 0.30: temperature scaling deflates probabilities so
  // genuine other_leaf images rarely scored above 0.50 after scaling.
  static const double otherLeafThreshold = 0.30;

  double _cropTotalThreshold = defaultCropTotalThreshold;
  double _otherLeafAbsoluteFloor = defaultOtherLeafAbsoluteFloor;

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
      await tempFile.writeAsBytes(
        modelData.buffer.asUint8List(),
        flush: true,
      );
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
      final thresholds = decoded['thresholds'] as Map<String, dynamic>?;
      if (thresholds == null) return;

      final cropTotal = (thresholds['cropTotalThreshold'] as num?)?.toDouble();
      final otherLeafFloor = (thresholds['otherLeafAbsoluteFloor'] as num?)
          ?.toDouble();

      if (cropTotal != null && cropTotal > 0 && cropTotal < 1) {
        _cropTotalThreshold = cropTotal;
      }
      if (otherLeafFloor != null && otherLeafFloor >= 0 && otherLeafFloor < 1) {
        _otherLeafAbsoluteFloor = otherLeafFloor;
      }
    } catch (_) {
      // Use compiled defaults when threshold config is absent or invalid.
      _cropTotalThreshold = defaultCropTotalThreshold;
      _otherLeafAbsoluteFloor = defaultOtherLeafAbsoluteFloor;
    }
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

    // ── Step 3: Apply temperature scaling ──
    final probabilities = _applySoftmaxWithTemperature(
      rawOutputs,
      temperatureScaling,
    );

    // ── Step 4: Build probability map ──
    Map<String, double> allProbs = {};
    for (int i = 0; i < _labels.length && i < probabilities.length; i++) {
      allProbs[_labels[i]] = probabilities[i];
    }

    final otherLeafProb = allProbs['other_leaf'] ?? 0.0;

    final cropProbs = _aggregateByCrop(probabilities);
    final sortedCrops = cropProbs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final bestCandidateCrop = sortedCrops.isNotEmpty
        ? sortedCrops[0].key
        : 'unknown';
    final bestCandidateCropTotal = sortedCrops.isNotEmpty
        ? sortedCrops[0].value
        : 0.0;

    // ── Step 5: Early exit — other_leaf direct softmax winner ──
    // Threshold lowered 0.50 → 0.30 because temperature scaling deflates
    // probabilities; genuine other_leaf images rarely scored above 0.50.
    if (otherLeafProb >= otherLeafThreshold) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        candidateCrop: bestCandidateCrop,
        candidateCropTotal: bestCandidateCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowLight',
      );
      if (rescue != null) return rescue;

      _logDecision(
        gate: 'G5_other_leaf_winner',
        resultType: 'other_leaf',
        allProbs: allProbs,
        cropName: 'unknown',
        confidence: otherLeafProb,
        note: 'other_leaf=${otherLeafProb.toStringAsFixed(3)}',
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
    if (otherLeafProb > _otherLeafAbsoluteFloor) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        candidateCrop: bestCandidateCrop,
        candidateCropTotal: bestCandidateCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescuePossibleCropRetake',
      );
      if (rescue != null) return rescue;

      _logDecision(
        gate: 'G5b_other_leaf_floor',
        resultType: 'other_leaf',
        allProbs: allProbs,
        cropName: 'unknown',
        confidence: otherLeafProb,
        note:
            'other_leaf=${otherLeafProb.toStringAsFixed(3)} floor=${_otherLeafAbsoluteFloor.toStringAsFixed(3)}',
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

    final bestCrop = sortedCrops[0].key;
    final bestCropTotal = sortedCrops[0].value;
    final secondCropTotal = sortedCrops.length > 1 ? sortedCrops[1].value : 0.0;
    final cropGap = bestCropTotal - secondCropTotal;

    // ── Step 7: DECISION LOGIC ──

    // 7a: Is this a supported crop?
    // Calibrated 0.78 → 0.90 (calibrate_thresholds.py, 2314 val samples).
    if (bestCropTotal < _cropTotalThreshold) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowConfidence',
      );
      if (rescue != null) return rescue;

      _logDecision(
        gate: 'G7a_crop_total',
        resultType: 'unsupported',
        allProbs: allProbs,
        cropName: _formatCropName(bestCrop),
        confidence: bestCropTotal,
        note:
            'bestCropTotal=${bestCropTotal.toStringAsFixed(3)} threshold=${_cropTotalThreshold.toStringAsFixed(3)}',
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
        ? otherLeafProb / bestCropTotal
        : 0.0;
    if (otherLeafVsCropRatio > otherLeafVsCropRatioThreshold) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescuePossibleCropRetake',
      );
      if (rescue != null) return rescue;

      _logDecision(
        gate: 'G7a_other_leaf_ratio',
        resultType: 'other_leaf',
        allProbs: allProbs,
        cropName: _formatCropName(bestCrop),
        confidence: otherLeafProb,
        note:
            'ratio=${otherLeafVsCropRatio.toStringAsFixed(3)} threshold=${otherLeafVsCropRatioThreshold.toStringAsFixed(3)}',
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
    if (cropGap < uncertainGapThreshold) {
      _logDecision(
        gate: 'G7c_crop_gap',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'gap=${cropGap.toStringAsFixed(3)} threshold=${uncertainGapThreshold.toStringAsFixed(3)}',
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
    if (secondCropTotal > secondCropAmbiguityThreshold) {
      _logDecision(
        gate: 'G7c_second_crop',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'secondCrop=${secondCropTotal.toStringAsFixed(3)} threshold=${secondCropAmbiguityThreshold.toStringAsFixed(3)}',
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
    if (entropy > maxEntropyThreshold) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowConfidence',
      );
      if (rescue != null) return rescue;

      _logDecision(
        gate: 'G7c_entropy',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'entropy=${entropy.toStringAsFixed(3)} threshold=${maxEntropyThreshold.toStringAsFixed(3)}',
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
    if (classTotalRatio < 0.60) {
      final rescue = _buildBeansPotatoRescue(
        imagePath: imagePath,
        allProbs: allProbs,
        candidateCrop: bestCrop,
        candidateCropTotal: bestCropTotal,
        otherLeafProb: otherLeafProb,
        reasonKey: 'rescueLikelyCropLowConfidence',
      );
      if (rescue != null) return rescue;

      _logDecision(
        gate: 'G7c_class_ratio',
        resultType: 'uncertain',
        allProbs: allProbs,
        cropName: cropDisplayName,
        confidence: bestClassProb,
        note:
            'classRatio=${classTotalRatio.toStringAsFixed(3)} threshold=0.600',
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
    if (bestClassProb < confidentClassThreshold) {
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
              'bestClassProb=${bestClassProb.toStringAsFixed(3)} threshold=${confidentClassThreshold.toStringAsFixed(3)}',
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
            'bestClassProb=${bestClassProb.toStringAsFixed(3)} threshold=${confidentClassThreshold.toStringAsFixed(3)}',
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

  ScanResult? _buildBeansPotatoRescue({
    required String imagePath,
    required Map<String, double> allProbs,
    required String candidateCrop,
    required double candidateCropTotal,
    required double otherLeafProb,
    required String reasonKey,
  }) {
    // Only rescue beans/potato; maize/banana behavior remains unchanged.
    if (!_isBeansOrPotato(candidateCrop)) return null;

    // Require at least a minimal crop signal to avoid turning true unsupported into uncertain.
    if (candidateCropTotal < 0.04) return null;

    // If other_leaf is overwhelmingly dominant, keep unsupported.
    if (otherLeafProb > 0.96) return null;

    _logDecision(
      gate: 'RESCUE_beans_potato',
      resultType: 'uncertain',
      allProbs: allProbs,
      cropName: _formatCropName(candidateCrop),
      confidence: candidateCropTotal,
      note: reasonKey,
    );

    return ScanResult(
      imagePath: imagePath,
      cropName: _formatCropName(candidateCrop),
      diseaseName: reasonKey,
      confidence: candidateCropTotal,
      resultType: 'uncertain',
      allProbabilities: allProbs,
      dateTime: DateTime.now().toIso8601String(),
    );
  }

  String _formatCropName(String crop) {
    return crop[0].toUpperCase() + crop.substring(1);
  }

  String _formatDiseaseName(String label) {
    final parts = label.split('_');
    if (parts.length > 1) {
      return parts
          .sublist(1)
          .map((p) => p[0].toUpperCase() + p.substring(1))
          .join(' ');
    }
    return label;
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
