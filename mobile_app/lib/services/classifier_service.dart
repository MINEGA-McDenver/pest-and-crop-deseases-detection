import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import '../models/scan_result.dart';
import '../data/disease_info.dart';

class ClassifierService {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isInitialized = false;

  static const int inputSize = 224;
  static const double temperatureScaling = 1.8;

  // Thresholds
  static const double confidentClassThreshold = 0.45;
  static const double cropTotalThreshold = 0.55;
  static const double uncertainGapThreshold = 0.10;
  static const double imageQualityMinStdDev = 15.0;

  // Crop grouping: maps each label to its crop
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

  // Known healthy labels per crop
  static const Map<String, String> healthyLabels = {
    'banana': 'banana_healthy',
    'beans': 'beans_healthy',
    'maize': 'maize_healthy',
    'potato': 'potato_healthy',
  };

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      _interpreter = await Interpreter.fromAsset(
        'models/crop_disease_model.tflite',
      );

      final labelData = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelData
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();

      _isInitialized = true;
    } catch (e) {
      throw Exception('Failed to initialize classifier: $e');
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
      issues.add('Image is too dark. Move to a well-lit area or use flash.');
      isAcceptable = false;
    } else if (meanBrightness > 240) {
      issues.add(
        'Image is too bright/overexposed. Avoid direct sunlight on the lens.',
      );
      isAcceptable = false;
    }

    if (stdDev < imageQualityMinStdDev) {
      issues.add(
        'Image lacks detail. Move closer to the leaf and ensure focus.',
      );
      isAcceptable = false;
    }

    if (greenRatio < 0.05) {
      issues.add('No plant leaf detected. Please photograph a leaf directly.');
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
        input[pixelIndex++] = (pixel.r / 127.5) - 1.0;
        input[pixelIndex++] = (pixel.g / 127.5) - 1.0;
        input[pixelIndex++] = (pixel.b / 127.5) - 1.0;
      }
    }
    return input;
  }

  // ─── Softmax with Temperature ───────────────────────────────
  List<double> _applySoftmaxWithTemperature(
    List<double> logits,
    double temperature,
  ) {
    final scaled = logits.map((l) => l / temperature).toList();
    final maxVal = scaled.reduce(max);
    final exps = scaled.map((s) => exp(s - maxVal)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExps).toList();
  }

  // ─── Aggregate by Crop ──────────────────────────────────────
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

  // ─── Main Classification ───────────────────────────────────
  Future<ScanResult> classifyImage(String imagePath) async {
    if (!_isInitialized) await initialize();

    final imageFile = File(imagePath);
    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Could not decode image');

    // ── Step 1: Check image quality ──
    final quality = _checkImageQuality(image);
    if (!quality.isAcceptable) {
      return ScanResult(
        imagePath: imagePath,
        cropName: 'Unknown',
        diseaseName: 'Poor Image Quality',
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

    // ── Step 5: Aggregate by crop ──
    final cropProbs = _aggregateByCrop(probabilities);

    // Sort crops by total probability
    final sortedCrops = cropProbs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    if (sortedCrops.isEmpty) {
      return ScanResult(
        imagePath: imagePath,
        cropName: 'Unknown',
        diseaseName: 'Classification Error',
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

    // ── Step 6: DECISION LOGIC ──

    // 6a: Is this even a supported crop?
    if (bestCropTotal < cropTotalThreshold) {
      return ScanResult(
        imagePath: imagePath,
        cropName: 'Unknown',
        diseaseName: 'Unsupported Crop',
        confidence: bestCropTotal,
        resultType: 'unsupported',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 6b: Crop identified! Now find the best class within this crop
    final cropClasses = _getClassesForCrop(bestCrop, probabilities);
    final sortedClasses = cropClasses.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final bestClass = sortedClasses[0].key;
    final bestClassProb = sortedClasses[0].value;
    final cropDisplayName = _formatCropName(bestCrop);

    // 6c: Is the crop identification itself uncertain? (two crops too close)
    if (cropGap < uncertainGapThreshold) {
      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'Uncertain',
        confidence: bestClassProb,
        resultType: 'uncertain',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 6d: Is the specific class confident enough?
    if (bestClassProb < confidentClassThreshold) {
      // Crop is identified but we can't determine the exact condition
      // Check if the healthy class is the strongest
      final healthyLabel = healthyLabels[bestCrop];
      final healthyProb = cropClasses[healthyLabel] ?? 0.0;

      if (healthyProb > 0 && healthyProb == bestClassProb) {
        // Healthy is the top class but low confidence
        return ScanResult(
          imagePath: imagePath,
          cropName: cropDisplayName,
          diseaseName: 'Likely Healthy',
          confidence: bestClassProb,
          resultType: 'uncertain',
          allProbabilities: allProbs,
          dateTime: DateTime.now().toIso8601String(),
        );
      }

      // The crop has some condition but confidence is too low to determine what
      // This could be an unsupported disease/pest
      return ScanResult(
        imagePath: imagePath,
        cropName: cropDisplayName,
        diseaseName: 'Unidentified Condition',
        confidence: bestClassProb,
        resultType: 'unknown_disease',
        allProbabilities: allProbs,
        dateTime: DateTime.now().toIso8601String(),
      );
    }

    // 6e: Confident prediction — check if it's in our disease database
    final isHealthy = bestClass.contains('healthy');
    final diseaseName = isHealthy ? 'Healthy' : _formatDiseaseName(bestClass);

    // Verify the disease exists in our database
    final diseaseKey = bestClass;
    final diseaseExists = DiseaseInfo.all.containsKey(diseaseKey);

    if (!isHealthy && !diseaseExists) {
      // Crop identified, disease detected but NOT in our database
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

    // 6f: Everything checks out — confident result
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
