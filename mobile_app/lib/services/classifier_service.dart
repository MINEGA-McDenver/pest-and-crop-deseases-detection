import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class ClassificationResult {
  final String className;
  final String cropName;
  final String diseaseName;
  final double confidence;
  final String resultType;
  final List<MapEntry<String, double>> topPredictions;

  ClassificationResult({
    required this.className,
    required this.cropName,
    required this.diseaseName,
    required this.confidence,
    required this.resultType,
    required this.topPredictions,
  });
}

class ClassifierService {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isLoaded = false;

  static const double _confidenceThreshold = 0.70;
  static const double _blurThreshold = 15.0;
  static const double _darkThreshold = 40.0;
  static const double _brightThreshold = 230.0;

  bool get isLoaded => _isLoaded;
  List<String> get labels => _labels;

  Future<void> initialize() async {
    if (_isLoaded) return;

    try {
      final labelData = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelData
          .split('\n')
          .map((l) => l.trim())
          .where((l) => l.isNotEmpty)
          .toList();

      _interpreter = await Interpreter.fromAsset(
        'models/crop_disease_model.tflite',
      );

      _isLoaded = true;
    } catch (e) {
      throw Exception('Failed to load model: $e');
    }
  }

  Future<ClassificationResult> classify(File imageFile) async {
    if (!_isLoaded) await initialize();

    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) {
      return _unclearImageResult();
    }

    final qualityIssue = _checkImageQuality(image);
    if (qualityIssue != null) {
      return _unclearImageResult();
    }

    final resized = img.copyResize(image, width: 224, height: 224);

    final input = Float32List(1 * 224 * 224 * 3);
    int pixelIndex = 0;
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        input[pixelIndex++] = (pixel.r.toDouble() / 127.5) - 1.0;
        input[pixelIndex++] = (pixel.g.toDouble() / 127.5) - 1.0;
        input[pixelIndex++] = (pixel.b.toDouble() / 127.5) - 1.0;
      }
    }

    final inputTensor = input.reshape([1, 224, 224, 3]);
    final output = List.filled(
      1 * _labels.length,
      0.0,
    ).reshape([1, _labels.length]);
    _interpreter!.run(inputTensor, output);

    final probabilities = List<double>.from(output[0]);

    final predictions = <MapEntry<String, double>>[];
    for (int i = 0; i < probabilities.length; i++) {
      predictions.add(MapEntry(_labels[i], probabilities[i]));
    }
    predictions.sort((a, b) => b.value.compareTo(a.value));

    final topClass = predictions[0].key;
    final topConfidence = predictions[0].value;
    final topCrop = _extractCropName(topClass);
    final topDisease = _extractDiseaseName(topClass);
    //final top3 = predictions.take(3).toList();

    if (topConfidence >= _confidenceThreshold) {
      final isHealthy = topClass.contains('healthy');
      return ClassificationResult(
        className: topClass,
        cropName: topCrop,
        diseaseName: topDisease,
        confidence: topConfidence,
        resultType: isHealthy ? 'healthy' : 'disease',
        topPredictions: predictions.take(5).toList(),
      );
    }

    return _analyzeLowConfidence(predictions);
  }

  ClassificationResult _analyzeLowConfidence(
    List<MapEntry<String, double>> predictions,
  ) {
    final top3 = predictions.take(3).toList();
    final top3Crops = top3.map((p) => _extractCropName(p.key)).toSet();

    if (top3Crops.length == 1) {
      final crop = top3Crops.first;
      return ClassificationResult(
        className: top3[0].key,
        cropName: crop,
        diseaseName: 'Unknown Issue',
        confidence: top3[0].value,
        resultType: 'uncertain',
        topPredictions: predictions.take(5).toList(),
      );
    }

    return ClassificationResult(
      className: 'unsupported',
      cropName: 'Unknown',
      diseaseName: 'Not Supported',
      confidence: predictions[0].value,
      resultType: 'unsupported',
      topPredictions: predictions.take(5).toList(),
    );
  }

  String? _checkImageQuality(img.Image image) {
    final small = img.copyResize(image, width: 100, height: 100);

    double totalBrightness = 0;
    int count = 0;
    for (int y = 0; y < small.height; y++) {
      for (int x = 0; x < small.width; x++) {
        final pixel = small.getPixel(x, y);
        totalBrightness += (pixel.r + pixel.g + pixel.b) / 3.0;
        count++;
      }
    }
    final avgBrightness = totalBrightness / count;
    if (avgBrightness < _darkThreshold) return 'too_dark';
    if (avgBrightness > _brightThreshold) return 'too_bright';

    final grayscale = img.grayscale(small);
    double laplacianSum = 0;
    int lapCount = 0;
    for (int y = 1; y < grayscale.height - 1; y++) {
      for (int x = 1; x < grayscale.width - 1; x++) {
        final center = grayscale.getPixel(x, y).r.toDouble();
        final laplacian =
            grayscale.getPixel(x, y - 1).r.toDouble() +
            grayscale.getPixel(x, y + 1).r.toDouble() +
            grayscale.getPixel(x - 1, y).r.toDouble() +
            grayscale.getPixel(x + 1, y).r.toDouble() -
            4 * center;
        laplacianSum += laplacian * laplacian;
        lapCount++;
      }
    }
    if (laplacianSum / lapCount < _blurThreshold) return 'blurry';

    return null;
  }

  ClassificationResult _unclearImageResult() {
    return ClassificationResult(
      className: 'unclear',
      cropName: 'Unknown',
      diseaseName: 'Unclear Image',
      confidence: 0.0,
      resultType: 'unclear_image',
      topPredictions: [],
    );
  }

  String _extractCropName(String className) {
    final parts = className.split('_');
    return parts[0][0].toUpperCase() + parts[0].substring(1);
  }

  String _extractDiseaseName(String className) {
    final parts = className.split('_');
    if (parts.length <= 1) return className;
    final disease = parts.sublist(1).join(' ');
    return disease[0].toUpperCase() + disease.substring(1);
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isLoaded = false;
  }
}
