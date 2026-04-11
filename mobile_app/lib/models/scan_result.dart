import 'dart:convert';

class ScanResult {
  final int? id;
  final String imagePath;
  final String cropName;
  final String diseaseName;
  final double confidence;
  final String resultType;
  // resultType values:
  // 'healthy'              - Supported crop, healthy
  // 'disease'              - Supported crop, known disease detected
  // 'unknown_disease'      - Supported crop identified, but disease not in database
  // 'uncertain'            - Low confidence, retake recommended
  // 'unsupported'          - Not a supported crop
  // 'poor_quality'         - Image too dark/blurry/no leaf detected
  final Map<String, double> allProbabilities;
  final String dateTime;
  final bool isSaved;
  final List<String>? qualityIssues;

  ScanResult({
    this.id,
    required this.imagePath,
    required this.cropName,
    required this.diseaseName,
    required this.confidence,
    required this.resultType,
    required this.allProbabilities,
    required this.dateTime,
    this.isSaved = false,
    this.qualityIssues,
  });

  Map<String, dynamic> toMap() {
    return {
      'imagePath': imagePath,
      'cropName': cropName,
      'diseaseName': diseaseName,
      'confidence': confidence,
      'resultType': resultType,
      'allProbabilities': jsonEncode(allProbabilities),
      'dateTime': dateTime,
      'isSaved': isSaved ? 1 : 0,
    };
  }

  factory ScanResult.fromMap(Map<String, dynamic> map) {
    final probs = _parseProbabilities(map['allProbabilities']);
    final confidenceRaw = map['confidence'];
    final confidence = confidenceRaw is num
        ? confidenceRaw.toDouble()
        : double.tryParse(confidenceRaw?.toString() ?? '') ?? 0.0;

    final savedRaw = map['isSaved'];
    final isSaved = savedRaw == 1 || savedRaw == true || savedRaw == '1';

    return ScanResult(
      id: map['id'] as int?,
      imagePath: (map['imagePath'] as String?) ?? '',
      cropName: (map['cropName'] as String?) ?? 'Unknown',
      diseaseName: (map['diseaseName'] as String?) ?? 'unknownCondition',
      confidence: confidence,
      resultType: (map['resultType'] as String?) ?? 'unsupported',
      allProbabilities: probs,
      dateTime:
          (map['dateTime'] as String?) ?? DateTime.now().toIso8601String(),
      isSaved: isSaved,
    );
  }

  static Map<String, double> _parseProbabilities(dynamic raw) {
    final probs = <String, double>{};
    if (raw == null) return probs;

    final text = raw.toString().trim();
    if (text.isEmpty) return probs;

    // Preferred format: JSON map serialized by jsonEncode.
    try {
      final decoded = jsonDecode(text);
      if (decoded is Map<String, dynamic>) {
        for (final entry in decoded.entries) {
          final value = entry.value;
          if (value is num) {
            probs[entry.key] = value.toDouble();
          } else {
            final parsed = double.tryParse(value.toString());
            if (parsed != null) probs[entry.key] = parsed;
          }
        }
        return probs;
      }
    } catch (_) {
      // Fall back to legacy comma-separated format below.
    }

    // Legacy format: class1:0.9,class2:0.1
    for (final entry in text.split(',')) {
      final parts = entry.split(':');
      if (parts.length == 2) {
        final parsed = double.tryParse(parts[1]);
        if (parsed != null) probs[parts[0]] = parsed;
      }
    }

    return probs;
  }
}
