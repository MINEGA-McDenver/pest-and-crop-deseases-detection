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
      'allProbabilities': allProbabilities.entries
          .map((e) => '${e.key}:${e.value}')
          .join(','),
      'dateTime': dateTime,
      'isSaved': isSaved ? 1 : 0,
    };
  }

  factory ScanResult.fromMap(Map<String, dynamic> map) {
    Map<String, double> probs = {};
    if (map['allProbabilities'] != null &&
        map['allProbabilities'].toString().isNotEmpty) {
      for (var entry in map['allProbabilities'].toString().split(',')) {
        final parts = entry.split(':');
        if (parts.length == 2) {
          probs[parts[0]] = double.tryParse(parts[1]) ?? 0.0;
        }
      }
    }
    return ScanResult(
      id: map['id'] as int?,
      imagePath: map['imagePath'] as String,
      cropName: map['cropName'] as String,
      diseaseName: map['diseaseName'] as String,
      confidence: (map['confidence'] as num).toDouble(),
      resultType: map['resultType'] as String,
      allProbabilities: probs,
      dateTime: map['dateTime'] as String,
      isSaved: map['isSaved'] == 1,
    );
  }
}
