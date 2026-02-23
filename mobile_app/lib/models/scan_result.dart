/// Represents the result of a crop disease scan.
class ScanResult {
  final int? id;
  final String imagePath;
  final String className;
  final String cropName;
  final String diseaseName;
  final double confidence;
  final String
  resultType; // 'disease', 'healthy', 'uncertain', 'unsupported', 'unclear_image'
  final DateTime scannedAt;
  final bool isSaved; // permanently saved vs auto-delete

  ScanResult({
    this.id,
    required this.imagePath,
    required this.className,
    required this.cropName,
    required this.diseaseName,
    required this.confidence,
    required this.resultType,
    required this.scannedAt,
    this.isSaved = false,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'imagePath': imagePath,
      'className': className,
      'cropName': cropName,
      'diseaseName': diseaseName,
      'confidence': confidence,
      'resultType': resultType,
      'scannedAt': scannedAt.toIso8601String(),
      'isSaved': isSaved ? 1 : 0,
    };
  }

  factory ScanResult.fromMap(Map<String, dynamic> map) {
    return ScanResult(
      id: map['id'] as int?,
      imagePath: map['imagePath'] as String,
      className: map['className'] as String,
      cropName: map['cropName'] as String,
      diseaseName: map['diseaseName'] as String,
      confidence: (map['confidence'] as num).toDouble(),
      resultType: map['resultType'] as String,
      scannedAt: DateTime.parse(map['scannedAt'] as String),
      isSaved: (map['isSaved'] as int) == 1,
    );
  }

  ScanResult copyWith({bool? isSaved}) {
    return ScanResult(
      id: id,
      imagePath: imagePath,
      className: className,
      cropName: cropName,
      diseaseName: diseaseName,
      confidence: confidence,
      resultType: resultType,
      scannedAt: scannedAt,
      isSaved: isSaved ?? this.isSaved,
    );
  }
}
