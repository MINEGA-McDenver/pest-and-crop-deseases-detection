import 'package:flutter/material.dart';

/// A visual confidence bar showing prediction confidence level.
class ConfidenceBar extends StatelessWidget {
  final double confidence;
  final double height;

  const ConfidenceBar({super.key, required this.confidence, this.height = 12});

  Color get _color {
    if (confidence >= 0.9) return Colors.green.shade700;
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.7) return Colors.orange;
    if (confidence >= 0.5) return Colors.orange.shade700;
    return Colors.red;
  }

  String get _label {
    if (confidence >= 0.9) return 'Very High';
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.7) return 'Moderate';
    if (confidence >= 0.5) return 'Low';
    return 'Very Low';
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: _color.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _label,
                style: TextStyle(
                  color: _color,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(height / 2),
          child: LinearProgressIndicator(
            value: confidence,
            minHeight: height,
            backgroundColor: Colors.grey.shade200,
            valueColor: AlwaysStoppedAnimation<Color>(_color),
          ),
        ),
      ],
    );
  }
}
