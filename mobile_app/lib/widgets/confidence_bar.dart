import 'package:flutter/material.dart';
import '../l10n/app_strings.dart';

class ConfidenceBar extends StatelessWidget {
  final double confidence;
  final String? label;

  const ConfidenceBar({super.key, required this.confidence, this.label});

  Color get _color {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.5) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: Text(
                label ?? AppStrings.tr(context, 'confidence'),
                style: Theme.of(context).textTheme.bodySmall,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Text(
              '${(confidence * 100).toStringAsFixed(1)}%',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: confidence,
            backgroundColor: _color.withValues(alpha: 0.15),
            valueColor: AlwaysStoppedAnimation<Color>(_color),
            minHeight: 8,
          ),
        ),
      ],
    );
  }
}
