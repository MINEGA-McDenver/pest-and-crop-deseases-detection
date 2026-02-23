import 'package:flutter/material.dart';
import '../data/disease_info.dart';

class DiseaseInfoScreen extends StatelessWidget {
  const DiseaseInfoScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final crops = ['Banana', 'Beans', 'Maize', 'Potato'];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Disease Guide'),
        backgroundColor: const Color(0xFF2E7D32),
        foregroundColor: Colors.white,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Learn about crop diseases we can detect',
            style: TextStyle(fontSize: 14, color: Colors.grey.shade600),
          ),
          const SizedBox(height: 16),
          ...crops.map((crop) => _buildCropSection(context, crop)),
        ],
      ),
    );
  }

  Widget _buildCropSection(BuildContext context, String crop) {
    final diseases = DiseaseInfo.getForCrop(crop);
    final emoji = _getCropEmoji(crop);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: _getCropColor(crop).withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Row(
            children: [
              Text(emoji, style: const TextStyle(fontSize: 28)),
              const SizedBox(width: 10),
              Text(
                crop,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: _getCropColor(crop),
                ),
              ),
              const Spacer(),
              Text(
                '${diseases.length} conditions',
                style: TextStyle(fontSize: 13, color: Colors.grey.shade600),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        ...diseases.map((d) => _buildDiseaseCard(context, d)),
        const SizedBox(height: 20),
      ],
    );
  }

  Widget _buildDiseaseCard(BuildContext context, DiseaseInfo disease) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ExpansionTile(
        leading: _getSeverityIcon(disease),
        title: Text(
          disease.displayName,
          style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
        ),
        subtitle: disease.isHealthy
            ? null
            : Text(
                'Severity: ${disease.severity}',
                style: TextStyle(
                  fontSize: 12,
                  color: _getSeverityColor(disease.severity),
                ),
              ),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  disease.description,
                  style: const TextStyle(fontSize: 14, height: 1.5),
                ),
                const SizedBox(height: 12),
                _buildSubSection(
                  disease.isHealthy ? 'Maintenance Tips' : 'What to do',
                  disease.whatToDo,
                  disease.isHealthy ? Colors.green : Colors.orange,
                ),
                const SizedBox(height: 8),
                _buildSubSection('Prevention', disease.prevention, Colors.blue),
                const SizedBox(height: 10),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.blue.shade50,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    DiseaseInfo.professionalAdvice,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.blue.shade800,
                      height: 1.3,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSubSection(String title, List<String> items, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: color,
            fontSize: 14,
          ),
        ),
        const SizedBox(height: 4),
        ...items.map(
          (item) => Padding(
            padding: const EdgeInsets.only(bottom: 3),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Icon(
                    Icons.circle,
                    size: 5,
                    color: Colors.grey.shade500,
                  ),
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    item,
                    style: const TextStyle(fontSize: 13, height: 1.3),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _getSeverityIcon(DiseaseInfo disease) {
    if (disease.isHealthy) {
      return const Icon(Icons.check_circle, color: Colors.green);
    }
    switch (disease.severity) {
      case 'critical':
        return Icon(Icons.error, color: Colors.red.shade700);
      case 'high':
        return const Icon(Icons.warning, color: Colors.orange);
      case 'medium':
        return Icon(Icons.info, color: Colors.amber.shade700);
      default:
        return const Icon(Icons.info_outline, color: Colors.blue);
    }
  }

  Color _getSeverityColor(String severity) {
    switch (severity) {
      case 'critical':
        return Colors.red.shade700;
      case 'high':
        return Colors.orange;
      case 'medium':
        return Colors.amber.shade700;
      default:
        return Colors.green;
    }
  }

  String _getCropEmoji(String crop) {
    switch (crop) {
      case 'Banana':
        return '🍌';
      case 'Beans':
        return '🫘';
      case 'Maize':
        return '🌽';
      case 'Potato':
        return '🥔';
      default:
        return '🌱';
    }
  }

  Color _getCropColor(String crop) {
    switch (crop) {
      case 'Banana':
        return Colors.amber.shade800;
      case 'Beans':
        return Colors.green.shade800;
      case 'Maize':
        return Colors.orange.shade800;
      case 'Potato':
        return Colors.brown.shade700;
      default:
        return Colors.green;
    }
  }
}
