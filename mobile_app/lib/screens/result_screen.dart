import 'dart:io';
import 'package:flutter/material.dart';
import '../models/scan_result.dart';
import '../services/classifier_service.dart';
import '../services/storage_service.dart';
import '../data/disease_info.dart';
import '../widgets/confidence_bar.dart';

class ResultScreen extends StatelessWidget {
  final ScanResult scanResult;
  final ClassificationResult classificationResult;

  const ResultScreen({
    super.key,
    required this.scanResult,
    required this.classificationResult,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Result'),
        backgroundColor: _getAppBarColor(),
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            _buildResultHeader(context),
            Padding(
              padding: const EdgeInsets.all(16),
              child: _buildResultBody(context),
            ),
          ],
        ),
      ),
      bottomNavigationBar: _buildBottomActions(context),
    );
  }

  Color _getAppBarColor() {
    switch (scanResult.resultType) {
      case 'disease':
        return Colors.orange.shade700;
      case 'healthy':
        return Colors.green.shade700;
      case 'uncertain':
        return Colors.amber.shade700;
      case 'unsupported':
        return Colors.blueGrey.shade600;
      case 'unclear_image':
        return Colors.grey.shade600;
      default:
        return Colors.green.shade700;
    }
  }

  Widget _buildResultHeader(BuildContext context) {
    return Container(
      width: double.infinity,
      color: _getAppBarColor(),
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
      child: Column(
        children: [
          // Image preview
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.file(
              File(scanResult.imagePath),
              height: 200,
              width: double.infinity,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => Container(
                height: 200,
                color: Colors.grey.shade300,
                child: const Icon(Icons.broken_image, size: 64),
              ),
            ),
          ),
          const SizedBox(height: 16),
          // Status icon and text
          _buildStatusBadge(),
        ],
      ),
    );
  }

  Widget _buildStatusBadge() {
    IconData icon;
    String title;

    switch (scanResult.resultType) {
      case 'disease':
        icon = Icons.warning_amber_rounded;
        title = 'Disease Detected';
        break;
      case 'healthy':
        icon = Icons.check_circle;
        title = 'Healthy Crop';
        break;
      case 'uncertain':
        icon = Icons.help_outline;
        title = 'Possible Health Issue';
        break;
      case 'unsupported':
        icon = Icons.block;
        title = 'Crop Not Supported';
        break;
      case 'unclear_image':
        icon = Icons.photo_camera;
        title = 'Image Not Clear';
        break;
      default:
        icon = Icons.info;
        title = 'Result';
    }

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(icon, color: Colors.white, size: 28),
        const SizedBox(width: 10),
        Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildResultBody(BuildContext context) {
    switch (scanResult.resultType) {
      case 'disease':
        return _buildDiseaseResult();
      case 'healthy':
        return _buildHealthyResult();
      case 'uncertain':
        return _buildUncertainResult();
      case 'unsupported':
        return _buildUnsupportedResult();
      case 'unclear_image':
        return _buildUnclearImageResult();
      default:
        return const Text('Unknown result type');
    }
  }

  // ── DISEASE DETECTED ─────────────────────────────────────
  Widget _buildDiseaseResult() {
    final info = DiseaseInfo.all[scanResult.className];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Crop and disease info
        _buildInfoCard(
          children: [
            _buildInfoRow('Crop', scanResult.cropName),
            const SizedBox(height: 8),
            _buildInfoRow(
              'Condition',
              info?.displayName ?? scanResult.diseaseName,
            ),
            const SizedBox(height: 12),
            ConfidenceBar(confidence: scanResult.confidence),
            if (info != null && info.severity == 'critical') ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.shade300),
                ),
                child: Row(
                  children: [
                    Icon(Icons.error, color: Colors.red.shade700),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'CRITICAL — Act immediately!',
                        style: TextStyle(
                          color: Colors.red.shade800,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
        const SizedBox(height: 16),

        // What is this?
        if (info != null) ...[
          _buildSectionCard(
            icon: Icons.info_outline,
            title: 'What is this?',
            color: Colors.blue,
            child: Text(
              info.description,
              style: const TextStyle(fontSize: 14, height: 1.5),
            ),
          ),
          const SizedBox(height: 12),

          // What to do
          _buildSectionCard(
            icon: Icons.medical_services,
            title: 'What to do',
            color: Colors.orange,
            child: Column(
              children: info.whatToDo
                  .map((tip) => _buildBulletItem(tip))
                  .toList(),
            ),
          ),
          const SizedBox(height: 12),

          // Prevention
          _buildSectionCard(
            icon: Icons.shield,
            title: 'Prevention',
            color: Colors.green,
            child: Column(
              children: info.prevention
                  .map((tip) => _buildBulletItem(tip))
                  .toList(),
            ),
          ),
          const SizedBox(height: 12),
        ],

        // Professional advice
        _buildProfessionalAdvice(),
      ],
    );
  }

  // ── HEALTHY CROP ─────────────────────────────────────────
  Widget _buildHealthyResult() {
    final info = DiseaseInfo.all[scanResult.className];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildInfoCard(
          children: [
            _buildInfoRow('Crop', scanResult.cropName),
            _buildInfoRow('Condition', 'Healthy'),
            const SizedBox(height: 12),
            ConfidenceBar(confidence: scanResult.confidence),
          ],
        ),
        const SizedBox(height: 16),

        // Healthy message
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.green.shade50,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.green.shade200),
          ),
          child: Column(
            children: [
              Icon(Icons.eco, color: Colors.green.shade600, size: 40),
              const SizedBox(height: 8),
              Text(
                'Your ${scanResult.cropName.toLowerCase()} plant looks healthy!',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.green.shade800,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // Keep it healthy tips
        if (info != null)
          _buildSectionCard(
            icon: Icons.tips_and_updates,
            title: 'Keep it Healthy',
            color: Colors.green,
            child: Column(
              children: info.whatToDo
                  .map((tip) => _buildBulletItem(tip))
                  .toList(),
            ),
          ),
        const SizedBox(height: 12),

        // Regular check-up advice
        _buildSectionCard(
          icon: Icons.calendar_month,
          title: 'Regular Check-ups',
          color: Colors.blue,
          child: Text(
            DiseaseInfo.healthyCheckup,
            style: const TextStyle(fontSize: 14, height: 1.5),
          ),
        ),
        const SizedBox(height: 12),

        _buildProfessionalAdvice(),
      ],
    );
  }

  // ── UNCERTAIN (Possible Health Issue) ────────────────────
  Widget _buildUncertainResult() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.amber.shade50,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.amber.shade300),
          ),
          child: Column(
            children: [
              Icon(Icons.warning_amber, color: Colors.amber.shade700, size: 40),
              const SizedBox(height: 12),
              Text(
                'The system detected that your crop may not be healthy, '
                'but could not identify the specific disease with enough confidence.',
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.amber.shade900,
                  height: 1.5,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              Text(
                'This could mean the disease or pest affecting your crop '
                'is not yet in our database.',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.amber.shade800,
                  height: 1.4,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        _buildSectionCard(
          icon: Icons.medical_services,
          title: 'What To Do',
          color: Colors.red,
          child: Column(
            children: [
              _buildBulletItem(
                'Do NOT ignore this — your crop may need urgent care',
              ),
              _buildBulletItem(
                'Contact your nearest agricultural extension officer as soon as possible',
              ),
              _buildBulletItem(
                'Take the affected leaf or plant sample to a local agricultural center',
              ),
              _buildBulletItem(
                'Take clear photos of the symptoms from different angles to show the expert',
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),

        _buildFindHelpCard(),
      ],
    );
  }

  // ── UNSUPPORTED CROP ─────────────────────────────────────
  Widget _buildUnsupportedResult() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.blueGrey.shade50,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.blueGrey.shade200),
          ),
          child: Column(
            children: [
              Icon(Icons.block, color: Colors.blueGrey.shade600, size: 40),
              const SizedBox(height: 12),
              Text(
                'This crop is not currently supported by the system.',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.blueGrey.shade800,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // Supported crops list
        _buildSectionCard(
          icon: Icons.eco,
          title: 'Supported Crops',
          color: Colors.green,
          child: Column(
            children: [
              _buildCropItem(
                '🍌',
                'Banana',
                'Cordana, Pestalotiopsis, Sigatoka',
              ),
              _buildCropItem('🫘', 'Beans', 'Angular Leaf Spot, Rust'),
              _buildCropItem(
                '🌽',
                'Maize',
                'Common Rust, Gray Leaf Spot, Northern Leaf Blight',
              ),
              _buildCropItem('🥔', 'Potato', 'Early Blight, Late Blight'),
            ],
          ),
        ),
        const SizedBox(height: 12),

        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.grey.shade100,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            'If you are scanning one of these crops and got this message, '
            'please try again with a clearer photo of a single leaf.',
            style: TextStyle(
              fontSize: 13,
              color: Colors.grey.shade700,
              height: 1.4,
            ),
          ),
        ),
        const SizedBox(height: 12),

        _buildFindHelpCard(),
      ],
    );
  }

  // ── UNCLEAR IMAGE ────────────────────────────────────────
  Widget _buildUnclearImageResult() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.grey.shade100,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.grey.shade300),
          ),
          child: Column(
            children: [
              Icon(Icons.photo_camera, color: Colors.grey.shade600, size: 40),
              const SizedBox(height: 12),
              Text(
                'The image quality is too low for accurate analysis.',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.grey.shade800,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        _buildSectionCard(
          icon: Icons.tips_and_updates,
          title: 'Tips for a Better Photo',
          color: Colors.blue,
          child: Column(
            children: [
              _buildBulletItem('Hold the phone steady to avoid blur'),
              _buildBulletItem('Use natural daylight (not direct sunlight)'),
              _buildBulletItem('Focus on a single leaf'),
              _buildBulletItem('Make sure the leaf fills most of the frame'),
              _buildBulletItem('Avoid shadows on the leaf'),
              _buildBulletItem('Clean the camera lens'),
              _buildBulletItem('Keep 15-20 cm distance from the leaf'),
            ],
          ),
        ),
      ],
    );
  }

  // ── SHARED WIDGETS ───────────────────────────────────────

  Widget _buildInfoCard({required List<Widget> children}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade200,
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: children,
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      children: [
        Text(
          '$label: ',
          style: TextStyle(fontSize: 15, color: Colors.grey.shade600),
        ),
        Text(
          value,
          style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  Widget _buildSectionCard({
    required IconData icon,
    required String title,
    required Color color,
    required Widget child,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade200,
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 22),
              const SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }

  Widget _buildBulletItem(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 6),
            child: Icon(Icons.circle, size: 6, color: Colors.grey),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(fontSize: 14, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCropItem(String emoji, String name, String diseases) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(emoji, style: const TextStyle(fontSize: 24)),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
                Text(
                  diseases,
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProfessionalAdvice() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.blue.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.person, color: Colors.blue.shade700, size: 20),
              const SizedBox(width: 8),
              Text(
                'Consult a Professional',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.blue.shade800,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            DiseaseInfo.professionalAdvice,
            style: TextStyle(
              fontSize: 13,
              color: Colors.blue.shade900,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFindHelpCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.red.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.local_hospital, color: Colors.red.shade700, size: 20),
              const SizedBox(width: 8),
              Text(
                'Find Help',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.red.shade800,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          _buildBulletItem(
            'Contact your nearest agricultural extension officer',
          ),
          _buildBulletItem(
            'Visit your local RAB (Rwanda Agriculture Board) center',
          ),
          _buildBulletItem(
            'Bring a sample of the affected plant for examination',
          ),
        ],
      ),
    );
  }

  Widget _buildBottomActions(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade300,
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          if (scanResult.resultType != 'unclear_image')
            Expanded(
              child: OutlinedButton.icon(
                onPressed: () async {
                  if (scanResult.id != null) {
                    final storage = StorageService();
                    await storage.savePermanently(scanResult.id!);
                    if (context.mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Saved to history permanently'),
                        ),
                      );
                    }
                  }
                },
                icon: const Icon(Icons.bookmark_add),
                label: const Text('Save'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: const Color(0xFF2E7D32),
                  side: const BorderSide(color: Color(0xFF2E7D32)),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                ),
              ),
            ),
          if (scanResult.resultType != 'unclear_image')
            const SizedBox(width: 12),
          Expanded(
            child: ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Scan Again'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2E7D32),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
