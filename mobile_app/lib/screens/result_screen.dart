import 'dart:io';
import 'package:flutter/material.dart';
import '../models/scan_result.dart';
import '../data/disease_info.dart';
import '../widgets/confidence_bar.dart';
import '../services/storage_service.dart';
import '../l10n/app_strings.dart';

class ResultScreen extends StatefulWidget {
  final ScanResult result;

  const ResultScreen({super.key, required this.result});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  final StorageService _storage = StorageService();
  bool _saved = false;

  static const Set<String> _diagnosisKeys = {
    'uncertain',
    'poorImageQuality',
    'unsupportedCrop',
    'classificationError',
    'likelyHealthy',
    'unidentifiedCondition',
    'likelyHealthyVerify',
    'rescueLikelyCropLowLight',
    'rescuePossibleCropRetake',
    'rescueLikelyCropLowConfidence',
  };

  static const Set<String> _diagnosisKeysWithCropArg = {
    'rescueLikelyCropLowLight',
    'rescuePossibleCropRetake',
    'rescueLikelyCropLowConfidence',
  };

  String _localizeStoredDiagnosis(String stored, {String? cropName}) {
    final normalized = stored.trim();
    if (normalized.isEmpty) return stored;

    if (_diagnosisKeysWithCropArg.contains(normalized)) {
      final crop = cropName ?? widget.result.cropName;
      return AppStrings.tr(context, normalized, args: {'crop': crop});
    }

    if (_diagnosisKeys.contains(normalized)) {
      return AppStrings.tr(context, normalized);
    }

    final byClass = DiseaseInfo.all[normalized.toLowerCase()];
    if (byClass != null) {
      return DiseaseInfo.localizeDiseaseName(
        byClass.displayName,
        classKey: byClass.className,
      );
    }

    return DiseaseInfo.localizeDiseaseName(normalized);
  }

  String _qualityIssueText(String issue) {
    final localized = AppStrings.tr(context, issue);
    return localized == issue ? issue : localized;
  }

  String _resolvedCurrentDiseaseLabel() {
    final info = DiseaseInfo.resolveByCropAndDiseaseName(
      widget.result.cropName,
      widget.result.diseaseName,
    );
    if (info != null) return info.displayName;
    return _localizeStoredDiagnosis(widget.result.diseaseName);
  }

  String _predictionLabel(String classKey) {
    if (classKey == 'other_leaf') {
      return AppStrings.tr(context, 'unsupportedCrop');
    }

    final info = DiseaseInfo.all[classKey];
    if (info != null) {
      final diseaseLabel = info.isHealthy
          ? AppStrings.tr(context, 'healthy')
          : DiseaseInfo.localizeDiseaseName(
              info.displayName,
              classKey: info.className,
            );
      return '${info.cropName} - $diseaseLabel';
    }

    return AppStrings.tr(context, 'unknownCondition');
  }

  @override
  void initState() {
    super.initState();
    _saved = widget.result.isSaved;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(AppStrings.tr(context, 'scanResult')),
        backgroundColor: _getAppBarColor(),
        foregroundColor: Colors.white,
        actions: [
          if (widget.result.resultType != 'poor_quality')
            IconButton(
              icon: Icon(_saved ? Icons.bookmark : Icons.bookmark_border),
              onPressed: _saveResult,
            ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [_buildImageSection(), _buildResultContent()],
        ),
      ),
    );
  }

  Color _getAppBarColor() {
    switch (widget.result.resultType) {
      case 'healthy':
        return Colors.green;
      case 'disease':
        return Colors.orange.shade800;
      case 'uncertain':
        return Colors.amber.shade700;
      case 'unknown_disease':
        return Colors.deepOrange;
      case 'unsupported':
        return Colors.blueGrey;
      case 'poor_quality':
        return Colors.red.shade700;
      default:
        return const Color(0xFF2E7D32);
    }
  }

  Widget _buildImageSection() {
    return Stack(
      children: [
        SizedBox(
          height: 250,
          width: double.infinity,
          child: Image.file(
            File(widget.result.imagePath),
            fit: BoxFit.cover,
            errorBuilder: (_, __, ___) => Container(
              color: Colors.grey.shade200,
              child: const Icon(Icons.broken_image, size: 64),
            ),
          ),
        ),
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: [
                  Colors.black.withValues(alpha: 0.7),
                  Colors.transparent,
                ],
              ),
            ),
            padding: const EdgeInsets.all(16),
            child: Text(
              _getHeaderText(),
              style: const TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
      ],
    );
  }

  String _getHeaderText() {
    final localizedDisease = _resolvedCurrentDiseaseLabel();

    switch (widget.result.resultType) {
      case 'poor_quality':
        return '⚠️ ${AppStrings.tr(context, 'poorQualityHeader')}';
      case 'unsupported':
        return '❓ ${AppStrings.tr(context, 'unsupportedHeader')}';
      case 'uncertain':
        return '🤔 ${AppStrings.tr(context, 'uncertainHeader')}';
      case 'unknown_disease':
        return '🔬 ${AppStrings.tr(context, 'unknownConditionHeader', args: {'crop': widget.result.cropName})}';
      case 'healthy':
        return '✅ ${AppStrings.tr(context, 'healthyHeader', args: {'crop': widget.result.cropName})}';
      case 'disease':
        return '⚠️ ${AppStrings.tr(context, 'diseaseHeader', args: {'crop': widget.result.cropName, 'disease': localizedDisease})}';
      default:
        return widget.result.cropName;
    }
  }

  Widget _buildResultContent() {
    switch (widget.result.resultType) {
      case 'poor_quality':
        return _buildPoorQualityCard();
      case 'unsupported':
        return _buildUnsupportedCard();
      case 'uncertain':
        return _buildUncertainCard();
      case 'unknown_disease':
        return _buildUnknownDiseaseCard();
      default:
        return _buildConfidentResult();
    }
  }

  // ─── POOR QUALITY ──────────────────────────────────────────
  Widget _buildPoorQualityCard() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Card(
            color: Colors.red.shade50,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Icon(
                    Icons.photo_camera,
                    size: 64,
                    color: Colors.red.shade300,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    AppStrings.tr(context, 'imageQualityTooLow'),
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    AppStrings.tr(context, 'imageNotReliable'),
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 14, color: Colors.black87),
                  ),
                  const SizedBox(height: 16),
                  if (widget.result.qualityIssues != null)
                    ...widget.result.qualityIssues!.map(
                      (issue) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(
                              Icons.error_outline,
                              size: 18,
                              color: Colors.red.shade600,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _qualityIssueText(issue),
                                style: TextStyle(color: Colors.red.shade800),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _buildPhotoTipsCard(),
          const SizedBox(height: 16),
          _buildRetakeButton(),
        ],
      ),
    );
  }

  // ─── UNSUPPORTED CROP ──────────────────────────────────────
  Widget _buildUnsupportedCard() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Card(
            color: Colors.blueGrey.shade50,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Icon(
                    Icons.help_outline,
                    size: 64,
                    color: Colors.blueGrey.shade300,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    AppStrings.tr(context, 'unsupportedHeader'),
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    AppStrings.tr(context, 'unsupportedIntro'),
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 14, color: Colors.black87),
                  ),
                  const SizedBox(height: 16),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    alignment: WrapAlignment.center,
                    children: [
                      _buildCropChip('🍌', AppStrings.tr(context, 'banana')),
                      _buildCropChip('🫘', AppStrings.tr(context, 'beans')),
                      _buildCropChip('🌽', AppStrings.tr(context, 'maize')),
                      _buildCropChip('🥔', AppStrings.tr(context, 'potato')),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    AppStrings.tr(context, 'unsupportedRetryHint'),
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.grey.shade600,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _buildPhotoTipsCard(),
          const SizedBox(height: 16),
          _buildRetakeButton(),
        ],
      ),
    );
  }

  // ─── UNKNOWN DISEASE (crop identified, disease not supported) ──
  Widget _buildUnknownDiseaseCard() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Main alert card
          Card(
            color: Colors.deepOrange.shade50,
            elevation: 3,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Icon(
                    Icons.warning_rounded,
                    size: 64,
                    color: Colors.deepOrange.shade400,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    AppStrings.tr(
                      context,
                      'cropDetected',
                      args: {'crop': widget.result.cropName},
                    ),
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  ConfidenceBar(
                    confidence: widget.result.confidence,
                    label: AppStrings.tr(context, 'cropDetectionConfidence'),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.deepOrange.shade100,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.deepOrange.shade300),
                    ),
                    child: Column(
                      children: [
                        const Icon(
                          Icons.medical_services,
                          size: 32,
                          color: Colors.deepOrange,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          AppStrings.tr(
                            context,
                            'yourCropUnhealthy',
                            args: {
                              'crop': widget.result.cropName.toLowerCase(),
                            },
                          ),
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.deepOrange,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          AppStrings.tr(
                            context,
                            'unknownDiseaseMessage',
                            args: {
                              'crop': widget.result.cropName.toLowerCase(),
                            },
                          ),
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.deepOrange.shade900,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 12),

          // Supported diseases for this crop
          _buildSupportedDiseasesCard(),

          const SizedBox(height: 12),

          // Urgent action card
          Card(
            color: Colors.red.shade50,
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.emergency, color: Colors.red.shade700),
                      const SizedBox(width: 8),
                      Text(
                        AppStrings.tr(context, 'whatYouShouldDo'),
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.red.shade800,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _buildActionItem(
                    Icons.person_search,
                    AppStrings.tr(context, 'contactProfessionalNow'),
                    AppStrings.tr(context, 'contactProfessionalDesc'),
                  ),
                  _buildActionItem(
                    Icons.timer,
                    AppStrings.tr(context, 'actQuickly'),
                    AppStrings.tr(context, 'actQuicklyDesc'),
                  ),
                  _buildActionItem(
                    Icons.camera_alt,
                    AppStrings.tr(context, 'takeMultiplePhotos'),
                    AppStrings.tr(context, 'takeMultiplePhotosDesc'),
                  ),
                  _buildActionItem(
                    Icons.note_alt,
                    AppStrings.tr(context, 'noteSymptoms'),
                    AppStrings.tr(context, 'noteSymptomsDesc'),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 12),

          // Contact info
          Card(
            color: Colors.blue.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Icon(Icons.contact_phone, color: Colors.blue.shade700),
                  const SizedBox(height: 8),
                  Text(
                    AppStrings.tr(context, 'contactAgriPros'),
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    AppStrings.tr(context, 'contactAgriList'),
                    style: TextStyle(fontSize: 13, height: 1.6),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),
          _buildRetakeButton(),
          const SizedBox(height: 8),
          Text(
            AppStrings.tr(context, 'professionalAdvice'),
            style: TextStyle(
              fontSize: 11,
              color: Colors.grey.shade500,
              fontStyle: FontStyle.italic,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildSupportedDiseasesCard() {
    final cropName = widget.result.cropName.toLowerCase();
    final diseases = DiseaseInfo.all.values
        .where((d) => d.cropName.toLowerCase() == cropName && !d.isHealthy)
        .toList();

    if (diseases.isEmpty) return const SizedBox.shrink();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              AppStrings.tr(
                context,
                'detectableDiseasesForCrop',
                args: {'crop': widget.result.cropName},
              ),
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            ...diseases.map(
              (d) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Row(
                  children: [
                    Icon(
                      Icons.check_circle,
                      size: 16,
                      color: Colors.green.shade600,
                    ),
                    const SizedBox(width: 8),
                    Text(d.displayName, style: const TextStyle(fontSize: 13)),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              AppStrings.tr(context, 'conditionNotInList'),
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey.shade600,
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionItem(IconData icon, String title, String subtitle) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 22, color: Colors.red.shade600),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ─── UNCERTAIN ─────────────────────────────────────────────
  Widget _buildUncertainCard() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Card(
            color: Colors.amber.shade50,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Icon(
                    Icons.warning_amber_rounded,
                    size: 48,
                    color: Colors.amber.shade700,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    AppStrings.tr(context, 'uncertainHeader'),
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    AppStrings.tr(
                      context,
                      'uncertainBody',
                      args: {
                        'crop': widget.result.cropName,
                        'disease': _resolvedCurrentDiseaseLabel(),
                        'confidence': (widget.result.confidence * 100)
                            .toStringAsFixed(1),
                      },
                    ),
                    textAlign: TextAlign.center,
                    style: const TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 12),
                  ConfidenceBar(
                    confidence: widget.result.confidence,
                    label: AppStrings.tr(context, 'confidence'),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          _buildPhotoTipsCard(),
          const SizedBox(height: 16),
          _buildRetakeButton(),
          const SizedBox(height: 8),
          Text(
            AppStrings.tr(context, 'professionalAdvice'),
            style: TextStyle(
              fontSize: 11,
              color: Colors.grey.shade500,
              fontStyle: FontStyle.italic,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  // ─── CONFIDENT RESULT (healthy or known disease) ───────────
  Widget _buildConfidentResult() {
    final localizedDisease = _resolvedCurrentDiseaseLabel();
    final diseaseInfo = DiseaseInfo.resolveByCropAndDiseaseName(
      widget.result.cropName,
      widget.result.diseaseName,
    );

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Main result card
          Card(
            color: widget.result.resultType == 'healthy'
                ? Colors.green.shade50
                : Colors.orange.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Text(
                    widget.result.resultType == 'healthy' ? '✅' : '⚠️',
                    style: const TextStyle(fontSize: 40),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    widget.result.resultType == 'healthy'
                        ? AppStrings.tr(
                            context,
                            'healthyHeader',
                            args: {'crop': widget.result.cropName},
                          )
                        : '${widget.result.cropName} - $localizedDisease',
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 12),
                  ConfidenceBar(
                    confidence: widget.result.confidence,
                    label: AppStrings.tr(context, 'detectionConfidence'),
                  ),
                ],
              ),
            ),
          ),

          if (diseaseInfo != null) ...[
            const SizedBox(height: 12),
            _buildDiseaseInfoCard(diseaseInfo),
          ],

          const SizedBox(height: 12),
          _buildTopPredictions(),

          const SizedBox(height: 16),
          Text(
            AppStrings.tr(context, 'professionalAdvice'),
            style: TextStyle(
              fontSize: 11,
              color: Colors.grey.shade500,
              fontStyle: FontStyle.italic,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildDiseaseInfoCard(DiseaseInfo info) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              info.description,
              style: const TextStyle(fontSize: 14, height: 1.5),
            ),
            const Divider(height: 24),
            Text(
              '🛠️ ${AppStrings.tr(context, 'whatToDo')}',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            ...info.whatToDo.map(
              (tip) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('• ', style: TextStyle(fontSize: 14)),
                    Expanded(
                      child: Text(tip, style: const TextStyle(fontSize: 13)),
                    ),
                  ],
                ),
              ),
            ),
            if (!info.isHealthy) ...[
              const Divider(height: 24),
              Text(
                '🛡️ ${AppStrings.tr(context, 'prevention')}',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ...info.prevention.map(
                (tip) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('• ', style: TextStyle(fontSize: 14)),
                      Expanded(
                        child: Text(tip, style: const TextStyle(fontSize: 13)),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildTopPredictions() {
    final sorted = widget.result.allProbabilities.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final top5 = sorted.take(5).toList();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              AppStrings.tr(context, 'topPredictions'),
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            ...top5.map(
              (entry) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: ConfidenceBar(
                  confidence: entry.value,
                  label: _predictionLabel(entry.key),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─── SHARED WIDGETS ────────────────────────────────────────
  Widget _buildPhotoTipsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.lightbulb_outline, color: Colors.amber.shade700),
                const SizedBox(width: 8),
                Text(
                  AppStrings.tr(context, 'tipsHeader'),
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _buildTip(
              Icons.wb_sunny_outlined,
              AppStrings.tr(context, 'tipDaylight'),
            ),
            _buildTip(
              Icons.center_focus_strong,
              AppStrings.tr(context, 'tipSingleLeaf'),
            ),
            _buildTip(Icons.straighten, AppStrings.tr(context, 'tipDistance')),
            _buildTip(Icons.blur_off, AppStrings.tr(context, 'tipSteady')),
            _buildTip(Icons.contrast, AppStrings.tr(context, 'tipBackground')),
            _buildTip(
              Icons.crop_original,
              AppStrings.tr(context, 'tipAffected'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTip(IconData icon, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(icon, size: 18, color: const Color(0xFF2E7D32)),
          const SizedBox(width: 10),
          Expanded(child: Text(text, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }

  Widget _buildCropChip(String emoji, String name) {
    return Chip(
      avatar: Text(emoji, style: const TextStyle(fontSize: 18)),
      label: Text(name),
      backgroundColor: Colors.green.shade50,
    );
  }

  Widget _buildRetakeButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: () => Navigator.pop(context),
        icon: const Icon(Icons.camera_alt),
        label: Text(AppStrings.tr(context, 'takeNewPhoto')),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF2E7D32),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 14),
        ),
      ),
    );
  }

  Future<void> _saveResult() async {
    if (widget.result.id != null) {
      await _storage.savePermanently(widget.result.id!);
    } else {
      await _storage.saveScan(widget.result);
    }
    setState(() => _saved = true);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(AppStrings.tr(context, 'resultSaved'))),
      );
    }
  }
}
