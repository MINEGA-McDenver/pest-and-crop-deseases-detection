import 'dart:io';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/scan_result.dart';
import '../services/storage_service.dart';
import '../l10n/app_strings.dart';
import '../data/disease_info.dart';
import 'result_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen>
    with SingleTickerProviderStateMixin {
  final StorageService _storage = StorageService();
  static const bool _enableLocaleDebugLogs = true;
  late TabController _tabController;
  List<ScanResult> _recentScans = [];
  List<ScanResult> _savedScans = [];
  bool _isLoading = true;
  String? _loadError;
  String? _lastLocaleCode;

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

  void _debugLocaleState(String stage, {ScanResult? scan}) {
    if (!_enableLocaleDebugLogs) return;
    final localeCode = AppStrings.localeCodeOf(context);
    debugPrint(
      '[LocaleDebug][History][$stage] locale=$localeCode active=${AppStrings.activeLanguageCode} crop=${scan?.cropName ?? '-'} disease=${scan?.diseaseName ?? '-'}',
    );
  }

  String _displayCropName(String cropName) {
    final cropKey = DiseaseInfo.canonicalCropKey(cropName);
    switch (cropKey) {
      case 'banana':
        return AppStrings.tr(context, 'banana');
      case 'beans':
        return AppStrings.tr(context, 'beans');
      case 'maize':
        return AppStrings.tr(context, 'maize');
      case 'potato':
        return AppStrings.tr(context, 'potato');
      case 'unknown':
        return AppStrings.tr(context, 'unknownCrop');
      default:
        return cropName;
    }
  }

  String _localizeStoredDiagnosis(ScanResult scan) {
    final stored = scan.diseaseName.trim();
    if (stored.isEmpty) return scan.diseaseName;

    if (_diagnosisKeysWithCropArg.contains(stored)) {
      return AppStrings.tr(
        context,
        stored,
        args: {'crop': _displayCropName(scan.cropName).toLowerCase()},
      );
    }

    if (_diagnosisKeys.contains(stored)) {
      return AppStrings.tr(context, stored);
    }

    final byClass = DiseaseInfo.all[stored.toLowerCase()];
    if (byClass != null) {
      return DiseaseInfo.localizeDiseaseName(
        byClass.displayName,
        classKey: byClass.className,
      );
    }

    return DiseaseInfo.localizeDiseaseName(scan.diseaseName);
  }

  String _resolvedDiseaseLabel(ScanResult scan) {
    try {
      final info = DiseaseInfo.resolveByCropAndDiseaseName(
        scan.cropName,
        scan.diseaseName,
      );
      if (info != null) return info.displayName;
      return _localizeStoredDiagnosis(scan);
    } catch (_) {
      return scan.diseaseName;
    }
  }

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _loadScans();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final normalized = AppStrings.localeCodeOf(context);
    if (_lastLocaleCode != normalized) {
      _lastLocaleCode = normalized;
      AppStrings.setActiveLanguageCode(normalized);
      _debugLocaleState('didChangeDependencies');
      _loadScans();
    }
  }

  Future<void> _loadScans() async {
    setState(() {
      _isLoading = true;
      _loadError = null;
    });

    try {
      _recentScans = await _storage.getRecentScans();
      _savedScans = await _storage.getSavedScans();
      if (!mounted) return;
      setState(() => _isLoading = false);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _loadError = '${AppStrings.tr(context, 'scanError')}: $e';
      });
    }
  }

  String _formatScanDate(DateTime scannedAt) {
    final localeCode = AppStrings.localeCodeOf(context);
    const pattern = 'MMM d, yyyy - h:mm a';

    try {
      return DateFormat(pattern, localeCode).format(scannedAt);
    } catch (_) {
      return DateFormat(pattern, 'en').format(scannedAt);
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    _debugLocaleState('build');
    return Scaffold(
      appBar: AppBar(
        title: Text(AppStrings.tr(context, 'scanHistory')),
        backgroundColor: const Color(0xFF2E7D32),
        foregroundColor: Colors.white,
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: Colors.white,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.white70,
          tabs: [
            Tab(
              text: AppStrings.tr(
                context,
                'recentTab',
                args: {'count': '${_recentScans.length}'},
              ),
            ),
            Tab(
              text: AppStrings.tr(
                context,
                'savedTab',
                args: {'count': '${_savedScans.length}'},
              ),
            ),
          ],
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _loadError != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red, size: 48),
                    const SizedBox(height: 12),
                    Text(
                      _loadError!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.red),
                    ),
                    const SizedBox(height: 12),
                    ElevatedButton.icon(
                      onPressed: _loadScans,
                      icon: const Icon(Icons.refresh),
                      label: Text(AppStrings.tr(context, 'retry')),
                    ),
                  ],
                ),
              ),
            )
          : TabBarView(
              controller: _tabController,
              children: [
                _buildScanList(_recentScans, isRecent: true),
                _buildScanList(_savedScans, isRecent: false),
              ],
            ),
    );
  }

  Widget _buildScanList(List<ScanResult> scans, {required bool isRecent}) {
    if (scans.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isRecent ? Icons.history : Icons.bookmark_border,
              size: 64,
              color: Colors.grey.shade300,
            ),
            const SizedBox(height: 16),
            Text(
              isRecent
                  ? AppStrings.tr(context, 'noRecentScans')
                  : AppStrings.tr(context, 'noSavedScans'),
              style: TextStyle(fontSize: 18, color: Colors.grey.shade500),
            ),
            const SizedBox(height: 8),
            Text(
              isRecent
                  ? AppStrings.tr(context, 'startScanningHint')
                  : AppStrings.tr(context, 'saveScanHint'),
              style: TextStyle(fontSize: 14, color: Colors.grey.shade400),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadScans,
      child: ListView.builder(
        padding: const EdgeInsets.all(12),
        itemCount: scans.length,
        itemBuilder: (context, index) {
          try {
            return _buildScanCard(scans[index], isRecent: isRecent);
          } catch (_) {
            return Card(
              margin: const EdgeInsets.only(bottom: 10),
              child: ListTile(
                leading: const Icon(Icons.warning_amber_rounded, color: Colors.orange),
                title: Text(AppStrings.tr(context, 'classificationError')),
                subtitle: Text(AppStrings.tr(context, 'imageNotReliable')),
              ),
            );
          }
        },
      ),
    );
  }

  Widget _buildScanCard(ScanResult scan, {required bool isRecent}) {
    _debugLocaleState('buildCard', scan: scan);
    final scannedAt = DateTime.tryParse(scan.dateTime) ?? DateTime.now();
    final dateStr = _formatScanDate(scannedAt);
    final localizedDisease = _resolvedDiseaseLabel(scan);
    final localizedCrop = _displayCropName(scan.cropName);
    final daysOld = DateTime.now().difference(scannedAt).inDays;

    Color statusColor;
    IconData statusIcon;
    switch (scan.resultType) {
      case 'disease':
        statusColor = Colors.orange.shade700;
        statusIcon = Icons.warning_amber_rounded;
        break;
      case 'healthy':
        statusColor = Colors.green;
        statusIcon = Icons.check_circle;
        break;
      case 'uncertain':
        statusColor = Colors.amber.shade700;
        statusIcon = Icons.help_outline;
        break;
      default:
        statusColor = Colors.grey;
        statusIcon = Icons.info;
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () {
          // Navigate to result detail view
          Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => ResultScreen(result: scan)),
          );
        },
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(
                  File(scan.imagePath),
                  width: 64,
                  height: 64,
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) => Container(
                    width: 64,
                    height: 64,
                    color: Colors.grey.shade200,
                    child: const Icon(Icons.broken_image),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(statusIcon, size: 16, color: statusColor),
                        const SizedBox(width: 4),
                        Expanded(
                          child: Text(
                            scan.resultType == 'healthy'
                                ? '$localizedCrop - ${AppStrings.tr(context, 'healthy')}'
                                : '$localizedCrop - $localizedDisease',
                            style: const TextStyle(
                              fontWeight: FontWeight.w600,
                              fontSize: 14,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${(scan.confidence * 100).toStringAsFixed(1)}% ${AppStrings.tr(context, 'confidence').toLowerCase()}',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Row(
                      children: [
                        Text(
                          dateStr,
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.grey.shade500,
                          ),
                        ),
                        if (isRecent && daysOld >= 25)
                          Container(
                            margin: const EdgeInsets.only(left: 6),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 6,
                              vertical: 1,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.red.shade50,
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              AppStrings.tr(
                                context,
                                'expiresInDays',
                                args: {'days': '${30 - daysOld}'},
                              ),
                              style: TextStyle(
                                fontSize: 10,
                                color: Colors.red.shade700,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              Column(
                children: [
                  if (isRecent)
                    IconButton(
                      icon: const Icon(Icons.bookmark_add_outlined),
                      iconSize: 20,
                      color: const Color(0xFF2E7D32),
                      tooltip: AppStrings.tr(context, 'savePermanently'),
                      onPressed: () async {
                        if (scan.id != null) {
                          await _storage.savePermanently(scan.id!);
                          _loadScans();
                          if (mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text(
                                  AppStrings.tr(context, 'savedPermanently'),
                                ),
                              ),
                            );
                          }
                        }
                      },
                    ),
                  IconButton(
                    icon: const Icon(Icons.delete_outline),
                    iconSize: 20,
                    color: Colors.red.shade400,
                    tooltip: AppStrings.tr(context, 'delete'),
                    onPressed: () => _confirmDelete(scan),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _confirmDelete(ScanResult scan) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(AppStrings.tr(context, 'deleteScan')),
        content: Text(AppStrings.tr(context, 'deleteScanConfirm')),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(AppStrings.tr(context, 'cancel')),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              if (scan.id != null) {
                await _storage.deleteScan(scan.id!);
                _loadScans();
              }
            },
            child: Text(
              AppStrings.tr(context, 'delete'),
              style: TextStyle(color: Colors.red.shade600),
            ),
          ),
        ],
      ),
    );
  }
}
