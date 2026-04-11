import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:io';
import '../services/classifier_service.dart';
import '../services/storage_service.dart';
import '../l10n/app_strings.dart';
import 'result_screen.dart';
import 'history_screen.dart';

enum _SupportedCrop { banana, beans, maize, potato }

class _CropGuidance {
  final String title;
  final String healthy;
  final String photo;

  const _CropGuidance({
    required this.title,
    required this.healthy,
    required this.photo,
  });
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ClassifierService _classifier = ClassifierService();
  final StorageService _storage = StorageService();
  final ImagePicker _picker = ImagePicker();
  static const bool _enableLocaleDebugLogs = true;
  bool _isInitialized = false;
  bool _isProcessing = false;
  String? _errorMessage;
  String? _lastLocaleCode;

  static const List<_SupportedCrop> _supportedCrops = [
    _SupportedCrop.banana,
    _SupportedCrop.beans,
    _SupportedCrop.maize,
    _SupportedCrop.potato,
  ];

  @override
  void initState() {
    super.initState();
    _initializeClassifier();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final normalized = AppStrings.localeCodeOf(context);
    if (_lastLocaleCode != normalized) {
      _lastLocaleCode = normalized;
      AppStrings.setActiveLanguageCode(normalized);
      _debugLocaleState('didChangeDependencies');
    }
  }

  void _debugLocaleState(String stage, {String? cropKey}) {
    if (!_enableLocaleDebugLogs) return;
    final localeCode = AppStrings.localeCodeOf(context);
    debugPrint(
      '[LocaleDebug][Home][$stage] locale=$localeCode active=${AppStrings.activeLanguageCode} crop=${cropKey ?? '-'}',
    );
  }

  Future<void> _initializeClassifier() async {
    try {
      await _classifier.initialize();
      if (mounted) {
        setState(() {
          _isInitialized = true;
          _errorMessage = null;
        });
      }
    } catch (e) {
      debugPrint('Model initialization failed: $e');
      final message = _mapInitializationError(e.toString());
      if (mounted) {
        setState(() {
          _errorMessage = message;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(message),
            duration: const Duration(seconds: 10),
          ),
        );
      }
    }
  }

  String _mapInitializationError(String rawError) {
    final normalized = rawError.toLowerCase();
    if (normalized.contains('labels.txt') ||
        normalized.contains('other_leaf')) {
      return AppStrings.tr(context, 'modelMismatch');
    }
    if (normalized.contains('asset') || normalized.contains('unable to load')) {
      return AppStrings.tr(context, 'modelAssetsMissing');
    }
    if (normalized.contains('memory') || normalized.contains('oom')) {
      return AppStrings.tr(context, 'lowMemory');
    }
    return AppStrings.tr(context, 'modelRetryInstall');
  }

  Future<void> _pickImage(ImageSource source) async {
    if (!_isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(AppStrings.tr(context, 'modelWait'))),
      );
      return;
    }

    if (source == ImageSource.camera) {
      final status = await Permission.camera.request();
      if (!status.isGranted) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(AppStrings.tr(context, 'cameraPermissionRequired')),
            ),
          );
        }
        return;
      }
    } else {
      final photoStatus = await Permission.photos.request();
      if (!photoStatus.isGranted && !photoStatus.isLimited) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                AppStrings.tr(context, 'galleryPermissionRequired'),
              ),
            ),
          );
        }
        return;
      }
    }

    final XFile? image = await _picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 85,
    );

    if (image == null) return;

    final imageFile = File(image.path);
    if (!await imageFile.exists()) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(AppStrings.tr(context, 'imageUnavailable'))),
        );
      }
      return;
    }

    setState(() => _isProcessing = true);

    try {
      // Classify image using the new 6-tier decision logic
      final result = await _classifier.classifyImage(image.path);

      // Save to database
      final storedResult = await _storage.saveScan(result);

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => ResultScreen(result: storedResult)),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('${AppStrings.tr(context, 'scanError')}: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  void _showCropInfo(_SupportedCrop crop) {
    _debugLocaleState('cropTap', cropKey: crop.name);
    final details = _cropGuidanceFor(crop);

    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return SafeArea(
          child: FractionallySizedBox(
            heightFactor: 0.82,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 28),
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Container(
                        width: 42,
                        height: 4,
                        margin: const EdgeInsets.only(bottom: 16),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade400,
                          borderRadius: BorderRadius.circular(999),
                        ),
                      ),
                    ),
                    Text(
                      details.title,
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      AppStrings.tr(context, 'keepHealthy'),
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(details.healthy),
                    const SizedBox(height: 14),
                    Text(
                      AppStrings.tr(context, 'takeGoodPhoto'),
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(details.photo),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  _CropGuidance _cropGuidanceFor(_SupportedCrop crop) {
    switch (crop) {
      case _SupportedCrop.banana:
        return _CropGuidance(
          title: AppStrings.tr(context, 'bananaGuidanceTitle'),
          healthy: AppStrings.tr(context, 'bananaHealthyGuide'),
          photo: AppStrings.tr(context, 'bananaPhotoGuide'),
        );
      case _SupportedCrop.beans:
        return _CropGuidance(
          title: AppStrings.tr(context, 'beansGuidanceTitle'),
          healthy: AppStrings.tr(context, 'beansHealthyGuide'),
          photo: AppStrings.tr(context, 'beansPhotoGuide'),
        );
      case _SupportedCrop.maize:
        return _CropGuidance(
          title: AppStrings.tr(context, 'maizeGuidanceTitle'),
          healthy: AppStrings.tr(context, 'maizeHealthyGuide'),
          photo: AppStrings.tr(context, 'maizePhotoGuide'),
        );
      case _SupportedCrop.potato:
        return _CropGuidance(
          title: AppStrings.tr(context, 'potatoGuidanceTitle'),
          healthy: AppStrings.tr(context, 'potatoHealthyGuide'),
          photo: AppStrings.tr(context, 'potatoPhotoGuide'),
        );
    }
  }

  String _cropLabel(_SupportedCrop crop) {
    switch (crop) {
      case _SupportedCrop.banana:
        return AppStrings.tr(context, 'banana');
      case _SupportedCrop.beans:
        return AppStrings.tr(context, 'beans');
      case _SupportedCrop.maize:
        return AppStrings.tr(context, 'maize');
      case _SupportedCrop.potato:
        return AppStrings.tr(context, 'potato');
    }
  }

  Widget _cropInfoButton({required _SupportedCrop crop}) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton(
        onPressed: () => _showCropInfo(crop),
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: Text(
          _cropLabel(crop),
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    _debugLocaleState('build');
    return Scaffold(
      appBar: AppBar(
        title: Text('🌿 ${AppStrings.tr(context, 'appTitle')}'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
        actions: [
          IconButton(
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const HistoryScreen()),
            ),
            tooltip: AppStrings.tr(context, 'scanHistory'),
            icon: Icon(
              Icons.history,
              color: Theme.of(context).colorScheme.onPrimaryContainer,
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const SizedBox(height: 20),
            Icon(
              Icons.local_florist,
              size: 80,
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(height: 16),
            Text(
              AppStrings.tr(context, 'cropDetectionTitle'),
              style: Theme.of(
                context,
              ).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              AppStrings.tr(context, 'cropDetectionSubtitle'),
              textAlign: TextAlign.center,
              style: Theme.of(
                context,
              ).textTheme.bodyLarge?.copyWith(color: Colors.grey[600]),
            ),
            const SizedBox(height: 40),
            if (_errorMessage != null)
              Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    const Icon(Icons.error, color: Colors.red, size: 48),
                    const SizedBox(height: 16),
                    Text(
                      AppStrings.tr(context, 'modelInitFail'),
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: Colors.red,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _errorMessage!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.red),
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton.icon(
                      onPressed: () {
                        setState(() => _errorMessage = null);
                        _initializeClassifier();
                      },
                      icon: const Icon(Icons.refresh),
                      label: Text(AppStrings.tr(context, 'retry')),
                    ),
                  ],
                ),
              ),
            if (!_isInitialized && _errorMessage == null)
              Padding(
                padding: EdgeInsets.all(20),
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text(AppStrings.tr(context, 'modelLoading')),
                  ],
                ),
              ),
            if (_isProcessing)
              Padding(
                padding: EdgeInsets.all(20),
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text(AppStrings.tr(context, 'analyzing')),
                  ],
                ),
              ),
            if (_isInitialized && !_isProcessing) ...[
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt, size: 28),
                  label: Text(
                    AppStrings.tr(context, 'takePhoto'),
                    style: TextStyle(fontSize: 18),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Theme.of(
                      context,
                    ).colorScheme.primaryContainer,
                    foregroundColor: Theme.of(
                      context,
                    ).colorScheme.onPrimaryContainer,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: OutlinedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library, size: 28),
                  label: Text(
                    AppStrings.tr(context, 'uploadGallery'),
                    style: TextStyle(fontSize: 18),
                  ),
                  style: OutlinedButton.styleFrom(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                ),
              ),
            ],
            const SizedBox(height: 40),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    Text(
                      AppStrings.tr(context, 'supportedCrops'),
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: _cropInfoButton(
                            crop: _supportedCrops[0],
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: _cropInfoButton(
                            crop: _supportedCrops[1],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: _cropInfoButton(
                            crop: _supportedCrops[2],
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: _cropInfoButton(
                            crop: _supportedCrops[3],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
