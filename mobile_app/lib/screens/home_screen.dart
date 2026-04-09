import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:io';
import '../services/classifier_service.dart';
import '../services/storage_service.dart';
import '../l10n/app_strings.dart';
import 'result_screen.dart';
import 'history_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ClassifierService _classifier = ClassifierService();
  final StorageService _storage = StorageService();
  final ImagePicker _picker = ImagePicker();
  bool _isInitialized = false;
  bool _isProcessing = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeClassifier();
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
      return 'Model files are mismatched. Reinstall the app package from the latest pilot build.';
    }
    if (normalized.contains('asset') || normalized.contains('unable to load')) {
      return 'Model assets are missing in this build. Please reinstall the app.';
    }
    if (normalized.contains('memory') || normalized.contains('oom')) {
      return 'Phone memory is low. Close other apps and retry.';
    }
    return 'Failed to load AI model. Please retry or reinstall the latest build.';
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
              content: Text(AppStrings.tr(context, 'galleryPermissionRequired')),
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
          SnackBar(
            content: Text(AppStrings.tr(context, 'imageUnavailable')),
          ),
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
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(
          SnackBar(
            content: Text('${AppStrings.tr(context, 'scanError')}: $e'),
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  void _showCropInfo(String crop) {
    final details = _cropGuidance[crop];
    if (details == null) return;

    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 28),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  details['title']!,
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 16),
                Text(
                  'How to keep it healthy',
                  style: Theme.of(
                    context,
                  ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 6),
                Text(details['healthy']!),
                const SizedBox(height: 14),
                Text(
                  'How to take a good diagnosis photo',
                  style: Theme.of(
                    context,
                  ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 6),
                Text(details['photo']!),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _cropInfoButton(String cropName) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton(
        onPressed: () => _showCropInfo(cropName),
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: Text(
          cropName,
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
      ),
    );
  }

  static const Map<String, Map<String, String>> _cropGuidance = {
    'Banana': {
      'title': 'Banana crop guidance',
      'healthy':
          'Use clean planting material, remove heavily infected leaves early, keep good spacing for airflow, and avoid wetting leaves late in the day.',
      'photo':
          'Take one clear leaf, fill most of the frame, stay 20-30 cm away, use daylight if possible, and avoid blur or shadows.',
    },
    'Beans': {
      'title': 'Beans crop guidance',
      'healthy':
          'Rotate crops, remove crop residue after harvest, avoid overhead irrigation when possible, and monitor leaves regularly for early spots.',
      'photo':
          'Photograph a single affected leaf front-on, include both healthy and diseased parts, keep background simple, and focus before capture.',
    },
    'Maize': {
      'title': 'Maize crop guidance',
      'healthy':
          'Plant at recommended spacing, manage weeds early, use balanced fertilizer, and scout often so leaf diseases are treated early.',
      'photo':
          'Capture one representative leaf with visible lesions, avoid backlight, keep the leaf centered, and ensure the image is sharp.',
    },
    'Potato': {
      'title': 'Potato crop guidance',
      'healthy':
          'Use healthy seed tubers, avoid prolonged leaf wetness, improve airflow, and remove severely infected leaves to reduce spread.',
      'photo':
          'Take a close photo of one leaf with clear symptom edges, keep 20-30 cm distance, avoid mixed crops in frame, and use natural light.',
    },
  };

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('🌿 ${AppStrings.tr(context, 'appTitle')}'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const HistoryScreen()),
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
                        Expanded(child: _cropInfoButton('Banana')),
                        const SizedBox(width: 8),
                        Expanded(child: _cropInfoButton('Beans')),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(child: _cropInfoButton('Maize')),
                        const SizedBox(width: 8),
                        Expanded(child: _cropInfoButton('Potato')),
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
