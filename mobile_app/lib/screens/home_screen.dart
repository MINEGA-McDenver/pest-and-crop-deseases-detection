import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/classifier_service.dart';
import '../services/storage_service.dart';
import '../models/scan_result.dart';
import 'result_screen.dart';
import 'history_screen.dart';
import 'disease_info_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ClassifierService _classifier = ClassifierService();
  final StorageService _storage = StorageService();
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeClassifier();
    _cleanupOldScans();
  }

  Future<void> _initializeClassifier() async {
    try {
      await _classifier.initialize();
      setState(() => _isInitialized = true);
    } catch (e) {
      debugPrint('Model initialization failed: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to load AI model: $e'),
            duration: const Duration(seconds: 10),
          ),
        );
      }
    }
  }

  Future<void> _cleanupOldScans() async {
    final deleted = await _storage.autoDeleteOldScans();
    if (deleted > 0) {
      debugPrint('Auto-deleted $deleted old scans');
    }

    // Check for expiring scans
    final expiring = await _storage.getExpiringScans();
    if (expiring.isNotEmpty && mounted) {
      _showExpiringNotification(expiring);
    }
  }

  void _showExpiringNotification(List<ScanResult> expiring) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '${expiring.length} scan(s) will be deleted soon. '
            'Open History to save them permanently.',
          ),
          duration: const Duration(seconds: 5),
          action: SnackBarAction(
            label: 'View',
            onPressed: () => _openHistory(),
          ),
        ),
      );
    });
  }

  Future<void> _pickImage(ImageSource source) async {
    if (!_isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model is still loading, please wait...')),
      );
      return;
    }

    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 90,
      );

      if (pickedFile == null) return;

      setState(() => _isLoading = true);

      // Classify the image
      final result = await _classifier.classify(File(pickedFile.path));

      // Save image to app storage
      final savedImagePath = await _storage.saveImage(File(pickedFile.path));

      // Create scan result
      final scanResult = ScanResult(
        imagePath: savedImagePath,
        className: result.className,
        cropName: result.cropName,
        diseaseName: result.diseaseName,
        confidence: result.confidence,
        resultType: result.resultType,
        scannedAt: DateTime.now(),
      );

      // Save to database
      final id = await _storage.saveScan(scanResult);
      final savedResult = ScanResult(
        id: id,
        imagePath: savedImagePath,
        className: result.className,
        cropName: result.cropName,
        diseaseName: result.diseaseName,
        confidence: result.confidence,
        resultType: result.resultType,
        scannedAt: scanResult.scannedAt,
      );

      setState(() => _isLoading = false);

      if (!mounted) return;

      // Navigate to result screen
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            scanResult: savedResult,
            classificationResult: result,
          ),
        ),
      );
    } catch (e) {
      setState(() => _isLoading = false);
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error: ${e.toString()}')));
    }
  }

  void _openHistory() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const HistoryScreen()),
    );
  }

  void _openDiseaseInfo() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const DiseaseInfoScreen()),
    );
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 20),
                  // App header
                  _buildHeader(),
                  const SizedBox(height: 32),
                  // Main action buttons
                  _buildActionButtons(),
                  const SizedBox(height: 32),
                  // Supported crops
                  _buildSupportedCrops(),
                  const SizedBox(height: 24),
                  // Quick actions
                  _buildQuickActions(),
                  const SizedBox(height: 24),
                  // Tips
                  _buildPhotoTips(),
                  const SizedBox(height: 20),
                ],
              ),
            ),
            // Loading overlay
            if (_isLoading) _buildLoadingOverlay(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      children: [
        Container(
          width: 80,
          height: 80,
          decoration: BoxDecoration(
            color: Colors.green.shade50,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Icon(Icons.eco, size: 48, color: Colors.green.shade700),
        ),
        const SizedBox(height: 16),
        const Text(
          'Crop Doctor',
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            color: Color(0xFF2E7D32),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          'Detect crop diseases instantly with your camera',
          style: TextStyle(fontSize: 15, color: Colors.grey.shade600),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 4),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.wifi_off, size: 14, color: Colors.grey.shade500),
            const SizedBox(width: 4),
            Text(
              'Works offline',
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey.shade500,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildActionButtons() {
    return Column(
      children: [
        // Take Photo button
        SizedBox(
          width: double.infinity,
          height: 56,
          child: ElevatedButton.icon(
            onPressed: () => _pickImage(ImageSource.camera),
            icon: const Icon(Icons.camera_alt, size: 24),
            label: const Text(
              'Take Photo',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF2E7D32),
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
              elevation: 2,
            ),
          ),
        ),
        const SizedBox(height: 14),
        // Upload from Gallery button
        SizedBox(
          width: double.infinity,
          height: 56,
          child: OutlinedButton.icon(
            onPressed: () => _pickImage(ImageSource.gallery),
            icon: const Icon(Icons.photo_library, size: 24),
            label: const Text(
              'Upload from Gallery',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
            ),
            style: OutlinedButton.styleFrom(
              foregroundColor: const Color(0xFF2E7D32),
              side: const BorderSide(color: Color(0xFF2E7D32), width: 2),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSupportedCrops() {
    final crops = [
      {'name': 'Banana', 'icon': '🍌'},
      {'name': 'Beans', 'icon': '🫘'},
      {'name': 'Maize', 'icon': '🌽'},
      {'name': 'Potato', 'icon': '🥔'},
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Supported Crops',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Color(0xFF2E7D32),
          ),
        ),
        const SizedBox(height: 12),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: crops.map((crop) {
            return Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                color: Colors.green.shade50,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.green.shade100),
              ),
              child: Column(
                children: [
                  Text(crop['icon']!, style: const TextStyle(fontSize: 28)),
                  const SizedBox(height: 4),
                  Text(
                    crop['name']!,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Colors.green.shade800,
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  Widget _buildQuickActions() {
    return Row(
      children: [
        Expanded(
          child: _buildQuickActionCard(
            icon: Icons.history,
            label: 'History',
            onTap: _openHistory,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildQuickActionCard(
            icon: Icons.menu_book,
            label: 'Disease Guide',
            onTap: _openDiseaseInfo,
          ),
        ),
      ],
    );
  }

  Widget _buildQuickActionCard({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.grey.shade200),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.shade100,
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: const Color(0xFF2E7D32), size: 22),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                fontWeight: FontWeight.w600,
                color: Color(0xFF2E7D32),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPhotoTips() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.amber.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb, color: Colors.amber.shade700, size: 20),
              const SizedBox(width: 8),
              Text(
                'Tips for Best Results',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.amber.shade900,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          _buildTipItem('Photograph a single leaf clearly'),
          _buildTipItem('Use natural daylight (not direct sun)'),
          _buildTipItem('Hold phone 15-20 cm from the leaf'),
          _buildTipItem('Make sure the leaf fills most of the frame'),
        ],
      ),
    );
  }

  Widget _buildTipItem(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('• ', style: TextStyle(color: Colors.amber.shade800)),
          Expanded(
            child: Text(
              text,
              style: TextStyle(fontSize: 13, color: Colors.amber.shade900),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingOverlay() {
    return Container(
      color: Colors.black54,
      child: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
              strokeWidth: 3,
            ),
            SizedBox(height: 20),
            Text(
              'Analyzing image...',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w500,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'This may take a few seconds',
              style: TextStyle(color: Colors.white70, fontSize: 14),
            ),
          ],
        ),
      ),
    );
  }
}
