import 'dart:io';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/scan_result.dart';
import '../services/storage_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen>
    with SingleTickerProviderStateMixin {
  final StorageService _storage = StorageService();
  late TabController _tabController;
  List<ScanResult> _recentScans = [];
  List<ScanResult> _savedScans = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _loadScans();
  }

  Future<void> _loadScans() async {
    setState(() => _isLoading = true);
    _recentScans = await _storage.getRecentScans();
    _savedScans = await _storage.getSavedScans();
    setState(() => _isLoading = false);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History'),
        backgroundColor: const Color(0xFF2E7D32),
        foregroundColor: Colors.white,
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: Colors.white,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.white70,
          tabs: [
            Tab(text: 'Recent (${_recentScans.length})'),
            Tab(text: 'Saved (${_savedScans.length})'),
          ],
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
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
              isRecent ? 'No recent scans' : 'No saved scans',
              style: TextStyle(fontSize: 18, color: Colors.grey.shade500),
            ),
            const SizedBox(height: 8),
            Text(
              isRecent
                  ? 'Start scanning crops to see results here'
                  : 'Tap the bookmark icon on scans to save permanently',
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
          return _buildScanCard(scans[index], isRecent: isRecent);
        },
      ),
    );
  }

  Widget _buildScanCard(ScanResult scan, {required bool isRecent}) {
    final dateStr = DateFormat('MMM d, yyyy • h:mm a').format(scan.scannedAt);
    final daysOld = DateTime.now().difference(scan.scannedAt).inDays;

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
          // Could navigate to detail view
        },
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              // Image thumbnail
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
              // Info
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
                                ? '${scan.cropName} — Healthy'
                                : '${scan.cropName} — ${scan.diseaseName}',
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
                      '${(scan.confidence * 100).toStringAsFixed(1)}% confidence',
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
                              'Expires in ${30 - daysOld}d',
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
              // Actions
              Column(
                children: [
                  if (isRecent)
                    IconButton(
                      icon: const Icon(Icons.bookmark_add_outlined),
                      iconSize: 20,
                      color: const Color(0xFF2E7D32),
                      tooltip: 'Save permanently',
                      onPressed: () async {
                        if (scan.id != null) {
                          await _storage.savePermanently(scan.id!);
                          _loadScans();
                          if (mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Saved permanently'),
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
                    tooltip: 'Delete',
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
        title: const Text('Delete Scan'),
        content: const Text(
          'This will permanently delete this scan and its image. Continue?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              if (scan.id != null) {
                await _storage.deleteScan(scan.id!);
                _loadScans();
              }
            },
            child: Text('Delete', style: TextStyle(color: Colors.red.shade600)),
          ),
        ],
      ),
    );
  }
}
