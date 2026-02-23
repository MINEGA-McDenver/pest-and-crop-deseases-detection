import 'dart:io';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import '../models/scan_result.dart';

class StorageService {
  static Database? _database;
  static const String _tableName = 'scan_history';
  static const int _autoDeleteDays = 30;
  static const int _reminderDays = 25;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'crop_doctor.db');

    return await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE $_tableName (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            imagePath TEXT NOT NULL,
            className TEXT NOT NULL,
            cropName TEXT NOT NULL,
            diseaseName TEXT NOT NULL,
            confidence REAL NOT NULL,
            resultType TEXT NOT NULL,
            scannedAt TEXT NOT NULL,
            isSaved INTEGER NOT NULL DEFAULT 0
          )
        ''');
      },
    );
  }

  Future<int> saveScan(ScanResult result) async {
    final db = await database;
    return await db.insert(_tableName, result.toMap());
  }

  Future<List<ScanResult>> getAllScans() async {
    final db = await database;
    final maps = await db.query(_tableName, orderBy: 'scannedAt DESC');
    return maps.map((m) => ScanResult.fromMap(m)).toList();
  }

  Future<List<ScanResult>> getRecentScans() async {
    final db = await database;
    final maps = await db.query(
      _tableName,
      where: 'isSaved = 0',
      orderBy: 'scannedAt DESC',
    );
    return maps.map((m) => ScanResult.fromMap(m)).toList();
  }

  Future<List<ScanResult>> getSavedScans() async {
    final db = await database;
    final maps = await db.query(
      _tableName,
      where: 'isSaved = 1',
      orderBy: 'scannedAt DESC',
    );
    return maps.map((m) => ScanResult.fromMap(m)).toList();
  }

  Future<void> savePermanently(int id) async {
    final db = await database;
    await db.update(
      _tableName,
      {'isSaved': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<void> deleteScan(int id) async {
    final db = await database;
    final maps = await db.query(_tableName, where: 'id = ?', whereArgs: [id]);
    if (maps.isNotEmpty) {
      final imagePath = maps[0]['imagePath'] as String;
      final file = File(imagePath);
      if (await file.exists()) {
        await file.delete();
      }
    }
    await db.delete(_tableName, where: 'id = ?', whereArgs: [id]);
  }

  Future<int> autoDeleteOldScans() async {
    final db = await database;
    final cutoff = DateTime.now().subtract(
      const Duration(days: _autoDeleteDays),
    );

    final maps = await db.query(
      _tableName,
      where: 'isSaved = 0 AND scannedAt < ?',
      whereArgs: [cutoff.toIso8601String()],
    );

    for (final map in maps) {
      final imagePath = map['imagePath'] as String;
      final file = File(imagePath);
      if (await file.exists()) {
        await file.delete();
      }
    }

    return await db.delete(
      _tableName,
      where: 'isSaved = 0 AND scannedAt < ?',
      whereArgs: [cutoff.toIso8601String()],
    );
  }

  Future<List<ScanResult>> getExpiringScans() async {
    final db = await database;
    final reminderCutoff = DateTime.now().subtract(
      const Duration(days: _reminderDays),
    );
    final deleteCutoff = DateTime.now().subtract(
      const Duration(days: _autoDeleteDays),
    );

    final maps = await db.query(
      _tableName,
      where: 'isSaved = 0 AND scannedAt < ? AND scannedAt >= ?',
      whereArgs: [
        reminderCutoff.toIso8601String(),
        deleteCutoff.toIso8601String(),
      ],
      orderBy: 'scannedAt ASC',
    );
    return maps.map((m) => ScanResult.fromMap(m)).toList();
  }

  Future<String> saveImage(File imageFile) async {
    final appDir = await getApplicationDocumentsDirectory();
    final imagesDir = Directory(join(appDir.path, 'scan_images'));
    if (!await imagesDir.exists()) {
      await imagesDir.create(recursive: true);
    }

    final fileName =
        'scan_${DateTime.now().millisecondsSinceEpoch}${extension(imageFile.path)}';
    final savedFile = await imageFile.copy(join(imagesDir.path, fileName));
    return savedFile.path;
  }
}
