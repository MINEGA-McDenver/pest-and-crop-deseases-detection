import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'dart:io';
import '../models/scan_result.dart';

class StorageService {
  static Database? _database;

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
      version: 3,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cropName TEXT NOT NULL,
            diseaseName TEXT NOT NULL,
            confidence REAL NOT NULL,
            resultType TEXT NOT NULL,
            allProbabilities TEXT NOT NULL DEFAULT '',
            imagePath TEXT NOT NULL,
            dateTime TEXT NOT NULL,
            isSaved INTEGER NOT NULL DEFAULT 0
          )
        ''');
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute(
            'ALTER TABLE scan_history ADD COLUMN isSaved INTEGER NOT NULL DEFAULT 0',
          );
        }
        if (oldVersion < 3) {
          await db.execute(
            'ALTER TABLE scan_history ADD COLUMN allProbabilities TEXT NOT NULL DEFAULT \'\'',
          );
          // className column remains but is no longer used
        }
      },
    );
  }

  /// Save a ScanResult to the database
  Future<ScanResult> saveScan(ScanResult result) async {
    final db = await database;
    final map = result.toMap();
    final id = await db.insert('scan_history', map);

    return ScanResult(
      id: id,
      imagePath: result.imagePath,
      cropName: result.cropName,
      diseaseName: result.diseaseName,
      confidence: result.confidence,
      resultType: result.resultType,
      allProbabilities: result.allProbabilities,
      dateTime: result.dateTime,
      isSaved: false,
    );
  }

  /// Get recent scans (all scans, ordered by most recent)
  Future<List<ScanResult>> getRecentScans() async {
    await cleanupExpiredRecentScans();
    final db = await database;
    final maps = await db.query('scan_history', orderBy: 'id DESC');
    return maps.map((m) => ScanResult.fromMap(m)).toList();
  }

  /// Get only permanently saved scans
  Future<List<ScanResult>> getSavedScans() async {
    final db = await database;
    final maps = await db.query(
      'scan_history',
      where: 'isSaved = ?',
      whereArgs: [1],
      orderBy: 'id DESC',
    );
    return maps.map((m) => ScanResult.fromMap(m)).toList();
  }

  /// Mark a scan as permanently saved
  Future<void> savePermanently(int id) async {
    final db = await database;
    await db.update(
      'scan_history',
      {'isSaved': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  /// Delete a single scan by id
  Future<void> deleteScan(int id) async {
    final db = await database;
    final rows = await db.query(
      'scan_history',
      columns: ['imagePath'],
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );

    if (rows.isNotEmpty) {
      final path = (rows.first['imagePath'] as String?) ?? '';
      if (path.isNotEmpty) {
        final imageFile = File(path);
        if (await imageFile.exists()) {
          try {
            await imageFile.delete();
          } catch (_) {
            // Ignore file delete errors; DB deletion must still complete.
          }
        }
      }
    }

    await db.delete('scan_history', where: 'id = ?', whereArgs: [id]);
  }

  /// Delete recent non-bookmarked scans older than [maxAgeDays].
  Future<void> cleanupExpiredRecentScans({int maxAgeDays = 30}) async {
    final db = await database;
    final cutoff = DateTime.now().subtract(Duration(days: maxAgeDays));
    final candidateRows = await db.query(
      'scan_history',
      columns: ['id', 'dateTime'],
      where: 'isSaved = ?',
      whereArgs: [0],
    );

    for (final row in candidateRows) {
      final id = row['id'] as int?;
      final storedTime = row['dateTime'] as String?;
      final parsedTime =
          storedTime == null ? null : DateTime.tryParse(storedTime)?.toLocal();
      if (id != null && parsedTime != null && parsedTime.isBefore(cutoff)) {
        await deleteScan(id);
      }
    }
  }

  /// Get all history
  Future<List<ScanResult>> getHistory() async {
    return getRecentScans();
  }

  /// Clear all history
  Future<void> clearHistory() async {
    final db = await database;
    final allRows = await db.query('scan_history', columns: ['id']);
    for (final row in allRows) {
      final id = row['id'] as int?;
      if (id != null) {
        await deleteScan(id);
      }
    }
  }
}
