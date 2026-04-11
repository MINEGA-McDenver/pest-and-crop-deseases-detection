import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:crop_doctor/services/classifier_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  bool isMissingTfLiteWindowsRuntime(Object error) {
    final msg = error.toString().toLowerCase();
    return msg.contains('libtensorflowlite_c-win.dll') ||
        (msg.contains('failed to load dynamic library') &&
            msg.contains('tensorflowlite'));
  }

  Future<List<File>> sampleImagesForCrop(
    Directory testRoot,
    List<String> classFolders,
    int target,
  ) async {
    final perClassFiles = <String, List<File>>{};

    for (final className in classFolders) {
      final dir = Directory(
        '${testRoot.path}${Platform.pathSeparator}$className',
      );
      if (!await dir.exists()) continue;

      final files =
          dir
              .listSync()
              .whereType<File>()
              .where(
                (f) =>
                    f.path.toLowerCase().endsWith('.jpg') ||
                    f.path.toLowerCase().endsWith('.jpeg') ||
                    f.path.toLowerCase().endsWith('.png'),
              )
              .toList()
            ..sort((a, b) => a.path.compareTo(b.path));

      perClassFiles[className] = files;
    }

    final selected = <File>[];
    var index = 0;
    while (selected.length < target) {
      var addedThisRound = false;
      for (final className in classFolders) {
        final files = perClassFiles[className] ?? const <File>[];
        if (index < files.length && selected.length < target) {
          selected.add(files[index]);
          addedThisRound = true;
        }
      }
      if (!addedThisRound) break;
      index++;
    }

    return selected;
  }

  bool isRejected(String resultType) {
    return resultType == 'uncertain' ||
        resultType == 'unsupported' ||
        resultType == 'other_leaf';
  }

  test(
    'must-pass pre-install smoke set',
    () async {
      final repoRoot = Directory.current.parent;
      final testRoot = Directory(
        '${repoRoot.path}${Platform.pathSeparator}datasets${Platform.pathSeparator}model_ready${Platform.pathSeparator}test',
      );

      expect(
        await testRoot.exists(),
        isTrue,
        reason: 'Expected dataset test root at ${testRoot.path}',
      );

      final service = ClassifierService();
      try {
        await service.initialize();
      } catch (e) {
        if (Platform.isWindows && isMissingTfLiteWindowsRuntime(e)) {
          // Flutter unit-test VM can miss native TFLite runtime on some setups.
          // Keep CI/local tests unblocked and run this smoke set on device.
          // ignore: avoid_print
          print(
            '[SMOKE][SKIPPED] TensorFlow Lite Windows runtime not available in flutter test VM. '
            'Run device smoke validation with "flutter run --release -d <device-id>" '
            'and execute the 20/20/10/10 crop checklist manually.',
          );
          return;
        }
        rethrow;
      }

      final plan = <String, Map<String, Object>>{
        'beans': {
          'folders': <String>[
            'beans_angular_leaf_spot',
            'beans_healthy',
            'beans_rust',
          ],
          'target': 20,
        },
        'potato': {
          'folders': <String>[
            'potato_early_blight',
            'potato_healthy',
            'potato_late_blight',
          ],
          'target': 20,
        },
        'banana': {
          'folders': <String>[
            'banana_cordana',
            'banana_healthy',
            'banana_pestalotiopsis',
            'banana_sigatoka',
          ],
          'target': 10,
        },
        'maize': {
          'folders': <String>[
            'maize_common_rust',
            'maize_gray_leaf_spot',
            'maize_healthy',
            'maize_northern_leaf_blight',
          ],
          'target': 10,
        },
      };

      final summary = <String, Map<String, Object>>{};

      for (final entry in plan.entries) {
        final crop = entry.key;
        final folders = entry.value['folders'] as List<String>;
        final target = entry.value['target'] as int;

        final samples = await sampleImagesForCrop(testRoot, folders, target);
        expect(
          samples.length,
          target,
          reason:
              'Expected $target images for $crop but found ${samples.length}.',
        );

        var rejected = 0;
        var analyzed = 0;

        for (final file in samples) {
          final result = await service.classifyImage(file.path);
          analyzed++;
          if (isRejected(result.resultType)) {
            rejected++;
          }
        }

        final rejectionRate = analyzed == 0 ? 1.0 : rejected / analyzed;
        summary[crop] = {
          'analyzed': analyzed,
          'rejected': rejected,
          'rejectionRate': rejectionRate,
        };

        // Must-pass gate for beans and potato before pilot install.
        if (crop == 'beans' || crop == 'potato') {
          expect(
            rejectionRate < 0.20,
            isTrue,
            reason:
                '$crop rejection rate ${(rejectionRate * 100).toStringAsFixed(1)}% >= 20%. Do not install pilot build yet.',
          );
        }
      }

      // Print summary in test logs for quick deployment decision.
      for (final entry in summary.entries) {
        final crop = entry.key;
        final analyzed = entry.value['analyzed'] as int;
        final rejected = entry.value['rejected'] as int;
        final rejectionRate = entry.value['rejectionRate'] as double;
        // ignore: avoid_print
        print(
          '[SMOKE] $crop -> analyzed=$analyzed rejected=$rejected rejectionRate=${(rejectionRate * 100).toStringAsFixed(1)}%',
        );
      }
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );
}
