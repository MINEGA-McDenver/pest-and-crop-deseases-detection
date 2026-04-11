import 'package:crop_doctor/data/disease_info.dart';
import 'package:crop_doctor/l10n/app_strings.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Locale parity', () {
    test('canonical crop mapping supports EN and RW labels', () {
      expect(DiseaseInfo.canonicalCropKey('Banana'), 'banana');
      expect(DiseaseInfo.canonicalCropKey('Igitoki'), 'banana');
      expect(DiseaseInfo.canonicalCropKey('Ibishyimbo'), 'beans');
      expect(DiseaseInfo.canonicalCropKey('Unknown Crop'), 'unknown');
    });

    test('disease localization resolves canonical and localized aliases in rw', () {
      AppStrings.setActiveLanguageCode('rw');

      expect(DiseaseInfo.localizeDiseaseName('healthy'), AppStrings.trCode('healthy'));
      expect(DiseaseInfo.localizeDiseaseName('cyiza'), AppStrings.trCode('healthy'));
      expect(
        DiseaseInfo.localizeDiseaseName('unknown condition'),
        AppStrings.trCode('unknownCondition'),
      );

      final resolvedByKey = DiseaseInfo.resolveByCropAndDiseaseName(
        'Igitoki',
        'banana_sigatoka',
      );
      expect(resolvedByKey, isNotNull);

      final resolvedByLabel = DiseaseInfo.resolveByCropAndDiseaseName(
        'Igitoki',
        'Sigatoka Leaf Spot',
      );
      expect(resolvedByLabel, isNotNull);
    });

    test('AppStrings follows language switch en -> rw', () {
      AppStrings.setActiveLanguageCode('en');
      expect(AppStrings.trCode('banana'), 'Banana');

      AppStrings.setActiveLanguageCode('rw');
      expect(AppStrings.trCode('banana'), 'Igitoki');
    });
  });
}
