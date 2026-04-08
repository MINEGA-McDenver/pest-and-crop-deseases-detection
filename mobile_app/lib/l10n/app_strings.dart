import 'package:flutter/material.dart';

class AppStrings {
  static const Map<String, String> _en = {
    'appTitle': 'Crop Doctor',
    'scanHistory': 'Scan History',
    'modelLoading': 'Loading AI model...',
    'modelWait': 'Model is still loading, please wait...',
    'retry': 'Retry',
    'modelInitFail': 'Failed to load AI model',
    'cameraPermissionRequired': 'Camera permission is required',
    'galleryPermissionRequired': 'Gallery permission is required to pick images',
    'imageUnavailable': 'Selected image could not be accessed. Please try again.',
    'scanError': 'Error',
    'analyzing': 'Analyzing image...',
    'cropDetectionTitle': 'Crop Disease Detection',
    'cropDetectionSubtitle':
        'Take a photo or upload an image of your crop to detect diseases',
    'takePhoto': 'Take Photo',
    'uploadGallery': 'Upload from Gallery',
    'supportedCrops': 'Supported Crops',
    'scanResult': 'Scan Result',
    'resultSaved': 'Result saved!',
    'takeNewPhoto': 'Take New Photo',
    'tipsHeader': 'Tips for Better Photos',
    'tipDaylight': 'Take photos in natural daylight',
    'tipSingleLeaf': 'Focus on a single leaf, fill the frame',
    'tipDistance': 'Hold phone 15-30cm from the leaf',
    'tipSteady': 'Hold steady to avoid blur',
    'tipBackground': 'Use a plain background if possible',
    'tipAffected': 'Show the affected area clearly',
  };

  static const Map<String, String> _rw = {
    'appTitle': 'Muganga w\'Ibihingwa',
    'scanHistory': 'Amateka y\'Isesengura',
    'modelLoading': 'AI iri gutangira...',
    'modelWait': 'AI iracyatangiye, tegereza gato...',
    'retry': 'Ongera ugerageze',
    'modelInitFail': 'AI yanze gutangira',
    'cameraPermissionRequired':
        'Emera uburenganzira bwa camera kugira ngo bifunguke',
    'galleryPermissionRequired':
        'Emera uburenganzira bwa gallery kugira ngo uhitemo ifoto',
    'imageUnavailable':
        'Ifoto wahisemo ntibashije kuyibona. Ongera ugerageze.',
    'scanError': 'Ikosa',
    'analyzing': 'Ifoto iri gusesengurwa...',
    'cropDetectionTitle': 'Kumenya Indwara z\'Ibihingwa',
    'cropDetectionSubtitle':
        'Fata ifoto cyangwa wohereze ifoto y\'ikibabi kugira ngo umenye indwara',
    'takePhoto': 'Fata Ifoto',
    'uploadGallery': 'Hitamo muri Gallery',
    'supportedCrops': 'Ibihingwa Bishyigikiwe',
    'scanResult': 'Igisubizo cy\'Isesengura',
    'resultSaved': 'Igisubizo cyabitswe!',
    'takeNewPhoto': 'Fata Ifoto Nshya',
    'tipsHeader': 'Inama zo Gufata Ifoto Nziza',
    'tipDaylight': 'Fata ifoto ku manywa hari urumuri ruhagije',
    'tipSingleLeaf': 'Fata ikibabi kimwe kigaragara neza',
    'tipDistance': 'Shyira telefoni hagati ya cm 15-30 ku kibabi',
    'tipSteady': 'Fata telefoni idahinda kugira ngo ifoto itajyamo urujijo',
    'tipBackground': 'Niba bishoboka, koresha inyuma itari urujijo',
    'tipAffected': 'Erekana neza aharwaye ku kibabi',
  };

  static String tr(BuildContext context, String key) {
    final code = Localizations.localeOf(context).languageCode.toLowerCase();
    if (code == 'rw') {
      return _rw[key] ?? _en[key] ?? key;
    }
    return _en[key] ?? key;
  }
}
