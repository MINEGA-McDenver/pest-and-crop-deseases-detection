import 'package:flutter/material.dart';
import 'app_locale_scope.dart';

class AppStrings {
  static String _activeLanguageCode = 'en';

  static const Map<String, String> _en = {
    'appTitle': 'Crop Doctor',
    'language': 'Language',
    'english': 'English',
    'kinyarwanda': 'Kinyarwanda',
    'launchHeadline': 'Keep your crops healthy and your harvest secure',
    'launchSubheadline':
        'Choose a language to continue with diagnosis, history, and guidance in your preferred language.',
    'chooseLanguage': 'Choose Language',
    'gettingStarted': 'Getting Started',
    'scanHistory': 'Scan History',
    'recentTab': 'Recent ({count})',
    'savedTab': 'Saved ({count})',
    'noRecentScans': 'No recent scans',
    'noSavedScans': 'No saved scans',
    'startScanningHint': 'Start scanning crops to see results here',
    'saveScanHint': 'Tap the bookmark icon on scans to save permanently',
    'deleteScan': 'Delete Scan',
    'deleteScanConfirm':
        'This will permanently delete this scan and its image. Continue?',
    'cancel': 'Cancel',
    'delete': 'Delete',
    'savePermanently': 'Save permanently',
    'savedPermanently': 'Saved permanently',
    'expiresInDays': 'Expires in {days}d',
    'confidence': 'Confidence',
    'healthy': 'Healthy',
    'uncertain': 'Uncertain',
    'unknownCondition': 'Unknown Condition',
    'modelLoading': 'Loading AI model...',
    'modelWait': 'Model is still loading, please wait...',
    'retry': 'Retry',
    'modelInitFail': 'Failed to load AI model',
    'modelMismatch':
        'Model files are mismatched. Reinstall the app package from the latest pilot build.',
    'modelAssetsMissing':
        'Model assets are missing in this build. Please reinstall the app.',
    'lowMemory': 'Phone memory is low. Close other apps and retry.',
    'modelRetryInstall':
        'Failed to load AI model. Please retry or reinstall the latest build.',
    'cameraPermissionRequired': 'Camera permission is required',
    'galleryPermissionRequired':
        'Gallery permission is required to pick images',
    'imageUnavailable':
        'Selected image could not be accessed. Please try again.',
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
    'keepHealthy': 'How to keep it healthy',
    'takeGoodPhoto': 'How to take a good diagnosis photo',
    'banana': 'Banana',
    'beans': 'Beans',
    'maize': 'Maize',
    'potato': 'Potato',
    'bananaGuidanceTitle': 'Banana crop guidance',
    'bananaHealthyGuide':
        'Use clean planting material, remove heavily infected leaves early, keep good spacing for airflow, and avoid wetting leaves late in the day.',
    'bananaPhotoGuide':
        'Take one clear leaf, fill most of the frame, stay 20-30 cm away, use daylight if possible, and avoid blur or shadows.',
    'beansGuidanceTitle': 'Beans crop guidance',
    'beansHealthyGuide':
        'Rotate crops, remove crop residue after harvest, avoid overhead irrigation when possible, and monitor leaves regularly for early spots.',
    'beansPhotoGuide':
        'Photograph a single affected leaf front-on, include both healthy and diseased parts, keep background simple, and focus before capture.',
    'maizeGuidanceTitle': 'Maize crop guidance',
    'maizeHealthyGuide':
        'Plant at recommended spacing, manage weeds early, use balanced fertilizer, and scout often so leaf diseases are treated early.',
    'maizePhotoGuide':
        'Capture one representative leaf with visible lesions, avoid backlight, keep the leaf centered, and ensure the image is sharp.',
    'potatoGuidanceTitle': 'Potato crop guidance',
    'potatoHealthyGuide':
        'Use healthy seed tubers, avoid prolonged leaf wetness, improve airflow, and remove severely infected leaves to reduce spread.',
    'potatoPhotoGuide':
        'Take a close photo of one leaf with clear symptom edges, keep 20-30 cm distance, avoid mixed crops in frame, and use natural light.',
    'poorQualityHeader': 'Image Quality Issue',
    'unsupportedHeader': 'Crop Not Recognized',
    'uncertainHeader': 'Uncertain Result',
    'unknownConditionHeader': '{crop} - Unknown Condition',
    'healthyHeader': '{crop} - Healthy',
    'diseaseHeader': '{crop} - {disease}',
    'imageQualityTooLow': 'Image Quality Too Low',
    'imageNotReliable':
        'The image could not be analyzed reliably. Please take a new photo following the tips below.',
    'unsupportedIntro':
        'This image does not appear to match any of the crops supported by this app. Currently we support:',
    'unsupportedRetryHint':
        'If this is one of these crops, try taking a clearer photo of a single leaf in good lighting.',
    'cropDetected': '{crop} Detected',
    'cropDetectionConfidence': 'Crop Detection Confidence',
    'yourCropUnhealthy': 'Your {crop} appears to be unhealthy',
    'unknownDiseaseMessage':
        'Sadly, the disease or pest affecting your {crop} is not currently supported by this app. Our app can only detect the following diseases:',
    'whatYouShouldDo': 'What You Should Do',
    'contactProfessionalNow':
        'Contact an agricultural professional immediately',
    'contactProfessionalDesc':
        'Visit your local agricultural extension office or RAB center for expert diagnosis.',
    'actQuickly': 'Act quickly to prevent spread',
    'actQuicklyDesc':
        'Isolate affected plants if possible. Do not wait - diseases can spread rapidly.',
    'takeMultiplePhotos': 'Take multiple photos',
    'takeMultiplePhotosDesc':
        'Photograph affected leaves from different angles to show the professional.',
    'noteSymptoms': 'Note the symptoms',
    'noteSymptomsDesc':
        'Record when you first noticed the problem, which plants are affected, and how fast it spreads.',
    'contactAgriPros': 'Contact Agriculture Professionals',
    'contactAgriList':
        '• Visit your nearest RAB (Rwanda Agriculture Board) center\n• Contact your sector agronomist\n• Call the agriculture helpline for guidance\n• Visit a local agro-dealer for recommended treatments',
    'detectableDiseasesForCrop': 'Diseases we can detect for {crop}:',
    'conditionNotInList':
        'The condition on your crop does not match any of the above.',
    'uncertainBody':
        'The app thinks this might be {crop} ({disease}), but confidence is low at {confidence}%.',
    'detectionConfidence': 'Detection Confidence',
    'whatToDo': 'What To Do',
    'prevention': 'Prevention',
    'topPredictions': 'Top Predictions',
    'qualityTooDark':
        'Image is too dark. Move to a well-lit area or use flash.',
    'qualityTooBright':
        'Image is too bright/overexposed. Avoid direct sunlight on the lens.',
    'qualityLowDetail':
        'Image lacks detail. Move closer to the leaf and ensure focus.',
    'qualityNoLeaf':
        'No plant leaf detected. Please photograph a leaf directly.',
    'selectedImageMissing':
        'Selected image file is missing. Please retake the photo.',
    'couldNotDecodeImage': 'Could not decode image',
    'unknown': 'Unknown',
    'unknownCrop': 'Unknown Crop',
    'poorImageQuality': 'Poor Image Quality',
    'unsupportedCrop': 'Unsupported Crop',
    'classificationError': 'Classification Error',
    'likelyHealthy': 'Likely Healthy',
    'unidentifiedCondition': 'Unidentified Condition',
    'likelyHealthyVerify': 'Likely Healthy - verify manually',
    'rescueLikelyCropLowLight':
        'Likely {crop} leaf, but model is uncertain. Retake close photo in good light.',
    'rescuePossibleCropRetake':
        'Possible {crop} leaf detected. Retake from 20-30cm and include full affected area.',
    'rescueLikelyCropLowConfidence':
        'Likely {crop} leaf, but confidence is low. Retake with one clear leaf centered.',
    'professionalAdvice':
        'For more detailed diagnosis and treatment options, contact your local agricultural extension officer or visit your nearest RAB center. This app provides initial guidance only and does not replace professional advice.',
    'healthyCheckup':
        'Even when your crop looks healthy, schedule regular visits from your sector agronomist for soil testing and nutrition advice.',
    'continueMonitoringCrop': 'Continue monitoring your crop regularly',
  };

  static const Map<String, String> _rw = {
    'appTitle': 'Muganga w\'Ibihingwa',
    'language': 'Ururimi',
    'english': 'Icyongereza',
    'kinyarwanda': 'Ikinyarwanda',
    'launchHeadline':
        'Rinda ibihingwa byawe kandi urinde umusaruro wawe',
    'launchSubheadline':
        'Hitamo ururimi kugira ngo ukomeze ukoresha isesengura, amateka, n\'ubujyanama mu rurimi wahisemo.',
    'chooseLanguage': 'Hitamo Ururimi',
    'gettingStarted': 'Tangira',
    'scanHistory': 'Amateka y\'Isesengura',
    'recentTab': 'Ibya vuba ({count})',
    'savedTab': 'Byabitswe ({count})',
    'noRecentScans': 'Nta bisesengura bya vuba bihari',
    'noSavedScans': 'Nta bisesengura byabitswe bihari',
    'startScanningHint': 'Tangira gusesengura ibihingwa ubone ibisubizo hano',
    'saveScanHint':
        'Kanda ku kimenyetso cyo kubika kugira ngo ibisubizo bibikwe burundu',
    'deleteScan': 'Siba Isesengura',
    'deleteScanConfirm':
        'Ibi bisiba burundu iri sesengura n\'ifoto yaryo. Ukomeze?',
    'cancel': 'Reka',
    'delete': 'Siba',
    'savePermanently': 'Bika burundu',
    'savedPermanently': 'Byabitswe burundu',
    'expiresInDays': 'Bizasibwa mu {days}d',
    'confidence': 'Icyizere',
    'healthy': 'Cyiza',
    'uncertain': 'Ntibirasobanuka',
    'unknownCondition': 'Indwara itazwi',
    'modelLoading': 'AI iri gutangira...',
    'modelWait': 'AI iracyatangiye, tegereza gato...',
    'retry': 'Ongera ugerageze',
    'modelInitFail': 'AI yanze gutangira',
    'modelMismatch':
        'Dosiye za model ntizihuye. Ongera ushyiremo porogaramu nshya y\'igerageza.',
    'modelAssetsMissing':
        'Dosiye za model ntiziri muri iyi version. Ongera ushyiremo porogaramu.',
    'lowMemory':
        'Ububiko bwa telefoni ni buke. Funga izindi porogaramu wongere ugerageze.',
    'modelRetryInstall':
        'AI yanze gufunguka. Ongera ugerageze cyangwa ushyiremo version nshya.',
    'cameraPermissionRequired':
        'Emera uburenganzira bwa camera kugira ngo bifunguke',
    'galleryPermissionRequired':
        'Emera uburenganzira bwa gallery kugira ngo uhitemo ifoto',
    'imageUnavailable': 'Ifoto wahisemo ntibashije kuyibona. Ongera ugerageze.',
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
    'keepHealthy': 'Uko wagumana igihingwa gifite ubuzima bwiza',
    'takeGoodPhoto': 'Uko wafata ifoto nziza yo gusesengura',
    'banana': 'Igitoki',
    'beans': 'Ibishyimbo',
    'maize': 'Ibigori',
    'potato': 'Ibirayi',
    'bananaGuidanceTitle': 'Inama ku gihingwa cy\'igitoki',
    'bananaHealthyGuide':
        'Koresha ingemwe nziza, kura amababi arwaye cyane hakiri kare, usige intera ihagije kugira ngo umwuka winjire, kandi wirinde kuhira ku mababi nimugoroba.',
    'bananaPhotoGuide':
        'Fata ikibabi kimwe kigaragara neza, cyuzuze hafi ifoto yose, jya hagati ya cm 20-30, ukoreshe urumuri rw\'amanywa kandi wirinde igihu cyangwa igicucu.',
    'beansGuidanceTitle': 'Inama ku bihingwa by\'ibishyimbo',
    'beansHealthyGuide':
        'Hinduranya ibihingwa, kura ibisigazwa by\'umusaruro nyuma yo gusarura, wirinde kuhira hejuru y\'amababi bishoboka, kandi ukurikire amababi kenshi kugira ngo ubone ibimenyetso hakiri kare.',
    'beansPhotoGuide':
        'Fata ifoto y\'ikibabi kimwe gifite ibimenyetso, erekana igice kizima n\'kirwaye, ukoreshe inyuma itarangaza kandi ukore focus mbere yo gufata.',
    'maizeGuidanceTitle': 'Inama ku bihingwa by\'ibigori',
    'maizeHealthyGuide':
        'Tera ku ntera isabwa, rwanya urumamfu hakiri kare, koresha ifumbire iboneye, kandi ukurikire kenshi kugira ngo indwara zifatwe hakiri kare.',
    'maizePhotoGuide':
        'Fata ikibabi gihagarariye ibimenyetso, wirinde gufata urumuri ruturutse inyuma, shyira ikibabi hagati kandi urebe ko ifoto ityaye.',
    'potatoGuidanceTitle': 'Inama ku bihingwa by\'ibirayi',
    'potatoHealthyGuide':
        'Koresha imbuto z\'ibirayi nziza, wirinde ko amababi ahora atose igihe kirekire, ongera uko umwuka winjira, kandi ukure amababi arwaye cyane kugira ngo indwara idakwirakwira.',
    'potatoPhotoGuide':
        'Fata ifoto ya hafi y\'ikibabi kimwe gifite ibimenyetso bisobanutse, jya hagati ya cm 20-30, wirinde kuvanga ibindi bihingwa mu ifoto, kandi ukoreshe urumuri rusanzwe.',
    'poorQualityHeader': 'Ikibazo cy\'ubuziranenge bw\'ifoto',
    'unsupportedHeader': 'Igihingwa ntikimenyekanye',
    'uncertainHeader': 'Igisubizo ntikirasobanuka',
    'unknownConditionHeader': '{crop} - Indwara itazwi',
    'healthyHeader': '{crop} - Gifite ubuzima bwiza',
    'diseaseHeader': '{crop} - {disease}',
    'imageQualityTooLow': 'Ubuziranenge bw\'ifoto buri hasi cyane',
    'imageNotReliable':
        'Iyi foto ntiyashoboye gusesengurwa neza. Fata indi foto ukurikije inama zikurikira.',
    'unsupportedIntro':
        'Iyi foto isa n\'idahuye n\'ibihingwa porogaramu yacu ishyigikira. Ubu dushyigikiye:',
    'unsupportedRetryHint':
        'Niba ari kimwe muri ibi bihingwa, ongera ufate ifoto isobanutse y\'ikibabi kimwe ahari urumuri rwiza.',
    'cropDetected': '{crop} cyamenyekanye',
    'cropDetectionConfidence': 'Icyizere mu kumenya igihingwa',
    'yourCropUnhealthy': '{crop} cyawe gisa n\'ikidafite ubuzima bwiza',
    'unknownDiseaseMessage':
        'Birababaje, indwara cyangwa udukoko byateye {crop} yawe ntibirashyirwa muri iyi porogaramu. Porogaramu yacu ishobora kumenya izi ndwara zikurikira gusa:',
    'whatYouShouldDo': 'Ibyo ugomba gukora',
    'contactProfessionalNow': 'Hamagara inzobere mu buhinzi ako kanya',
    'contactProfessionalDesc':
        'Sura ibiro by\'ubujyanama bw\'ubuhinzi biri hafi yawe cyangwa ikigo cya RAB kugira ngo ubone ubufasha bw\'inzobere.',
    'actQuickly': 'Kora vuba kugira ngo indwara idakwirakwira',
    'actQuicklyDesc':
        'Niba bishoboka, tandukanya ibimera byanduye. Ntutinde - indwara ishobora gukwirakwira vuba.',
    'takeMultiplePhotos': 'Fata amafoto menshi',
    'takeMultiplePhotosDesc':
        'Fata amababi arwaye ku mpande zitandukanye kugira ngo ubereke inzobere.',
    'noteSymptoms': 'Andika ibimenyetso',
    'noteSymptomsDesc':
        'Andika igihe waboneye ikibazo bwa mbere, ibimera byafashwe, n\'uko gikwirakwira.',
    'contactAgriPros': 'Vugisha inzobere mu buhinzi',
    'contactAgriList':
        '• Sura ikigo cya RAB kikwegereye\n• Vugana n\'agronome w\'umurenge wawe\n• Hamagara umurongo ufasha mu buhinzi\n• Sura umucuruzi w\'inyongeramusaruro wegereye',
    'detectableDiseasesForCrop': 'Indwara dushobora kumenya kuri {crop}:',
    'conditionNotInList':
        'Indwara iri ku gihingwa cyawe ntijya muri izi zavuzwe haruguru.',
    'uncertainBody':
        'Porogaramu ibona ko ishobora kuba {crop} ({disease}), ariko icyizere ni gito: {confidence}%.',
    'detectionConfidence': 'Icyizere cy\'isesengura',
    'whatToDo': 'Ibyo wakora',
    'prevention': 'Uko wakwirinda',
    'topPredictions': 'Ibisubizo byo hejuru',
    'qualityTooDark':
        'Ifoto yijimye cyane. Jya ahari urumuri ruhagije cyangwa ukoreshe flash.',
    'qualityTooBright':
        'Ifoto irakeye cyane. Irinde izuba rikubita lens ya camera.',
    'qualityLowDetail':
        'Ifoto ntigaragaza neza ibisobanuro. Egera ikibabi kandi ukore focus neza.',
    'qualityNoLeaf':
        'Nta kibabi cy\'igihingwa cyabonetse. Fata ifoto y\'ikibabi neza.',
    'selectedImageMissing': 'Ifoto wahisemo ntiyabonetse. Ongera ufate ifoto.',
    'couldNotDecodeImage': 'Ifoto ntiyashoboye gusomwa',
    'unknown': 'Ntibizwi',
    'unknownCrop': 'Igihingwa kitazwi',
    'poorImageQuality': 'Ubuziranenge buke bw\'ifoto',
    'unsupportedCrop': 'Igihingwa kidashyigikiwe',
    'classificationError': 'Ikosa mu isesengura',
    'likelyHealthy': 'Gisa n\'igifite ubuzima bwiza',
    'unidentifiedCondition': 'Indwara itarasobanurwa',
    'likelyHealthyVerify':
        'Bisa n\'igifite ubuzima bwiza - genzura n\'inzobere',
    'rescueLikelyCropLowLight':
        'Birasa n\'{crop}, ariko model nticyizeye. Ongera ufate ifoto ya hafi ahari urumuri rwiza.',
    'rescuePossibleCropRetake':
        'Hashobora kuba ari {crop}. Ongera ufate ifoto uri hagati ya cm 20-30 kandi werekane ahafashwe hose.',
    'rescueLikelyCropLowConfidence':
        'Birasa n\'{crop}, ariko icyizere ni gito. Ongera ufate ifoto y\'ikibabi kimwe gishyizwe hagati.',
    'professionalAdvice':
        'Ku bisobanuro birambuye ku isuzuma n\'ubuvuzi, vugana n\'ushinzwe ubujyanama bw\'ubuhinzi cyangwa usure ikigo cya RAB kikwegereye. Iyi porogaramu itanga ubuyobozi bw\'ibanze gusa kandi ntisimbura inama y\'inzobere.',
    'healthyCheckup':
        'N\'ubwo igihingwa gisa neza, teganya gusurwa kenshi n\'agronome w\'umurenge kugira ngo agufashe ku butaka n\'ifumbire.',
    'continueMonitoringCrop': 'Komeza gukurikirana igihingwa cyawe buri gihe',
  };

    static String normalizeLanguageCode(String languageCode) {
        return languageCode.toLowerCase() == 'rw' ? 'rw' : 'en';
    }

  static void setActiveLanguageCode(String languageCode) {
        _activeLanguageCode = normalizeLanguageCode(languageCode);
  }

  static String get activeLanguageCode => _activeLanguageCode;

    static String localeCodeOf(BuildContext context) {
        final scoped = AppLocaleScope.maybeOf(context);
        if (scoped != null) {
            return normalizeLanguageCode(scoped.locale.languageCode);
        }

        try {
            return normalizeLanguageCode(Localizations.localeOf(context).languageCode);
        } catch (_) {
            return _activeLanguageCode;
        }
    }

  static String tr(
    BuildContext context,
    String key, {
    Map<String, String>? args,
  }) {
      final code = localeCodeOf(context);
        setActiveLanguageCode(code);
        return trCode(key, code: code, args: args);
  }

  static String trCode(String key, {String? code, Map<String, String>? args}) {
        final lang = normalizeLanguageCode(code ?? _activeLanguageCode);
    final raw = lang == 'rw'
        ? (_rw[key] ?? _en[key] ?? key)
        : (_en[key] ?? key);
    if (args == null || args.isEmpty) return raw;

    var rendered = raw;
    for (final entry in args.entries) {
      rendered = rendered.replaceAll('{${entry.key}}', entry.value);
    }
    return rendered;
  }
}
