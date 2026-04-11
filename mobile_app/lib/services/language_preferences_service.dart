import 'package:shared_preferences/shared_preferences.dart';
import '../l10n/app_strings.dart';

class LanguagePreferencesService {
  static const String _languageCodeKey = 'app_language_code';

  Future<String?> getSavedLanguageCode() async {
    final prefs = await SharedPreferences.getInstance();
    final code = prefs.getString(_languageCodeKey);
    if (code == null) return null;
    final normalized = AppStrings.normalizeLanguageCode(code);
    if (normalized == 'rw' || normalized == 'en') return normalized;
    return null;
  }

  Future<void> saveLanguageCode(String languageCode) async {
    final prefs = await SharedPreferences.getInstance();
    final normalized = AppStrings.normalizeLanguageCode(languageCode);
    await prefs.setString(_languageCodeKey, normalized);
  }
}
