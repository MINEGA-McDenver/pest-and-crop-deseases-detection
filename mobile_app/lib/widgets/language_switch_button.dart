import 'package:flutter/material.dart';
import '../l10n/app_strings.dart';
import '../l10n/app_locale_scope.dart';

class LanguageSwitchButton extends StatelessWidget {
  const LanguageSwitchButton({super.key});

  @override
  Widget build(BuildContext context) {
    final localeState = AppLocaleScope.of(context);
    final currentCode = localeState.locale.languageCode.toLowerCase();
    final label = AppStrings.tr(context, 'language');

    return PopupMenuButton<String>(
      tooltip: label,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
        child: Text(
          '${currentCode.toUpperCase()} | $label',
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
      ),
      onSelected: (value) {
        localeState.setLocale(Locale(value));
      },
      itemBuilder: (context) => [
        PopupMenuItem<String>(
          value: 'rw',
          child: Text(
            '${currentCode == 'rw' ? '[x]' : '[ ]'} ${AppStrings.tr(context, 'kinyarwanda')}',
          ),
        ),
        PopupMenuItem<String>(
          value: 'en',
          child: Text(
            '${currentCode == 'en' ? '[x]' : '[ ]'} ${AppStrings.tr(context, 'english')}',
          ),
        ),
      ],
    );
  }
}
