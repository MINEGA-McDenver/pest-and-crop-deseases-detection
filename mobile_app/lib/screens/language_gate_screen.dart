import 'package:flutter/material.dart';
import '../l10n/app_strings.dart';
import 'home_screen.dart';

class LanguageGateScreen extends StatefulWidget {
  final Locale currentLocale;
  final Future<void> Function(Locale locale) onLocaleConfirmed;

  const LanguageGateScreen({
    super.key,
    required this.currentLocale,
    required this.onLocaleConfirmed,
  });

  @override
  State<LanguageGateScreen> createState() => _LanguageGateScreenState();
}

class _LanguageGateScreenState extends State<LanguageGateScreen> {
  static const bool _enableLocaleDebugLogs = true;
  late String _selectedCode;
  bool _submitting = false;

  @override
  void initState() {
    super.initState();
    _selectedCode = widget.currentLocale.languageCode.toLowerCase() == 'rw'
        ? 'rw'
        : 'en';
  }

  @override
  Widget build(BuildContext context) {
    String tr(String key, {Map<String, String>? args}) {
      return AppStrings.trCode(key, code: _selectedCode, args: args);
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          const DecoratedBox(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Color(0xFF153F28), Color(0xFF2E7D32), Color(0xFF77A95A)],
              ),
            ),
          ),
          Positioned(
            top: -70,
            left: -40,
            child: Container(
              width: 220,
              height: 220,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withValues(alpha: 0.08),
              ),
            ),
          ),
          Positioned(
            bottom: -90,
            right: -30,
            child: Container(
              width: 240,
              height: 240,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFFFFD54F).withValues(alpha: 0.18),
              ),
            ),
          ),
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 24, 24, 32),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
                  child: Container(
                    padding: const EdgeInsets.fromLTRB(20, 22, 20, 24),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.20),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.24),
                        width: 1.2,
                      ),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const Icon(
                          Icons.agriculture,
                          color: Colors.white,
                          size: 52,
                        ),
                        const SizedBox(height: 14),
                        Text(
                          tr('launchHeadline'),
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 28,
                            fontWeight: FontWeight.w800,
                            height: 1.2,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          tr('launchSubheadline'),
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.93),
                            fontSize: 15,
                            fontWeight: FontWeight.w500,
                            height: 1.35,
                          ),
                        ),
                        const SizedBox(height: 18),
                        Wrap(
                          alignment: WrapAlignment.center,
                          spacing: 8,
                          runSpacing: 8,
                          children: const [
                            _CropChip(icon: '🍌', label: 'Banana'),
                            _CropChip(icon: '🫘', label: 'Beans'),
                            _CropChip(icon: '🌽', label: 'Maize'),
                            _CropChip(icon: '🥔', label: 'Potato'),
                          ],
                        ),
                        const SizedBox(height: 22),
                        Text(
                          tr('chooseLanguage'),
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 24,
                            fontWeight: FontWeight.w400,
                          ),
                        ),
                        const SizedBox(height: 14),
                        _languageSelector(
                          code: 'en',
                          label: AppStrings.trCode('english', code: 'en').toUpperCase(),
                        ),
                        const SizedBox(height: 10),
                        _languageSelector(
                          code: 'rw',
                          label: AppStrings.trCode('kinyarwanda', code: 'rw').toUpperCase(),
                        ),
                        const SizedBox(height: 22),
                        SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            onPressed: _submitting ? null : _continue,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFFFFD54F),
                              foregroundColor: const Color(0xFF173322),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(30),
                              ),
                            ),
                            child: _submitting
                                ? const SizedBox(
                                    height: 22,
                                    width: 22,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2.2,
                                      color: Color(0xFF173322),
                                    ),
                                  )
                                : Text(
                                    tr('gettingStarted').toUpperCase(),
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w800,
                                      letterSpacing: 0.5,
                                    ),
                                  ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _languageSelector({required String code, required String label}) {
    final selected = _selectedCode == code;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => setState(() => _selectedCode = code),
        borderRadius: BorderRadius.circular(8),
        child: Container(
          height: 62,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: selected
                ? Colors.white
                : Colors.white.withValues(alpha: 0.10),
            border: Border.all(
              color: selected
                  ? const Color(0xFF173322)
                  : Colors.white.withValues(alpha: 0.65),
              width: selected ? 2 : 1.4,
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: selected ? const Color(0xFF1B5E20) : Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _continue() async {
    if (_enableLocaleDebugLogs) {
      debugPrint(
        '[LocaleDebug][LanguageGate][confirm] selected=$_selectedCode active=${AppStrings.activeLanguageCode}',
      );
    }
    setState(() => _submitting = true);
    try {
      await widget.onLocaleConfirmed(Locale(_selectedCode));
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const HomeScreen()),
      );
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }
}

class _CropChip extends StatelessWidget {
  final String icon;
  final String label;

  const _CropChip({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withValues(alpha: 0.22)),
      ),
      child: Text(
        '$icon $label',
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w600,
          fontSize: 12,
        ),
      ),
    );
  }
}