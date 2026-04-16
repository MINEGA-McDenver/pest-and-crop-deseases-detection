import 'dart:async';

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
  static const Duration _autoSlideInterval = Duration(seconds: 5);

  final PageController _pageController = PageController();

  static const List<_LanguageSlide> _slides = [
    _LanguageSlide(
      imagePath: 'assets/images/language_slide_potato.jpg',
      titleKey: 'languageSlide2Title',
      subtitleKey: 'languageSlide2Subtitle',
    ),
    _LanguageSlide(
      imagePath: 'assets/images/language_slide_maize.jpg',
      titleKey: 'languageSlide3Title',
      subtitleKey: 'languageSlide3Subtitle',
    ),
    _LanguageSlide(
      imagePath: 'assets/images/language_slide_beans.jpg',
      titleKey: 'languageSlide4Title',
      subtitleKey: 'languageSlide4Subtitle',
    ),
    _LanguageSlide(
      imagePath: 'assets/images/language_slide_banana.jpg',
      titleKey: 'languageSlide5Title',
      subtitleKey: 'languageSlide5Subtitle',
    ),
  ];

  late String _selectedCode;
  late int _currentSlideIndex;
  Timer? _autoSlideTimer;
  bool _submitting = false;

  @override
  void initState() {
    super.initState();
    _selectedCode = widget.currentLocale.languageCode.toLowerCase() == 'rw'
        ? 'rw'
        : 'en';
    _currentSlideIndex = 0;
    _autoSlideTimer = Timer.periodic(_autoSlideInterval, (_) {
      if (!mounted || !_pageController.hasClients || _slides.isEmpty) return;
      final next = (_currentSlideIndex + 1) % _slides.length;
      _pageController.animateToPage(
        next,
        duration: const Duration(milliseconds: 450),
        curve: Curves.easeInOut,
      );
    });
  }

  @override
  void dispose() {
    _autoSlideTimer?.cancel();
    _pageController.dispose();
    super.dispose();
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
          PageView.builder(
            controller: _pageController,
            itemCount: _slides.length,
            onPageChanged: (index) {
              setState(() {
                _currentSlideIndex = index;
              });
            },
            itemBuilder: (context, index) {
              final slide = _slides[index];
              return Image.asset(
                slide.imagePath,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) {
                  return const DecoratedBox(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [
                          Color(0xFF1B5E20),
                          Color(0xFF2E7D32),
                          Color(0xFF4CAF50),
                        ],
                      ),
                    ),
                  );
                },
              );
            },
          ),
          Container(
            color: Colors.black.withValues(alpha: 0.45),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(24, 32, 24, 28),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 16),
                  Text(
                    AppStrings.trCode('appTitle', code: 'en'),
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 34,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0.6,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    tr(_slides[_currentSlideIndex].titleKey),
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.95),
                      fontSize: 17,
                      fontWeight: FontWeight.w600,
                      height: 1.25,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    tr(_slides[_currentSlideIndex].subtitleKey),
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.9),
                      fontSize: 14,
                      fontWeight: FontWeight.w400,
                      height: 1.3,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    tr('chooseLanguage'),
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 34,
                      fontWeight: FontWeight.w300,
                    ),
                  ),
                  const SizedBox(height: 14),
                  _buildLanguageSegmentedControl(),
                  const SizedBox(height: 26),
                  SizedBox(
                    height: 56,
                    child: OutlinedButton(
                      onPressed: _submitting ? null : _continue,
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(
                          color: Colors.white.withValues(alpha: 0.95),
                          width: 2,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(28),
                        ),
                        foregroundColor: Colors.white,
                      ),
                      child: _submitting
                          ? const SizedBox(
                              height: 22,
                              width: 22,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.2,
                                color: Colors.white,
                              ),
                            )
                          : Text(
                              tr('gettingStarted').toUpperCase(),
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 0.5,
                              ),
                            ),
                    ),
                  ),
                  const SizedBox(height: 18),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(_slides.length, (index) {
                      final isActive = index == _currentSlideIndex;
                      return AnimatedContainer(
                        duration: const Duration(milliseconds: 250),
                        margin: const EdgeInsets.symmetric(horizontal: 5),
                        width: isActive ? 12 : 9,
                        height: isActive ? 12 : 9,
                        decoration: BoxDecoration(
                          color: isActive
                              ? const Color(0xFF4CAF50)
                              : Colors.white,
                          shape: BoxShape.circle,
                        ),
                      );
                    }),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLanguageSegmentedControl() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: Colors.white, width: 2),
      ),
      child: Row(
        children: [
          Expanded(
            child: _languageSegment(
              code: 'en',
              label: AppStrings.trCode('english', code: 'en').toUpperCase(),
            ),
          ),
          Expanded(
            child: _languageSegment(
              code: 'rw',
              label: AppStrings.trCode('kinyarwanda', code: 'rw').toUpperCase(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _languageSegment({required String code, required String label}) {
    final selected = _selectedCode == code;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => setState(() => _selectedCode = code),
        child: Container(
          height: 64,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: selected ? const Color(0xFF1E1E1E) : Colors.white,
          ),
          child: Text(
            label,
            style: TextStyle(
              color: selected ? Colors.white : const Color(0xFF1E3A2D),
              fontSize: 20,
              fontWeight: FontWeight.w600,
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

class _LanguageSlide {
  final String imagePath;
  final String titleKey;
  final String subtitleKey;

  const _LanguageSlide({
    required this.imagePath,
    required this.titleKey,
    required this.subtitleKey,
  });
}
