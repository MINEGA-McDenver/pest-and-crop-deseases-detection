import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:io';
import 'dart:ui';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'screens/auth_gate_screen.dart';
import 'screens/reauth_screen.dart';
import 'screens/language_gate_screen.dart';
import 'l10n/app_strings.dart';
import 'l10n/app_locale_scope.dart';
import 'services/language_preferences_service.dart';
import 'services/session_guard_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  FlutterError.onError = (FlutterErrorDetails details) {
    _logCrash('FlutterError', details.exceptionAsString(), details.stack);
  };

  PlatformDispatcher.instance.onError = (error, stack) {
    _logCrash('PlatformError', error.toString(), stack);
    return true;
  };

  runZonedGuarded(
    () => runApp(const CropDoctorApp()),
    (error, stack) => _logCrash('ZonedError', error.toString(), stack),
  );
}

Future<void> _logCrash(String source, String message, StackTrace? stack) async {
  try {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/crash.log');
    final entry =
        '[${DateTime.now().toIso8601String()}] $source\n$message\n${stack ?? ''}\n\n';
    await file.writeAsString(entry, mode: FileMode.append, flush: true);
  } catch (_) {
    // Keep app alive even if crash logging fails.
  }
}

class CropDoctorApp extends StatefulWidget {
  const CropDoctorApp({super.key});

  @override
  State<CropDoctorApp> createState() => _CropDoctorAppState();
}

class _CropDoctorAppState extends State<CropDoctorApp>
  with WidgetsBindingObserver {
  final LanguagePreferencesService _languagePrefs =
      LanguagePreferencesService();
  static const bool _enableLocaleDebugLogs = true;
  static const Duration _languageResetAfter = Duration(seconds: 30);

  final GlobalKey<NavigatorState> _navigatorKey = GlobalKey<NavigatorState>();

  Locale _locale = const Locale('rw');
  bool _loadedPreference = false;

  DateTime? _backgroundedAt;
  bool _resumeGateInProgress = false;

  void _debugLocale(String stage, {String? value}) {
    if (!_enableLocaleDebugLogs) return;
    debugPrint(
      '[LocaleDebug][App][$stage] locale=${_locale.languageCode} active=${AppStrings.activeLanguageCode} value=${value ?? '-'}',
    );
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadLocalePreference();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _onLifecycleState(state);
  }

  void _onLifecycleState(AppLifecycleState state) {
    switch (state) {
      case AppLifecycleState.resumed:
        WidgetsBinding.instance.addPostFrameCallback((_) {
          _handleResumeGate();
        });
        return;
      case AppLifecycleState.paused:
      case AppLifecycleState.detached:
      case AppLifecycleState.hidden:
        _backgroundedAt ??= DateTime.now();
        return;
      case AppLifecycleState.inactive:
        // Ignore transient inactive states (e.g., permission dialogs).
        return;
    }
  }

  Future<void> _handleResumeGate() async {
    if (_resumeGateInProgress) return;
    if (SessionGuardService.instance.isInExternalOperation) {
      _backgroundedAt = null;
      return;
    }

    final leftAt = _backgroundedAt;
    if (leftAt == null) return;

    _resumeGateInProgress = true;
    _backgroundedAt = null;

    try {
      final elapsed = DateTime.now().difference(leftAt);
      final navigator = _navigatorKey.currentState;
      if (navigator == null) return;

      if (elapsed > _languageResetAfter) {
        navigator.pushAndRemoveUntil(
          MaterialPageRoute(builder: (_) => _buildAuthThenLanguageGate()),
          (route) => false,
        );
        return;
      }

      await navigator.push<bool>(
        MaterialPageRoute(
          builder: (_) => ReauthScreen(
            languageCode: _locale.languageCode.toLowerCase(),
          ),
        ),
      );
    } finally {
      _resumeGateInProgress = false;
    }
  }

  Widget _buildAuthThenLanguageGate() {
    return AuthGateScreen(
      languageCode: _locale.languageCode.toLowerCase(),
      child: LanguageGateScreen(
        currentLocale: _locale,
        onLocaleConfirmed: _setLocale,
      ),
    );
  }

  Future<void> _loadLocalePreference() async {
    String? savedCode;
    try {
      savedCode = await _languagePrefs.getSavedLanguageCode();
    } catch (_) {
      savedCode = null;
    }
    if (!mounted) return;

    final resolvedCode = AppStrings.normalizeLanguageCode(savedCode ?? 'rw');
    setState(() {
      _locale = Locale(resolvedCode);
      _loadedPreference = true;
    });
    AppStrings.setActiveLanguageCode(resolvedCode);
    _debugLocale('loadPreference', value: resolvedCode);
  }

  Future<void> _setLocale(Locale locale) async {
    final normalized = AppStrings.normalizeLanguageCode(locale.languageCode);
    _debugLocale('setLocale.start', value: normalized);
    if (_locale.languageCode.toLowerCase() == normalized) {
      AppStrings.setActiveLanguageCode(normalized);
      await _languagePrefs.saveLanguageCode(normalized);
      _debugLocale('setLocale.noop', value: normalized);
      return;
    }

    setState(() {
      _locale = Locale(normalized);
    });
    AppStrings.setActiveLanguageCode(normalized);
    await _languagePrefs.saveLanguageCode(normalized);
    _debugLocale('setLocale.done', value: normalized);
  }

  @override
  Widget build(BuildContext context) {
    if (!_loadedPreference) {
      return const MaterialApp(
        debugShowCheckedModeBanner: false,
        home: Scaffold(body: Center(child: CircularProgressIndicator())),
      );
    }

    return AppLocaleScope(
      locale: _locale,
      setLocale: _setLocale,
      child: MaterialApp(
        navigatorKey: _navigatorKey,
        locale: const Locale('en'),
        onGenerateTitle: (context) => AppStrings.tr(context, 'appTitle'),
        debugShowCheckedModeBanner: false,
        supportedLocales: const [Locale('en')],
        localizationsDelegates: const [
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        theme: ThemeData(
          colorSchemeSeed: const Color(0xFF2E7D32),
          useMaterial3: true,
          scaffoldBackgroundColor: const Color(0xFFF5F5F5),
          appBarTheme: const AppBarTheme(centerTitle: true, elevation: 0),
          cardTheme: CardThemeData(
            elevation: 1,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
        home: _buildAuthThenLanguageGate(),
      ),
    );
  }
}
