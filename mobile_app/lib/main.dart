import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:io';
import 'dart:ui';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'screens/home_screen.dart';
import 'l10n/app_strings.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  FlutterError.onError = (FlutterErrorDetails details) {
    _logCrash(
      'FlutterError',
      details.exceptionAsString(),
      details.stack,
    );
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

class CropDoctorApp extends StatelessWidget {
  const CropDoctorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      onGenerateTitle: (context) => AppStrings.tr(context, 'appTitle'),
      debugShowCheckedModeBanner: false,
      supportedLocales: const [Locale('en'), Locale('rw')],
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
      home: const HomeScreen(),
    );
  }
}
