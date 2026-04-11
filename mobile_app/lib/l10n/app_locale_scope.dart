import 'package:flutter/material.dart';

class AppLocaleScope extends InheritedWidget {
  final Locale locale;
  final Future<void> Function(Locale locale) setLocale;

  const AppLocaleScope({
    super.key,
    required this.locale,
    required this.setLocale,
    required super.child,
  });

  static AppLocaleScope of(BuildContext context) {
    final scope = context.dependOnInheritedWidgetOfExactType<AppLocaleScope>();
    assert(scope != null, 'AppLocaleScope not found in context');
    return scope!;
  }

  static AppLocaleScope? maybeOf(BuildContext context) {
    return context.dependOnInheritedWidgetOfExactType<AppLocaleScope>();
  }

  @override
  bool updateShouldNotify(AppLocaleScope oldWidget) {
    return locale != oldWidget.locale;
  }
}
