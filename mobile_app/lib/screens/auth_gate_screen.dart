import 'package:flutter/material.dart';

import '../l10n/app_strings.dart';
import '../services/auth_service.dart';

class AuthGateScreen extends StatefulWidget {
  final Widget child;
  final String languageCode;

  const AuthGateScreen({
    super.key,
    required this.child,
    required this.languageCode,
  });

  @override
  State<AuthGateScreen> createState() => _AuthGateScreenState();
}

class _AuthGateScreenState extends State<AuthGateScreen> {
  final AuthService _auth = AuthService();

  bool _unlocked = false;
  bool _authInProgress = false;
  String? _errorKey;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _unlock();
    });
  }

  String _tr(String key, {Map<String, String>? args}) {
    return AppStrings.trCode(key, code: widget.languageCode, args: args);
  }

  Future<void> _unlock() async {
    if (_authInProgress) return;

    setState(() {
      _authInProgress = true;
      _errorKey = null;
    });

    final supported = await _auth.isSupported();
    if (!supported) {
      if (!mounted) return;
      setState(() {
        _authInProgress = false;
        _errorKey = 'authUnavailable';
      });
      return;
    }

    final attempt = await _auth.authenticate(reason: _tr('authReasonApp'));
    if (!mounted) return;

    if (attempt.authenticated) {
      setState(() {
        _authInProgress = false;
        _unlocked = true;
      });
      return;
    }

    setState(() {
      _authInProgress = false;
      _unlocked = false;
      _errorKey = attempt.errorCode != null ? 'authUnavailable' : 'authFailed';
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_unlocked) return widget.child;

    final theme = Theme.of(context);

    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    Icons.lock,
                    size: 72,
                    color: theme.colorScheme.primary,
                  ),
                  const SizedBox(height: 18),
                  Text(
                    _tr('authUnlockTitle'),
                    textAlign: TextAlign.center,
                    style: theme.textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    _tr('authUnlockSubtitle'),
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurface.withValues(alpha: 0.75),
                      height: 1.25,
                    ),
                  ),
                  if (_errorKey != null) ...[
                    const SizedBox(height: 14),
                    Text(
                      _tr(_errorKey!),
                      textAlign: TextAlign.center,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.error,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                  const SizedBox(height: 20),
                  SizedBox(
                    width: double.infinity,
                    height: 52,
                    child: ElevatedButton(
                      onPressed: _authInProgress ? null : _unlock,
                      child: _authInProgress
                          ? const SizedBox(
                              height: 22,
                              width: 22,
                              child: CircularProgressIndicator(strokeWidth: 2.4),
                            )
                          : Text(
                              _tr('authUnlockButton').toUpperCase(),
                              style: const TextStyle(
                                fontWeight: FontWeight.w800,
                                letterSpacing: 0.4,
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
    );
  }
}
