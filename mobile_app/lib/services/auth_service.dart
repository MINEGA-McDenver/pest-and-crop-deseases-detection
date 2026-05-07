import 'package:flutter/services.dart';
import 'package:local_auth/local_auth.dart';

class AuthAttempt {
  final bool authenticated;
  final String? errorCode;

  const AuthAttempt({required this.authenticated, this.errorCode});
}

class AuthService {
  final LocalAuthentication _auth = LocalAuthentication();

  Future<bool> isSupported() async {
    try {
      return await _auth.isDeviceSupported();
    } catch (_) {
      return false;
    }
  }

  Future<AuthAttempt> authenticate({required String reason}) async {
    try {
      final didAuthenticate = await _auth.authenticate(
        localizedReason: reason,
        options: const AuthenticationOptions(
          biometricOnly: false,
          stickyAuth: true,
          useErrorDialogs: true,
        ),
      );
      return AuthAttempt(authenticated: didAuthenticate);
    } on PlatformException catch (e) {
      return AuthAttempt(authenticated: false, errorCode: e.code);
    } catch (_) {
      return const AuthAttempt(authenticated: false);
    }
  }
}
