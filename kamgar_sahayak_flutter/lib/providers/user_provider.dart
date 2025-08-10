// lib/providers/user_provider.dart
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../utils/auth_utils.dart';

class UserProvider extends ChangeNotifier {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  User? _user;
  bool _initialized = false;

  User? get user => _user;
  bool get isLoggedIn => _user != null;
  bool get isInitialized => _initialized;

  UserProvider() {
    _init();
  }

  Future<void> _init() async {
    // Listen firebase auth changes
    _auth.authStateChanges().listen((u) async {
      _user = u;
      if (u != null) {
        // persist uid
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('uid', u.uid);
      } else {
        final prefs = await SharedPreferences.getInstance();
        await prefs.remove('uid');
      }
      _initialized = true;
      notifyListeners();
    });

    // load if firebase already has user
    _user = _auth.currentUser;
    if (_user != null) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('uid', _user!.uid);
    }
    _initialized = true;
    notifyListeners();
  }

  /// Google login wrapper
  Future<bool> loginWithGoogle() async {
    try {
      final u = await signInWithGoogle();
      if (u != null) {
        _user = u;
        notifyListeners();
        return true;
      }
    } catch (e) {
      debugPrint('loginWithGoogle error: $e');
    }
    return false;
  }

  /// Start phone OTP (sends code)
  Future<void> startPhoneOtp({
    required String phoneNumber,
    required void Function(String verificationId) codeSent,
    required void Function(FirebaseAuthException e) verificationFailed,
  }) async {
    await startPhoneVerification(
      phoneNumber: phoneNumber,
      codeSent: codeSent,
      verificationFailed: verificationFailed,
    );
  }

  /// Verify OTP and sign in
  Future<bool> verifyOtpAndSignIn({
    required String verificationId,
    required String smsCode,
  }) async {
    try {
      final u = await verifySmsCode(verificationId: verificationId, smsCode: smsCode);
      if (u != null) {
        _user = u;
        notifyListeners();
        return true;
      }
    } catch (e) {
      debugPrint('verifyOtpAndSignIn error: $e');
    }
    return false;
  }

  /// Sign out for logout action
  Future<void> signOutUser() async {
    await signOutAll();
    _user = null;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('uid');
  }
}
