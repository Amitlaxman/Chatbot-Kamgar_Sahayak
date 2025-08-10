// lib/utils/auth_utils.dart
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:google_sign_in/google_sign_in.dart';

final FirebaseAuth _auth = FirebaseAuth.instance;

/// Sign in with Google. Returns Firebase [User] on success, null on cancel.
Future<User?> signInWithGoogle() async {
  if (kIsWeb) {
    // Web: use popup
    final GoogleAuthProvider provider = GoogleAuthProvider();
    UserCredential credential = await _auth.signInWithPopup(provider);
    return credential.user;
  } else {
    final GoogleSignIn googleSignIn = GoogleSignIn(scopes: ['email', 'profile']);
    final GoogleSignInAccount? googleUser = await googleSignIn.signIn();
    if (googleUser == null) return null; // user cancelled
    final GoogleSignInAuthentication googleAuth = await googleUser.authentication;
    final credential = GoogleAuthProvider.credential(
      idToken: googleAuth.idToken,
      accessToken: googleAuth.accessToken,
    );
    final UserCredential userCredential = await _auth.signInWithCredential(credential);
    return userCredential.user;
  }
}

/// Start phone verification. Provide [phoneNumber] in international format (+<country><number>).
/// [codeSent] receives verificationId to be used for final OTP verification.
Future<void> startPhoneVerification({
  required String phoneNumber,
  required void Function(String verificationId) codeSent,
  required void Function(FirebaseAuthException e) verificationFailed,
}) async {
  await _auth.verifyPhoneNumber(
    phoneNumber: phoneNumber,
    timeout: const Duration(seconds: 60),
    verificationCompleted: (PhoneAuthCredential credential) async {
      // Automatic handling on Android (instant or auto-retrieval)
      await _auth.signInWithCredential(credential);
    },
    verificationFailed: verificationFailed,
    codeSent: (String verificationId, int? resendToken) {
      codeSent(verificationId);
    },
    codeAutoRetrievalTimeout: (String verificationId) {},
  );
}

/// Verify OTP using [verificationId] and [smsCode], signs in the user.
Future<User?> verifySmsCode({
  required String verificationId,
  required String smsCode,
}) async {
  PhoneAuthCredential credential = PhoneAuthProvider.credential(
    verificationId: verificationId,
    smsCode: smsCode,
  );
  final userCredential = await _auth.signInWithCredential(credential);
  return userCredential.user;
}

/// Sign out Google + Firebase
Future<void> signOutAll() async {
  try {
    if (!kIsWeb) {
      await GoogleSignIn().signOut();
    }
  } catch (_) {}
  await _auth.signOut();
}
