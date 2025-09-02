// lib/screens/auth_screen.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import '../locale_provider.dart';
import '../widgets/app_header.dart';
import '../l10n/app_localizations.dart';
import 'main_screen.dart';

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final TextEditingController phoneController = TextEditingController();
  final TextEditingController otpController = TextEditingController();
  String? _verificationId;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _googleSignIn() async {
    setState(() => _loading = true);
    final ok = await context.read<UserProvider>().loginWithGoogle();
    setState(() => _loading = false);
    if (ok && mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const MainScreen()),
      );
    } else {
      final loc = AppLocalizations.of(context)!;
      _showError(loc.googleSignInFailed);
    }
  }

  Future<void> _sendOtp() async {
    final phone = phoneController.text.trim();
    final loc = AppLocalizations.of(context)!;
    if (phone.isEmpty) {
      _showError(loc.enterPhoneNumber);
      return;
    }
    setState(() => _loading = true);
    try {
      await context.read<UserProvider>().startPhoneOtp(
            phoneNumber: phone,
            codeSent: (verificationId) {
              setState(() {
                _verificationId = verificationId;
              });
            },
            verificationFailed: (e) {
              _showError(e.message ?? loc.phoneVerificationFailed);
            },
          );
    } catch (e) {
      _showError('${loc.errorSendingOtp} $e');
    }
    setState(() => _loading = false);
  }

  Future<void> _verifyOtp() async {
    final otp = otpController.text.trim();
    final loc = AppLocalizations.of(context)!;
    if (_verificationId == null) {
      _showError(loc.pleaseSendOtpFirst);
      return;
    }
    if (otp.isEmpty) {
      _showError(loc.enterOtpPrompt);
      return;
    }
    setState(() => _loading = true);
    final ok = await context.read<UserProvider>().verifyOtpAndSignIn(
          verificationId: _verificationId!,
          smsCode: otp,
        );
    setState(() => _loading = false);
    if (ok && mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const MainScreen()),
      );
    } else {
      _showError(loc.otpVerificationFailed);
    }
  }

  void _guestLogin() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const MainScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final userProvider = context.watch<UserProvider>();
    final localeProvider = Provider.of<LocaleProvider>(context);
    final loc = AppLocalizations.of(context)!;

    // Auto-redirect if logged in
    if (userProvider.isLoggedIn) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const MainScreen()),
        );
      });
    }

    return Scaffold(
      backgroundColor: const Color(0xFF1A3C5A), // Blue bg
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(130),
        child: SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AppHeader(showProfileButton: false),
              TabBar(
                controller: _tabController,
                tabs: [
                  Tab(
                    icon: const Icon(Icons.login, color: Colors.white),
                    child: Text(loc.signInWithGoogle, style: const TextStyle(color: Colors.white)),
                  ),
                  Tab(
                    icon: const Icon(Icons.phone, color: Colors.white),
                    child: Text(loc.phoneNumber, style: const TextStyle(color: Colors.white)),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Colors.orange))
          : TabBarView(
              controller: _tabController,
              children: [
                // Google sign-in tab
                Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      ElevatedButton.icon(
                        icon: const Icon(Icons.login),
                        label: Text(loc.signInWithGoogle),
                        onPressed: _googleSignIn,
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        icon: const Icon(Icons.person_outline),
                        label: Text(loc.guestLogin ?? "Log in as Guest"),
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.grey.shade700),
                        onPressed: _guestLogin,
                      ),
                    ],
                  ),
                ),

                // Phone sign-in tab
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        TextField(
                          controller: phoneController,
                          keyboardType: TextInputType.phone,
                          style: const TextStyle(color: Colors.white),
                          decoration: InputDecoration(
                            labelText: loc.phoneNumber,
                            labelStyle: const TextStyle(color: Colors.white70),
                            border: const OutlineInputBorder(),
                            enabledBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.orange.shade300),
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderSide: BorderSide(color: Colors.orange.shade600),
                            ),
                            hintText: '+91xxxxxxxxxx',
                            hintStyle: const TextStyle(color: Colors.white54),
                          ),
                        ),
                        const SizedBox(height: 12),
                        ElevatedButton(
                          onPressed: _sendOtp,
                          child: Text(loc.sendOtp),
                        ),
                        if (_verificationId != null) ...[
                          const SizedBox(height: 12),
                          TextField(
                            controller: otpController,
                            keyboardType: TextInputType.number,
                            style: const TextStyle(color: Colors.white),
                            decoration: InputDecoration(
                              labelText: loc.enterOtpPrompt,
                              labelStyle: const TextStyle(color: Colors.white70),
                              border: const OutlineInputBorder(),
                              enabledBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Colors.orange.shade300),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderSide: BorderSide(color: Colors.orange.shade600),
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          ElevatedButton(
                            onPressed: _verifyOtp,
                            child: Text(loc.verifyOtp),
                          ),
                        ],
                        const SizedBox(height: 16),
                        ElevatedButton.icon(
                          icon: const Icon(Icons.person_outline),
                          label: Text(loc.guestLogin ?? "Log in as Guest"),
                          style: ElevatedButton.styleFrom(backgroundColor: Colors.grey.shade700),
                          onPressed: _guestLogin,
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}
