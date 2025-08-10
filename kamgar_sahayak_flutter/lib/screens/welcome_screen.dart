// lib/screens/welcome_screen.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../locale_provider.dart';
import '../l10n/app_localizations.dart';

class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final localeProvider = Provider.of<LocaleProvider>(context);
    final loc = AppLocalizations.of(context)!;

    return Scaffold(
      backgroundColor: const Color(0xFF1A3C5A),
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset('assets/logo.png', height: 120),
              const SizedBox(height: 24),
              Text(
                loc.appTitle,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                loc.legalAdvice,
                style: const TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () {
                  Navigator.pushReplacementNamed(context, '/auth');
                },
                child: Text(loc.getStarted),
              ),
              const SizedBox(height: 20),
              TextButton.icon(
                onPressed: () {
                  localeProvider.toggleLocale();
                },
                icon: const Icon(Icons.language, color: Colors.white),
                label: Text(
                  localeProvider.locale.languageCode == 'en'
                      ? 'हिन्दी'
                      : 'English',
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
