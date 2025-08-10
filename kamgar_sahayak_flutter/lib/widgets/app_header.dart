import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../locale_provider.dart';

class AppHeader extends StatelessWidget {
  final bool showProfileButton;

  const AppHeader({super.key, this.showProfileButton = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Back button or profile button on left
          if (showProfileButton)
            IconButton(
              icon: const Icon(Icons.person, color: Colors.orange),
              onPressed: () {
                Navigator.pushNamed(context, '/profile');
              },
            )
          else
            IconButton(
              icon: const Icon(Icons.arrow_back, color: Colors.orange),
              onPressed: () {
                Navigator.pop(context);
              },
            ),

          // Centered logo
          Expanded(
            child: Center(
              child: Image.asset('assets/logo.png', height: 40),
            ),
          ),

          // Language toggle button
          Consumer<LocaleProvider>(
            builder: (context, localeProvider, _) {
              return TextButton(
                onPressed: () {
                  localeProvider.toggleLocale();
                },
                child: Text(
                  localeProvider.locale.languageCode == 'en' ? 'हिन्दी' : 'English',
                  style: const TextStyle(color: Colors.blue),
                ),
              );
            },
          ),
        ],
      ),
    );
  }
}
