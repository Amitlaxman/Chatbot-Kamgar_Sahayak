import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../locale_provider.dart';
import '../l10n/app_localizations.dart';

class AppHeader extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final localeProvider = Provider.of<LocaleProvider>(context);
    return Container(
      color: Colors.white,
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          IconButton(
            icon: Icon(Icons.arrow_back, color: Colors.orange),
            onPressed: () {
              Navigator.pop(context);
            },
          ),
          Image.asset('assets/logo.png', height: 40),
          TextButton(
            onPressed: () {
              localeProvider.toggleLocale();
            },
            child: Text(
              localeProvider.locale.languageCode == 'en' ? 'हिन्दी' : 'English',
              style: TextStyle(color: Colors.blue),
            ),
          ),
        ],
      ),
    );
  }
}