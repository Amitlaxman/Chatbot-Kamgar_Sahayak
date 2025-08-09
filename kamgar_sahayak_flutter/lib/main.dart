import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'locale_provider.dart';
import '../l10n/app_localizations.dart';
import 'screens/legal_rights_screen.dart';
import 'screens/main_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/welcome_screen.dart';
import 'screens/sign_in_screen.dart';


void main() {
  runApp(
    ChangeNotifierProvider(
      create: (context) => LocaleProvider(),
      child: MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Consumer<LocaleProvider>(
      builder: (context, localeProvider, child) {
        return MaterialApp(
          title: 'Kamgar Sahayak',
          locale: localeProvider.locale,
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: WelcomeScreen(),
          routes: {
            '/main': (context) => MainScreen(),
            '/legal_rights': (context) => LegalRightsScreen(),
            '/sign_in': (context) => SignInScreen(),
            '/profile': (context) => ProfileScreen(),
          },
        );
      },
    );
  }
}