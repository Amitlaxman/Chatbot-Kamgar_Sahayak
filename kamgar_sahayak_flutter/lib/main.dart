// lib/main.dart
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:kamgar_sahayak/screens/profile_screen.dart';
import 'package:provider/provider.dart';

import 'providers/user_provider.dart';
import 'locale_provider.dart';
import 'screens/welcome_screen.dart';
import 'screens/auth_screen.dart';
import 'screens/main_screen.dart';
import 'firebase_options.dart';
import 'l10n/app_localizations.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => UserProvider()),
        ChangeNotifierProvider(
          create: (_) => LocaleProvider(),
        ), // << add LocaleProvider here
      ],
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final localeProvider = Provider.of<LocaleProvider>(context);

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Kamgar Sahayak',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        // Optional: you can also customize orange accents here if you want consistency
        colorScheme: ColorScheme.fromSwatch(
          primarySwatch: Colors.blue,
        ).copyWith(secondary: Colors.orange),
      ),
      locale: localeProvider.locale, // << pass locale here to rebuild on change
      supportedLocales: const [Locale('en'), Locale('hi')],
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      home: const WelcomeScreen(),
      routes: {
        '/auth': (context) => const AuthScreen(),
        '/main': (context) => const MainScreen(),
        '/profile': (context) => const ProfileScreen(),
      },
    );
  }
}
