import 'package:flutter/material.dart';
import 'welcome.dart';
import 'sign_up.dart';
import 'sign_in.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kamgar Sahayak',
      theme: ThemeData(primarySwatch: Colors.orange),
      debugShowCheckedModeBanner: false,
      initialRoute: '/welcome',
      routes: {
        '/welcome': (context) => const WelcomeScreen(),
        '/signup': (context) => const SignUp(),
        '/signin': (context) => const SignIn(),
      },
    );
  }
}
