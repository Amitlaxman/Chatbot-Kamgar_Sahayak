import 'package:flutter/material.dart';
import '../l10n/app_localizations.dart';
import '../widgets/app_header.dart';

class SignInScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final localizations = AppLocalizations.of(context)!;

    return Scaffold(
      backgroundColor: Color(0xFF1A3C5A),
      body: SafeArea(
        child: Column(
          children: [
            AppHeader(),
            SizedBox(height: 20),
            Container(
              padding: EdgeInsets.all(16),
              margin: EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: Color(0xFF1A3C5A),
                border: Border.all(color: Colors.orange),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                children: [
                  Text(
                    localizations.signIn,
                    style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 20),
                  TextField(
                    decoration: InputDecoration(
                      hintText: localizations.name,
                      hintStyle: TextStyle(color: Colors.grey),
                      filled: true,
                      fillColor: Colors.white,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(5)),
                    ),
                  ),
                  SizedBox(height: 10),
                  TextField(
                    decoration: InputDecoration(
                      hintText: localizations.phoneNumber,
                      hintStyle: TextStyle(color: Colors.grey),
                      filled: true,
                      fillColor: Colors.white,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(5)),
                      suffixIcon: Icon(Icons.phone, color: Colors.orange),
                    ),
                  ),
                  SizedBox(height: 10),
                  TextField(
                    obscureText: true,
                    decoration: InputDecoration(
                      hintText: localizations.password,
                      hintStyle: TextStyle(color: Colors.grey),
                      filled: true,
                      fillColor: Colors.white,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(5)),
                      suffixIcon: Icon(Icons.visibility, color: Colors.orange),
                    ),
                  ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: () {},
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.orange,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    child: Text(localizations.signIn, style: TextStyle(color: Colors.white)),
                  ),
                  SizedBox(height: 10),
                  Text(localizations.or, style: TextStyle(color: Colors.white)),
                  SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: () {},
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    child: Text(localizations.continueAsGuest, style: TextStyle(color: Colors.blue[900])),
                  ),
                ],
              ),
            ),
            Spacer(),
            TextButton(
              onPressed: () {},
              child: Text(
                localizations.alreadyHaveAccount,
                style: TextStyle(color: Colors.blue),
              ),
            ),
            SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}