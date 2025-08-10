// lib/screens/main_screen.dart
import 'package:flutter/material.dart';
import 'package:kamgar_sahayak/locale_provider.dart';
import 'package:provider/provider.dart';
import '../widgets/app_header.dart';
import 'legal_rights_screen.dart';

class MainScreen extends StatelessWidget {
  const MainScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A3C5A),
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60),
        child: const AppHeader(
          showProfileButton: true,
        ),
      ),

      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 20),
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(color: Colors.orange),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
                        Text(
                          'Kamgar Sahayak',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.black,
                          ),
                        ),
                        SizedBox(height: 6),
                        Text(
                          'Legal advice for Everyone.',
                          style: TextStyle(fontSize: 14, color: Colors.black),
                        ),
                      ],
                    ),
                  ),
                  Image.asset('assets/illustration.png', height: 60),
                ],
              ),
            ),
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _actionBtn(context, 'Live Chat', Icons.chat, () {}),
                _actionBtn(context, 'Your Rights', Icons.book, () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const LegalRightsScreen()),
                  );
                }),
              ],
            ),
            const SizedBox(height: 20),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: TextField(
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Colors.white,
                  hintText: 'Search Legal Document...',
                  prefixIcon: const Icon(Icons.filter_list),
                  suffixIcon: const Icon(Icons.search),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Container(
              color: Colors.white,
              width: double.infinity,
              padding: const EdgeInsets.all(8),
              child: const Text(
                'Recently Searched Documents',
                style: TextStyle(color: Colors.blue),
              ),
            ),
            Expanded(
              child: ListView(
                children: [
                  _docRow(context, 'PAYMENT OF WAGES...'),
                  _docRow(context, 'MINIMUM WAGES A...'),
                  _docRow(context, 'THE CODE ON WA...'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _actionBtn(BuildContext context, String label, IconData icon, VoidCallback onTap) {
    return Column(
      children: [
        CircleAvatar(
          radius: 30,
          backgroundColor: Colors.white,
          child: IconButton(icon: Icon(icon, color: Colors.orange), onPressed: onTap),
        ),
        const SizedBox(height: 6),
        Text(label, style: const TextStyle(color: Colors.white)),
      ],
    );
  }

  Widget _docRow(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
      child: Row(
        children: [
          Expanded(child: Text(title, style: const TextStyle(color: Colors.white))),
          IconButton(icon: const Icon(Icons.download, color: Colors.white), onPressed: () {}),
          IconButton(icon: const Icon(Icons.visibility, color: Colors.white), onPressed: () {}),
        ],
      ),
    );
  }
}
