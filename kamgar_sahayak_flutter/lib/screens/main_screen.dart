// lib/screens/main_screen.dart
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../widgets/app_header.dart';
import 'legal_rights_screen.dart';
import 'chat_screen.dart'; // <-- Import ChatScreen

class MainScreen extends StatelessWidget {
  const MainScreen({super.key});

  // Map of document titles to their PDF URLs
final Map<String, String> documentLinks = const {
  'CIS Information': 'https://drive.google.com/uc?export=download&id=1WsV2OLdwgkssW4duoRdt2zcDR6_dJzay',
  'CM Sambal Scheme': 'https://drive.google.com/uc?export=download&id=1dKlApGsl5pxPgKfmey3yaYokPLVP2tEY',
  'Minimum Wages - Working Professionals': 'https://drive.google.com/uc?export=download&id=17qDIMmv3w7V-cGXkoSIPTgVWeK2w9oBX',
  'Minimum Wages Act': 'https://drive.google.com/uc?export=download&id=1oBUd7zc5mu3xkkBjnBFu5a3eL8iT0evZ',
  'National Database': 'https://drive.google.com/uc?export=download&id=1s3EYIolVVST13-rBlUbQERvF9FnUJxFb',
  'Shops & Establishments Act, 1958': 'https://drive.google.com/uc?export=download&id=1j1aqg27efJ5SIl5kq0cOFP3o1MjnywWd',
  'Labour Welfare Act - Part 10': 'https://drive.google.com/uc?export=download&id=1nj8EyEQ6k3hGqLYWuM0SpDJpKBRa4HPR',
  'Labour Welfare Act - Part 11': 'https://drive.google.com/uc?export=download&id=1K_3BJ227T1KwdbNzBpCgyDBRvDSXeYyW',
  'Labour Welfare Act - Part 12': 'https://drive.google.com/uc?export=download&id=1YynfJup-XfYgp9RjpZzcql1Euq9zx5oB',
  'Labour Welfare Act - Part 13': 'https://drive.google.com/uc?export=download&id=1Dn0jY7xLAqRdGYnUvmf0n-ndVvH-PAx-',
  'Labour Welfare Act - Part 14': 'https://drive.google.com/uc?export=download&id=1EktWVFGXa7M9CSCZPVBB8fNx1WvpJUZm',
  'Labour Welfare Act - Part 15': 'https://drive.google.com/uc?export=download&id=1YKecYOEiZpBUP6C0dQAH3_GH8cXkm3Ao',
  'Labour Welfare Act - Part 16': 'https://drive.google.com/uc?export=download&id=1JB5W8r1fnGaR2L-rPp-1ATUbaQENzltQ',
  'Labour Welfare Act - Part 17': 'https://drive.google.com/uc?export=download&id=19IP5IWUYjzdN12_d9XONTZCX7XScley8',
  'Labour Welfare Act - Part 2': 'https://drive.google.com/uc?export=download&id=18hgNfnNQbORbU91gk6lh4EhIAfaQP3pr',
  'Labour Welfare Act - Part 3': 'https://drive.google.com/uc?export=download&id=14rC845WPCrj6-nmIxR_OGQEC83cuzQk-',
  'Labour Welfare Act - Part 4': 'https://drive.google.com/uc?export=download&id=134sbLm8Mp5aS3xeNqgPPdIxvCTwZUA_8',
  'Labour Welfare Act - Part 5': 'https://drive.google.com/uc?export=download&id=1r7gj2JiDzNr2jIuQ2v8MKjPh6RwnG12K',
  'Labour Welfare Act - Part 6': 'https://drive.google.com/uc?export=download&id=1cDbkj2FKgMeV0SOrUubdExhWngUH2Reg',
  'Labour Welfare Act - Part 7': 'https://drive.google.com/uc?export=download&id=14Sx4LosMVqgohfy6EsbvgKL1a9KK7zBG',
  'Labour Welfare Act - Part 8': 'https://drive.google.com/uc?export=download&id=1Qki9yTvQttcoNYmgrSqpeV30GyQ99HCE',
  'Labour Welfare Act - Part 9': 'https://drive.google.com/uc?export=download&id=1igcLqFkgxq4pL7QK2MaLAb56YI-0q6lq',
  'DIHS Guidelines': 'https://drive.google.com/uc?export=download&id=1MU2wnUO-FHDGNhGv6NTWxLrcHdmhX9pY',
  'Outsourced Labour Policy': 'https://drive.google.com/uc?export=download&id=1mPcsR4upgMOux4_QGTUnvUqcmPICzlbS',
  'Additional Labour Guidelines': 'https://drive.google.com/uc?export=download&id=14UKzujyG1a_jgCoLnLyMnY9n-VfWHWL9',
  'Labour Act - AXD Form': 'https://drive.google.com/uc?export=download&id=1TC7I5dsQ_IwhwubnbmnwAoCC9OFnxP5F',
  'Yuva Sangam Program': 'https://drive.google.com/uc?export=download&id=1lxR0qwNc5NhJbMnEKupYlK8Ah_UrKElB',
  'Consumer Price Index': 'https://drive.google.com/uc?export=download&id=1N4_qSMKTih2mc6oGlgxt6eFsbFaMVU7z',
  'Fourth Class Post Seniority List': 'https://drive.google.com/uc?export=download&id=1zN36myfflPrtVUooUbZk2X5wFvlu76rp',
  'Time Scale Pay': 'https://drive.google.com/uc?export=download&id=16SwqaSQ_5ZPlgs6gwsCcmcYCTPBcpWyK',
};

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
                _actionBtn(context, 'Live Chat', Icons.chat, () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const ChatScreen()), // <-- Route to ChatScreen
                  );
                }),
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
                children: documentLinks.entries
                    .map((entry) => _docRow(context, entry.key, entry.value))
                    .toList(),
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

  Widget _docRow(BuildContext context, String title, String url) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
      child: Row(
        children: [
          Expanded(
            child: GestureDetector(
              onTap: () => _launchURL(url),
              child: Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  decoration: TextDecoration.underline,
                ),
              ),
            ),
          ),
          IconButton(icon: const Icon(Icons.download, color: Colors.white), onPressed: () => _launchURL(url)),
          IconButton(icon: const Icon(Icons.visibility, color: Colors.white), onPressed: () => _launchURL(url)),
        ],
      ),
    );
  }

  Future<void> _launchURL(String url) async {
    final Uri uri = Uri.parse(url);
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      throw 'Could not launch $url';
    }
  }
}