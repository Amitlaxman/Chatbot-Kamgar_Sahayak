import 'package:flutter/material.dart';
import '../l10n/app_localizations.dart';
import '../widgets/app_header.dart';

class MainScreen extends StatelessWidget {
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
                color: Colors.white,
                border: Border.all(color: Colors.orange),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          localizations.appTitle,
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.black,
                          ),
                        ),
                        SizedBox(height: 5),
                        Text(
                          localizations.legalAdvice,
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.black,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Image.asset('assets/illustration.png', height: 60),
                ],
              ),
            ),
            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildActionButton(context, localizations.liveChat, Icons.chat, () {}),
                _buildActionButton(context, localizations.yourRights, Icons.book, () {
                  Navigator.pushNamed(context, '/legal_rights');
                }),
              ],
            ),
            SizedBox(height: 20),
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: TextField(
                decoration: InputDecoration(
                  hintText: localizations.searchLegalDocument,
                  hintStyle: TextStyle(color: Colors.grey),
                  prefixIcon: Icon(Icons.filter_list, color: Colors.grey),
                  suffixIcon: Icon(Icons.search, color: Colors.grey),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                    borderSide: BorderSide(color: Colors.grey),
                  ),
                ),
              ),
            ),
            SizedBox(height: 20),
            Container(
              color: Colors.blue[900],
              padding: EdgeInsets.all(8),
              child: Text(
                localizations.recentlySearchedDocuments,
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
            Expanded(
              child: ListView(
                children: [
                  _buildDocumentRow(context, 'PAYMENT OF WAGES...'),
                  _buildDocumentRow(context, 'MINIMUM WAGES A...'),
                  _buildDocumentRow(context, 'THE CODE ON WA...'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButton(BuildContext context, String text, IconData icon, VoidCallback onPressed) {
    return Column(
      children: [
        CircleAvatar(
          radius: 30,
          backgroundColor: Colors.white,
          child: IconButton(
            icon: Icon(icon, color: Colors.orange, size: 30),
            onPressed: onPressed,
          ),
        ),
        SizedBox(height: 5),
        Text(text, style: TextStyle(color: Colors.white)),
      ],
    );
  }

  Widget _buildDocumentRow(BuildContext context, String documentName) {
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(child: Text(documentName, style: TextStyle(color: Colors.white))),
          IconButton(icon: Icon(Icons.download, color: Colors.white), onPressed: () {}),
          IconButton(icon: Icon(Icons.visibility, color: Colors.white), onPressed: () {}),
        ],
      ),
    );
  }
}