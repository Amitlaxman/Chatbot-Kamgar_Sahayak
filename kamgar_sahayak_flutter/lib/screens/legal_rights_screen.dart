import 'package:flutter/material.dart';
import '../l10n/app_localizations.dart';
import '../widgets/app_header.dart';

class LegalRightsScreen extends StatelessWidget {
  const LegalRightsScreen({super.key});

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
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    localizations.yourLegalRights,
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 5),
                  Text(localizations.importantDocuments),
                ],
              ),
            ),
            SizedBox(height: 20),
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

  Widget _buildDocumentRow(BuildContext context, String documentName) {
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(child: Text(documentName, style: TextStyle(color: Colors.white))),
          IconButton(icon: Icon(Icons.download, color: Colors.orange), onPressed: () {}),
          IconButton(icon: Icon(Icons.visibility, color: Colors.orange), onPressed: () {}),
        ],
      ),
    );
  }
}