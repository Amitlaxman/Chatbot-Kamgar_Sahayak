import 'dart:convert';
import 'package:http/http.dart' as http;

class ChatService {
  // Use your ngrok public URL
  static const String baseUrl = "http://192.168.29.152:8000";

  static Future<String> sendMessage(String userMessage) async {
    final String chatUrl = "$baseUrl/chat";

    try {
      final res = await http.post(
        Uri.parse(chatUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"query": userMessage}),
      );

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);

        // Always return only the "response" field as a String
        return data["response"]?.toString() ?? "⚠️ No response from bot";
      } else {
        throw Exception("Backend error: ${res.statusCode} - ${res.body}");
      }
    } catch (e) {
      return "Error: $e";
    }
  }
}
