import 'dart:convert';
import 'package:http/http.dart' as http;

class ChatService {
  // ⚠️ Use your LAN IP (not localhost), same as Node server
  static const String chatUrl = "http://192.168.29.152:5000/api/chat";

  static Future<String> sendMessage(String userMessage) async {
    try {
      final res = await http.post(
        Uri.parse(chatUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"message": userMessage}),
      );

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        return data["reply"] ?? "";
      } else {
        throw Exception("Node backend error: ${res.statusCode} ${res.body}");
      }
    } catch (e) {
      // Show real error instead of generic
      return "⚠️ Error: $e";
    }
  }
}
