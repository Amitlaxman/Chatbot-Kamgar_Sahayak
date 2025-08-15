import 'dart:convert';
import 'package:http/http.dart' as http;

class ChatService {
  static const String nlpUrl = "http://192.168.29.152:5001/get_answer"; // Replace with your PC IP
  static const String adminReportUrl = "http://192.168.29.152:5000/api/admin/report";

  static Future<String> sendMessage(String userMessage) async {
    try {
      final res = await http.post(
        Uri.parse(nlpUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"query": userMessage}),
      );

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final answer = data["answer"] ?? "";

        if (answer == "ASK_ADMIN") {
          await _reportToAdmin(userMessage);
          return "Sorry — I don't have that answer. We've forwarded your query to the admin.";
        }

        return answer;
      } else {
        throw Exception("NLP backend error: ${res.statusCode}");
      }
    } catch (e) {
      return "Error fetching answer. Please try again later.";
    }
  }

  static Future<void> _reportToAdmin(String question) async {
    try {
      final res = await http.post(
        Uri.parse(adminReportUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"question": question}),
      );
      if (res.statusCode != 200) {
        throw Exception("Admin report failed");
      }
    } catch (e) {
      print("Admin report failed: $e");
    }
  }
}
