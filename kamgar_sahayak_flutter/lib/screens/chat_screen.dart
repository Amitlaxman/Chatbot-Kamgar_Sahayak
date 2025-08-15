import 'package:flutter/material.dart';
import '../chat_service.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<Map<String, String>> _messages = [];
  final TextEditingController _controller = TextEditingController();
  bool _loading = false;

  void _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _messages.add({"sender": "user", "text": text});
      _loading = true;
    });
    _controller.clear();

    final reply = await ChatService.sendMessage(text);
    setState(() {
      _messages.add({"sender": "bot", "text": reply});
      _loading = false;
    });
  }

  Widget _bubble(Map<String, String> msg) {
    bool isUser = msg["sender"] == "user";
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.blue[200] : Colors.grey[200],
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(msg["text"] ?? ""),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("MP Labour Chatbot")),
      body: Column(
        children: [
          Expanded(
            child: ListView(
              children: _messages.map(_bubble).toList(),
            ),
          ),
          if (_loading)
            const Padding(
              padding: EdgeInsets.all(8),
              child: Text("Typing..."),
            ),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _controller,
                  decoration:
                      const InputDecoration(hintText: "Type a message..."),
                  onSubmitted: (_) => _send(),
                ),
              ),
              IconButton(onPressed: _send, icon: const Icon(Icons.send)),
            ],
          ),
        ],
      ),
    );
  }
}
