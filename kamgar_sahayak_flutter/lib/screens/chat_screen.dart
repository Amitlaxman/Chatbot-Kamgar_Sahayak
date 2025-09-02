import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
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

  // Speech-to-text instance
  late stt.SpeechToText _speech;
  bool _isListening = false;

  // Language toggle: 0 = English, 1 = Hindi
  int _selectedLang = 0;

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
  }

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


  void _listen() async {
    if (!_isListening) {
      bool available = await _speech.initialize();
      if (available) {
        setState(() => _isListening = true);

        // Set language based on toggle
        String locale = _selectedLang == 0 ? "en-US" : "hi-IN";

        _speech.listen(
          localeId: locale,
          onResult: (result) {
            setState(() {
              _controller.text = result.recognizedWords; // fill text box
            });
          },
        );
      }
    } else {
      setState(() => _isListening = false);
      _speech.stop();
    }
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
      appBar: AppBar(title: const Text("Kaamgar Sahayak Chatbot")),
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

          // 🔥 Language toggle slider
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              children: [
                const Text("Choose Speaking Language:"),
                ToggleButtons(
                  isSelected: [_selectedLang == 0, _selectedLang == 1],
                  onPressed: (index) {
                    setState(() {
                      _selectedLang = index;
                    });
                  },
                  children: const [
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 16),
                      child: Text("English"),
                    ),
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 16),
                      child: Text("Hindi"),
                    ),
                  ],
                ),
              ],
            ),
          ),

          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _controller,
                  decoration: const InputDecoration(
                      hintText: "Type or speak a message..."),
                  onSubmitted: (_) => _send(),
                ),
              ),
              IconButton(
                icon: Icon(
                  _isListening ? Icons.mic : Icons.mic_none,
                  color: _isListening ? Colors.red : null,
                ),
                onPressed: _listen,
              ),
              IconButton(onPressed: _send, icon: const Icon(Icons.send)),
            ],
          ),
        ],
      ),
    );
  }
}