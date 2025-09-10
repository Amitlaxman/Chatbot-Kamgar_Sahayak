import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart'; // 👈 for TTS
import '../chat_service.dart';
import 'package:flutter_linkify/flutter_linkify.dart';
import 'package:url_launcher/url_launcher.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<Map<String, String>> _messages = [];
  final TextEditingController _controller = TextEditingController();
  bool _loading = false;

  late stt.SpeechToText _speech;
  bool _isListening = false;

  int _selectedLang = 0; // 0 = English, 1 = Hindi

  // 👇 TTS instance
  final FlutterTts _flutterTts = FlutterTts();
  bool _isPlaying = false;
  String _currentText = "";

  final Color _primaryBlue = const Color(0xFF1A3C5A);
  final Color _accentOrange = Colors.orange;

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();

    _flutterTts.setCompletionHandler(() {
      setState(() => _isPlaying = false);
    });

    _flutterTts.setStartHandler(() {
      setState(() => _isPlaying = true);
    });

    _flutterTts.setPauseHandler(() {
      setState(() => _isPlaying = false);
    });
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

        String locale = _selectedLang == 0 ? "en-US" : "hi-IN";

        _speech.listen(
          localeId: locale,
          onResult: (result) {
            setState(() {
              _controller.text = result.recognizedWords;
            });
          },
        );
      }
    } else {
      setState(() => _isListening = false);
      _speech.stop();
    }
  }

  // 🔊 Speak selected text
  void _speak(String text) async {
    await _flutterTts.stop();

    // 👇 Fix: Set voice language properly
    String langCode = _selectedLang == 0 ? "en-US" : "hi-IN";
    await _flutterTts.setLanguage(langCode);

    setState(() {
      _currentText = text;
      _isPlaying = true;
    });
    await _flutterTts.speak(text);
  }

  // ⏸ Pause speaking
  void _pause() async {
    await _flutterTts.pause();
    setState(() => _isPlaying = false);
  }

  Widget _bubble(Map<String, String> msg) {
    bool isUser = msg["sender"] == "user";
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: isUser ? _accentOrange.withOpacity(0.8) : _primaryBlue.withOpacity(0.1),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SelectableLinkify(
              text: msg["text"] ?? "",
              onOpen: (link) async {
                if (!await launchUrl(Uri.parse(link.url),
                    mode: LaunchMode.externalApplication)) {
                  throw 'Could not launch ${link.url}';
                }
              },
              style: TextStyle(
                fontSize: 16,
                color: isUser ? Colors.white : _primaryBlue,
              ),
              linkStyle: TextStyle(
                color: _accentOrange,
                decoration: TextDecoration.underline,
              ),
            ),

            if (!isUser)
              Row(
                children: [
                  IconButton(
                    icon: Icon(Icons.volume_up, color: _primaryBlue),
                    onPressed: () => _speak(msg["text"] ?? ""),
                  ),
                  IconButton(
                    icon: Icon(Icons.pause, color: Colors.redAccent),
                    onPressed: _pause,
                  ),
                ],
              )
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: _primaryBlue,
        foregroundColor: Colors.white,
        title: const Text("Kaamgar Sahayak Chatbot", style: TextStyle(color: Colors.white)),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(8),
              children: _messages.map(_bubble).toList(),
            ),
          ),
          if (_loading)
            const Padding(
              padding: EdgeInsets.all(8),
              child: Text("Typing..."),
            ),

          // 🔘 Language Toggle
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ChoiceChip(
                  label: const Text("English"),
                  selected: _selectedLang == 0,
                  onSelected: (_) => setState(() => _selectedLang = 0),
                  selectedColor: _accentOrange,
                  labelStyle: TextStyle(
                    color: _selectedLang == 0 ? Colors.white : _primaryBlue,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 12),
                ChoiceChip(
                  label: const Text("हिन्दी"),
                  selected: _selectedLang == 1,
                  onSelected: (_) => setState(() => _selectedLang = 1),
                  selectedColor: _accentOrange,
                  labelStyle: TextStyle(
                    color: _selectedLang == 1 ? Colors.white : _primaryBlue,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),

          // 📝 Input Row
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
            color: Colors.white,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: const InputDecoration(
                        hintText: "Type or speak a message...",
                        border: InputBorder.none),
                    onSubmitted: (_) => _send(),
                  ),
                ),
                IconButton(
                  icon: Icon(
                    _isListening ? Icons.mic : Icons.mic_none,
                    color: _isListening ? Colors.red : _primaryBlue,
                  ),
                  onPressed: _listen,
                ),
                IconButton(
                  onPressed: _send,
                  icon: Icon(Icons.send, color: _accentOrange),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
