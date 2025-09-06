// src/components/ChatInput.jsx
import React, { useState, useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import './ChatInput.css'; // optional - your CSS

const ChatInput = ({ onSend, language = 'hi' }) => {
  const [inputText, setInputText] = useState('');
  const { transcript, listening, resetTranscript } = useSpeechRecognition();

  // When transcript changes, update inputText (but don't auto-send)
  useEffect(() => {
    if (listening) {
      setInputText(transcript);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [transcript]);

  const handleSendClick = () => {
    const trimmed = inputText.trim();
    if (trimmed !== '') {
      onSend(trimmed);
      setInputText('');
      resetTranscript();
      if (listening) SpeechRecognition.stopListening();
    }
  };

  const handleMicClick = () => {
    if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
      alert(language === 'hi' ? 'आपका ब्राउज़र आवाज़ पहचान का समर्थन नहीं करता।' : 'Your browser does not support speech recognition.');
      return;
    }
    const langCode = language === 'hi' ? 'hi-IN' : 'en-US';
    if (!listening) {
      SpeechRecognition.startListening({ continuous: true, language: langCode });
      resetTranscript();
    } else {
      SpeechRecognition.stopListening();
    }
  };

  return (
    <div className="input-container" style={{ display: 'flex', gap: 8, marginTop: 12 }}>
      <input
        type="text"
        placeholder={language === 'hi' ? 'अपना प्रश्न टाइप करें...' : 'Type your question...'}
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter') handleSendClick(); }}
        className="chat-input"
        style={{ flex: 1, padding: '10px 12px', borderRadius: 6, border: '1px solid #ccc' }}
        autoComplete="off"
      />
      <button onClick={handleSendClick} className="send-btn" style={{ padding: '10px 12px', borderRadius: 6, border: 'none', background: '#1976d2', color: '#fff' }}>
        {language === 'hi' ? 'भेजें' : 'Send'}
      </button>
      <button onClick={handleMicClick} className="mic-btn" style={{ padding: '10px 12px', borderRadius: 6, border: '1px solid #1976d2', background: listening ? '#e3f2fd' : '#fff', cursor: 'pointer' }}>
        {listening ? '🎙️...' : '🎤'}
      </button>
    </div>
  );
};

export default ChatInput;
