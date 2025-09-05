import React, { useState, useRef } from 'react';
import ChatInput from './ChatInput';
import '../styles/ChatBot.css'; 

const PY_NLP_URL = 'http://localhost:8000/chat_api/chat';   

const ADMIN_REPORT_URL = 'http://localhost:5000/api/admin/report'; // optional Node API for admin reporting

const ChatBot = ({ language = 'hi' }) => {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: language === 'hi' ? 'नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?' : 'Hello! How can I help you?' }
  ]);
  const [loading, setLoading] = useState(false);
  const boxRef = useRef(null);

  const addMessage = (sender, text) => {
    setMessages(prev => [...prev, { sender, text }]);
    // scroll to bottom
    setTimeout(() => {
      if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }, 50);
  };

  // Render message text with clickable links and line breaks
  const renderMessageText = (text) => {
    if (!text) return null;
    // Replace <br> and \n with <br />
    let html = text.replace(/\n/g, '<br />').replace(/<br\s*\/?>(?!\s*<br)/gi, '<br />');
    // Convert URLs to clickable links
    html = html.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    // Optionally, convert www. links
    html = html.replace(/(^|[^\/])(www\.[^\s<]+)/g, '$1<a href="http://$2" target="_blank" rel="noopener noreferrer">$2</a>');
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
  };

  const reportToAdmin = async (question) => {
    try {
      const res = await fetch(ADMIN_REPORT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      if (!res.ok) throw new Error('Admin report failed');
      return true;
    } catch (err) {
      console.warn('Could not send admin report (no backend?), logging locally:', err);
      console.log('Unanswered saved (frontend fallback):', question);
      return false;
    }
  };

  const handleSend = async (userText) => {
    if (!userText || !userText.trim()) return;
    addMessage('user', userText);
    setLoading(true);

    // Get user email from localStorage
    const user = JSON.parse(localStorage.getItem("user"));
    const userId = user?.email || "anonymous_user";

    try {
      const resp = await fetch(PY_NLP_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_text: userText,
          user_id: userId,
          language: language
        }),
      });
      if (!resp.ok) {
        throw new Error(`NLP backend error: ${resp.status}`);
      }

      const data = await resp.json();
      const answer = data.bot_response ?? '';

      if (answer === 'ASK_ADMIN') {
        // Inform user and send to admin
        addMessage('bot', language === 'hi' ? 'माफ़ कीजिए — मैं इसका उत्तर नहीं दे पा रहा/रही। हमने आपका सवाल एडमिन को भेज दिया है।' : "Sorry — I don't have that answer. We've forwarded your query to the admin.");
        await reportToAdmin(userText);
      } else {
        // Show the answer (multi-line friendly)
        addMessage('bot', answer);
      }
    } catch (err) {
      console.error('Error fetching answer:', err);
      addMessage('bot', language === 'hi' ? 'उत्तर लाने में त्रुटि हुई। कृपया बाद में कोशिश करें।' : 'Error fetching answer. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Function to speak text aloud
  const speakText = (text, lang) => {
    if (!window.speechSynthesis) return;
    const utter = new window.SpeechSynthesisUtterance(text);
    utter.lang = lang === 'hi' ? 'hi-IN' : 'en-US';
    window.speechSynthesis.speak(utter);
  };

  return (
    <div className="chat-container">
      <h2>{language === 'hi' ? 'MP Labour Chatbot' : 'MP Labour Chatbot'}</h2>

      <div className="chat-box" ref={boxRef} style={{ maxHeight: '60vh', overflowY: 'auto', padding: '12px', background: '#f8f8f8', borderRadius: 8 }}>
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.sender}`} style={{
            display: 'block',
            margin: '8px 0',
            textAlign: msg.sender === 'user' ? 'right' : 'left'
          }}>
            <div style={{
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: 12,
              background: msg.sender === 'user' ? '#d1e7ff' : '#ffffff',
              boxShadow: msg.sender === 'bot' ? '0 1px 1px rgba(0,0,0,0.05)' : 'none',
              maxWidth: '85%'
            }}>
              {renderMessageText(msg.text)}
            </div>
            {/* Speaker button for bot messages only */}
            {msg.sender === 'bot' && msg.text && (
              <button
                onClick={() => speakText(msg.text, language)}
                title={language === 'hi' ? 'उत्तर पढ़ें' : 'Read answer aloud'}
                style={{
                  marginTop: 4,
                  marginLeft: 4,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  verticalAlign: 'middle',
                  padding: 0
                }}
              >
                {/* Speaker SVG icon */}
                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" style={{ verticalAlign: 'middle' }}>
                  <path d="M3 8v4h4l5 5V3L7 8H3zm13.5 2a5.5 5.5 0 0 0-1.5-3.9v7.8A5.5 5.5 0 0 0 16.5 10zm-2-7.7v2.06A7.5 7.5 0 0 1 18 10a7.5 7.5 0 0 1-3.5 6.64v2.06A9.5 9.5 0 0 0 20 10a9.5 9.5 0 0 0-5.5-8.7z" />
                </svg>
              </button>
            )}
          </div>
        ))}
        {loading && (
          <div className="message bot" style={{ marginTop: 8 }}>
            <div style={{ display: 'inline-block', padding: '8px 12px', borderRadius: 12, background: '#fff' }}>
              {language === 'hi' ? 'टाइप कर रहे हैं...' : 'Typing...'}
            </div>
          </div>
        )}
      </div>

      <ChatInput onSend={handleSend} language={language} />
    </div>
  );
};

export default ChatBot;
