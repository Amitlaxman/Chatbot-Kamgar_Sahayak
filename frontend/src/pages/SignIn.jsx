import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Auth.css';

const BACKEND_URL = "http://127.0.0.1:5000";

export const SignIn = () => {
  const [phone, setPhone] = useState('');
  const navigate = useNavigate();

  const handleSignIn = async () => {
    if (phone.length !== 10) {
      alert('कृपया 10 अंकों का मोबाइल नंबर दर्ज करें');
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/send-otp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone })
      });

      if (!res.ok) throw new Error("OTP भेजने में समस्या आई");

      const data = await res.json();
      console.log("OTP Sent Response:", data);

      navigate('/verify-otp', { state: { phone } });
    } catch (err) {
      console.error(err);
      alert("सर्वर से कनेक्ट करने में समस्या");
    }
  };

  return (
    <div className="auth-container">
      <h2>लॉगिन करें</h2>
      <input
        type="text"
        placeholder="मोबाइल नंबर"
        value={phone}
        onChange={(e) => setPhone(e.target.value)}
      />
      <button onClick={handleSignIn}>OTP भेजें</button>
    </div>
  );
};

export default SignIn;
