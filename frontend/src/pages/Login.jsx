// import React, { useState } from 'react';
// import { useNavigate } from 'react-router-dom';
// import '../styles/Auth.css';

// const BACKEND_URL = "http://127.0.0.1:8000";  // Correct backend URL & port

// const Login = () => {
//   const [email, setEmail] = useState('');
//   const [password, setPassword] = useState('');
//   const navigate = useNavigate();

//   const handleLogin = async () => {
//     if (!email || !password) {
//       alert('कृपया सभी फ़ील्ड भरें');
//       return;
//     }

//     try {
//       const res = await fetch(`${BACKEND_URL}/login_api/login`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ email, password })
//       });
//       if (!res.ok) {
//         const errorData = await res.json();
//         throw new Error(errorData.detail || "लॉगिन विफल");
//       }
//       const data = await res.json();
//       console.log("Login Response:", data);
//       alert("लॉगिन सफल");
//       localStorage.setItem("isUserLoggedIn", "true");
//       localStorage.setItem("userToken", data.token);
//       localStorage.setItem("user", JSON.stringify(data.user));
//       navigate('/'); // go to home page or dashboard after login
//     } catch (err) {
//       console.error(err);
//       alert(`त्रुटि: ${err.message}`);
//     }
//   };

//   return (
//     <div className="auth-container">
//       <h2>साइन इन करें</h2>
//       <input
//         type="email"
//         placeholder="ईमेल"
//         value={email}
//         onChange={(e) => setEmail(e.target.value)}
//       />
//       <input
//         type="password"
//         placeholder="पासवर्ड"
//         value={password}
//         onChange={(e) => setPassword(e.target.value)}
//       />
//       <button onClick={handleLogin}>साइन इन</button>
//     </div>
//   );
// };

// export default Login;





// src/pages/Login.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Auth.css';

const BACKEND_URL = "http://127.0.0.1:8000";  // Backend URL

const Login = ({ language }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  // Translation object
  const translations = {
    en: {
      title: "Sign In",
      email: "Email",
      password: "Password",
      button: "Sign In",
      fillAll: "Please fill all fields",
      success: "Login Successful",
      fail: "Login Failed",
      error: "Error"
    },
    hi: {
      title: "साइन इन करें",
      email: "ईमेल",
      password: "पासवर्ड",
      button: "साइन इन",
      fillAll: "कृपया सभी फ़ील्ड भरें",
      success: "लॉगिन सफल",
      fail: "लॉगिन विफल",
      error: "त्रुटि"
    }
  };

  const t = translations[language] || translations.hi;

  const handleLogin = async () => {
    if (!email || !password) {
      alert(t.fillAll);
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/login_api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || t.fail);
      }
      const data = await res.json();
      console.log("Login Response:", data);
      alert(t.success);
      localStorage.setItem("isUserLoggedIn", "true");
      localStorage.setItem("userToken", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));
      navigate('/');
    } catch (err) {
      console.error(err);
      alert(`${t.error}: ${err.message}`);
    }
  };

  return (
    <div className="auth-container">
      <h2>{t.title}</h2>
      <input
        type="email"
        placeholder={t.email}
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        type="password"
        placeholder={t.password}
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button onClick={handleLogin}>{t.button}</button>
    </div>
  );
};

export default Login;



