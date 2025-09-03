// import React, { useState } from 'react';
// import { useNavigate } from 'react-router-dom';

// const Register = ({language}) => {
//   const [formData, setFormData] = useState({
//     name: '',
//     email: '',
//     password: '',
//     address: '',
//     workType: ''
//   });

//   const navigate = useNavigate();

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     try {
//     const response = await fetch("http://localhost:8000/register_api/register-user", {

//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(formData)
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         throw new Error(errorData.detail || "Registration failed");
//       }

//       const data = await response.json();
//       alert(data.message);

//   // Registration successful, redirect to login page
//   navigate("/login");
//     } catch (error) {
//       alert(`Error: ${error.message}`);
//     }
//   };

//   return (
//     <div className="auth-container">
//       <h2>Registration Form</h2>
//       <form onSubmit={handleSubmit}>
//         <input
//           type="text"
//           placeholder="Full Name"
//           value={formData.name}
//           onChange={(e) => setFormData({ ...formData, name: e.target.value })}
//           required
//         />
//         <input
//           type="email"
//           placeholder="Email Address"
//           value={formData.email}
//           onChange={(e) => setFormData({ ...formData, email: e.target.value })}
//           required
//         />
//         <input
//           type="password"
//           placeholder="Password"
//           value={formData.password}
//           onChange={(e) => setFormData({ ...formData, password: e.target.value })}
//           required
//         />
//         <input
//           type="text"
//           placeholder="Address"
//           value={formData.address}
//           onChange={(e) => setFormData({ ...formData, address: e.target.value })}
//         />
//         <input
//           type="text"
//           placeholder="Work Type"
//           value={formData.workType}
//           onChange={(e) => setFormData({ ...formData, workType: e.target.value })}
//         />
//         <button type="submit">Register</button>
//       </form>
//     </div>
//   );
// };

// export default Register;




// src/pages/Register.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Register = ({ language }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    address: '',
    workType: ''
  });

  const navigate = useNavigate();

  const content = {
    hi: {
      title: 'पंजीकरण फॉर्म',
      name: 'पूरा नाम',
      email: 'ईमेल पता',
      password: 'पासवर्ड',
      address: 'पता',
      workType: 'काम का प्रकार',
      button: 'रजिस्टर करें',
      success: 'पंजीकरण सफल हुआ!',
      error: 'पंजीकरण असफल रहा'
    },
    en: {
      title: 'Registration Form',
      name: 'Full Name',
      email: 'Email Address',
      password: 'Password',
      address: 'Address',
      workType: 'Work Type',
      button: 'Register',
      success: 'Registration successful!',
      error: 'Registration failed'
    }
  };

  const lang = content[language];

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("http://localhost:8000/register_api/register-user", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || lang.error);
      }

      const data = await response.json();
      alert(data.message || lang.success);

      // Registration successful, redirect to login page
      navigate("/signin");
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  return (
    <div className="auth-container">
      <h2>{lang.title}</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder={lang.name}
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          required
        />
        <input
          type="email"
          placeholder={lang.email}
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          required
        />
        <input
          type="password"
          placeholder={lang.password}
          value={formData.password}
          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
          required
        />
        <input
          type="text"
          placeholder={lang.address}
          value={formData.address}
          onChange={(e) => setFormData({ ...formData, address: e.target.value })}
        />
        <input
          type="text"
          placeholder={lang.workType}
          value={formData.workType}
          onChange={(e) => setFormData({ ...formData, workType: e.target.value })}
        />
        <button type="submit">{lang.button}</button>
      </form>
    </div>
  );
};

export default Register;

