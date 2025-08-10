import React, { useState } from 'react';

const Register = () => {
  const [formData, setFormData] = useState({
    name: '',
    address: '',
    workType: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Labour Registered (Dummy):', formData);
    alert('पंजीकरण सफल (डेमो)');
  };

  return (
    <div className="auth-container">
      <h2>पंजीकरण फॉर्म</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="पूरा नाम"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
        />
        <input
          type="text"
          placeholder="पता"
          value={formData.address}
          onChange={(e) => setFormData({ ...formData, address: e.target.value })}
        />
        <input
          type="text"
          placeholder="कार्य का प्रकार"
          value={formData.workType}
          onChange={(e) => setFormData({ ...formData, workType: e.target.value })}
        />
        <button type="submit">पंजीकरण करें</button>
      </form>
    </div>
  );
};

export default Register;
