// src/pages/AdminDashboard.jsx
import React, { useEffect, useState } from "react";
import "../styles/Admin.css";

const AdminDashboard = () => {
  const [queries, setQueries] = useState([]);
  const [answers, setAnswers] = useState({});

  useEffect(() => {
    fetchQueries();
  }, []);

  const fetchQueries = async () => {
    try {
      const token = localStorage.getItem("adminToken");
      const res = await fetch("http://localhost:5000/api/admin/queries", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setQueries(data);
    } catch (error) {
      console.error(error);
    }
  };

  const handleAnswerSubmit = async (id) => {
    try {
      const token = localStorage.getItem("adminToken");
      const res = await fetch(`http://localhost:5000/api/admin/answer/${id}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ answer: answers[id] }),
      });

      if (res.ok) {
        alert("Answer submitted");
        fetchQueries(); // refresh list
      } else {
        alert("Failed to submit answer");
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="admin-dashboard-container">
      <h2>Pending Queries</h2>
      {queries.length === 0 ? (
        <p>No pending queries</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Question</th>
              <th>Asked By</th>
              <th>Answer</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {queries.map((q) => (
              <tr key={q._id}>
                <td>{q.question}</td>
                <td>{q.askedBy}</td>
                <td>
                  <textarea
                    value={answers[q._id] || ""}
                    onChange={(e) =>
                      setAnswers({ ...answers, [q._id]: e.target.value })
                    }
                  />
                </td>
                <td>
                  <button onClick={() => handleAnswerSubmit(q._id)}>
                    Submit
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default AdminDashboard;
