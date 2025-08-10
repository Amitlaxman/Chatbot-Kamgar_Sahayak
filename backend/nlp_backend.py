from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import difflib

app = Flask(__name__)
CORS(app)  # Allow requests from any origin (React frontend)

# Load dataset
df = pd.read_excel("data/labour_data.xlsx")

def search_dataset(query):
    questions = df['Question'].astype(str).tolist()
    answers = df['Answer'].astype(str).tolist()
    match = difflib.get_close_matches(query, questions, n=1, cutoff=0.6)
    if match:
        idx = questions.index(match[0])
        return answers[idx]
    else:
        return "ASK_ADMIN"

@app.route("/get_answer", methods=["POST"])
def get_answer():
    data = request.get_json()
    query = data.get("query", "")
    answer = search_dataset(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
