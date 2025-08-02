import streamlit as st
import pandas as pd
from rapidfuzz import process

# Load FAQs
faqs = pd.read_csv("labours_faqs.csv", encoding="cp1252")

# Map UI language to CSV codes
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Hinglish": "hinglish"
}

# Search function
def find_best_answer(user_question: str, lang: str = "English") -> str:
    lang_code = lang_map.get(lang, "en")  # default to English if not found
    
    subset = faqs[faqs['language'] == lang_code]
    questions = subset['Question'].tolist()

    if not questions:
        return f"❌ No FAQs found for {lang} (code: {lang_code})"

    result = process.extractOne(user_question, questions)
    if result:
        best_match, score, idx = result
        if score > 60:
            return str(subset.iloc[idx]['Answer'])
    return "❌ Sorry, I could not find an answer. Please contact the labour helpline."


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="MP Labour Chatbot", page_icon="🛠️")
st.title("🛠️ MP Labour Chatbot")
st.write("Ask your questions about wages, health, rights, and government schemes.")

user_q = st.text_input("Enter your question:")
lang = st.radio("Choose language:", ["English", "Hindi", "Hinglish"])

if st.button("Submit"):
    if user_q.strip():
        answer = find_best_answer(user_q, lang)
        st.success(answer)
    else:
        st.warning("⚠️ Please enter a question.")
