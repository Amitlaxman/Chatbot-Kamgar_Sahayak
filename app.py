
import streamlit as st

st.title("🛠️ MP Labour Chatbot")

user_q = st.text_input("Ask your question:")
lang = st.radio("Language:", ["English", "Hindi", "Hinglish"])

if st.button("Submit"):
    answer = find_best_answer(user_q, lang)
    st.write("**Answer:** ", answer)
