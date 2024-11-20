import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data.json", "r") as f:
    faq_data = json.load(f)

faq_questions = [item['question'] for item in faq_data]
faq_answers = [item['answer'] for item in faq_data]

faq_embeddings = model.encode(faq_questions)

def get_best_faq_answer(user_question):
    user_embedding = model.encode([user_question])

    similarities = cosine_similarity(user_embedding, faq_embeddings)

    most_similar_idx = np.argmax(similarities)

    most_similar_question = faq_questions[most_similar_idx]
    answer = faq_answers[most_similar_idx]

    return most_similar_question, answer

def run_app():
    st.title("JNTUH FAQ Chatbot")

    st.write("Hello! I'm your FAQ chatbot. Ask me anything about JNTUH services.")

    user_input = st.text_input("You:", "")

    if user_input:
        question, answer = get_best_faq_answer(user_input)
        st.write(f"I found a similar question: '{question}'")
        st.write(f"Answer: {answer}")
    
    if st.button("Exit"):
        st.write("Goodbye!")

if __name__ == "__main__":
    run_app()
