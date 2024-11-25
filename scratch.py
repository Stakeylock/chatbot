import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import base64

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data.json", "r") as f:
    faq_data = json.load(f)

faq_questions = [item['question'] for item in faq_data]
faq_answers = [item['answer'] for item in faq_data]

faq_embeddings = model.encode(faq_questions)

def get_similar_questions(user_question, top_n=5):
    user_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_embedding, faq_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    similar_questions = [
        (faq_questions[idx], similarities[idx])
        for idx in top_indices
    ]
    return similar_questions

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-color: black;
    background-repeat: no-repeat;
    background-size: 150px 150px;
    background-position-y:60px;
    background-position-x:center;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def run_app():
    set_background("jntu_logo.png")
    st.title(":violet[JNTUH FAQ Chatbot]")
    st.write(":green[Hello! I'm your FAQ chatbot. Start typing your query, and I'll show the most relevant answer.]")

    user_input = st.text_input(":orange[Start typing your question:]", "")
    similar_questions = []

    if user_input.strip():
        similar_questions = get_similar_questions(user_input, top_n=5)

    suggested_questions = [question for question, _ in similar_questions]
    selected_question = st.selectbox("Suggested Questions:", options=[""] + suggested_questions, index=0)

    if user_input:
        if selected_question != "":
            user_input = selected_question

        similarities = cosine_similarity([model.encode([user_input])[0]], faq_embeddings)
        idx = np.argmax(similarities)
        st.write(f"Answer: {faq_answers[idx]}")

    if user_input.strip() and not selected_question:
        st.write(":green[Here are some similar questions:]")
        for question in suggested_questions:
            if st.button(question):
                user_input = question
                selected_question = question

        if selected_question:
            similarities = cosine_similarity([model.encode([user_input])[0]], faq_embeddings)
            idx = np.argmax(similarities)
            st.write(f"Answer: {faq_answers[idx]}")

    if st.button("Exit"):
        st.write("Goodbye!")

if __name__ == "__main__":
    run_app()
