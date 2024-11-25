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

with open("embeddings.json", "w") as f:
    json.dump(faq_embeddings.tolist(), f)


def get_best_faq_answer(user_question):
    user_embedding = model.encode([user_question])
    print(user_embedding)
    print(faq_embeddings)
    with open("user_embeddings.json", "w") as f:
        json.dump(user_embedding.tolist(), f)

    similarities = cosine_similarity(user_embedding, faq_embeddings)

    most_similar_idx = np.argmax(similarities)

    most_similar_question = faq_questions[most_similar_idx]
    answer = faq_answers[most_similar_idx]

    return most_similar_question, answer
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-color: white;
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
    #st.logo("jntu_logo.png", size="large", link=None)
    set_background("jntu_logo.png")
    st.title("")
    st.title(":violet[JNTUH FAQ Chatbot]")
    

    st.write(":green[Hello! I'm your FAQ chatbot. Ask me anything about JNTUH services.]")

    user_input = st.text_input(":orange[You:]", "")
    button = st.button("Ask!")

    if user_input or button:
        question, answer = get_best_faq_answer(user_input)
        st.write(f":red[I found a similar question: '{question}']")
        st.write(f":green[Answer: {answer}]")
    
    if st.button("Exit"):
        st.write("Goodbye!")

if __name__ == "__main__":
    run_app()
