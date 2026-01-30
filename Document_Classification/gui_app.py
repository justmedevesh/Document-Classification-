import streamlit as st
import pickle
from utils import preprocess

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

reverse_label_map = {
    0: "Business",
    1: "Entertainment",
    2: "Health"
}

st.title("📄 Document Classification System")

user_text = st.text_area("Enter a document")

if st.button("Classify"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(user_text)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]

        st.success(f"Predicted Category: **{reverse_label_map[pred]}**")