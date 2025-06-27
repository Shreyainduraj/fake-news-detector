import streamlit as st
import pickle

st.markdown("""
    <style>
    body {
        background-color: #fff0f5;
    }
    .stApp {
        background-color: #fff0f5;
    }
    h1 {
        color: #cc3366;
    }
    </style>
""", unsafe_allow_html=True)

import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")

# Load model and vectorizer
model = pickle.load(open("nb_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Input section
title = st.text_input("Enter News Title")
text = st.text_area("Enter Full News Content")

if st.button("Classify"):
    combined_input = title + " " + text
    transformed_input = vectorizer.transform([combined_input])
    prediction = model.predict(transformed_input)[0]
    st.success("✅ Real News" if prediction == 1 else "🚫 Fake News")
