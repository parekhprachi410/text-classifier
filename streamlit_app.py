import streamlit as st
import pickle
import re

# LOAD MODEL
model = pickle.load(
    open("model.pkl","rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl","rb")
)

# CLEAN FUNCTION
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z ]',
        '',
        text
    )

    return text

# PAGE CONFIG
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered"
)

# UI
st.title("📰 BBC News Category Predictor")

st.markdown(
"""
Predict news categories:

✔ Business  
✔ Entertainment  
✔ Politics  
✔ Sport  
✔ Tech
"""
)

text = st.text_area(
    "Enter News Article",
    height=200
)

if st.button("Predict Category"):

    if text.strip()=="":

        st.warning(
            "Please enter some text."
        )

    else:

        cleaned = clean_text(text)

        vectorized = vectorizer.transform(
            [cleaned]
        )

        prediction = model.predict(
            vectorized
        )[0]

        st.success(
            f"Prediction: {prediction.upper()}"
        )