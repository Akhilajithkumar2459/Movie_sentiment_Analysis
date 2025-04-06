import streamlit as st
import pickle

# Load the trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define label mapping
label_mapping = {0: "Negative", 1: "Positive"}

# Streamlit UI
st.title("Sentiment Analysis App 😊😡")

st.write("Enter a sentence below, and the model will predict whether it's **Positive** or **Negative**.")

# User input text
user_input = st.text_area("Type your sentence here:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        # Transform input using the vectorizer
        user_input_vectorized = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(user_input_vectorized)[0]

        # Display result
        st.success(f"Sentiment: **{label_mapping[prediction]}**")
    else:
        st.warning("Please enter a valid sentence!")

