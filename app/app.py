import streamlit as st
import requests

st.title("Carbon Risk Predictor")

# Example inputs
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=[f1, f2])

    if response.status_code == 200:
        result = response.json()
        st.write("Prediction:", result["prediction"])
    else:
        st.error("API error")
