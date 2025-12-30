import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model & columns
model = joblib.load("water_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Water Potability Predictor", page_icon="")
st.title(" Water Potability Prediction System")

st.write("Enter water quality values:")

inputs = {}

for col in columns:
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):

    df = pd.DataFrame([inputs])

    prob = model.predict_proba(df)[0][1]

    threshold = 0.35

    if prob >= threshold:
        result = "Potable"
        st.success(f" Water is Potable\n\nConfidence: {prob*100:.2f}%")
    else:
        result = "Not Potable"
        st.error(f" Water is Not Potable\n\nConfidence: {prob*100:.2f}%")
