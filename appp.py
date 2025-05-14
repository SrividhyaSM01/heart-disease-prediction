import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide"
)

# Load model
with open("bestmodelforheartdisease1.pkl", "rb") as f:
    model = pickle.load(f)

with open("heart_feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# App Title
st.markdown("<h1 style='text-align: center; color: white;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details below to predict heart disease</p>", unsafe_allow_html=True)

st.write("---")

# Create two columns for inputs
left_col, right_col = st.columns(2)

user_input = []
invalid_input = False  # To track if any input is invalid

for i, col in enumerate(feature_columns):
    with (left_col if i % 2 == 0 else right_col):
        val = st.text_input(f"{col}", value=" ")
        try:
            user_input.append(float(val))
        except ValueError:
            invalid_input = True  # Flag the input as invalid
            # Instead of st.warning inside the loop, we'll show a custom alert later

st.write("---")

# Predict Button
if st.button("Predict disease"):
    if invalid_input:
        st.error("Please enter valid numbers for all fields. Invalid inputs detected!")
    else:
        input_array = np.array(user_input).reshape(1, -1)

        # Safety check
        if input_array.shape[1] != 13:
            st.error(f"Feature mismatch: Expected 12 values, got {input_array.shape[1]}")
        else:
            prediction = model.predict(input_array)
            result_text = "no disease" if prediction[0] == 1 else "heart disease"

            st.markdown(
                f"<div style='text-align:center; padding:20px; border-radius:10px; background-color:#e6f2ff;'>"
                f"<h2 style='color:black;'>Prediction: {result_text}</h2></div>",
                unsafe_allow_html=True
            )

# Footer
st.write(" ")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
