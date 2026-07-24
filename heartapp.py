import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("heart_model.pkl")
model_columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction System")

# User Inputs
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
cholesterol = st.number_input("Cholesterol", 0, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar (>120 mg/dl)", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")