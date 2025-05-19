import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and feature columns
model = joblib.load("mental_health_best_model.joblib")
feature_columns = joblib.load("feature_columns.joblib")

# Title
st.set_page_config(page_title="Student Mental Health Predictor", layout="centered")
st.title("🎓 Student Mental Health Risk Predictor")
st.markdown("Use this tool to assess the mental health risk of students based on academic and personal factors.")

# User Inputs
gender = st.selectbox("Choose your gender:", ["Male", "Female"])
age = st.slider("Select your age:", min_value=15, max_value=40, step=1, value=20)
course = st.text_input("Enter your course name (e.g., CSE, BIT, Law):")
year = st.selectbox("Select your current year of study:", ["Year 1", "Year 2", "Year 3", "Year 4"])
cgpa = st.slider("What is your CGPA?", min_value=1.0, max_value=10.0, step=0.1, value=7.5)
married = st.selectbox("Are you married?", ["No", "Yes"])
depression = st.selectbox("Do you have Depression?", ["No", "Yes"])
anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
panic_attack = st.selectbox("Do you have Panic Attacks?", ["No", "Yes"])
specialist = st.selectbox("Have you sought specialist treatment?", ["No", "Yes"])

# Encode input
input_dict = {
    "Age": age,
    "What is your CGPA?": cgpa,
    "Choose your gender_" + gender: 1,
    "What is your course_" + course: 1,
    "your_current_year_of_study_" + year: 1,
    "marital_status_" + married: 1,
    "do_you_have_depression_" + depression: 1,
    "do_you_have_anxiety_" + anxiety: 1,
    "do_you_have_panic_attack_" + panic_attack: 1,
    "did_you_seek_any_specialist_for_a_treatment_" + specialist: 1,
}

# Initialize full input with all 0s
full_input = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

# Update with user values
for key, value in input_dict.items():
    if key in full_input.columns:
        full_input.at[0, key] = value

# Make Prediction
if st.button("🔍 Predict Mental Health Risk"):
    prediction = model.predict(full_input)[0]
    proba = model.predict_proba(full_input)[0]

    if prediction == 1:
        st.error("⚠️ The student is **at risk of mental health issues**.")
        confidence = round(proba[1] * 100, 2)
    else:
        st.success("✅ The student is **not at immediate risk** of mental health issues.")
        confidence = round(proba[0] * 100, 2)

    st.markdown(f"**Confidence Score:** `{confidence}%`")














