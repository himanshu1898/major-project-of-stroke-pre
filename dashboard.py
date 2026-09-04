"""Local dashboard for stroke-risk screening predictions.

Run with: streamlit run dashboard.py
This tool is for educational screening support only; it is not a diagnosis.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from feature_engineering import create_engineered_features
from train_dashboard_model import MODEL_PATH, train_and_save_model

st.set_page_config(page_title="Stroke Risk Dashboard", page_icon="🫀", layout="wide")


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        with st.spinner("Preparing the prediction model (first run only)..."):
            train_and_save_model()
    return joblib.load(MODEL_PATH)


def probability_label(probability):
    if probability < 0.20:
        return "Lower estimated risk", "success"
    if probability < 0.50:
        return "Moderate estimated risk", "warning"
    return "Higher estimated risk", "error"


st.title("🫀 Stroke Risk Dashboard")
st.caption("Local screening-support tool based on the project’s XGBoost ML pipeline.")
st.warning("This estimate is not a medical diagnosis. Consult a qualified clinician for medical decisions.")

with st.form("patient_form"):
    st.subheader("Patient information")
    left, middle, right = st.columns(3)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
    with middle:
        ever_married = st.selectbox("Ever married", ["No", "Yes"])
        work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
        smoking_status = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    with right:
        glucose = st.number_input("Average glucose level (mg/dL)", min_value=1.0, max_value=600.0, value=105.0, step=0.1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        submitted = st.form_submit_button("Estimate stroke risk", type="primary", use_container_width=True)

if submitted:
    raw_record = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": int(hypertension == "Yes"),
        "heart_disease": int(heart_disease == "Yes"),
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }])
    bundle = load_model_bundle()
    engineered_record = create_engineered_features(raw_record)
    transformed_record = bundle["preprocessor"].transform(engineered_record)
    selected_record = transformed_record[:, bundle["feature_mask"]]
    probability = float(bundle["model"].predict_proba(selected_record)[0, 1])
    label, alert_type = probability_label(probability)

    st.divider()
    result, details = st.columns([1, 2])
    with result:
        st.metric("Estimated stroke-risk probability", f"{probability:.1%}")
        getattr(st, alert_type)(label)
    with details:
        st.subheader("Interpretation")
        st.write(
            "The score estimates the model’s predicted likelihood from the provided "
            "information. It must be interpreted alongside clinical assessment."
        )
        st.caption("Model inputs selected by Boruta: " + ", ".join(bundle["selected_features"]))

with st.expander("About this dashboard"):
    st.write(
        "The dashboard uses feature engineering, training-only preprocessing, Boruta "
        "feature selection, SMOTE balancing, and an XGBoost classifier from this project."
    )
