# Stroke Risk Prediction Dashboard

This project provides a local Streamlit dashboard for screening-support stroke
risk estimates. It is not a diagnostic device and must not replace a qualified
clinician's assessment.

## Run locally

From this folder, install the dependencies and start the dashboard:

```powershell
python -m pip install -r requirements.txt
streamlit run dashboard.py
```

The first prediction automatically trains and saves the final project model at
`models/stroke_risk_model.joblib`. Later runs reuse that saved bundle.

## Dashboard workflow

1. Enter patient demographics, medical history, glucose level, BMI, and smoking status.
2. Select **Estimate stroke risk**.
3. Review the risk probability and screening label.

The model follows the project pipeline: engineered features, training-only
preprocessing, Boruta feature selection, SMOTE balancing, and XGBoost.
