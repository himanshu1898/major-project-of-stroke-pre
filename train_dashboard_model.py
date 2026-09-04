"""Train and save the model used by the local Streamlit dashboard.

This script deliberately uses only the training partition to fit transformations,
feature selection and SMOTE.  The saved bundle contains everything needed to
make a prediction from one raw patient record.
"""

from pathlib import Path
import joblib
from imblearn.over_sampling import SMOTE

from config import RANDOM_STATE
from preprocessing import load_data, split_data, preprocess_data
from feature_engineering import create_engineered_features, get_engineered_columns
from feature_selection import run_boruta, apply_feature_selection
from xgboost_model import _get_xgb_model

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "stroke_risk_model.joblib"


def train_and_save_model():
    """Fit the final E experiment and persist its inference components."""
    df = load_data()
    X_train, _, y_train, _ = split_data(df)

    X_train_engineered = create_engineered_features(X_train)
    numerical_cols, categorical_cols = get_engineered_columns()
    X_train_processed, _, feature_names, preprocessor = preprocess_data(
        X_train_engineered, X_train_engineered, numerical_cols, categorical_cols
    )

    mask, selected_features = run_boruta(
        X_train_processed, y_train, feature_names
    )
    X_selected = apply_feature_selection(X_train_processed, mask)
    X_resampled, y_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_selected, y_train)

    model = _get_xgb_model(use_scale_pos_weight=False)
    model.fit(X_resampled, y_resampled)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_mask": mask,
            "selected_features": selected_features,
        },
        MODEL_PATH,
    )
    print(f"Saved dashboard model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
