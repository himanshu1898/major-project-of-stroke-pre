# -*- coding: utf-8 -*-
"""
feature_engineering.py -- Feature Engineering for Stroke Prediction

Creates clinically meaningful features from raw dataset variables:
  Interaction features (numerical):
    1. age_glucose          -- age x avg_glucose_level
    2. age_hypertension     -- age x hypertension
    3. age_heart_disease    -- age x heart_disease
    4. age_bmi              -- age x bmi
    5. glucose_hypertension -- avg_glucose_level x hypertension
    6. cardio_risk          -- hypertension + heart_disease (0-2 indicator)
  
  Categorical bins:
    7. age_group            -- child/young_adult/adult/middle_aged/senior
    8. bmi_category         -- underweight/normal/overweight/obese
    9. glucose_risk         -- low/normal/pre_diabetic/diabetic
"""

import pandas as pd
import numpy as np

from config import NUMERICAL_COLS, CATEGORICAL_COLS

ENGINEERED_NUMERICAL = [
    'age_glucose',
    'age_hypertension',
    'age_heart_disease',
    'age_bmi',
    'glucose_hypertension',
    'cardio_risk',
]

ENGINEERED_CATEGORICAL = [
    'age_group',
    'bmi_category',
    'glucose_risk',
]


def create_engineered_features(df):
    """
    Create 9 engineered features from raw clinical variables.
    Deterministic transform (no fitting required).
    """
    df = df.copy()

    # 1. Interaction Features
    df['age_glucose'] = df['age'] * df['avg_glucose_level']
    df['age_hypertension'] = df['age'] * df['hypertension']
    df['age_heart_disease'] = df['age'] * df['heart_disease']
    df['age_bmi'] = df['age'] * df['bmi']
    df['glucose_hypertension'] = df['avg_glucose_level'] * df['hypertension']
    df['cardio_risk'] = df['hypertension'] + df['heart_disease']

    # 2. Categorical Bins
    df['age_group'] = pd.cut(
        df['age'],
        bins=[-np.inf, 17, 35, 50, 65, np.inf],
        labels=['child', 'young_adult', 'adult', 'middle_aged', 'senior']
    )

    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
        labels=['underweight', 'normal', 'overweight', 'obese']
    )

    df['glucose_risk'] = pd.cut(
        df['avg_glucose_level'],
        bins=[-np.inf, 70, 99, 125, np.inf],
        labels=['low', 'normal', 'pre_diabetic', 'diabetic']
    )

    print(f"[feature_engineering] Created {len(ENGINEERED_NUMERICAL)} numerical features")
    print(f"[feature_engineering] Created {len(ENGINEERED_CATEGORICAL)} categorical features")
    print(f"[feature_engineering] Total columns: {df.shape[1]}")

    return df


# Alias for compatibility
engineer_features = create_engineered_features


def get_engineered_columns():
    """
    Return full lists of numerical and categorical columns after engineering.
    """
    numerical = NUMERICAL_COLS + ENGINEERED_NUMERICAL
    categorical = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL
    return numerical, categorical
