# -*- coding: utf-8 -*-
"""
preprocessing.py -- Data Loading, Splitting, and Preprocessing

Fixes original LabelEncoder bugs:
1. Uses ColumnTransformer (OneHotEncoder + StandardScaler)
2. Fits preprocessor ONLY on training data to prevent data leakage
3. Uses SimpleImputer for missing BMI values instead of global median imputation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import (
    RANDOM_STATE, TEST_SIZE, DATA_PATH,
    NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL
)


def load_data(filepath=DATA_PATH):
    """
    Load the stroke dataset and remove the 'id' column.
    """
    df = pd.read_csv(filepath)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    print(f"[preprocessing] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into training and testing sets.
    Uses stratified splitting to preserve class distribution (~95:5 ratio).
    """
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[preprocessing] Train set: {X_train.shape[0]} samples")
    print(f"[preprocessing] Test set:  {X_test.shape[0]} samples")
    print(f"[preprocessing] Train stroke rate: {y_train.mean():.4f}")
    print(f"[preprocessing] Test stroke rate:  {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test


def build_preprocessor(numerical_cols, categorical_cols):
    """
    Build a ColumnTransformer preprocessing pipeline.
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        # Engineered bins can be missing when BMI is missing.  Impute before
        # encoding so OneHotEncoder receives a consistent categorical value.
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            drop='if_binary',
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def preprocess_data(X_train, X_test, numerical_cols, categorical_cols):
    """
    Build, fit, and transform data using the preprocessing pipeline.
    Fit ONLY on training data to prevent data leakage.
    """
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

    print(f"[preprocessing] Preprocessed features: {len(feature_names)}")

    return X_train_processed, X_test_processed, feature_names, preprocessor
