# -*- coding: utf-8 -*-
"""
config.py -- Central Configuration File

Centralizes all shared constants, hyperparameters, and directory paths
to ensure consistency and reproducibility across the pipeline.
"""

import os

# Reproducibility
RANDOM_STATE = 42

# Data Splitting & Cross-Validation
TEST_SIZE = 0.20
CV_FOLDS = 5

# File Paths
DATA_PATH = 'healthcare-dataset-stroke-data.csv'
RESULTS_DIR = 'results'
EDA_DIR = os.path.join(RESULTS_DIR, 'eda')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
FEATURE_SEL_DIR = os.path.join(RESULTS_DIR, 'feature_selection')

# Feature definitions
NUMERICAL_COLS = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
TARGET_COL = 'stroke'
