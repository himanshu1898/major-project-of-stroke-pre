# -*- coding: utf-8 -*-
"""
main.py -- Stroke Prediction 50% Milestone Pipeline Entry Point

Orchestrates the clean, research-oriented ML pipeline:
  1. Load dataset
  2. Validate dataset
  3. Run EDA
  4. Split data (Stratified Train/Test Split)
  5. Preprocess (Original features)
  6. Feature Engineering (9 derived features)
  7. Boruta Feature Selection
  8. SMOTE Experiment
  9. XGBoost Experiments A-E
  10. Comparison Models (XGBoost, Random Forest, ExtraTrees, LightGBM, CatBoost)
  11. Model Evaluation & Comparison Reports
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Config & Modules
from config import (
    DATA_PATH, EDA_DIR, METRICS_DIR, PLOTS_DIR, FEATURE_SEL_DIR,
    NUMERICAL_COLS, CATEGORICAL_COLS, RANDOM_STATE
)
from preprocessing import load_data, split_data, preprocess_data
from eda import run_eda
from feature_engineering import create_engineered_features, get_engineered_columns
from feature_selection import run_boruta, apply_feature_selection
from imbalance import run_smote_experiment
from xgboost_model import run_all_xgboost_experiments
from comparison_models import run_comparison
from evaluation import save_metrics_table
from model_comparison import generate_comparison_report


def create_directories():
    """Create output directories."""
    for d in [EDA_DIR, METRICS_DIR, PLOTS_DIR, FEATURE_SEL_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[main] Output directories verified")


def validate_data(df):
    """Validate dataset structure."""
    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    expected_rows = 5110
    expected_cols = 11

    assert df.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df.shape[0]}"
    assert df.shape[1] == expected_cols, f"Expected {expected_cols} cols, got {df.shape[1]}"
    assert 'stroke' in df.columns, "'stroke' column missing"
    assert 'id' not in df.columns, "'id' column should have been dropped"

    stroke_count = df['stroke'].sum()
    assert stroke_count == 249, f"Expected 249 stroke cases, got {stroke_count}"

    print(f"[OK] Shape: {df.shape}")
    print(f"[OK] Target: {stroke_count} stroke / {len(df) - stroke_count} no-stroke")
    print(f"[OK] Missing BMI: {df['bmi'].isnull().sum()} values")
    print("[main] Data validation passed")


def main():
    """Run the 50% milestone stroke prediction pipeline."""

    print("+" + "=" * 68 + "+")
    print("|  STROKE PREDICTION -- 50% MILESTONE ML PIPELINE                  |")
    print("|  Primary Model: XGBoost                                          |")
    print("|  Dataset: healthcare-dataset-stroke-data.csv                      |")
    print("|  Random Seed: 42                                                 |")
    print("+" + "=" * 68 + "+")

    create_directories()

    # 1. Load Data
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)
    df = load_data(DATA_PATH)

    # 2. Validate Data
    validate_data(df)

    # 3. Perform EDA
    print("\n" + "=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    run_eda(df, output_dir=EDA_DIR)

    # 4. Train/Test Split
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("=" * 70)
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)

    # 5. Preprocess Original Features
    print("\n" + "=" * 70)
    print("STEP 5: PREPROCESS ORIGINAL FEATURES")
    print("=" * 70)
    X_train_orig, X_test_orig, feature_names_orig, preprocessor_orig = preprocess_data(
        X_train_raw, X_test_raw, NUMERICAL_COLS, CATEGORICAL_COLS
    )

    # 6. Feature Engineering
    print("\n" + "=" * 70)
    print("STEP 6: FEATURE ENGINEERING")
    print("=" * 70)
    X_train_eng_raw = create_engineered_features(X_train_raw)
    X_test_eng_raw = create_engineered_features(X_test_raw)

    eng_num_cols, eng_cat_cols = get_engineered_columns()

    X_train_eng, X_test_eng, feature_names_eng, preprocessor_eng = preprocess_data(
        X_train_eng_raw, X_test_eng_raw, eng_num_cols, eng_cat_cols
    )

    # 7. Boruta Feature Selection
    print("\n" + "=" * 70)
    print("STEP 7: BORUTA FEATURE SELECTION")
    print("=" * 70)
    boruta_mask, boruta_features = run_boruta(
        X_train_eng, y_train, feature_names_eng, output_dir=FEATURE_SEL_DIR
    )

    X_train_boruta = apply_feature_selection(X_train_eng, boruta_mask)
    X_test_boruta = apply_feature_selection(X_test_eng, boruta_mask)

    print(f"\n[main] Boruta: {X_train_eng.shape[1]} -> {X_train_boruta.shape[1]} features")

    # 8. SMOTE Experiment
    print("\n" + "=" * 70)
    print("STEP 8: SMOTE EXPERIMENT")
    print("=" * 70)
    y_original = df['stroke']
    X_train_smote, y_train_smote = run_smote_experiment(
        X_train_boruta, y_train, y_original, output_dir=PLOTS_DIR
    )

    # 9. XGBoost Experiments (A-E)
    print("\n" + "=" * 70)
    print("STEP 9: XGBOOST EXPERIMENTS (A-E)")
    print("=" * 70)
    xgb_results, xgb_models = run_all_xgboost_experiments(
        X_train_orig=X_train_orig, X_test_orig=X_test_orig,
        X_train_eng=X_train_eng, X_test_eng=X_test_eng,
        X_train_boruta=X_train_boruta, X_test_boruta=X_test_boruta,
        y_train=y_train, y_test=y_test,
        output_dir=PLOTS_DIR
    )

    save_metrics_table(xgb_results, os.path.join(METRICS_DIR, 'xgboost_experiments.csv'))

    # 10. Comparison Models
    print("\n" + "=" * 70)
    print("STEP 10: COMPARISON MODELS")
    print("=" * 70)
    comparison_results = run_comparison(
        X_train_boruta, X_test_boruta, y_train, y_test,
        use_smote=True, output_dir=PLOTS_DIR
    )

    save_metrics_table(comparison_results, os.path.join(METRICS_DIR, 'comparison_models.csv'))

    # 11. Model Comparison Report
    print("\n" + "=" * 70)
    print("STEP 11: MODEL COMPARISON REPORT")
    print("=" * 70)
    generate_comparison_report(xgb_results, comparison_results, output_dir=METRICS_DIR)

    # Final Summary
    _print_final_summary(xgb_results, comparison_results, boruta_features)

    print("\n" + "+" + "=" * 68 + "+")
    print("|  50% MILESTONE PIPELINE COMPLETED SUCCESSFULLY                   |")
    print("+" + "=" * 68 + "+")


def _print_final_summary(xgb_results, comparison_results, boruta_features):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n--- XGBoost Experiments ---")
    for name, metrics in xgb_results.items():
        print(f"  {name}:")
        print(f"    Recall={metrics['Recall']:.4f}  F1={metrics['F1']:.4f}  "
              f"ROC-AUC={metrics['ROC-AUC']:.4f}  PR-AUC={metrics['PR-AUC']:.4f}")

    best_xgb = max(xgb_results.keys(), key=lambda k: xgb_results[k].get('F1', 0))
    print(f"\n  -> Best XGBoost (by F1): {best_xgb}")

    if 'XGBoost A (Original)' in xgb_results and 'XGBoost B (Original+SMOTE)' in xgb_results:
        recall_diff = (xgb_results['XGBoost B (Original+SMOTE)']['Recall'] -
                      xgb_results['XGBoost A (Original)']['Recall'])
        print(f"\n  SMOTE effect on Recall: {recall_diff:+.4f}")

    if 'XGBoost C (Engineered)' in xgb_results and 'XGBoost D (Eng+Boruta)' in xgb_results:
        f1_diff = (xgb_results['XGBoost D (Eng+Boruta)']['F1'] -
                  xgb_results['XGBoost C (Engineered)']['F1'])
        print(f"  Boruta effect on F1:    {f1_diff:+.4f}")
        print(f"  Boruta selected: {len(boruta_features)} features")

    if comparison_results:
        print("\n--- Comparison Models ---")
        best_comp_f1 = max(comparison_results.keys(), key=lambda k: comparison_results[k].get('F1', 0))
        best_comp_recall = max(comparison_results.keys(), key=lambda k: comparison_results[k].get('Recall', 0))
        best_comp_prauc = max(comparison_results.keys(), key=lambda k: comparison_results[k].get('PR-AUC', 0))

        print(f"  Best model by F1:     {best_comp_f1} (F1={comparison_results[best_comp_f1]['F1']:.4f})")
        print(f"  Best model by Recall: {best_comp_recall} (Recall={comparison_results[best_comp_recall]['Recall']:.4f})")
        print(f"  Best model by PR-AUC: {best_comp_prauc} (PR-AUC={comparison_results[best_comp_prauc]['PR-AUC']:.4f})")

    print("\n--- Output Files ---")
    for root, dirs, files in os.walk('results'):
        for f in sorted(files):
            filepath = os.path.join(root, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {filepath} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
