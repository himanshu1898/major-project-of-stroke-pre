# -*- coding: utf-8 -*-
"""
comparison_models.py -- Classical ML Model Comparison

Trains classical ML models for comparison against XGBoost:
  1. Random Forest
  2. ExtraTrees
  3. LightGBM
  4. CatBoost

All models use the same data, same train/test split, and same evaluation.
XGBoost is excluded here because it is already covered by Experiments A-E.
"""

import numpy as np
import warnings

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from evaluation import evaluate_model
from config import RANDOM_STATE, CV_FOLDS, PLOTS_DIR


def get_comparison_models(y_train=None):
    """
    Return a dictionary of model name -> model instance.
    XGBoost is excluded (covered by xgboost_model.py experiments).
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_seed=RANDOM_STATE,
            verbose=0
        ),
    }

    return models


def run_comparison(X_train, X_test, y_train, y_test, use_smote=True,
                   output_dir=PLOTS_DIR):
    """
    Train and evaluate comparison models.
    """
    print("\n" + "=" * 70)
    print("COMPARISON MODELS")
    print("=" * 70)

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_final, y_train_final = smote.fit_resample(X_train, y_train_arr)
        print(f"[comparison] SMOTE applied: {X_train.shape[0]} -> {X_train_final.shape[0]} samples")
        models = get_comparison_models(y_train=None)
    else:
        X_train_final = X_train
        y_train_final = y_train_arr
        models = get_comparison_models(y_train=y_train_arr)

    all_results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = ['accuracy', 'recall', 'f1', 'roc_auc']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if use_smote:
                print(f"  [CV Validation] 5-Fold Stratified CV: SMOTE applied strictly inside training folds via ImbPipeline (validation folds untouched)")
            cv_estimator = (
                ImbPipeline([('smote', SMOTE(random_state=RANDOM_STATE)), ('model', model)])
                if use_smote else model
            )
            cv_results = cross_validate(
                cv_estimator, X_train, y_train_arr,
                cv=cv, scoring=scoring,
                return_train_score=False,
                n_jobs=1
            )

        for metric in scoring:
            key = f'test_{metric}'
            scores = cv_results[key]
            print(f"  CV {metric:>10s}: {scores.mean():.4f} +/- {scores.std():.4f}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_final, y_train_final)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(
            y_test_arr, y_pred, y_proba,
            name, output_dir
        )

        all_results[name] = metrics

    return all_results
