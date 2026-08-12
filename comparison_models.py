# -*- coding: utf-8 -*-
"""
comparison_models.py -- Classical ML Model Comparison

Trains classical ML models for comparison against XGBoost:
  1. XGBoost (Main Model)
  2. Random Forest
  3. ExtraTrees
  4. LightGBM
  5. CatBoost

All models use the same data, same train/test split, and same evaluation.
"""

import numpy as np
import warnings

from xgboost import XGBClassifier
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
    Focuses strictly on requested comparison models.
    """
    if y_train is not None:
        n_neg = np.sum(np.array(y_train) == 0)
        n_pos = np.sum(np.array(y_train) == 1)
        scale_weight = n_neg / max(n_pos, 1)
    else:
        scale_weight = 1

    models = {
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=scale_weight,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            use_label_encoder=False
        ),
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

            # Resample within each fold.  Resampling before cross-validation
            # lets synthetic samples derived from a validation fold leak into
            # that fold's training data and inflates CV scores.
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
