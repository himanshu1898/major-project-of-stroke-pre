# -*- coding: utf-8 -*-
"""
xgboost_model.py -- XGBoost Primary Model Experiments

XGBoost is the PRIMARY model for this project.
Five controlled experiments isolate the effect of each pipeline component:

  Experiment A: XGBoost A (Original features + class weight)
  Experiment B: XGBoost B (Original features + SMOTE)
  Experiment C: XGBoost C (Engineered features + class weight)
  Experiment D: XGBoost D (Engineered features + Boruta + class weight)
  Experiment E: XGBoost E (Engineered features + Boruta + SMOTE)
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from evaluation import evaluate_model
from config import RANDOM_STATE, CV_FOLDS, PLOTS_DIR

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
}


def _get_xgb_model(use_scale_pos_weight=True, y_train=None):
    params = XGBOOST_PARAMS.copy()

    if use_scale_pos_weight and y_train is not None:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        params['scale_pos_weight'] = n_neg / max(n_pos, 1)
        print(f"  scale_pos_weight = {params['scale_pos_weight']:.2f}")
    else:
        params['scale_pos_weight'] = 1

    return XGBClassifier(**params)


def _run_cv(model, X_train, y_train, use_smote=False, experiment_name=""):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if use_smote:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('model', model)
        ])
        estimator = pipeline
    else:
        estimator = model

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    cv_results = cross_validate(
        estimator, X_train, y_train,
        cv=cv, scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    print(f"\n  {experiment_name} -- {CV_FOLDS}-Fold CV Results:")
    for metric in scoring:
        key = f'test_{metric}'
        scores = cv_results[key]
        print(f"    {metric:>12s}: {scores.mean():.4f} +/- {scores.std():.4f}")

    return cv_results


def run_single_experiment(X_train, X_test, y_train, y_test,
                          experiment_name, use_smote=False,
                          output_dir=PLOTS_DIR):
    print(f"\n{'='*50}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*50}")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  SMOTE: {'Yes' if use_smote else 'No'}")

    y_train_arr = np.array(y_train)
    model = _get_xgb_model(
        use_scale_pos_weight=(not use_smote),
        y_train=y_train_arr
    )

    _run_cv(model, X_train, y_train_arr, use_smote=use_smote,
            experiment_name=experiment_name)

    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_final, y_train_final = smote.fit_resample(X_train, y_train_arr)
        model = _get_xgb_model(use_scale_pos_weight=False, y_train=None)
    else:
        X_train_final = X_train
        y_train_final = y_train_arr

    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(
        np.array(y_test), y_pred, y_proba,
        experiment_name, output_dir
    )

    return metrics, model


def run_all_xgboost_experiments(
    X_train_orig, X_test_orig,
    X_train_eng, X_test_eng,
    X_train_boruta, X_test_boruta,
    y_train, y_test,
    output_dir=PLOTS_DIR
):
    print("\n" + "=" * 70)
    print("XGBOOST EXPERIMENTS (A through E)")
    print("=" * 70)

    all_results = {}
    all_models = {}

    # Experiment A: Original features, no SMOTE
    metrics, model = run_single_experiment(
        X_train_orig, X_test_orig, y_train, y_test,
        "XGBoost A (Original)", use_smote=False, output_dir=output_dir
    )
    all_results['XGBoost A (Original)'] = metrics
    all_models['A'] = model

    # Experiment B: Original features + SMOTE
    metrics, model = run_single_experiment(
        X_train_orig, X_test_orig, y_train, y_test,
        "XGBoost B (Original+SMOTE)", use_smote=True, output_dir=output_dir
    )
    all_results['XGBoost B (Original+SMOTE)'] = metrics
    all_models['B'] = model

    # Experiment C: Engineered features, no SMOTE
    metrics, model = run_single_experiment(
        X_train_eng, X_test_eng, y_train, y_test,
        "XGBoost C (Engineered)", use_smote=False, output_dir=output_dir
    )
    all_results['XGBoost C (Engineered)'] = metrics
    all_models['C'] = model

    # Experiment D: Engineered features + Boruta, no SMOTE
    metrics, model = run_single_experiment(
        X_train_boruta, X_test_boruta, y_train, y_test,
        "XGBoost D (Eng+Boruta)", use_smote=False, output_dir=output_dir
    )
    all_results['XGBoost D (Eng+Boruta)'] = metrics
    all_models['D'] = model

    # Experiment E: Engineered features + Boruta + SMOTE (final pipeline)
    metrics, model = run_single_experiment(
        X_train_boruta, X_test_boruta, y_train, y_test,
        "XGBoost E (Eng+Boruta+SMOTE)", use_smote=True, output_dir=output_dir
    )
    all_results['XGBoost E (Eng+Boruta+SMOTE)'] = metrics
    all_models['E'] = model

    return all_results, all_models
