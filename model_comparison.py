# -*- coding: utf-8 -*-
"""
model_comparison.py -- Final Model Comparison Report

Generates two major comparison tables:
  1. xgboost_experiments.csv (XGBoost A, B, C, D, E)
  2. model_comparison.csv (XGBoost, Random Forest, ExtraTrees, LightGBM, CatBoost)

Saves all tables to results/metrics/
"""

import pandas as pd
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import METRICS_DIR, PLOTS_DIR


def generate_comparison_report(xgb_results, comparison_results, output_dir=METRICS_DIR):
    """
    Generate major comparison tables and save as CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Table 1: XGBoost Experiments (A-E)
    # ----------------------------------------------------------
    df_xgb = pd.DataFrame(xgb_results).T
    df_xgb.index.name = 'Experiment'
    df_xgb = df_xgb.round(4)

    xgb_path = os.path.join(output_dir, 'xgboost_experiments.csv')
    df_xgb.to_csv(xgb_path)

    print("\n--- Table 1: XGBoost Experiments (A-E) ---")
    print(df_xgb.to_string())
    print(f"\nSaved to {xgb_path}")

    # ----------------------------------------------------------
    # 2. Table 2: Model Comparison (XGBoost + 4 Classifiers)
    # ----------------------------------------------------------
    df_comp = pd.DataFrame(comparison_results).T
    df_comp.index.name = 'Model'
    df_comp = df_comp.round(4)

    comp_path = os.path.join(output_dir, 'model_comparison.csv')
    df_comp.to_csv(comp_path)

    print("\n--- Table 2: Model Comparison ---")
    print(df_comp.to_string())
    print(f"\nSaved to {comp_path}")

    # ----------------------------------------------------------
    # 3. Full Combined Comparison
    # ----------------------------------------------------------
    all_results = {}
    all_results.update(xgb_results)
    all_results.update(comparison_results)

    df_all = pd.DataFrame(all_results).T
    df_all.index.name = 'Model'
    df_all = df_all.round(4)

    full_path = os.path.join(output_dir, 'full_comparison.csv')
    df_all.to_csv(full_path)

    # ----------------------------------------------------------
    # Focused Comparisons
    # ----------------------------------------------------------
    smote_keys = {
        'Without SMOTE': 'XGBoost A (Original)',
        'With SMOTE': 'XGBoost B (Original+SMOTE)',
    }
    _print_focused_comparison(xgb_results, smote_keys,
                              "XGBoost: SMOTE Effect", output_dir,
                              'xgboost_smote_comparison.csv')

    boruta_keys = {
        'Without Boruta': 'XGBoost C (Engineered)',
        'With Boruta': 'XGBoost D (Eng+Boruta)',
    }
    _print_focused_comparison(xgb_results, boruta_keys,
                              "XGBoost: Boruta Effect", output_dir,
                              'xgboost_boruta_comparison.csv')

    pipeline_keys = {
        'Baseline (A)': 'XGBoost A (Original)',
        'Final Pipeline (E)': 'XGBoost E (Eng+Boruta+SMOTE)',
    }
    _print_focused_comparison(xgb_results, pipeline_keys,
                              "XGBoost: Baseline vs Final Pipeline", output_dir,
                              'xgboost_baseline_vs_final.csv')

    # Visualizations
    _plot_model_comparison(comparison_results, PLOTS_DIR)

    print(f"\n[model_comparison] All comparison tables saved to {output_dir}/")


def _print_focused_comparison(results, key_mapping, title, output_dir, filename):
    """Print and save a focused 2-model comparison."""
    print(f"\n--- {title} ---")

    comparison = {}
    for label, key in key_mapping.items():
        if key in results:
            comparison[label] = results[key]

    if len(comparison) < 2:
        print("  (Insufficient data for comparison)")
        return

    df = pd.DataFrame(comparison).T
    df.index.name = 'Configuration'
    df = df.round(4)

    keys = list(comparison.keys())
    delta = {}
    for col in df.columns:
        diff = df.loc[keys[1], col] - df.loc[keys[0], col]
        delta[col] = diff
    df.loc['Delta'] = delta
    df = df.round(4)

    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path)

    print(df.to_string())

    for metric in ['Recall', 'F1', 'ROC-AUC', 'PR-AUC']:
        if metric in delta:
            d = delta[metric]
            direction = "(UP)" if d > 0 else "(DOWN)" if d < 0 else "(SAME)"
            print(f"  {metric}: {d:+.4f} {direction}")

    print(f"Saved to {save_path}")


def _plot_model_comparison(comparison_results, plot_dir=PLOTS_DIR):
    """Generate bar chart comparing key metrics across comparison models."""
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.DataFrame(comparison_results).T

    key_metrics = ['Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'MCC']
    available = [m for m in key_metrics if m in df.columns]

    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.5 * len(available)))
    if len(available) == 1:
        axes = [axes]

    colors = sns.color_palette('viridis', len(df))

    for ax, metric in zip(axes, available):
        values = df[metric].sort_values(ascending=True)
        bars = ax.barh(range(len(values)), values.values, color=colors)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index, fontsize=10)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'{metric} - Model Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for j, v in enumerate(values.values):
            ax.text(v + 0.005, j, f'{v:.4f}', va='center', fontsize=9)

    plt.suptitle('Model Comparison - Key Metrics', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, 'model_comparison_overview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[model_comparison] Comparison plot saved to {save_path}")
