# -*- coding: utf-8 -*-
"""
evaluation.py -- Comprehensive Model Evaluation

Computes all 13 classification and probability metrics:
  1. Accuracy
  2. Precision
  3. Recall
  4. Specificity
  5. F1-score
  6. ROC-AUC
  7. PR-AUC
  8. MCC
  9. Cohen's Kappa
  10. False Positive Rate (FPR)
  11. False Negative Rate (FNR)
  12. Brier Score
  13. Log Loss

Also generates diagnostic plots: Confusion Matrix, ROC Curve, PR Curve.
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, roc_curve,
    precision_recall_curve, brier_score_loss, log_loss
)

from config import PLOTS_DIR


def compute_metrics(y_true, y_pred, y_proba):
    """
    Compute all 13 classification and probability metrics.
    """
    # Explicit labels keep the metric routine robust even if a model predicts
    # only one class on a small test split.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': specificity,
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'FPR': fpr,
        'FNR': fnr,
        'Brier Score': brier_score_loss(y_true, y_proba),
        'Log Loss': log_loss(y_true, y_proba),
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Stroke', 'Stroke'],
        yticklabels=['No Stroke', 'Stroke']
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_proba, title, save_path):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2196F3', linewidth=2,
             label=f'ROC AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, y_proba, title, save_path):
    """Generate and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)
    baseline = np.mean(y_true)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#4CAF50', linewidth=2,
             label=f'PR AUC = {ap_score:.4f}')
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                label=f'Baseline (prevalence) = {baseline:.4f}')
    plt.fill_between(recall, precision, alpha=0.1, color='#4CAF50')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(y_true, y_pred, y_proba, model_name, output_dir=PLOTS_DIR):
    """
    Consolidated evaluation: compute all metrics and generate diagnostic plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    safe_name = model_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '').lower()

    plot_confusion_matrix(
        y_true, y_pred,
        f'Confusion Matrix - {model_name}',
        os.path.join(output_dir, f'cm_{safe_name}.png')
    )
    plot_roc_curve(
        y_true, y_proba,
        f'ROC Curve - {model_name}',
        os.path.join(output_dir, f'roc_{safe_name}.png')
    )
    plot_pr_curve(
        y_true, y_proba,
        f'Precision-Recall Curve - {model_name}',
        os.path.join(output_dir, f'pr_{safe_name}.png')
    )

    print(f"\n[evaluation] {model_name}")
    print(f"  Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"  Precision:   {metrics['Precision']:.4f}")
    print(f"  Recall:      {metrics['Recall']:.4f}  <-- (critical for stroke detection)")
    print(f"  Specificity: {metrics['Specificity']:.4f}")
    print(f"  F1:          {metrics['F1']:.4f}")
    print(f"  ROC-AUC:     {metrics['ROC-AUC']:.4f}")
    print(f"  PR-AUC:      {metrics['PR-AUC']:.4f}")
    print(f"  MCC:         {metrics['MCC']:.4f}")
    print(f"  Kappa:       {metrics['Kappa']:.4f}")
    print(f"  FPR:         {metrics['FPR']:.4f}")
    print(f"  FNR:         {metrics['FNR']:.4f}")
    print(f"  Brier Score: {metrics['Brier Score']:.4f}")
    print(f"  Log Loss:    {metrics['Log Loss']:.4f}")

    return metrics


# Alias for backward compatibility
evaluate_and_plot = evaluate_model


def save_metrics_table(all_results, save_path):
    """
    Save comparison table of metrics for models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(all_results).T
    df.index.name = 'Model'
    df = df.round(4)
    df.to_csv(save_path)
    print(f"[evaluation] Metrics table saved to {save_path}")
    return df
