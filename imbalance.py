# -*- coding: utf-8 -*-
"""
imbalance.py -- SMOTE Imbalance Handling

Uses SMOTE to address severe class imbalance in stroke dataset (~95:5).
Applied ONLY to training data. Test set distribution is NEVER modified.
"""

import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE, PLOTS_DIR


def apply_smote(X_train, y_train, random_state=RANDOM_STATE):
    """
    Apply SMOTE to training data only.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def record_distributions(y_original, y_train, y_resampled):
    """
    Print class distributions at each stage.
    """
    print("\n--- Class Distribution Report ---")

    def _print_dist(name, y):
        total = len(y)
        if hasattr(y, 'value_counts'):
            counts = y.value_counts()
        else:
            unique, cnts = np.unique(y, return_counts=True)
            counts = dict(zip(unique, cnts))

        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        print(f"\n{name} (n={total}):")
        print(f"  No Stroke (0): {n0} ({n0/total*100:.1f}%)")
        print(f"  Stroke    (1): {n1} ({n1/total*100:.1f}%)")
        print(f"  Ratio:         {n0/max(n1,1):.1f}:1")

    _print_dist("Original Dataset", y_original)
    _print_dist("Training Set (before SMOTE)", y_train)
    _print_dist("Training Set (after SMOTE)", y_resampled)


def plot_distributions(y_original, y_train, y_resampled, output_dir=PLOTS_DIR):
    """
    Visualize class distributions: original, train pre-SMOTE, train post-SMOTE.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _get_counts(y):
        if hasattr(y, 'value_counts'):
            vc = y.value_counts().sort_index()
            return vc.get(0, 0), vc.get(1, 0)
        else:
            unique, cnts = np.unique(y, return_counts=True)
            d = dict(zip(unique, cnts))
            return d.get(0, 0), d.get(1, 0)

    orig_0, orig_1 = _get_counts(y_original)
    train_0, train_1 = _get_counts(y_train)
    smote_0, smote_1 = _get_counts(y_resampled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    labels = ['No Stroke\n(0)', 'Stroke\n(1)']
    colors = ['#4CAF50', '#F44336']

    datasets = [
        ('Original Dataset', [orig_0, orig_1]),
        ('Train (Before SMOTE)', [train_0, train_1]),
        ('Train (After SMOTE)', [smote_0, smote_1]),
    ]

    for ax, (title, counts) in zip(axes, datasets):
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                    str(count), ha='center', fontweight='bold', fontsize=11)

    plt.suptitle('SMOTE: Class Distribution Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'smote_distribution_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[imbalance] Distribution plot saved to {save_path}")


def run_smote_experiment(X_train, y_train, y_original, output_dir=PLOTS_DIR):
    """
    Full SMOTE experiment: apply SMOTE, record distributions, generate plots.
    """
    print("\n" + "=" * 70)
    print("SMOTE EXPERIMENT")
    print("=" * 70)

    X_resampled, y_resampled = apply_smote(X_train, y_train)
    record_distributions(y_original, y_train, y_resampled)
    plot_distributions(y_original, y_train, y_resampled, output_dir)

    return X_resampled, y_resampled
