# -*- coding: utf-8 -*-
"""
feature_selection.py -- Boruta Feature Selection

Uses Boruta (with Random Forest as estimator) to identify statistically
important features for stroke prediction.

CRITICAL: Fit ONLY on training data to prevent data leakage.
"""

import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from config import RANDOM_STATE, FEATURE_SEL_DIR


def run_boruta(X_train, y_train, feature_names, output_dir=FEATURE_SEL_DIR, return_ranking=False):
    """
    Perform Boruta feature selection on preprocessed training data.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("BORUTA FEATURE SELECTION")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    boruta_selector = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=100,
        random_state=RANDOM_STATE,
        verbose=0
    )

    print("[feature_selection] Running Boruta...")
    y_array = np.array(y_train)
    boruta_selector.fit(X_train, y_array)

    selected_mask = boruta_selector.support_
    tentative_mask = boruta_selector.support_weak_
    ranking = boruta_selector.ranking_

    feature_names = list(feature_names)
    selected_names = [f for f, s in zip(feature_names, selected_mask) if s]
    tentative_names = [f for f, s in zip(feature_names, tentative_mask) if s]
    rejected_names = [f for f, s, t in zip(feature_names, selected_mask, tentative_mask)
                      if not s and not t]

    print(f"\n--- Boruta Results ---")
    print(f"Original features:  {len(feature_names)}")
    print(f"Selected features:  {len(selected_names)}")
    print(f"Tentative features: {len(tentative_names)}")
    print(f"Rejected features:  {len(rejected_names)}")

    print(f"\nSelected features:")
    for name in selected_names:
        print(f"  [+] {name}")

    if tentative_names:
        print(f"\nTentative features (borderline):")
        for name in tentative_names:
            print(f"  [?] {name}")

    print(f"\nRejected features:")
    for name in rejected_names:
        print(f"  [-] {name}")

    final_mask = selected_mask | tentative_mask
    # A strict Boruta run can reject every feature on a small or noisy split.
    # Keep the best-ranked feature so the downstream estimators have a valid
    # feature matrix and the pipeline still produces a diagnostic result.
    if not final_mask.any():
        best_feature_index = int(np.argmin(ranking))
        final_mask[best_feature_index] = True
        tentative_names = [feature_names[best_feature_index]]
        print("[feature_selection] Boruta selected no features; "
              f"using best-ranked fallback: {tentative_names[0]}")
    final_names = [f for f, s in zip(feature_names, final_mask) if s]

    # 1. Save selected_features.txt
    txt_path = os.path.join(output_dir, 'selected_features.txt')
    with open(txt_path, 'w') as f:
        f.write("BORUTA FEATURE SELECTION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original features:  {len(feature_names)}\n")
        f.write(f"Selected features:  {len(selected_names)}\n")
        f.write(f"Tentative features: {len(tentative_names)}\n")
        f.write(f"Rejected features:  {len(rejected_names)}\n\n")

        f.write("SELECTED FEATURES:\n")
        for name in selected_names:
            f.write(f"  {name}\n")

        if tentative_names:
            f.write("\nTENTATIVE FEATURES:\n")
            for name in tentative_names:
                f.write(f"  {name}\n")

        f.write("\nREJECTED FEATURES:\n")
        for name in rejected_names:
            f.write(f"  {name}\n")

        f.write(f"\nFINAL FEATURES USED (selected + tentative):\n")
        for name in final_names:
            f.write(f"  {name}\n")

    # 2. Save feature_selection_results.csv
    status_list = []
    for f in feature_names:
        if f in selected_names:
            status_list.append('Selected')
        elif f in tentative_names:
            status_list.append('Tentative')
        else:
            status_list.append('Rejected')

    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Status': status_list,
        'Ranking': ranking
    }).sort_values('Ranking')

    csv_path = os.path.join(output_dir, 'feature_selection_results.csv')
    results_df.to_csv(csv_path, index=False)

    print(f"\n[feature_selection] Results saved to {txt_path} and {csv_path}")

    if return_ranking:
        return final_mask, final_names, ranking
    return final_mask, final_names


def analyze_top_n_features(feature_names, ranking, boruta_selected, n=10, output_dir=FEATURE_SEL_DIR):
    """
    Extract and analyze Top-N features based on Boruta ranking.
    Compares Boruta-selected features with the Top-N ranked features.

    Saves top_10_feature_ranking.csv to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        'Feature': list(feature_names),
        'Ranking': ranking
    })
    df['Boruta_Status'] = df['Feature'].apply(
        lambda f: 'Selected' if f in boruta_selected else 'Rejected'
    )
    # Sort by ranking ascending (rank 1 is best)
    df = df.sort_values(by=['Ranking', 'Feature']).reset_index(drop=True)
    df['In_Top_10'] = [True if i < n else False for i in range(len(df))]

    top_n_df = df.head(n).copy()

    csv_path = os.path.join(output_dir, f'top_{n}_feature_ranking.csv')
    top_n_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print(f"TOP-{n} FEATURE ANALYSIS")
    print("=" * 70)
    print(f"\nTop-{n} Features by Boruta Ranking:")
    print(f"{'Rank':<6} {'Feature':<32} {'Boruta Status':<16} {'In Top 10?':<10}")
    print("-" * 66)
    for _, row in top_n_df.iterrows():
        print(f"{row['Ranking']:<6} {row['Feature']:<32} {row['Boruta_Status']:<16} {'Yes':<10}")

    overlap = set(boruta_selected).intersection(set(top_n_df['Feature']))
    borderline = set(top_n_df['Feature']) - set(boruta_selected)

    print(f"\nSummary:")
    print(f"  - Boruta selected features: {len(boruta_selected)}")
    print(f"  - Top-{n} features captured: {len(top_n_df)}")
    print(f"  - Confirmed Boruta features in Top-{n}: {len(overlap)} / {len(boruta_selected)}")
    if borderline:
        borderline_sorted = sorted(list(borderline))
        print(f"  - Additional borderline features in Top-{n}: {len(borderline)} ({', '.join(borderline_sorted)})")
    print(f"[feature_selection] Top-{n} ranking saved to {csv_path}")

    return top_n_df


def apply_feature_selection(X, mask):
    """
    Apply Boruta selection mask to feature matrix.
    """
    return X[:, mask]

