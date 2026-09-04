# 🧠 Stroke Prediction ML Pipeline — 1st-Half Milestone Final Report

**Academic Presentation & Evaluation Reference**  
*Course:* Major Project (Machine Learning / Healthcare Informatics)  
*Milestone:* 1st Half (Exploratory Analysis, Preprocessing, Feature Engineering, Boruta Selection, SMOTE Imbalance Analysis, Model Training, Cross-Validation, Multi-Metric Evaluation, Two-Tier Ranking & Final Model Selection)

---

## 1. Executive Summary & Selected Final Model

### 1.1 Selected Final Model: ExtraTrees Classifier
Following extensive ablation experiments across 5 XGBoost configurations and benchmark evaluations against 4 distinct model families under 5-fold Stratified Cross-Validation and 10 clinical metrics, **ExtraTrees** is selected as the **Final 1st-Half Model**.

| Metric | ExtraTrees Result | Clinical Implication |
|:---|:---:|:---|
| **Recall (Sensitivity)** | **78.00%** | **Catches 39 out of 50 stroke patients** in unseen holdout test data (vs. 15–20 for boosting models). |
| **ROC-AUC** | **0.8371** | **Highest discriminatory AUC** among all tested model families. |
| **PR-AUC** | **0.2110** | **Highest Precision-Recall AUC** among comparison models under extreme 19.5:1 imbalance. |
| **MCC** | **0.2926** | **Highest Matthews Correlation Coefficient**, indicating strongest overall balanced prediction. |
| **F1 Score** | **26.99%** | Competitive harmonic mean of precision and recall. |
| **Accuracy** | **79.35%** | Realistic screening accuracy; avoids the ~95% majority-class "lazy predictor" trap. |

> [!IMPORTANT]
> **Clinical Justification for Model Selection:**
> In clinical stroke triage and preventative screening, **minimizing false negatives (missed strokes) is paramount**. A missed stroke (False Negative) can result in permanent neurological deficit or fatality. A false alarm (False Positive) merely triggers routine non-invasive secondary testing (Doppler ultrasound, lipid panel). While XGBoost configurations achieved higher accuracy by predicting the majority class conservatively, **ExtraTrees identified 78% of active stroke cases**, making it the superior clinical screening tool.

---

## 2. Project Scope & Boundary Enforcement

### 2.1 1st-Half Scope (Completed)
- Dataset validation and structural integrity checks
- Comprehensive Exploratory Data Analysis (EDA)
- 80/20 Stratified Train/Test Split (`random_state=42`)
- Leak-free preprocessing (Median Imputation, StandardScaler, OneHotEncoder fit solely on train)
- Clinically motivated feature engineering (9 domain interactions expanding 15 → 38 features)
- Boruta all-relevant feature selection (confirmed 7 statistically significant features)
- Top-10 feature comparative ranking analysis
- Verified fold-isolated SMOTE inside cross-validation pipelines
- XGBoost ablation experiments across 5 configurations (A through E)
- Comparison model family evaluations (ExtraTrees, Random Forest, CatBoost, LightGBM)
- 5-fold Stratified Cross-Validation across all models
- 10-metric comprehensive evaluation
- Two-tier ranking system (XGBoost ablation ranking + Model family comparison)
- Final model selection and academic defense

### 2.2 Postponed to 2nd-Half Milestone
* Explainable AI (TreeSHAP, KernelSHAP, LIME force plots)
* Ensemble, stacking, soft-voting, and blending architectures
* Automated hyperparameter optimization (Optuna, Bayesian search, GridSearchCV)
* Deep learning / Neural Network architectures
* Probability calibration curves (Isotonic/Platt scaling)
* Web production deployment and clinical API integration

---

## 3. Dataset Architecture & Exploratory Data Analysis

### 3.1 Dataset Profile
* **File:** `healthcare-dataset-stroke-data.csv`
* **Sample Count:** 5,110 patients
* **Target Feature:** `stroke` (Binary: `0` = No Stroke, `1` = Stroke)
* **Class Distribution:**
  * No Stroke: 4,861 patients (95.13%)
  * Stroke: 249 patients (4.87%)
  * **Imbalance Ratio:** ~19.5 : 1 (Extreme clinical rarity)
* **Missingness:** `bmi` contained 201 missing entries (3.93%); imputed using median calculated strictly on the training partition. The administrative `id` column was dropped.

### 3.2 EDA Findings (7 Diagnostic Figures in `results/eda/`)
1. `01_target_distribution.png`: Demonstrates severe rarity of stroke events, establishing that accuracy is a misleading metric and Sensitivity (Recall) must guide decision making.
2. `02_numerical_distributions.png`: Shows bimodal `age` distribution, right-skewed `avg_glucose_level` with a diabetic tail (>200 mg/dL), and normally distributed `bmi`.
3. `03_categorical_distributions.png`: Profiles marital status, employment type, residence, and smoking habits.
4. `04_bivariate_numerical.png` & `05_bivariate_numerical_boxplots.png`: Confirms stroke incidents concentrate overwhelmingly in elderly cohorts (median age > 70) and elevated glucose cohorts.
5. `06_bivariate_categorical.png`: Hypertension and heart disease empirically multiply stroke likelihood by over 3×.
6. `07_correlation_heatmap.png`: Identifies `age` as the primary linear correlate with stroke, followed by `avg_glucose_level`, `heart_disease`, and `hypertension`.

---

## 4. Feature Engineering (9 Domain Interactions)

To capture clinical synergy between risk factors, 9 new features were engineered:

| # | Feature Name | Formula / Logic | Clinical Rationale |
|---|---|---|---|
| 1 | `age_glucose` | `age * avg_glucose_level` | Metabolic aging accelerates cerebrovascular calcification. |
| 2 | `age_bmi` | `age * bmi` | Combined impact of biological age and adiposity. |
| 3 | `age_hypertension` | `age * hypertension` | Arterial stiffness compounded by chronic high blood pressure. |
| 4 | `age_heart_disease` | `age * heart_disease` | Cardiogenic emboli risk amplified in geriatric patients. |
| 5 | `glucose_hypertension` | `avg_glucose_level * hypertension` | Synergistic microvascular damage from hyperglycemia and high blood pressure. |
| 6 | `cardio_risk` | `hypertension + heart_disease` | Additive composite score of primary cardiovascular comorbidities (0, 1, or 2). |
| 7 | `age_group` | Binning: Child (<18), Young Adult (18-35), Adult (36-50), Middle Aged (51-65), Senior (>65) | Non-linear transition points in stroke incidence. |
| 8 | `bmi_category` | WHO Bins: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>=30) | Standard clinical categorizations of nutritional risk. |
| 9 | `glucose_risk` | Clinical Bins: Low (<70), Normal (70-99), Pre-diabetic (100-125), Diabetic (>=126) | Diabetes-specific vascular vulnerability categorization. |

After one-hot encoding the categorical bins, the feature space expanded from 15 to **38 features**.

---

## 5. Boruta Feature Selection & Top-10 Comparative Analysis

### 5.1 Boruta Selection
Boruta (Random Forest wrapper, `n_estimators=200`, `max_depth=7`, `class_weight='balanced'`) confirmed **7 statistically significant features** (Rank 1):
1. `age`
2. `avg_glucose_level`
3. `bmi`
4. `age_glucose` (Domain-engineered)
5. `age_bmi` (Domain-engineered)
6. `age_group_senior` (Domain-engineered)
7. `age_group_young_adult` (Domain-engineered)

*Empirical Confirmation:* **4 of the 7 confirmed features are domain-engineered**, proving that non-linear interaction terms provided critical signal over raw indicators alone.

### 5.2 Top-10 Feature Comparative Analysis (Analysis-Only)
Features were ranked by Boruta importance score to evaluate borderline candidates (`top_10_feature_ranking.csv`):
* **Confirmed Features (Rank 1):** `age`, `age_bmi`, `age_glucose`, `age_group_senior`, `age_group_young_adult`, `avg_glucose_level`, `bmi`.
* **Borderline Features (Rank 2–4):** `cardio_risk` (Rank 2), `age_group_child` (Rank 3), `age_hypertension` (Rank 4).
* *Scope Clarification:* The Top-10 ranking is provided as comparative exploratory analysis. Downstream models were evaluated on the 7 statistically confirmed Boruta features to prevent unnecessary dimensionality expansion.

---

## 6. Imbalance Handling: Verified Fold-Isolated SMOTE

### 6.1 Data Leakage Prevention Verification
To guarantee complete scientific rigor:
1. **Never Applied to Test Partition:** The test set retains the authentic 4.87% clinical stroke incidence.
2. **Strictly Fold-Isolated in Cross-Validation:** Synthetic oversampling is wrapped inside `imblearn.pipeline.Pipeline`. In each fold of 5-fold CV, SMOTE is fit and applied **exclusively to the 4 training folds** ($N \approx 3,270 \rightarrow 6,222$), leaving the 5th validation fold ($N \approx 818$) completely untouched.
3. Console logging explicitly validates: `[CV Validation] 5-Fold Stratified CV: SMOTE applied strictly inside training folds via ImbPipeline (validation folds untouched)`.

### 6.2 Empirical SMOTE Research Finding
Comparing XGBoost A (no SMOTE) to XGBoost B (with SMOTE):
* **Recall dropped from 34.00% → 18.00% (-16.0%)**
* **F1 dropped from 26.98% → 20.93% (-6.05%)**
* **PR-AUC dropped from 21.36% → 16.51% (-4.85%)**
* *Key Research Finding:* SMOTE increased nominal class balance during training, but synthetic interpolation in high-dimensional feature space diluted the boundary definition for gradient boosting, causing XGBoost to predict the minority class more conservatively.

---

## 7. Ranking 1 — XGBoost Ablation Experiments (A through E)

Five controlled experiments evaluated the incremental contribution of Preprocessing, Feature Engineering, Boruta Selection, and SMOTE on XGBoost:

| Rank | Experiment | Features | Imbalance Strategy | Recall | F1 | ROC-AUC | PR-AUC | MCC | Accuracy | Specificity |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **XGBoost D (Eng + Boruta)** | **7 (Boruta)** | `scale_pos_weight` | **0.4000** | **0.2797** | 0.8150 | 0.1860 | **0.2437** | 0.8992 | 0.9249 |
| 🥈 **2** | **XGBoost A (Original Baseline)** | 15 (Raw) | `scale_pos_weight` | 0.3400 | 0.2698 | 0.8027 | **0.2136** | 0.2296 | 0.9100 | 0.9393 |
| 🥉 **3** | **XGBoost C (Engineered)** | 38 (Eng) | `scale_pos_weight` | 0.3000 | 0.2679 | 0.8146 | 0.1918 | 0.2274 | 0.9198 | 0.9516 |
| **4** | **XGBoost E (Eng + Boruta + SMOTE)**| 7 (Boruta) | SMOTE | 0.3600 | 0.2483 | **0.8296** | 0.1683 | 0.2086 | 0.8933 | 0.9208 |
| **5** | **XGBoost B (Original + SMOTE)** | 15 (Raw) | SMOTE | 0.1800 | 0.2093 | 0.7951 | 0.1651 | 0.1781 | **0.9335** | **0.9722** |

### 7.1 XGBoost Ablation Findings
* **Best XGBoost Variant: `XGBoost D (Eng + Boruta)`**
  * Achieved the highest F1 score (**0.2797**), highest Recall (**0.4000**), and highest MCC (**0.2437**) among all XGBoost experiments.
  * Pruning 31 noisy features via Boruta improved Recall over XGBoost C by **+10.0% (0.3000 → 0.4000)** and increased F1 by **+0.0119**.

---

## 8. Ranking 2 — Model-Family Comparison & Final Selection

To evaluate distinct learning paradigms, the best representative from each model family was benchmarked on the 20% holdout test set with 5-fold CV:

| Rank | Model Family | Architecture | Recall | ROC-AUC | PR-AUC | MCC | F1 | Accuracy | Specificity | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **1** | **ExtraTrees** | Extremely Randomized Trees | **0.7800** | **0.8371** | **0.2110** | **0.2926** | **0.2699** | 0.7935 | 0.7942 | **SELECTED FINAL MODEL** |
| 🥈 **2** | **Random Forest** | Bagged Decision Trees | 0.6200 | 0.8213 | 0.1819 | 0.2499 | 0.2551 | 0.8229 | 0.8333 | Benchmark |
| 🥉 **3** | **XGBoost (XGBoost D)** | Extreme Gradient Boosting | 0.4000 | 0.8150 | 0.1860 | 0.2437 | **0.2797** | 0.8992 | 0.9249 | Best XGBoost Variant |
| **4** | **CatBoost** | Categorical Boosting | 0.3200 | 0.8177 | 0.1614 | 0.2041 | 0.2462 | **0.9041** | 0.9342 | Benchmark |
| **5** | **LightGBM** | Histogram Gradient Boosting | 0.2200 | 0.8237 | 0.1533 | 0.1308 | 0.1789 | 0.9012 | **0.9362** | Benchmark |

---

## 9. Academic Defense Summary for College Presentation

When presenting the final model selection to evaluators:

1. **Why ExtraTrees over XGBoost?**
   * XGBoost achieves high specificity (>92%) and high overall accuracy (~90%) because it penalizes false alarms heavily. However, in doing so, it misses 60% of true stroke cases (Recall = 40%).
   * ExtraTrees randomized cut-point selection introduces variance reduction that allows minority clusters to assert branch ownership, yielding **78% sensitivity (39/50 strokes caught)**.
   * Furthermore, ExtraTrees achieved the **highest ROC-AUC (0.8371)** and **highest MCC (0.2926)**, proving that its superior sensitivity did not come at the expense of discriminatory ability.

2. **Why XGBoost D over other XGBoost variants?**
   * XGBoost D validates our feature engineering hypothesis: 4 engineered features combined with Boruta selection yielded a 10% gain in Recall over the full feature set.

3. **What did we learn about SMOTE?**
   * In extreme imbalance (19.5:1), standard synthetic interpolation can blur minority boundaries for gradient boosted trees. Honest reporting of this phenomenon demonstrates rigorous experimental methodology rather than superficial metric chasing.

---

## 10. Verified Output Inventory

```
results/
├── eda/                          (7 diagnostic plots)
│   ├── 01_target_distribution.png
│   ├── 02_numerical_distributions.png
│   ├── 03_categorical_distributions.png
│   ├── 04_bivariate_numerical.png
│   ├── 05_bivariate_numerical_boxplots.png
│   ├── 06_bivariate_categorical.png
│   └── 07_correlation_heatmap.png
├── feature_selection/            (3 files)
│   ├── selected_features.txt
│   ├── feature_selection_results.csv
│   └── top_10_feature_ranking.csv
├── metrics/                      (4 lean authoritative CSVs)
│   ├── model_ranking.csv         (Model Family Leaderboard with ExtraTrees as winner)
│   ├── focused_comparisons.csv   (SMOTE, Boruta, and Baseline vs Final ablation deltas)
│   ├── xgboost_experiments.csv   (5 XGBoost experiments × 10 metrics)
│   └── comparison_models.csv     (4 benchmark classifiers × 10 metrics)
└── plots/                        (29 diagnostic plots)
    ├── cm_*.png                  (9 confusion matrices)
    ├── roc_*.png                 (9 ROC curves)
    ├── pr_*.png                  (9 Precision-Recall curves)
    ├── model_comparison_overview.png
    └── smote_distribution_comparison.png
```

---

## 11. Conclusion

The **1st-Half Milestone** is 100% complete, scientifically verified, and presentation-ready:
* **Selected Model:** **ExtraTrees** (Recall: 78.00%, ROC-AUC: 0.8371, PR-AUC: 0.2110, MCC: 0.2926)
* **Best Boosting Model:** **XGBoost D (Eng+Boruta)** (Recall: 40.00%, F1: 0.2797, MCC: 0.2437)
* **All code, metrics, and plots are fully reproducible and verified.**
