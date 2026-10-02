# Task 1 Report — Supervised Churn Classification

**Intern task:** Build and evaluate a supervised classifier  
**Problem:** Predict whether a telecom customer will churn (`Yes` / `No`)  
**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 columns)  
**Split:** Stratified 80/20 (5,634 train / 1,409 test) + 5-fold stratified cross-validation on train

## Preprocessing

- Dropped `customerID` (identifier, not a predictor).
- Converted `TotalCharges` to numeric; blank strings became missing and were median-imputed.
- Target encoded as `Yes=1`, `No=0`.
- Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`): median impute + StandardScaler.
- Categorical features: most-frequent impute + one-hot encoding.
- Class imbalance (~26.5% churn) handled with stratified splits and `class_weight="balanced"`.

## Algorithms compared

| Model | Why included |
| --- | --- |
| Logistic Regression | Linear baseline, calibrated probabilities, easy to explain |
| Random Forest (200 trees) | Non-linear interactions, feature importance |

## Cross-validation (train, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.749 ± 0.015 | 0.517 ± 0.019 | **0.801 ± 0.038** | **0.629 ± 0.023** | **0.846 ± 0.012** |
| Random Forest | **0.789 ± 0.012** | **0.595 ± 0.023** | 0.641 ± 0.024 | 0.617 ± 0.021 | 0.837 ± 0.011 |

## Held-out test metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression (selected)** | 0.737 | 0.503 | **0.783** | **0.613** | **0.841** |
| Random Forest | **0.779** | **0.575** | 0.636 | 0.604 | 0.835 |

## Model selection

**Selected: Logistic Regression.**

Random Forest is more accurate overall, but churn work is not an accuracy contest. Missing a customer who will leave (false negative) is more expensive than a false retention offer. Logistic Regression has the higher **recall (0.783 vs 0.636)**, higher **F1 (0.613 vs 0.604)**, and slightly higher **ROC-AUC (0.841 vs 0.835)** on the test set, matching the CV ranking on F1 and AUC.

Month-to-month contracts churn at **42.7%**, vs **11.3%** (one-year) and **2.8%** (two-year). Tenure, total charges, monthly charges, and month-to-month contract dominate Random Forest importance — consistent with the EDA.

## Responsible AI

- Imbalance was addressed; accuracy alone would overstate a “always retain” baseline (~73.5%).
- `customerID` was not used. Do not store raw PII in artifacts.
- Gender and `SeniorCitizen` should be fairness-audited before production (disparate impact / equalized odds).

## Deliverables

1. Notebook: `customer_churn_classification.ipynb`
2. This report
3. Interactive dashboard: `outputs/churn_dashboard.html`
4. Plots: `plots/`
5. Metrics JSON: `outputs/metrics.json`
