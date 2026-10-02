# Customer Churn Classification

This project focuses on predicting customer churn for a telecom company using supervised machine learning. The objective is to identify customers likely to leave the service so that retention strategies can be prioritized effectively.

## Project Overview

The solution includes:
- Data preprocessing and cleaning
- Feature engineering through encoding and scaling
- Model comparison between two classifiers
- Stratified cross-validation for robust evaluation
- Metric analysis and dashboard generation

## Dataset

The project uses the IBM Telco Customer Churn dataset, which contains customer information such as contract type, billing behavior, service usage, and churn status.

## Models Evaluated

- Logistic Regression
- Random Forest Classifier

## Selected Model

**Logistic Regression** was selected as the final model because it achieved a stronger balance of recall and F1-score, which is more valuable for churn prediction than raw accuracy alone.

- Test F1-score: 0.613
- ROC-AUC: 0.841
- Recall: 0.783

This makes it better suited for identifying customers at risk of churn, which is the primary business objective.

## Workflow

1. Load and clean the dataset
2. Remove irrelevant identifiers such as `customerID`
3. Handle missing values and encode categorical features
4. Scale numeric features
5. Train and compare Logistic Regression and Random Forest models
6. Use stratified 5-fold cross-validation
7. Evaluate test-set performance
8. Generate dashboard and plots

## How to Run

```powershell
cd "C:\Users\USER\OneDrive\Documents\AI  Task-1\Task1_Classification"
C:\Users\USER\anaconda3\python.exe train_and_dashboard.py
```

After execution, open the dashboard at:

`outputs/churn_dashboard.html`

You can also open the notebook file:

`customer_churn_classification.ipynb`

## Output Files

- `outputs/churn_dashboard.html` — interactive dashboard
- `outputs/metrics.json` — model metrics and evaluation summary
- `plots/` — visual analysis plots
- `REPORT.md` — project report

## Author

**Sounsdarya Umesh Barigidad**
**Information Science Student**
