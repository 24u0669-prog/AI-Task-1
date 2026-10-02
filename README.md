# Task 1 — Customer Churn Classification

Intern deliverable for Alfido Tech Task 1: supervised classification with preprocessing, two algorithms, cross-validation, and full metrics.

## How to run

```powershell
cd "C:\Users\USER\OneDrive\Documents\AI  Task-1\Task1_Classification"
C:\Users\USER\anaconda3\python.exe train_and_dashboard.py
```

Then open `outputs/churn_dashboard.html` in a browser.

Notebook: open `customer_churn_classification.ipynb` in Jupyter and run all cells.

## Selected result

**Logistic Regression** (test F1 0.613, ROC-AUC 0.841, recall 0.783) over Random Forest, because catching churners matters more than overall accuracy.
