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

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_and_dashboard.py
```

After the pipeline completes, open `outputs/churn_dashboard.html` in a browser. The generated dashboard is self-contained, including its chart images.

## View on GitHub

GitHub's repository file viewer does not execute HTML files. To view the interactive dashboard online, this project includes a GitHub Pages workflow at `.github/workflows/pages.yml`.

1. Generate the dashboard by running the local instructions above.
2. Commit and push the updated `outputs/churn_dashboard.html` to the repository's `main` or `master` branch.
3. In the repository, open **Settings → Pages** and set the build and deployment source to **GitHub Actions**.
4. Open the **Actions** tab and wait for the **Deploy dashboard to GitHub Pages** workflow to finish.
5. Open the Pages URL shown in the deployment job summary.

The workflow publishes the dashboard as the Pages home page. Subsequent pushes to `main` or `master` redeploy it automatically. The notebook is available as `customer_churn_classification.ipynb`.

## Output Files

- `outputs/churn_dashboard.html` — interactive dashboard
- `outputs/metrics.json` — model metrics and evaluation summary
- `plots/` — visual analysis plots
- `REPORT.md` — project report

## Author

**Soundarya Umesh Barigidad**  

**Information Science Engineering Student**
