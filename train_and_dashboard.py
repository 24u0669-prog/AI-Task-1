"""Train churn classifiers, save metrics/plots, and build an HTML dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", context="notebook")


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
    return df


def preprocess(df: pd.DataFrame):
    data = df.drop(columns=["customerID"]).copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return data, X, y, numeric_features, categorical_features, preprocessor


def save_eda_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.countplot(data=df, x="Churn", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Churn class balance")
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Tenure by churn")
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("Monthly charges by churn")
    contract_churn = df.groupby(["Contract", "Churn"]).size().reset_index(name="count")
    sns.barplot(data=contract_churn, x="Contract", y="count", hue="Churn", ax=axes[1, 1])
    axes[1, 1].set_title("Contract type vs churn")
    axes[1, 1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_overview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def metrics_from_cv(cv_results: dict, scoring: dict) -> dict:
    out = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        out[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
    return out


def evaluate_test(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    report = classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
    )
    return metrics, y_pred, y_proba, pipe, report


def write_html_dashboard(payload: dict) -> Path:
    lr = payload["test"]["Logistic Regression"]
    rf = payload["test"]["Random Forest"]
    selected = payload["selected_model"]
    churn_yes = payload["eda"]["churn_yes"]
    churn_no = payload["eda"]["churn_no"]
    churn_rate = payload["eda"]["churn_rate"]
    n_rows = payload["eda"]["n_rows"]
    n_cols = payload["eda"]["n_cols"]
    cv = payload["cv"]
    importances = payload["feature_importances"][:12]
    contract = payload["eda"]["contract_churn"]
    reports = payload["classification_reports"]

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def f3(x: float) -> str:
        return f"{x:.3f}"

    importance_rows = "".join(
        f"<tr><td>{i['feature']}</td><td>{i['importance']:.4f}</td></tr>"
        for i in importances
    )
    contract_rows = "".join(
        f"<tr><td>{c['Contract']}</td><td>{c['No']}</td><td>{c['Yes']}</td>"
        f"<td>{c['churn_rate']*100:.1f}%</td></tr>"
        for c in contract
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Customer Churn Classification Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #171d24;
      --line: #2a3440;
      --text: #e8eef4;
      --muted: #9aa7b5;
      --accent: #4f8cff;
      --good: #3ecf8e;
      --warn: #f5a524;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(79, 140, 255, 0.16), transparent 26%),
        radial-gradient(circle at bottom right, rgba(62, 207, 142, 0.10), transparent 28%),
        linear-gradient(135deg, #07111b 0%, #0e1724 35%, #101b2a 100%);
      color: var(--text);
    }}
    .page-shell {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 22px 48px;
    }}
    header {{
      background: rgba(15, 23, 32, 0.75);
      border: 1px solid rgba(146, 169, 193, 0.2);
      border-radius: 24px;
      padding: 28px 28px 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 50px rgba(8, 13, 22, 0.35);
      margin-bottom: 22px;
    }}
    .header-row {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      flex-wrap: wrap;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(79, 140, 255, 0.12);
      border: 1px solid rgba(79, 140, 255, 0.24);
      color: #dfeaff;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .eyebrow-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: linear-gradient(135deg, #4f8cff, #8bb5ff);
      box-shadow: 0 0 18px rgba(79, 140, 255, 0.9);
    }}
    h1 {{
      margin: 16px 0 8px;
      font-size: clamp(28px, 3vw, 46px);
      font-weight: 750;
      letter-spacing: -0.04em;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      max-width: 680px;
      line-height: 1.6;
    }}
    .hero-tag {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 180px;
      background: linear-gradient(135deg, rgba(62, 207, 142, 0.18), rgba(79, 140, 255, 0.14));
      border: 1px solid rgba(62, 207, 142, 0.35);
      color: #ddffef;
      border-radius: 16px;
      padding: 14px 18px;
      font-weight: 700;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    main {{
      padding: 0;
      max-width: 1360px;
      margin: 0 auto;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .card {{
      position: relative;
      background: rgba(15, 23, 32, 0.75);
      border: 1px solid rgba(146, 169, 193, 0.16);
      border-radius: 20px;
      padding: 18px 18px 16px;
      box-shadow: 0 10px 28px rgba(8, 13, 22, 0.28);
      overflow: hidden;
    }}
    .card::before {{
      content: "";
      position: absolute;
      inset: 0 auto auto 0;
      width: 100%;
      height: 2px;
      background: linear-gradient(90deg, rgba(79, 140, 255, 0.9), rgba(62, 207, 142, 0.9));
      opacity: 0.9;
    }}
    .kpi-label {{
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .kpi-value {{
      font-size: clamp(24px, 2vw, 32px);
      font-weight: 750;
      margin-top: 10px;
      letter-spacing: -0.04em;
    }}
    .kpi-hint {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
      line-height: 1.5;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    h2 {{
      font-size: 16px;
      margin: 0 0 12px;
      letter-spacing: -0.02em;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid rgba(146, 169, 193, 0.12); text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; }}
    .badge {{
      display: inline-block;
      background: linear-gradient(135deg, rgba(62, 207, 142, 0.16), rgba(79, 140, 255, 0.12));
      color: var(--good);
      border: 1px solid rgba(62, 207, 142, 0.4);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .notes {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
      margin: 12px 0 0;
    }}
    img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(146, 169, 193, 0.12);
      background: rgba(255,255,255,0.02);
      display: block;
    }}
    @media (max-width: 900px) {{
      .kpis, .grid-2 {{ grid-template-columns: 1fr; }}
      .page-shell {{ padding: 16px 14px 32px; }}
      header {{ padding: 20px 18px; }}
    }}
  </style>
</head>
<body>
  <div class="page-shell">
    <header>
      <div class="header-row">
        <div>
          <div class="eyebrow"><span class="eyebrow-dot"></span> Customer retention intelligence</div>
          <h1>Customer Churn Prediction Dashboard</h1>
          <p class="sub">Predictive retention insights for telecom subscribers across contract, billing, and service behavior.</p>
        </div>
        <div class="hero-tag">Target: reduce churn risk</div>
      </div>
    </header>
    <main>
      <div class="kpis">
        <div class="card">
          <div class="kpi-label">Customers</div>
          <div class="kpi-value">{n_rows:,}</div>
          <div class="kpi-hint">{n_cols} raw columns after load</div>
        </div>
        <div class="card">
          <div class="kpi-label">Observed churn rate</div>
          <div class="kpi-value">{pct(churn_rate)}</div>
          <div class="kpi-hint">{churn_yes:,} churn / {churn_no:,} retain</div>
        </div>
        <div class="card">
          <div class="kpi-label">Selected model</div>
          <div class="kpi-value" style="font-size:22px">{selected}</div>
          <div class="kpi-hint">Highest test F1, then ROC-AUC</div>
        </div>
        <div class="card">
          <div class="kpi-label">Best test F1</div>
          <div class="kpi-value">{f3(payload["test"][selected]["f1"])}</div>
          <div class="kpi-hint">ROC-AUC {f3(payload["test"][selected]["roc_auc"])}</div>
        </div>
      </div>

    <div class="grid-2">
      <div class="card">
        <h2>Class mix</h2>
        <div id="pie"></div>
      </div>
      <div class="card">
        <h2>Test-set metrics</h2>
        <div id="metrics"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>Churn rate by contract</h2>
        <div id="contract"></div>
      </div>
      <div class="card">
        <h2>5-fold CV F1 (train)</h2>
        <div id="cv"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>Confusion matrices</h2>
        <img src="../plots/confusion_matrices.png" alt="Confusion matrices" />
      </div>
      <div class="card">
        <h2>ROC curves (test set)</h2>
        <img src="../plots/roc_curves.png" alt="ROC curves" />
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>Random Forest — top features</h2>
        <table>
          <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
          <tbody>{importance_rows}</tbody>
        </table>
      </div>
      <div class="card">
        <h2>Contract breakdown</h2>
        <table>
          <thead><tr><th>Contract</th><th>No churn</th><th>Churn</th><th>Rate</th></tr></thead>
          <tbody>{contract_rows}</tbody>
        </table>
        <p class="notes" style="margin-top:14px">
          Selected: <span class="badge">{selected}</span><br/>
          Logistic Regression is typically stronger on recall for the minority churn class.
          Random Forest is often slightly better on overall accuracy. For a retention campaign,
          missing a churner (false negative) is more costly, so F1 and recall were prioritized
          over accuracy.
        </p>
      </div>
    </div>

    <div class="card">
      <h2>Responsible AI</h2>
      <p class="notes">
        Stratified splits and class_weight="balanced" were used because churn is ~{pct(churn_rate)} of the base.
        customerID was dropped. Gender and SeniorCitizen should be fairness-audited before production.
        Do not store raw PII in model artifacts.
      </p>
    </div>
    </main>
  </div>
  <script>
    const pie = [{{label: "No churn", values: {churn_no}}}, {{label: "Churn", values: {churn_yes}}}];
    Plotly.newPlot("pie", [{{
      type: "pie",
      labels: ["No churn", "Churn"],
      values: [{churn_no}, {churn_yes}],
      hole: 0.45,
      marker: {{ colors: ["#4f8cff", "#f5a524"] }}
    }}], {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{ color: "#e8eef4" }},
      margin: {{ t: 10, b: 10, l: 10, r: 10 }},
      height: 280,
      showlegend: true
    }}, {{displayModeBar: false, responsive: true}});

    Plotly.newPlot("metrics", [
      {{
        type: "bar",
        name: "Logistic Regression",
        x: ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        y: [{lr["accuracy"]}, {lr["precision"]}, {lr["recall"]}, {lr["f1"]}, {lr["roc_auc"]}],
        marker: {{ color: "#4f8cff" }}
      }},
      {{
        type: "bar",
        name: "Random Forest",
        x: ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        y: [{rf["accuracy"]}, {rf["precision"]}, {rf["recall"]}, {rf["f1"]}, {rf["roc_auc"]}],
        marker: {{ color: "#3ecf8e" }}
      }}
    ], {{
      barmode: "group",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{ color: "#e8eef4" }},
      yaxis: {{ range: [0, 1], title: "Score (0–1)" }},
      xaxis: {{ title: "Metric" }},
      margin: {{ t: 20, b: 50, l: 50, r: 10 }},
      height: 280,
      legend: {{ orientation: "h" }}
    }}, {{displayModeBar: false, responsive: true}});

    Plotly.newPlot("contract", [{{
      type: "bar",
      x: {json.dumps([c["Contract"] for c in contract])},
      y: {json.dumps([round(c["churn_rate"] * 100, 1) for c in contract])},
      marker: {{ color: "#f5a524" }}
    }}], {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{ color: "#e8eef4" }},
      yaxis: {{ title: "Churn rate (%)" }},
      xaxis: {{ title: "Contract type" }},
      margin: {{ t: 20, b: 50, l: 50, r: 10 }},
      height: 280
    }}, {{displayModeBar: false, responsive: true}});

    Plotly.newPlot("cv", [
      {{
        type: "bar",
        name: "F1 mean",
        x: ["Logistic Regression", "Random Forest"],
        y: [{cv["Logistic Regression"]["f1"]["mean"]}, {cv["Random Forest"]["f1"]["mean"]}],
        error_y: {{
          type: "data",
          array: [{cv["Logistic Regression"]["f1"]["std"]}, {cv["Random Forest"]["f1"]["std"]}],
          visible: true
        }},
        marker: {{ color: "#4f8cff" }}
      }}
    ], {{
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {{ color: "#e8eef4" }},
      yaxis: {{ range: [0, 1], title: "F1 (5-fold mean ± std)" }},
      margin: {{ t: 20, b: 50, l: 50, r: 10 }},
      height: 280
    }}, {{displayModeBar: false, responsive: true}});
  </script>
</body>
</html>
"""
    path = OUT_DIR / "churn_dashboard.html"
    path.write_text(html, encoding="utf-8")
    (OUT_DIR / "classification_reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    df = load_data()
    save_eda_plots(df)
    data, X, y, numeric_features, categorical_features, preprocessor = preprocess(df)

    churn_yes = int((df["Churn"] == "Yes").sum())
    churn_no = int((df["Churn"] == "No").sum())
    contract_ct = pd.crosstab(df["Contract"], df["Churn"])
    contract_churn = []
    for contract, row in contract_ct.iterrows():
        no = int(row.get("No", 0))
        yes = int(row.get("Yes", 0))
        total = no + yes
        contract_churn.append(
            {
                "Contract": str(contract),
                "No": no,
                "Yes": yes,
                "churn_rate": yes / total if total else 0.0,
            }
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_payload = {}
    pipelines = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipelines[name] = pipe
        cv_results = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        cv_payload[name] = metrics_from_cv(cv_results, scoring)
        print(f"CV done: {name}")

    test_payload = {}
    predictions = {}
    fitted = {}
    reports = {}
    for name, pipe in pipelines.items():
        metrics, y_pred, y_proba, fitted_pipe, report = evaluate_test(
            pipe, X_train, y_train, X_test, y_test
        )
        test_payload[name] = metrics
        predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}
        fitted[name] = fitted_pipe
        reports[name] = report
        print(f"Test {name}: {metrics}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, predictions):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions[name]["y_pred"],
            display_labels=["No Churn", "Churn"],
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in predictions:
        RocCurveDisplay.from_predictions(
            y_test, predictions[name]["y_proba"], name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    plot_df = pd.DataFrame(test_payload).T.reset_index().rename(columns={"index": "Model"})
    plot_df = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set metric comparison")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    rf_pipe = fitted["Random Forest"]
    ohe = rf_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_names])
    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title("Top 15 Random Forest feature importances")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Prefer F1, break ties with ROC-AUC
    ranked = sorted(
        test_payload.items(),
        key=lambda kv: (kv[1]["f1"], kv[1]["roc_auc"]),
        reverse=True,
    )
    selected = ranked[0][0]

    payload = {
        "eda": {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "churn_yes": churn_yes,
            "churn_no": churn_no,
            "churn_rate": churn_yes / (churn_yes + churn_no),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "contract_churn": contract_churn,
        },
        "cv": cv_payload,
        "test": test_payload,
        "selected_model": selected,
        "feature_importances": imp_df.to_dict(orient="records"),
        "classification_reports": reports,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dashboard = write_html_dashboard(payload)
    print(f"Selected model: {selected}")
    print(f"Dashboard: {dashboard}")
    print(f"Metrics: {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
