"""Train and evaluate several loan-status classifiers."""
import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]

def load_data(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[target]).copy()
    X = df.drop(columns=[target])
    y = df[target].astype(str)
    if y.nunique() != 2:
        raise ValueError(f"This starter pipeline expects a binary target; found {y.nunique()} classes.")
    # Map common approval labels to 0/1, otherwise encode consistently.
    mapping = {"N": 0, "Y": 1, "NO": 0, "YES": 1, "0": 0, "1": 1,
               "REJECTED": 0, "APPROVED": 1}
    upper = y.str.upper()
    if set(upper.unique()).issubset(mapping):
        y = upper.map(mapping).astype(int)
    else:
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    return X, y

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/loan_data.csv")
    parser.add_argument("--target", default="Loan_Status")
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    X, y = load_data(data_path, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    results = []
    best_name, best_pipeline, best_f1 = None, None, -1
    preprocessor = build_preprocessor(X_train)
    reports_dir = ROOT / "reports"
    models_dir = ROOT / "models"
    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if proba is not None else float("nan"),
        }
        results.append(row)
        print(f"\n{name}\n{classification_report(y_test, pred, zero_division=0)}")
        if row["f1"] > best_f1:
            best_name, best_pipeline, best_f1 = name, pipe, row["f1"]
            ConfusionMatrixDisplay.from_predictions(y_test, pred)
            plt.title(f"Confusion Matrix — {name}")
            plt.tight_layout()
            plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
            plt.close()

    metrics = pd.DataFrame(results).sort_values("f1", ascending=False)
    metrics.to_csv(reports_dir / "model_metrics.csv", index=False)
    joblib.dump(best_pipeline, models_dir / "best_model.joblib")
    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\nSaved best F1 model: {best_name} (F1={best_f1:.3f})")
    print("Note: metrics on synthetic data are only a code smoke-test, not real-world evidence.")

if __name__ == "__main__":
    main()
