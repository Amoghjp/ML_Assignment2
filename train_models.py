"""Train five classifiers on the Wisconsin Breast Cancer dataset."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "target"
SEED = 42


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED)
        ),
    }


def main():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    MODEL_DIR.mkdir(exist_ok=True)
    metrics = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "Accuracy": float(accuracy_score(y_test, pred)),
            "AUC": float(roc_auc_score(y_test, scores)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "MCC": float(matthews_corrcoef(y_test, pred)),
        }
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_').replace('-', '')}.joblib")

    test_frame = X_test.copy()
    test_frame[TARGET] = y_test
    test_frame.to_csv(ROOT / "test_data.csv", index=False)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "metadata.json").write_text(json.dumps({
        "dataset": "Wisconsin Breast Cancer (Diagnostic)",
        "features": list(X.columns),
        "target": TARGET,
        "target_labels": {"0": "malignant", "1": "benign"},
        "positive_class": 1,
        "test_size": 0.20,
        "random_state": SEED,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).T.round(4).to_string())
    print(f"\nSaved test data: {ROOT / 'test_data.csv'} ({len(test_frame)} rows)")


if __name__ == "__main__":
    main()
