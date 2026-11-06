"""Streamlit interface for evaluating the five trained classifiers."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
LABELS = {0: "malignant", 1: "benign"}

st.set_page_config(page_title="Classifier Comparison", page_icon="🧪", layout="wide")
st.title("Wisconsin Breast Cancer Classifier Lab")
st.caption("Compare five classification models on a labelled test CSV. Target: 0 = malignant, 1 = benign.")

with (ROOT / "metadata.json").open(encoding="utf-8") as handle:
    metadata = json.load(handle)
with (ROOT / "metrics.json").open(encoding="utf-8") as handle:
    reference_metrics = json.load(handle)
feature_columns = metadata["features"]

@st.cache_resource
def load_model(path):
    return joblib.load(path)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload the included test_data.csv to view the evaluation, or use the sample file below.")
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button("Download sample test_data.csv", sample_path.read_bytes(), "test_data.csv", "text/csv")
    st.stop()

try:
    frame = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

missing = [column for column in feature_columns if column not in frame.columns]
if missing:
    st.error(f"CSV is missing {len(missing)} required feature columns. Example: {missing[:4]}")
    st.stop()
if TARGET not in frame.columns:
    st.error("CSV must include the labelled 'target' column so the app can calculate evaluation metrics.")
    st.stop()

X = frame[feature_columns]
y = pd.to_numeric(frame[TARGET], errors="coerce")
if y.isna().any() or not set(y.unique()).issubset({0, 1}):
    st.error("Target values must be 0 (malignant) or 1 (benign).")
    st.stop()
if X.isna().any().any():
    st.error("Feature columns contain missing values. Please clean the CSV first.")
    st.stop()

model_name = st.selectbox("Choose a model", list(MODEL_FILES))
model_path = MODEL_DIR / MODEL_FILES[model_name]
if not model_path.exists():
    st.error("Saved model files are missing. Run `python train_models.py` first.")
    st.stop()

model = load_model(model_path)
predictions = model.predict(X)
scores = model.predict_proba(X)[:, 1]
result = {
    "Accuracy": accuracy_score(y, predictions),
    "AUC": roc_auc_score(y, scores),
    "Precision": precision_score(y, predictions, zero_division=0),
    "Recall": recall_score(y, predictions, zero_division=0),
    "F1": f1_score(y, predictions, zero_division=0),
    "MCC": matthews_corrcoef(y, predictions),
}

st.subheader(f"{model_name} results")
cols = st.columns(6)
for col, (label, value) in zip(cols, result.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    matrix_frame = pd.DataFrame(
        matrix,
        index=["Actual malignant (0)", "Actual benign (1)"],
        columns=["Predicted malignant (0)", "Predicted benign (1)"],
    )
    st.dataframe(matrix_frame, use_container_width=True)
with right:
    st.markdown("#### Classification report")
    report = classification_report(y, predictions, labels=[0, 1], target_names=["malignant", "benign"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.subheader("Model comparison on the bundled test split")
comparison = pd.DataFrame(reference_metrics).T
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

st.subheader("Predictions for uploaded rows")
output = frame.copy()
output["predicted_target"] = predictions
output["predicted_class"] = [LABELS[int(value)] for value in predictions]
output["benign_probability"] = scores
st.dataframe(output.head(20), use_container_width=True)
st.download_button("Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv")
