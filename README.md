# Wisconsin Breast Cancer: five-classifier comparison

## Problem statement
Classify a breast mass as malignant or benign from numeric measurements computed from digitized fine-needle aspirate images. This is a learning demonstration and is not a medical diagnostic tool.

## Dataset description
The Wisconsin Breast Cancer Diagnostic dataset has 569 instances, 30 numeric features, and two classes. Target `0` means malignant and target `1` means benign. The data is provided by scikit-learn from the UCI ML Breast Cancer Wisconsin (Diagnostic) dataset. Dataset reference: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## GitHub repository link
`https://github.com/Amoghjp/ML_Assignment2`

## Models and metrics
The data is split once using a stratified 80/20 train/test split (`random_state=42`). All models use that same split. AUC is calculated for class 1 (benign).

<!-- Filled by train_models.py -->

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9386 | 0.9342 | 0.9577 | 0.9444 | 0.9510 | 0.8689 |
| K-Nearest Neighbors | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations
Observations from this fixed 80/20 split:

- Logistic Regression: tied for the highest accuracy (0.9825), AUC (0.9954), F1 (0.9861), and MCC (0.9623) on this split.
- Decision Tree: lowest accuracy (0.9386) and AUC (0.9342) in this run; the depth limit reduces tree complexity.
- K-Nearest Neighbors: accuracy is 0.9561; scaling is important because kNN compares feature distances.
- Gaussian Naive Bayes: accuracy is 0.9386, while AUC is 0.9878; threshold ranking is strong even though some hard-label predictions are wrong.
- Support Vector Machine: tied with Logistic Regression on accuracy, precision, recall, F1, and MCC; its AUC is 0.9950.
- Overall winner: Logistic Regression and SVM tie on the hard-label metrics; Logistic Regression has the slightly higher AUC. The winner can change with a different split or dataset. Since malignant is class 0, also inspect its recall in the classification report before making a screening-focused conclusion.

