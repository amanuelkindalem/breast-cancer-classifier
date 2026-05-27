# Breast Cancer Classifier

## Overview
Machine learning models to classify breast tumors as **benign** or **malignant** using the Wisconsin Breast Cancer dataset.

## Dataset
- 569 samples, 30 features
- Binary classification: 0=malignant, 1=benign

## Models
- Logistic Regression (97.8% accuracy)
- Decision Tree (93.3% accuracy)
- SVM (97.4% accuracy)

## Usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from breast_cancer_classifier import homework

# Load data
data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=['target'])
y = data.frame['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Evaluate
accuracy = homework(X_train, y_train, 'logistic_regression', 5)
print(f"Accuracy: {accuracy}")
