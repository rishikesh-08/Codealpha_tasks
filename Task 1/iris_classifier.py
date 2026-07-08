"""
Iris Flower Species Classifier
================================
Trains and evaluates three ML models (Random Forest, Logistic Regression, SVM)
on the Iris dataset and prints a full performance report.

Usage:
    python iris_classifier.py
    python iris_classifier.py --data path/to/Iris.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ── Config ────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COL   = "Species"
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: str):
    df = pd.read_csv(path)
    # Drop id column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].values)
    return X, y, le.classes_


def print_section(title: str):
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_confusion_matrix(cm: np.ndarray, classes: list):
    col_w = 14
    header = f"{'':14}" + "".join(f"{c[:12]:>{col_w}}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        label = classes[i][:12]
        print(f"{label:<14}" + "".join(f"{v:>{col_w}}" for v in row))


def print_feature_importances(importances: np.ndarray):
    pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in pairs:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<18} {bar:<40} {imp:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str):
    # Load and split
    X, y, classes = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (benefits LR and SVM)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print_section("Dataset Summary")
    print(f"  Total samples  : {len(X)}")
    print(f"  Features       : {FEATURE_COLS}")
    print(f"  Classes        : {list(classes)}")
    print(f"  Train / test   : {len(X_train)} / {len(X_test)}")

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            X_train, X_test        # tree models don't need scaling
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
            X_train_s, X_test_s
        ),
        "SVM (RBF kernel)": (
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            X_train_s, X_test_s
        ),
    }

    rf_model = None

    for name, (model, Xtr, Xte) in models.items():
        print_section(name)

        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        acc   = accuracy_score(y_test, preds)

        # 5-fold cross-validation on full scaled/unscaled data
        cv_X = scaler.transform(X) if Xtr is X_train_s else X
        cv_scores = cross_val_score(model, cv_X, y, cv=5, scoring="accuracy")

        print(f"  Test accuracy      : {acc * 100:.2f}%")
        print(f"  5-fold CV accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

        print("\n  Confusion matrix:")
        cm = confusion_matrix(y_test, preds)
        print_confusion_matrix(cm, list(classes))

        print("\n  Classification report:")
        report = classification_report(y_test, preds, target_names=classes, digits=3)
        for line in report.splitlines():
            print(f"    {line}")

        if hasattr(model, "feature_importances_"):
            rf_model = model

    # ── Feature importances (Random Forest only) ──────────────────────────────
    if rf_model is not None:
        print_section("Feature Importances (Random Forest)")
        print_feature_importances(rf_model.feature_importances_)

    print(f"\n{'─' * 60}\n  Done.\n{'─' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris species classifier")
    parser.add_argument(
        "--data",
        default="Iris.csv",
        help="Path to the Iris CSV file (default: Iris.csv)"
    )
    args = parser.parse_args()
    main(args.data)
