"""
ECG FINAL PRO — PHASE 1: STAGE 5
Baseline Benchmark Evaluation (Decision Tree under Record Separation)
File: 04_machine_learning/stage5_baseline_benchmark.py

Evaluates:
  - Legacy Decision Tree architecture (max_depth=10) on record-separated train/test splits
  - Measures generalization degradation and overfitting gap
  - Documents baseline metrics (accuracy, balanced accuracy, recall, specificity, F1)
  - Exports baseline results to 05_results/stage5_baseline_metrics.json
  - Exports report to 05_results/stage5_baseline_report.txt
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"

SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"
SCALER_PATH = RESULTS_DIR / "stage4_candidate_scaler.pkl"

REPORT_PATH = RESULTS_DIR / "stage5_baseline_report.txt"
METRICS_JSON_PATH = RESULTS_DIR / "stage5_baseline_metrics.json"

FEATURES = [
    "HeartRate",
    "RRInterval",
    "PRInterval",
    "QRSDuration",
    "QTInterval",
    "STSegment",
    "HRV",
    "PAmplitude",
    "RAmplitude",
    "TAmplitude",
]

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 72)
    log(lines, "ECG FINAL PRO — STAGE 5: BASELINE BENCHMARK EVALUATION")
    log(lines, "=" * 72)

    # Load splits and scaler
    splits = json.loads(SPLITS_JSON_PATH.read_text(encoding="utf-8"))
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATASET_PATH)

    train_df = df[df["Record"].isin(splits["train_records"])]
    test_df = df[df["Record"].isin(splits["test_records"])]

    X_train = scaler.transform(train_df[FEATURES].values)
    y_train = train_df["Label"].values
    X_test = scaler.transform(test_df[FEATURES].values)
    y_test = test_df["Label"].values

    log(lines, f"Training beats: {len(X_train):,} (38 records)")
    log(lines, f"Testing beats : {len(X_test):,} (10 records)")

    # Fit baseline decision tree
    log(lines, "\nTraining Baseline Decision Tree (max_depth=10, random_state=42)...")
    baseline_dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    baseline_dt.fit(X_train, y_train)

    train_preds = baseline_dt.predict(X_train)
    test_preds = baseline_dt.predict(X_test)

    # Metrics computation
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    bal_acc = balanced_accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
    rec = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
    spec = recall_score(y_test, test_preds, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, test_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])

    log(lines, "\n" + "-" * 72)
    log(lines, "BASELINE DECISION TREE PERFORMANCE METRICS")
    log(lines, "-" * 72)
    log(lines, f"  Training Accuracy     : {train_acc * 100:.2f}%")
    log(lines, f"  Independent Test Acc  : {test_acc * 100:.2f}%")
    log(lines, f"  Generalization Gap    : {(train_acc - test_acc) * 100:.2f}% (Evidence of single-tree overfitting)")
    log(lines, f"  Balanced Accuracy     : {bal_acc * 100:.2f}%")
    log(lines, f"  Sensitivity (Abnormal): {rec * 100:.2f}%")
    log(lines, f"  Specificity (Normal)  : {spec * 100:.2f}%")
    log(lines, f"  Precision (Abnormal)  : {prec * 100:.2f}%")
    log(lines, f"  F1-Score (Abnormal)   : {f1 * 100:.2f}%")

    log(lines, "\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    log(lines, f"                Pred Normal (0)   Pred Abnormal (1)")
    log(lines, f"  Actual Norm(0):    {cm[0, 0]:6d}           {cm[0, 1]:6d}")
    log(lines, f"  Actual Abn (1):    {cm[1, 0]:6d}           {cm[1, 1]:6d}")

    log(lines, "\nClassification Report:")
    log(lines, classification_report(y_test, test_preds, target_names=["Normal (0)", "Abnormal (1)"], zero_division=0))

    # Feature Importance
    log(lines, "Baseline Feature Importances:")
    importances = baseline_dt.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        log(lines, f"  {rank:2d}. {FEATURES[idx]:14s}: {importances[idx] * 100:6.2f}%")

    metrics = {
        "model_type": "DecisionTreeClassifier",
        "max_depth": 10,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "generalization_gap": float(train_acc - test_acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity_recall": float(rec),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "feature_importances": {FEATURES[i]: float(importances[i]) for i in range(len(FEATURES))},
    }

    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(lines, f"\nSaved baseline metrics to: {METRICS_JSON_PATH}")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 5 RESULT: PASSED")
    log(lines, "Baseline benchmarks established. Ready for Stage 6 (Random Forest Training).")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
