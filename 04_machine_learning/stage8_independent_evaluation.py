"""
ECG FINAL PRO — PHASE 1: STAGE 8
Independent Test Set Evaluation & Comprehensive Diagnostics
File: 04_machine_learning/stage8_independent_evaluation.py

Evaluates:
  - Trained Random Forest on 10 held-out independent test records (never seen during training)
  - Computes complete metric suite: Accuracy, Balanced Acc, Sensitivity, Specificity, Precision, F1, ROC-AUC
  - Performs direct side-by-side comparison with baseline Decision Tree
  - Exports metrics to 05_results/stage8_independent_evaluation_metrics.json
  - Exports report to 05_results/stage8_independent_evaluation_report.txt
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"

SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"
SCALER_PATH = RESULTS_DIR / "stage4_candidate_scaler.pkl"
RF_MODEL_PATH = RESULTS_DIR / "stage6_candidate_random_forest.pkl"
BASELINE_METRICS_PATH = RESULTS_DIR / "stage5_baseline_metrics.json"

REPORT_PATH = RESULTS_DIR / "stage8_independent_evaluation_report.txt"
METRICS_JSON_PATH = RESULTS_DIR / "stage8_independent_evaluation_metrics.json"

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
    log(lines, "ECG FINAL PRO — STAGE 8: INDEPENDENT TEST SET EVALUATION")
    log(lines, "=" * 72)

    # Load splits, scaler, and models
    splits = json.loads(SPLITS_JSON_PATH.read_text(encoding="utf-8"))
    test_records = splits["test_records"]

    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)

    df = pd.read_csv(DATASET_PATH)
    test_df = df[df["Record"].isin(test_records)].copy()

    X_test_scaled = scaler.transform(test_df[FEATURES].values)
    y_test = test_df["Label"].values

    log(lines, f"Independent Test Set: {len(X_test_scaled):,} beats across {len(test_records)} records.")
    log(lines, f"Test Normal records   : {splits['test_normal_records']}")
    log(lines, f"Test Abnormal records : {splits['test_abnormal_records']}")

    # Predict with Random Forest
    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    test_acc = accuracy_score(y_test, rf_preds)
    bal_acc = balanced_accuracy_score(y_test, rf_preds)
    sens = recall_score(y_test, rf_preds, pos_label=1, zero_division=0)
    spec = recall_score(y_test, rf_preds, pos_label=0, zero_division=0)
    prec = precision_score(y_test, rf_preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, rf_preds, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, rf_probs)
    pr_auc = average_precision_score(y_test, rf_probs)
    cm = confusion_matrix(y_test, rf_preds, labels=[0, 1])

    log(lines, "\n" + "-" * 72)
    log(lines, "RANDOM FOREST INDEPENDENT TEST METRICS")
    log(lines, "-" * 72)
    log(lines, f"  Overall Accuracy      : {test_acc * 100:.2f}%")
    log(lines, f"  Balanced Accuracy     : {bal_acc * 100:.2f}%")
    log(lines, f"  Sensitivity (Recall 1): {sens * 100:.2f}%")
    log(lines, f"  Specificity (Recall 0): {spec * 100:.2f}%")
    log(lines, f"  Precision (Abnormal)  : {prec * 100:.2f}%")
    log(lines, f"  F1-Score (Abnormal)   : {f1 * 100:.2f}%")
    log(lines, f"  ROC-AUC Score         : {roc_auc:.4f}")
    log(lines, f"  PR-AUC Score          : {pr_auc:.4f}")

    log(lines, "\nConfusion Matrix:")
    log(lines, f"                Pred Normal (0)   Pred Abnormal (1)")
    log(lines, f"  Actual Norm(0):    {cm[0, 0]:6d}           {cm[0, 1]:6d}")
    log(lines, f"  Actual Abn (1):    {cm[1, 0]:6d}           {cm[1, 1]:6d}")

    # Side-by-side comparison with Stage 5 baseline
    if BASELINE_METRICS_PATH.exists():
        base = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
        log(lines, "\n" + "-" * 72)
        log(lines, "HEAD-TO-HEAD COMPARISON: BASELINE DECISION TREE vs RANDOM FOREST")
        log(lines, "-" * 72)
        log(lines, f"  Metric               Baseline (Decision Tree)    New Random Forest        Delta")
        log(lines, f"  ---------------------------------------------------------------------------------")
        log(lines, f"  Accuracy             {base['test_accuracy']*100:6.2f}%                     {test_acc*100:6.2f}%                 {test_acc*100 - base['test_accuracy']*100:+6.2f}%")
        log(lines, f"  Balanced Accuracy    {base['balanced_accuracy']*100:6.2f}%                     {bal_acc*100:6.2f}%                 {bal_acc*100 - base['balanced_accuracy']*100:+6.2f}%")
        log(lines, f"  Sensitivity (Abn)    {base['sensitivity_recall']*100:6.2f}%                     {sens*100:6.2f}%                 {sens*100 - base['sensitivity_recall']*100:+6.2f}%")
        log(lines, f"  Specificity (Norm)   {base['specificity']*100:6.2f}%                     {spec*100:6.2f}%                 {spec*100 - base['specificity']*100:+6.2f}%")
        log(lines, f"  Precision (Abn)      {base['precision']*100:6.2f}%                     {prec*100:6.2f}%                 {prec*100 - base['precision']*100:+6.2f}%")
        log(lines, f"  F1-Score (Abn)       {base['f1_score']*100:6.2f}%                     {f1*100:6.2f}%                 {f1*100 - base['f1_score']*100:+6.2f}%")

    # Per-record breakdown
    log(lines, "\nPer-Record Test Performance Breakdown:")
    test_df["Predicted"] = rf_preds
    rec_perf = test_df.groupby("Record").agg(
        Label=("Label", "first"),
        TotalBeats=("Beat", "count"),
        CorrectBeats=("Predicted", lambda p: int((p == test_df.loc[p.index, "Label"]).sum()))
    ).reset_index()
    rec_perf["Accuracy"] = (rec_perf["CorrectBeats"] / rec_perf["TotalBeats"]) * 100.0
    log(lines, rec_perf.to_string(index=False))

    results = {
        "model": "RandomForestClassifier",
        "test_accuracy": float(test_acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity_recall": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "per_record_performance": rec_perf.to_dict(orient="records"),
    }

    METRICS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(lines, f"\nSaved independent evaluation metrics to: {METRICS_JSON_PATH}")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 8 RESULT: PASSED")
    log(lines, "Independent evaluation completed. Ready for Stage 9 (Feature Importance & Interpretability).")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
