"""
ECG FINAL PRO — PHASE 1: STAGE 6
Random Forest Architecture Design & Training
File: 04_machine_learning/stage6_train_random_forest.py

Trains:
  - Ensemble RandomForestClassifier with regularization to combat overfitting:
      * n_estimators = 150
      * max_depth = 12
      * min_samples_split = 10
      * min_samples_leaf = 4
      * max_features = 'sqrt' (essential to prevent HRV single-feature dominance)
      * class_weight = 'balanced'
      * oob_score = True
      * random_state = 42
  - Evaluates Out-Of-Bag (OOB) accuracy on training records
  - Exports candidate model to 05_results/stage6_candidate_random_forest.pkl
  - Exports report to 05_results/stage6_rf_training_report.txt
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"

SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"
SCALER_PATH = RESULTS_DIR / "stage4_candidate_scaler.pkl"

CANDIDATE_RF_PATH = RESULTS_DIR / "stage6_candidate_random_forest.pkl"
REPORT_PATH = RESULTS_DIR / "stage6_rf_training_report.txt"

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
    log(lines, "ECG FINAL PRO — STAGE 6: RANDOM FOREST TRAINING & ARCHITECTURE")
    log(lines, "=" * 72)

    # Load splits and scaler
    splits = json.loads(SPLITS_JSON_PATH.read_text(encoding="utf-8"))
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATASET_PATH)

    train_df = df[df["Record"].isin(splits["train_records"])]

    X_train = scaler.transform(train_df[FEATURES].values)
    y_train = train_df["Label"].values

    log(lines, f"Training Set: {len(X_train):,} beats across {len(splits['train_records'])} records.")
    log(lines, f"Feature Count: {X_train.shape[1]}")

    # Build Random Forest Classifier
    rf_params = {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "oob_score": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    log(lines, "\nRandom Forest Architecture Parameters:")
    for k, v in rf_params.items():
        log(lines, f"  - {k:<18}: {v}")

    log(lines, "\nTraining Random Forest ensemble...")
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)

    train_preds = rf_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    oob_acc = rf_model.oob_score_

    log(lines, "\n" + "-" * 72)
    log(lines, "TRAINING CONVERGENCE & OOB RESULTS")
    log(lines, "-" * 72)
    log(lines, f"  Number of Trees Fitted : {len(rf_model.estimators_)}")
    log(lines, f"  Training Set Accuracy  : {train_acc * 100:.2f}%")
    log(lines, f"  Out-of-Bag (OOB) Score : {oob_acc * 100:.2f}%")

    # Feature importances in RF
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    log(lines, "\nRandom Forest Feature Importances (Gini):")
    for rank, idx in enumerate(sorted_idx, 1):
        log(lines, f"  {rank:2d}. {FEATURES[idx]:14s}: {importances[idx] * 100:6.2f}%")

    # Check that no single feature dominates > 50%
    max_imp = np.max(importances)
    if max_imp > 0.50:
        log(lines, f"[WARNING] Single feature dominance remains high: {max_imp * 100:.2f}%")
    else:
        log(lines, f"[PASS] Balanced feature utilization achieved (highest single feature is {max_imp * 100:.2f}%).")

    # Save candidate model
    joblib.dump(rf_model, CANDIDATE_RF_PATH)
    log(lines, f"\nSaved candidate Random Forest model to: {CANDIDATE_RF_PATH}")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 6 RESULT: PASSED")
    log(lines, "Random Forest trained successfully. Ready for Stage 7 (Cross-Validation).")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
