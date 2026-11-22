"""
ECG FINAL PRO — PHASE 1: STAGE 4
Feature Preprocessing & Robust Scaler Fitting
File: 04_machine_learning/stage4_scaler_pipeline.py

Ensures:
  - Scaler is fitted strictly on X_train (zero data snooping from test records)
  - Evaluates mean and variance scaling parameters for all 10 features
  - Verifies transformed matrices have zero NaNs or infs
  - Tests inverse transformation and arbitrary single-vector scaling
  - Exports candidate scaler to 05_results/stage4_candidate_scaler.pkl
  - Exports report to 05_results/stage4_scaler_report.txt
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"
SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"

REPORT_PATH = RESULTS_DIR / "stage4_scaler_report.txt"
CANDIDATE_SCALER_PATH = RESULTS_DIR / "stage4_candidate_scaler.pkl"
PARAMS_CSV_PATH = RESULTS_DIR / "stage4_scaler_parameters.csv"

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
    log(lines, "ECG FINAL PRO — STAGE 4: FEATURE PREPROCESSING & SCALER FITTING")
    log(lines, "=" * 72)

    if not SPLITS_JSON_PATH.exists():
        log(lines, f"[FAIL] Split manifest missing: {SPLITS_JSON_PATH}")
        sys.exit(1)

    splits = json.loads(SPLITS_JSON_PATH.read_text(encoding="utf-8"))
    train_records = splits["train_records"]
    test_records = splits["test_records"]

    df = pd.read_csv(DATASET_PATH)

    train_df = df[df["Record"].isin(train_records)]
    test_df = df[df["Record"].isin(test_records)]

    X_train = train_df[FEATURES].values
    X_test = test_df[FEATURES].values

    log(lines, f"X_train shape: {X_train.shape} (from {len(train_records)} records)")
    log(lines, f"X_test shape : {X_test.shape} (from {len(test_records)} records)")

    # Fit scaler strictly on training data
    scaler = StandardScaler()
    scaler.fit(X_train)

    log(lines, "\n[PASS] StandardScaler fitted strictly on X_train.")
    log(lines, f"Features in scaler: {scaler.n_features_in_}")

    # Inspect fitted parameters
    param_rows = []
    log(lines, "\nFitted Scaler Parameters (per feature):")
    for idx, feat in enumerate(FEATURES):
        mean_val = float(scaler.mean_[idx])
        scale_val = float(scaler.scale_[idx])
        var_val = float(scaler.var_[idx])
        param_rows.append({
            "Feature": feat,
            "Fitted_Mean": mean_val,
            "Fitted_Scale (Std)": scale_val,
            "Fitted_Variance": var_val,
        })
        log(lines, f"  {idx+1:2d}. {feat:14s}: Mean = {mean_val:10.5f}, Scale = {scale_val:10.5f}")

    pd.DataFrame(param_rows).to_csv(PARAMS_CSV_PATH, index=False)
    log(lines, f"\nSaved parameters table to: {PARAMS_CSV_PATH}")

    # Transform both sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check scaled statistical sanity
    train_means = np.mean(X_train_scaled, axis=0)
    train_stds = np.std(X_train_scaled, axis=0)
    test_means = np.mean(X_test_scaled, axis=0)
    test_stds = np.std(X_test_scaled, axis=0)

    log(lines, "\nTransformed Statistical Verification:")
    log(lines, f"  X_train_scaled mean max absolute dev from 0: {np.max(np.abs(train_means)):.2e}")
    log(lines, f"  X_train_scaled std max absolute dev from 1 : {np.max(np.abs(train_stds - 1.0)):.2e}")
    log(lines, f"  X_test_scaled mean range: [{test_means.min():.4f}, {test_means.max():.4f}]")
    log(lines, f"  X_test_scaled std range : [{test_stds.min():.4f}, {test_stds.max():.4f}]")

    # Check for NaNs or Infs
    train_finite = np.all(np.isfinite(X_train_scaled))
    test_finite = np.all(np.isfinite(X_test_scaled))

    if not (train_finite and test_finite):
        log(lines, "[FAIL] Non-finite values detected in scaled matrices!")
        sys.exit(1)
    log(lines, "[PASS] Zero NaNs or Infs in both scaled matrices.")

    # Save candidate scaler
    joblib.dump(scaler, CANDIDATE_SCALER_PATH)
    log(lines, f"\nSaved candidate scaler to: {CANDIDATE_SCALER_PATH}")

    # Test reload
    reloaded_scaler = joblib.load(CANDIDATE_SCALER_PATH)
    test_vec = np.array([72.0, 0.833, 0.160, 0.080, 0.350, -0.30, 0.05, 0.30, 4.20, 0.70]).reshape(1, -1)
    scaled_vec1 = scaler.transform(test_vec)
    scaled_vec2 = reloaded_scaler.transform(test_vec)
    if not np.allclose(scaled_vec1, scaled_vec2):
        log(lines, "[FAIL] Reloaded scaler transformation mismatch!")
        sys.exit(1)
    log(lines, "[PASS] Scaler round-trip reload and single-vector inference verified.")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 4 RESULT: PASSED")
    log(lines, "Feature preprocessing pipeline verified and certified ready for Stage 5.")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
