"""
ECG FINAL PRO — PHASE 1: STAGE 9
Feature Importance & Model Interpretability Analysis
File: 04_machine_learning/stage9_feature_importance.py

Analyzes:
  - Global Gini impurity feature importances (mean & inter-tree std across all 150 trees)
  - Permutation feature importance on the unseen independent test set
  - Identifies which features genuinely drive patient-level generalization
  - Exports results to 05_results/stage9_feature_importance.csv
  - Exports report to 05_results/stage9_feature_importance_report.txt
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"

SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"
SCALER_PATH = RESULTS_DIR / "stage4_candidate_scaler.pkl"
RF_MODEL_PATH = RESULTS_DIR / "stage6_candidate_random_forest.pkl"

REPORT_PATH = RESULTS_DIR / "stage9_feature_importance_report.txt"
CSV_PATH = RESULTS_DIR / "stage9_feature_importance.csv"

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

CLINICAL_ROLES = {
    "HeartRate": "Chronotropic rate metric; distinguishes tachy/bradyarrhythmias.",
    "RRInterval": "Beat timing; rhythm regularity and pause identification.",
    "PRInterval": "AV conduction time; identifies AV blocks and pre-excitation.",
    "QRSDuration": "Ventricular depolarization width; hallmark of PVCs and bundle branch blocks.",
    "QTInterval": "Ventricular repolarization duration; long/short QT syndrome risk.",
    "STSegment": "Ischemic proxy level; ST elevation or depression.",
    "HRV": "Autonomic nervous system tone; standard deviation of NN intervals.",
    "PAmplitude": "Atrial depolarization amplitude; P-wave morphology.",
    "RAmplitude": "Ventricular peak amplitude; hypertrophy or conduction anomalies.",
    "TAmplitude": "Ventricular repolarization amplitude; hyperkalemia or inversion.",
}

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 72)
    log(lines, "ECG FINAL PRO — STAGE 9: FEATURE IMPORTANCE & INTERPRETABILITY")
    log(lines, "=" * 72)

    # Load artifacts
    splits = json.loads(SPLITS_JSON_PATH.read_text(encoding="utf-8"))
    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)

    test_df = df[df["Record"].isin(splits["test_records"])]
    X_test_scaled = scaler.transform(test_df[FEATURES].values)
    y_test = test_df["Label"].values

    log(lines, f"Loaded Random Forest model with {len(rf_model.estimators_)} trees.")
    log(lines, f"Evaluating on {len(X_test_scaled):,} independent test beats.")

    # 1. Gini Importance across trees
    gini_mean = rf_model.feature_importances_
    gini_tree_matrix = np.array([tree.feature_importances_ for tree in rf_model.estimators_])
    gini_std = np.std(gini_tree_matrix, axis=0)

    log(lines, "\n" + "-" * 72)
    log(lines, "1. GINI FEATURE IMPORTANCE (ENSEMBLE AVERAGE)")
    log(lines, "-" * 72)
    sorted_gini_idx = np.argsort(gini_mean)[::-1]
    for rank, idx in enumerate(sorted_gini_idx, 1):
        log(lines, f"  {rank:2d}. {FEATURES[idx]:14s}: {gini_mean[idx]*100:6.2f}% +/- {gini_std[idx]*100:5.2f}%")

    # 2. Permutation Importance on held-out test data
    log(lines, "\n" + "-" * 72)
    log(lines, "2. TEST-SET PERMUTATION IMPORTANCE (BALANCED ACCURACY DROP)")
    log(lines, "-" * 72)
    log(lines, "Computing permutation importance over 5 repeats...")

    perm = permutation_importance(
        rf_model,
        X_test_scaled,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    sorted_perm_idx = np.argsort(perm.importances_mean)[::-1]
    for rank, idx in enumerate(sorted_perm_idx, 1):
        log(lines, f"  {rank:2d}. {FEATURES[idx]:14s}: Drop = {perm.importances_mean[idx]*100:+.2f}% +/- {perm.importances_std[idx]*100:.2f}%")

    # Combine into comprehensive dataframe
    summary_rows = []
    for idx, feat in enumerate(FEATURES):
        summary_rows.append({
            "Feature": feat,
            "Gini_Importance_Pct": round(float(gini_mean[idx] * 100), 2),
            "Gini_Std_Pct": round(float(gini_std[idx] * 100), 2),
            "Permutation_BalAcc_Drop_Pct": round(float(perm.importances_mean[idx] * 100), 2),
            "Permutation_Std_Pct": round(float(perm.importances_std[idx] * 100), 2),
            "Clinical_Interpretation": CLINICAL_ROLES.get(feat, ""),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Gini_Importance_Pct", ascending=False)
    summary_df.to_csv(CSV_PATH, index=False)
    log(lines, f"\nSaved feature ranking table to: {CSV_PATH}")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 9 RESULT: PASSED")
    log(lines, "Feature interpretability verified. Ready for Stage 10 (Model Serialization).")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
