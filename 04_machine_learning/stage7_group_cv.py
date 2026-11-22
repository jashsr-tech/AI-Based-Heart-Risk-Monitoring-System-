"""
ECG FINAL PRO — PHASE 1: STAGE 7
Record-Grouped Cross-Validation (K-Fold Stability Assessment)
File: 04_machine_learning/stage7_group_cv.py

Evaluates:
  - 5-Fold GroupKFold strictly on the training set (grouped by Record)
  - Zero leakage across cross-validation folds
  - Measures fold-to-fold generalization stability (Accuracy, Balanced Acc, Sensitivity, Specificity, F1)
  - Exports results to 05_results/stage7_cv_results.json
  - Exports report to 05_results/stage7_cv_report.txt
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"

SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"
REPORT_PATH = RESULTS_DIR / "stage7_cv_report.txt"
CV_RESULTS_JSON_PATH = RESULTS_DIR / "stage7_cv_results.json"

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
    log(lines, "ECG FINAL PRO — STAGE 7: RECORD-GROUPED 5-FOLD CROSS-VALIDATION")
    log(lines, "=" * 72)

    splits = json.loads(SPLITS_JSON_PATH.read_text(encoding="utf-8"))
    train_records = splits["train_records"]

    df = pd.read_csv(DATASET_PATH)
    train_df = df[df["Record"].isin(train_records)].copy().reset_index(drop=True)

    log(lines, f"Training Set: {len(train_df):,} beats across {len(train_records)} records.")
    log(lines, "Setting up 5-Fold GroupKFold partitioned strictly by Record...")

    gkf = GroupKFold(n_splits=5)
    groups = train_df["Record"].values
    X_raw = train_df[FEATURES].values
    y = train_df["Label"].values

    fold_metrics = []

    for fold_num, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups=groups), 1):
        fold_train_recs = set(train_df.iloc[train_idx]["Record"].unique())
        fold_val_recs = set(train_df.iloc[val_idx]["Record"].unique())

        # Assert zero leakage
        assert len(fold_train_recs.intersection(fold_val_recs)) == 0, "Fold record leakage!"

        # Fit scaler on fold train only
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_raw[train_idx])
        y_fold_train = y[train_idx]

        X_fold_val = scaler.transform(X_raw[val_idx])
        y_fold_val = y[val_idx]

        # Train regularized Random Forest for this fold
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42 + fold_num,
            n_jobs=-1,
        )
        rf.fit(X_fold_train, y_fold_train)

        val_preds = rf.predict(X_fold_val)

        acc = accuracy_score(y_fold_val, val_preds)
        bal_acc = balanced_accuracy_score(y_fold_val, val_preds)
        rec = recall_score(y_fold_val, val_preds, pos_label=1, zero_division=0)
        spec = recall_score(y_fold_val, val_preds, pos_label=0, zero_division=0)
        f1 = f1_score(y_fold_val, val_preds, pos_label=1, zero_division=0)

        fold_data = {
            "fold": fold_num,
            "train_beats": int(len(train_idx)),
            "val_beats": int(len(val_idx)),
            "train_records_count": len(fold_train_recs),
            "val_records_count": len(fold_val_recs),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "sensitivity": float(rec),
            "specificity": float(spec),
            "f1_score": float(f1),
        }
        fold_metrics.append(fold_data)

        log(lines, f"\nFold {fold_num}: {len(fold_train_recs)} train recs, {len(fold_val_recs)} val recs")
        log(lines, f"  Acc: {acc*100:6.2f}% | Bal Acc: {bal_acc*100:6.2f}% | Sens: {rec*100:6.2f}% | Spec: {spec*100:6.2f}% | F1: {f1*100:6.2f}%")

    # Aggregate statistics
    accs = [m["accuracy"] for m in fold_metrics]
    bal_accs = [m["balanced_accuracy"] for m in fold_metrics]
    senss = [m["sensitivity"] for m in fold_metrics]
    specs = [m["specificity"] for m in fold_metrics]
    f1s = [m["f1_score"] for m in fold_metrics]

    log(lines, "\n" + "-" * 72)
    log(lines, "RECORD-GROUPED 5-FOLD CV SUMMARY (MEAN +/- STD)")
    log(lines, "-" * 72)
    log(lines, f"  Accuracy          : {np.mean(accs)*100:6.2f}% +/- {np.std(accs)*100:5.2f}%")
    log(lines, f"  Balanced Accuracy : {np.mean(bal_accs)*100:6.2f}% +/- {np.std(bal_accs)*100:5.2f}%")
    log(lines, f"  Sensitivity (Abn) : {np.mean(senss)*100:6.2f}% +/- {np.std(senss)*100:5.2f}%")
    log(lines, f"  Specificity (Norm): {np.mean(specs)*100:6.2f}% +/- {np.std(specs)*100:5.2f}%")
    log(lines, f"  F1-Score (Abn)    : {np.mean(f1s)*100:6.2f}% +/- {np.std(f1s)*100:5.2f}%")

    results_dict = {
        "n_folds": 5,
        "fold_details": fold_metrics,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "mean_sensitivity": float(np.mean(senss)),
        "std_sensitivity": float(np.std(senss)),
        "mean_specificity": float(np.mean(specs)),
        "std_specificity": float(np.std(specs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
    }

    CV_RESULTS_JSON_PATH.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
    log(lines, f"\nSaved cross-validation results to: {CV_RESULTS_JSON_PATH}")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 7 RESULT: PASSED")
    log(lines, "Record-grouped cross-validation completed. Ready for Stage 8 (Independent Test Evaluation).")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
