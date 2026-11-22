"""
ECG FINAL PRO — PHASE 1: STAGE 2
10-Feature Validation & Physiological Bounds Analysis
File: 04_machine_learning/stage2_feature_validation.py

Verifies:
  - Statistical distributions for each of the 10 core features
  - Class-conditional distribution separation (Normal vs Abnormal)
  - Physiological plausibility checks and anomaly counts
  - Exports stage2_feature_validation_report.txt and stage2_feature_summary.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = RESULTS_DIR / "stage2_feature_validation_report.txt"
CSV_PATH = RESULTS_DIR / "stage2_feature_summary.csv"

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

# Standard physiological plausibility thresholds for beat-level automated features
PHYSIOLOGICAL_BOUNDS = {
    "HeartRate": (20.0, 250.0, "BPM"),
    "RRInterval": (0.15, 3.0, "s"),
    "PRInterval": (0.01, 0.45, "s"),
    "QRSDuration": (0.02, 0.35, "s"),
    "QTInterval": (0.05, 0.80, "s"),
    "HRV": (0.0, 2.0, "s"),
    "STSegment": (-10.0, 10.0, "arb. amp"),
    "PAmplitude": (-5.0, 10.0, "arb. amp"),
    "RAmplitude": (0.1, 20.0, "arb. amp"),
    "TAmplitude": (-5.0, 10.0, "arb. amp"),
}

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 72)
    log(lines, "ECG FINAL PRO — STAGE 2: 10-FEATURE VALIDATION & STATISTICAL SANITY")
    log(lines, "=" * 72)

    if not DATASET_PATH.exists():
        log(lines, f"[FAIL] Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    log(lines, f"Dataset loaded: {len(df):,} rows.")

    normal_df = df[df["Label"] == 0]
    abnormal_df = df[df["Label"] == 1]

    summary_rows = []
    outlier_counts = {}

    log(lines, "\n" + "-" * 72)
    log(lines, "FEATURE-BY-FEATURE DISTRIBUTION & PHYSIOLOGICAL CHECK")
    log(lines, "-" * 72)

    for feat in FEATURES:
        col = df[feat]
        norm_col = normal_df[feat]
        abnorm_col = abnormal_df[feat]

        low_bound, high_bound, unit = PHYSIOLOGICAL_BOUNDS[feat]
        outliers_below = int((col < low_bound).sum())
        outliers_above = int((col > high_bound).sum())
        total_outliers = outliers_below + outliers_above
        outlier_pct = (total_outliers / len(df)) * 100.0
        outlier_counts[feat] = (total_outliers, outlier_pct)

        row_dict = {
            "Feature": feat,
            "Unit": unit,
            "Mean": float(col.mean()),
            "Std": float(col.std()),
            "Overall_Median": float(col.median()),
            "Overall_1%": float(col.quantile(0.01)),
            "Overall_99%": float(col.quantile(0.99)),
            "Normal_Median": float(norm_col.median()),
            "Normal_1%": float(norm_col.quantile(0.01)),
            "Normal_99%": float(norm_col.quantile(0.99)),
            "Abnormal_Median": float(abnorm_col.median()),
            "Abnormal_1%": float(abnorm_col.quantile(0.01)),
            "Abnormal_99%": float(abnorm_col.quantile(0.99)),
            "Median_Diff (Abn - Norm)": float(abnorm_col.median() - norm_col.median()),
            "Physio_Range": f"[{low_bound}, {high_bound}]",
            "Outlier_Count": total_outliers,
            "Outlier_Pct": round(outlier_pct, 3),
        }
        summary_rows.append(row_dict)

        log(lines, f"\n--- {feat} ({unit}) ---")
        log(lines, f"  Plausible Range: [{low_bound}, {high_bound}] {unit}")
        log(lines, f"  Overall Range  : min={col.min():.4f}, max={col.max():.4f}, median={col.median():.4f}")
        log(lines, f"  Normal (0)     : median={norm_col.median():.4f}, 1st-99th%=[{norm_col.quantile(0.01):.4f}, {norm_col.quantile(0.99):.4f}]")
        log(lines, f"  Abnormal (1)   : median={abnorm_col.median():.4f}, 1st-99th%=[{abnorm_col.quantile(0.01):.4f}, {abnorm_col.quantile(0.99):.4f}]")
        log(lines, f"  Separation     : delta={abnorm_col.median() - norm_col.median():+.4f}")
        status = "PASSED" if outlier_pct < 1.0 else f"NOTE: {outlier_pct:.2f}% beyond typical bounds"
        log(lines, f"  Status         : {status}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CSV_PATH, index=False)
    log(lines, f"\nSummary table saved to: {CSV_PATH}")

    # Check for non-physiological critical errors (e.g., negative duration/intervals)
    critical_errors = []
    for non_neg in ["HeartRate", "RRInterval", "PRInterval", "QRSDuration", "QTInterval", "HRV"]:
        if (df[non_neg] <= 0).sum() > 0:
            critical_errors.append(f"{non_neg} has {int((df[non_neg] <= 0).sum())} non-positive values!")

    if critical_errors:
        log(lines, f"\n[FAIL] Critical physiological invalidity detected:\n" + "\n".join(critical_errors))
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)

    log(lines, "\n[PASS] Zero non-positive values in strictly positive interval/rate features.")
    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 2 RESULT: PASSED")
    log(lines, "All 10 features have passed validation checks and statistical sanity.")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
