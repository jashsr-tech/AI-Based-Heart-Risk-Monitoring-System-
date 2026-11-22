"""
ECG FINAL PRO — PHASE 1: STAGE 1
Dataset Verification and Schema Audit
File: 04_machine_learning/stage1_dataset_audit.py

Verifies:
  - Existence of target training dataset
  - Column schema integrity (10 features + metadata + binary label)
  - Numeric conversions, null/NaN/inf absence
  - Label balance and record distribution
  - Generates audit report in 05_results/
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = RESULTS_DIR / "stage1_dataset_audit_report.txt"

EXPECTED_FEATURES = [
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

EXPECTED_METADATA = ["Record", "Label", "Condition", "Beat"]
ALL_EXPECTED = EXPECTED_METADATA + EXPECTED_FEATURES

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 72)
    log(lines, "ECG FINAL PRO — STAGE 1: DATASET VERIFICATION & SCHEMA AUDIT")
    log(lines, "=" * 72)
    log(lines, f"Target Dataset: {DATASET_PATH}")

    # Check 1: File existence
    if not DATASET_PATH.exists():
        log(lines, f"\n[FAIL] Dataset file does not exist: {DATASET_PATH}")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)
    log(lines, "[PASS] Dataset file exists.")

    # Check 2: Load CSV
    try:
        df = pd.read_csv(DATASET_PATH)
        log(lines, f"[PASS] Loaded CSV successfully. Total Rows: {len(df):,}, Total Columns: {len(df.columns)}")
    except Exception as exc:
        log(lines, f"\n[FAIL] Error reading CSV: {exc}")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)

    # Check 3: Column schema
    missing_cols = [c for c in ALL_EXPECTED if c not in df.columns]
    if missing_cols:
        log(lines, f"\n[FAIL] Missing required columns: {missing_cols}")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)
    log(lines, f"[PASS] All {len(ALL_EXPECTED)} expected columns are present.")

    # Check 4: Data types and numeric conversions
    numeric_cols = EXPECTED_FEATURES + ["Label", "Record"]
    coercion_issues = {}
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(converted.isna().sum())
        if nan_count > 0:
            coercion_issues[col] = nan_count

    if coercion_issues:
        log(lines, f"\n[FAIL] Non-numeric or missing values detected: {coercion_issues}")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)
    log(lines, "[PASS] All feature and label columns are fully numeric with zero missing values.")

    # Check 5: Check infinite values
    inf_count = int(np.isinf(df[EXPECTED_FEATURES].values).sum())
    if inf_count > 0:
        log(lines, f"\n[FAIL] Infinite values found in feature columns: {inf_count}")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)
    log(lines, "[PASS] Zero infinite values detected across all features.")

    # Check 6: Label analysis
    labels = df["Label"].unique()
    log(lines, f"\nTarget Labels present: {sorted(labels.tolist())}")
    if set(labels) != {0, 1}:
        log(lines, f"\n[FAIL] Expected binary labels {0, 1}, found: {labels}")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)

    counts = df["Label"].value_counts().sort_index()
    pcts = df["Label"].value_counts(normalize=True).sort_index() * 100
    log(lines, f"  - Label 0 (NORMAL)  : {counts[0]:,} beats ({pcts[0]:.2f}%)")
    log(lines, f"  - Label 1 (ABNORMAL): {counts[1]:,} beats ({pcts[1]:.2f}%)")

    # Check 7: Conditions breakdown
    log(lines, "\nConditions Breakdown:")
    cond_table = pd.crosstab(df["Condition"], df["Label"])
    cond_table.columns = ["Normal (0)", "Abnormal (1)"]
    log(lines, cond_table.to_string())

    # Check 8: Patient / Record Analysis
    records = df["Record"].unique()
    log(lines, f"\nPatient / Record Counts:")
    log(lines, f"  - Unique Records: {len(records)}")

    record_label_mix = df.groupby("Record")["Label"].nunique()
    mixed_records = record_label_mix[record_label_mix > 1]
    log(lines, f"  - Records with mixed labels: {len(mixed_records)}")

    norm_records = df[df["Label"] == 0]["Record"].unique()
    abnorm_records = df[df["Label"] == 1]["Record"].unique()
    log(lines, f"  - Records with purely Normal beats (0): {len(norm_records)} -> {sorted(norm_records.tolist())}")
    log(lines, f"  - Records with purely Abnormal beats (1): {len(abnorm_records)} -> {sorted(abnorm_records.tolist())}")

    # Summary Conclusion
    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 1 RESULT: PASSED")
    log(lines, "Dataset verified and certified ready for Stage 2 (Feature Validation).")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
