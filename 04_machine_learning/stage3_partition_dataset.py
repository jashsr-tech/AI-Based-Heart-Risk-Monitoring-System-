"""
ECG FINAL PRO — PHASE 1: STAGE 3
Patient / Record-Separated Partitioning (Zero-Leakage Splitting)
File: 04_machine_learning/stage3_partition_dataset.py

Ensures:
  - 100% record-isolated partitioning (no heartbeat from a test patient ever appears in training)
  - Balanced stratified representation of both Normal and Abnormal records
  - Deterministic reproducibility with random_state=42
  - Exports split manifest to 05_results/stage3_record_splits.json
  - Exports verification report to 05_results/stage3_partition_report.txt
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
RESULTS_DIR = PROJECT_ROOT / "05_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = RESULTS_DIR / "stage3_partition_report.txt"
SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 72)
    log(lines, "ECG FINAL PRO — STAGE 3: RECORD-SEPARATED DATASET PARTITIONING")
    log(lines, "=" * 72)

    df = pd.read_csv(DATASET_PATH)
    log(lines, f"Dataset loaded: {len(df):,} rows across {df['Record'].nunique()} records.")

    # Record-level metadata table
    rec_meta = df.groupby("Record").agg(
        Label=("Label", "first"),
        Condition=("Condition", "first"),
        BeatCount=("Beat", "count")
    ).reset_index()

    normal_recs = sorted(rec_meta[rec_meta["Label"] == 0]["Record"].tolist())
    abnormal_recs = sorted(rec_meta[rec_meta["Label"] == 1]["Record"].tolist())

    log(lines, f"Total Normal records:   {len(normal_recs)}")
    log(lines, f"Total Abnormal records: {len(abnormal_recs)}")

    # Stratified record split using fixed seed
    rng = np.random.default_rng(42)

    # Pick 3 Normal records for test (out of 13 ~ 23%) and 7 Abnormal records for test (out of 35 = 20%)
    shuffled_norm = rng.permutation(normal_recs)
    shuffled_abnorm = rng.permutation(abnormal_recs)

    test_norm_recs = sorted(shuffled_norm[:3].tolist())
    train_norm_recs = sorted(shuffled_norm[3:].tolist())

    test_abnorm_recs = sorted(shuffled_abnorm[:7].tolist())
    train_abnorm_recs = sorted(shuffled_abnorm[7:].tolist())

    train_records = sorted(train_norm_recs + train_abnorm_recs)
    test_records = sorted(test_norm_recs + test_abnorm_recs)

    # Verify zero leakage
    leakage = set(train_records).intersection(set(test_records))
    if len(leakage) > 0:
        log(lines, f"[FAIL] CRITICAL LEAKAGE DETECTED! Overlapping records: {leakage}")
        sys.exit(1)

    log(lines, "\n[PASS] Zero patient / record leakage verified: set(train) & set(test) is empty.")

    # Split dataframe
    train_df = df[df["Record"].isin(train_records)].copy()
    test_df = df[df["Record"].isin(test_records)].copy()

    log(lines, "\nPartition Summary:")
    log(lines, f"  Training set   : {len(train_df):,} beats ({len(train_records)} records)")
    log(lines, f"    - Normal (0) : {(train_df['Label'] == 0).sum():,} beats ({len(train_norm_recs)} records)")
    log(lines, f"    - Abnormal(1): {(train_df['Label'] == 1).sum():,} beats ({len(train_abnorm_recs)} records)")
    log(lines, f"  Independent Test: {len(test_df):,} beats ({len(test_records)} records)")
    log(lines, f"    - Normal (0) : {(test_df['Label'] == 0).sum():,} beats ({len(test_norm_recs)} records)")
    log(lines, f"    - Abnormal(1): {(test_df['Label'] == 1).sum():,} beats ({len(test_abnorm_recs)} records)")

    log(lines, f"\nTraining record IDs ({len(train_records)}):")
    log(lines, f"  Normal (0)  : {train_norm_recs}")
    log(lines, f"  Abnormal (1): {train_abnorm_recs}")

    log(lines, f"\nIndependent Test record IDs ({len(test_records)}):")
    log(lines, f"  Normal (0)  : {test_norm_recs}")
    log(lines, f"  Abnormal (1): {test_abnorm_recs}")

    # Save partition manifest to JSON
    manifest = {
        "random_seed": 42,
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_records": train_records,
        "test_records": test_records,
        "train_normal_records": train_norm_recs,
        "train_abnormal_records": train_abnorm_recs,
        "test_normal_records": test_norm_recs,
        "test_abnormal_records": test_abnorm_recs,
    }

    SPLITS_JSON_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(lines, f"\nSaved partition manifest to: {SPLITS_JSON_PATH}")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 3 RESULT: PASSED")
    log(lines, "Record-separated partition created and certified ready for Stage 4.")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
