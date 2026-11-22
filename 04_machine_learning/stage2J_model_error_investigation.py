
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET = (
    PROJECT_DIR
    / "04_machine_learning"
    / "ecg_features_balanced_clean_labeled.csv"
)

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

print("=" * 70)
print("STAGE 2J - MODEL ERROR INVESTIGATION")
print("=" * 70)

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

df = pd.read_csv(DATASET)

required = FEATURES + ["Label", "Record", "Condition"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in FEATURES + ["Label"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()

print(f"\nDataset rows: {len(df)}")
print(f"Unique records: {df['Record'].nunique()}")
print(f"Unique conditions: {df['Condition'].nunique()}")

print("\n1. LABEL COUNTS BY RECORD")
record_summary = (
    df.groupby(["Record", "Label"])
    .size()
    .unstack(fill_value=0)
)
print(record_summary.to_string())

print("\n2. LABEL COUNTS BY CONDITION")
condition_summary = pd.crosstab(
    df["Condition"], df["Label"]
)
print(condition_summary.to_string())

print("\n3. FEATURE RANGE CHECK")
print(
    df[FEATURES]
    .describe(percentiles=[0.01, 0.50, 0.99])
    .T
    .round(4)
    .to_string()
)

print("\n4. EXTREME VALUES")
for feature in FEATURES:
    low = df[feature].quantile(0.01)
    high = df[feature].quantile(0.99)

    low_count = int((df[feature] < low).sum())
    high_count = int((df[feature] > high).sum())

    print(
        f"{feature:15s} "
        f"1st percentile={low:10.4f}  "
        f"99th percentile={high:10.4f}  "
        f"below={low_count}  above={high_count}"
    )

print("\n5. NORMAL VS ABNORMAL FEATURE MEDIANS")
medians = df.groupby("Label")[FEATURES].median().T
medians.columns = ["Normal_0", "Abnormal_1"]
medians["Difference"] = (
    medians["Abnormal_1"] - medians["Normal_0"]
)
print(medians.round(4).to_string())

print("\n6. RECORD-LEVEL LABEL MIX")
record_mix = (
    df.groupby("Record")
    .agg(
        rows=("Label", "size"),
        labels=("Label", "nunique"),
        abnormal_fraction=("Label", "mean"),
    )
    .sort_values("abnormal_fraction")
)
print(record_mix.round(4).to_string())

print("\n" + "=" * 70)
print("STAGE 2J COMPLETE")
print("No model, scaler, or dataset was modified.")
print("=" * 70)
