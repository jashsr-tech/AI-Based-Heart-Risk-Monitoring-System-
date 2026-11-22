
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
    "HeartRate", "RRInterval", "PRInterval", "QRSDuration",
    "QTInterval", "STSegment", "HRV", "PAmplitude",
    "RAmplitude", "TAmplitude",
]

# Synthetic ECG values from your previous Stage 2E output.
# Keep the same feature units and definitions as that output.
SYNTHETIC = {
    "HeartRate": 71.91438763,
    "RRInterval": 0.83432540,
    "PRInterval": 0.16017214,
    "QRSDuration": 0.09229264,
    "QTInterval": 0.35105634,
    "STSegment": -0.31447483,
    "HRV": 0.05156791,
    "PAmplitude": 0.28571171,
    "RAmplitude": 4.15882634,
    "TAmplitude": 0.68568344,
}

print("=" * 70)
print("STAGE 2K - SYNTHETIC FEATURE COMPATIBILITY")
print("=" * 70)

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

df = pd.read_csv(DATASET)

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing dataset features: {missing}")

for feature in FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

df = df.dropna(subset=FEATURES)

rows = []

for feature in FEATURES:
    values = df[feature]
    synthetic_value = SYNTHETIC[feature]

    p01 = values.quantile(0.01)
    median = values.median()
    p99 = values.quantile(0.99)

    if synthetic_value < p01:
        status = "BELOW DATASET 1ST PERCENTILE"
    elif synthetic_value > p99:
        status = "ABOVE DATASET 99TH PERCENTILE"
    else:
        status = "WITHIN CENTRAL 98% RANGE"

    rows.append({
        "Feature": feature,
        "Synthetic": synthetic_value,
        "Dataset_1%": p01,
        "Dataset_Median": median,
        "Dataset_99%": p99,
        "Status": status,
    })

result = pd.DataFrame(rows)

print("\nFeature comparison:")
print(result.round(5).to_string(index=False))

print("\nImportant limitations:")
print("- A matching numerical range does not prove matching feature definitions.")
print("- The dataset contains beat-level rows; your synthetic values summarize a signal.")
print("- Do not use this comparison alone to claim that a prediction is reliable.")
print("\nSTAGE 2K COMPLETE")
print("No files were modified or overwritten.")
