
from pathlib import Path
import numpy as np
import pandas as pd

# ECG FINAL PRO — STAGE 2E.1
ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "04_machine_learning" / "ecg_features_balanced_clean_labeled.csv"
FEATURE_FILE = ROOT / "03_feature_extraction" / "ecg_ml_features.npy"

FEATURES = [
    "HeartRate", "RRInterval", "PRInterval", "QRSDuration",
    "QTInterval", "STSegment", "HRV", "PAmplitude",
    "RAmplitude", "TAmplitude"
]

print("=" * 65)
print("ECG FINAL PRO — FEATURE COMPATIBILITY CHECK")
print("=" * 65)

for path in [DATASET, FEATURE_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

df = pd.read_csv(DATASET)
vector = np.load(FEATURE_FILE).astype(float)

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

if len(vector) != len(FEATURES):
    raise ValueError("Feature vector length does not match feature list.")

print(f"Dataset rows: {len(df)}")
print(f"Feature columns checked: {len(FEATURES)}")

print("\nFEATURE DISTRIBUTION CHECK")
for name, value in zip(FEATURES, vector):
    series = pd.to_numeric(df[name], errors="coerce").dropna()

    if series.empty:
        print(f"\n{name}: no numeric training values")
        continue

    low = series.min()
    high = series.max()
    median = series.median()
    inside = low <= value <= high

    print(f"\n{name}")
    print(f"  New signal : {value:.6f}")
    print(f"  Train min  : {low:.6f}")
    print(f"  Train median: {median:.6f}")
    print(f"  Train max  : {high:.6f}")
    print(f"  Range check: {'INSIDE' if inside else 'OUTSIDE'}")

print("\nML FEATURE VECTOR ORDER")
for name, value in zip(FEATURES, vector):
    print(f"{name:14s}: {value:.6f}")

print("\nCHECK COMPLETE — do not run prediction yet.")
