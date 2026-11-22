
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_FILE = BASE_DIR / "03_feature_extraction" / "ecg_ml_features.npy"

FEATURE_NAMES = [
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

print("=" * 55)
print("STAGE 2E VECTOR VERIFICATION")
print("=" * 55)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

features = np.asarray(np.load(FEATURE_FILE), dtype=float).flatten()

print("Feature file:", FEATURE_FILE)
print("Feature count:", len(features))

if len(features) != len(FEATURE_NAMES):
    raise ValueError(
        f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
    )

if not np.all(np.isfinite(features)):
    raise ValueError("Feature vector contains NaN or infinite values.")

for name, value in zip(FEATURE_NAMES, features):
    print(f"{name:<15}: {value:.8f}")

print("\nAll 10 values are finite.")
print("This script performs no model prediction.")
