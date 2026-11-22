
from pathlib import Path
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
OLD_MODEL_DIR = Path(r"C:\Users\srjas\OneDrive\Desktop\proj code")

FEATURE_FILE = BASE_DIR / "03_feature_extraction" / "ecg_ml_features.npy"
MODEL_FILE = OLD_MODEL_DIR / "ecg_decision_tree_model.pkl"
SCALER_FILE = OLD_MODEL_DIR / "ecg_feature_scaler.pkl"
ORDER_FILE = OLD_MODEL_DIR / "model_feature_order.txt"

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

print("=" * 60)
print("STAGE 2F — NEW PIPELINE COMPATIBILITY TEST")
print("=" * 60)

# Load and validate the 10-feature vector.
features = np.asarray(np.load(FEATURE_FILE), dtype=float).flatten()

if len(features) != 10:
    raise ValueError(f"Expected 10 features; received {len(features)}.")

if not np.all(np.isfinite(features)):
    raise ValueError("Feature vector contains NaN or infinite values.")

# Verify the saved feature order.
saved_order = [
    line.strip()
    for line in ORDER_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if saved_order != FEATURE_NAMES:
    raise ValueError(
        f"Feature-order mismatch.\nExpected: {FEATURE_NAMES}\nSaved: {saved_order}"
    )

# Load the existing saved model and scaler.
# Only load these files if you trust their origin.
scaler = joblib.load(SCALER_FILE)
model = joblib.load(MODEL_FILE)

if scaler.n_features_in_ != 10 or model.n_features_in_ != 10:
    raise ValueError("The saved model or scaler does not expect 10 features.")

if hasattr(scaler, "feature_names_in_"):
    if list(scaler.feature_names_in_) != FEATURE_NAMES:
        raise ValueError("Scaler feature names do not match the expected order.")

print("\nFeature vector:")
for name, value in zip(FEATURE_NAMES, features):
    print(f"{name:<15}: {value:.8f}")

print("\nStructural checks passed.")
print("Model:", type(model).__name__)
print("Scaler:", type(scaler).__name__)
print("Model classes:", model.classes_)

# No prediction yet: feature-definition compatibility is not established.
print("\nPrediction intentionally NOT performed.")
print("The training feature definitions still require verification.")
