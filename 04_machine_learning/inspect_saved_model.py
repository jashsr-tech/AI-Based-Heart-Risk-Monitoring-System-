
from pathlib import Path
import joblib

MODEL_DIR = Path(r"C:\Users\srjas\OneDrive\Desktop\proj code")

model_path = MODEL_DIR / "ecg_decision_tree_model.pkl"
scaler_path = MODEL_DIR / "ecg_feature_scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("=" * 55)
print("SAVED ECG MODEL INSPECTION")
print("=" * 55)

print("\nModel type:", type(model).__name__)
print("Model parameters:", model.get_params())
print("Model classes:", model.classes_)
print("Expected input features:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("Model feature names:", model.feature_names_in_)

print("\nScaler type:", type(scaler).__name__)
print("Scaler feature count:", scaler.n_features_in_)
print("Scaler means:", scaler.mean_)
print("Scaler scales:", scaler.scale_)

if hasattr(scaler, "feature_names_in_"):
    print("Scaler feature names:", scaler.feature_names_in_)

print("\nInspection complete. No prediction performed.")
