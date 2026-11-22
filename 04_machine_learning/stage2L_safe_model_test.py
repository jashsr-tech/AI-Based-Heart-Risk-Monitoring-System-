
from pathlib import Path
import sys
import joblib
import numpy as np

# ============================================================
# STAGE 2L — SAFE SAVED MODEL TEST
# ============================================================

# Project root: ECG FINAL PROJ
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(
    r"C:\Users\srjas\OneDrive\Desktop\proj code"
)

MODEL_FILE = MODEL_DIR / "ecg_decision_tree_model.pkl"
SCALER_FILE = MODEL_DIR / "ecg_feature_scaler.pkl"
ORDER_FILE = MODEL_DIR / "model_feature_order.txt"

FEATURE_FILE = (
    PROJECT_DIR
    / "03_feature_extraction"
    / "ecg_ml_features.npy"
)

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


def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def main():
    print("=" * 65)
    print("STAGE 2L — SAFE SAVED MODEL TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Check that required files exist
    # --------------------------------------------------------
    required_files = {
        "Model": MODEL_FILE,
        "Scaler": SCALER_FILE,
        "Feature order": ORDER_FILE,
        "Stage 2E feature vector": FEATURE_FILE,
    }

    print("\n[1] Checking required files...")

    for name, path in required_files.items():
        if not path.is_file():
            stop(f"{name} file not found:\n{path}")
        print(f"FOUND: {name} -> {path}")

    # --------------------------------------------------------
    # 2. Read the saved feature order
    # --------------------------------------------------------
    print("\n[2] Checking feature order...")

    saved_features = [
        line.strip()
        for line in ORDER_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("Saved feature order:")
    for index, name in enumerate(saved_features, start=1):
        print(f"{index}. {name}")

    if saved_features != EXPECTED_FEATURES:
        stop(
            "The saved feature order does not match the expected "
            "10-feature order. Do not predict until this is resolved."
        )

    print("PASS: Feature order matches.")

    # --------------------------------------------------------
    # 3. Load the feature vector created by Stage 2E
    # --------------------------------------------------------
    print("\n[3] Loading Stage 2E feature vector...")

    features = np.load(FEATURE_FILE, allow_pickle=False)
    features = np.asarray(features, dtype=np.float64).reshape(-1)

    if features.size != len(EXPECTED_FEATURES):
        stop(
            f"Expected 10 features, but found {features.size}."
        )

    if not np.all(np.isfinite(features)):
        stop("The feature vector contains NaN or infinite values.")

    print("\nStage 2E feature vector:")
    for name, value in zip(EXPECTED_FEATURES, features):
        print(f"{name:15s}: {value:.8f}")

    # --------------------------------------------------------
    # 4. Load model and scaler using joblib
    # --------------------------------------------------------
    print("\n[4] Loading saved model and scaler with joblib...")

    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception as exc:
        stop(
            "Could not load the saved model or scaler.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Details: {exc}\n\n"
            "Do not delete or overwrite either file. Send me "
            "the complete error message."
        )

    print("PASS: Model loaded.")
    print("PASS: Scaler loaded.")

    # --------------------------------------------------------
    # 5. Check model and scaler compatibility
    # --------------------------------------------------------
    print("\n[5] Checking model/scaler compatibility...")

    if not hasattr(model, "predict"):
        stop("Loaded model does not provide predict().")

    if not hasattr(scaler, "transform"):
        stop("Loaded scaler does not provide transform().")

    scaler_count = getattr(scaler, "n_features_in_", None)
    model_count = getattr(model, "n_features_in_", None)

    print(f"Scaler expected features: {scaler_count}")
    print(f"Model expected features : {model_count}")

    if scaler_count is not None and scaler_count != len(features):
        stop("Scaler expects a different number of features.")

    if model_count is not None and model_count != len(features):
        stop("Model expects a different number of features.")

    scaler_names = getattr(scaler, "feature_names_in_", None)
    model_names = getattr(model, "feature_names_in_", None)

    if scaler_names is not None:
        if list(scaler_names) != EXPECTED_FEATURES:
            stop("Scaler feature names do not match the expected order.")

    if model_names is not None:
        if list(model_names) != EXPECTED_FEATURES:
            stop("Model feature names do not match the expected order.")

    print("PASS: Feature counts are compatible.")

    # --------------------------------------------------------
    # 6. Scale the feature vector and predict
    # --------------------------------------------------------
    print("\n[6] Running exploratory model prediction...")

    X = features.reshape(1, -1)

    try:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
    except Exception as exc:
        stop(
            "Prediction failed.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Details: {exc}"
        )

    label = int(prediction[0])

    print("\n" + "=" * 65)
    print("EXPLORATORY PREDICTION RESULT")
    print("=" * 65)

    if label == 0:
        print("Model output: NORMAL (label 0)")
    elif label == 1:
        print("Model output: ABNORMAL (label 1)")
    else:
        print(f"Model output: Unknown label ({label})")

    # --------------------------------------------------------
    # 7. Show probabilities if the model supports them
    # --------------------------------------------------------
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_scaled)[0]

            print("\nModel class probabilities:")
            for class_label, probability in zip(
                model.classes_, probabilities
            ):
                print(
                    f"Label {class_label}: "
                    f"{probability * 100:.2f}%"
                )

            print(
                "\nNote: These are model probabilities, not "
                "clinically validated probabilities of heart disease."
            )
        except Exception as exc:
            print(
                "\nProbability display unavailable: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 65)
    print("STAGE 2L TEST FINISHED")
    print("=" * 65)

    print(
        "\nIMPORTANT:\n"
        "- This is an exploratory software test, not a medical diagnosis.\n"
        "- A successful prediction does not prove the model is accurate.\n"
        "- Synthetic ECG input is not a real patient's ECG.\n"
        "- Dataset feature definitions and units still need validation."
    )


if __name__ == "__main__":
    main()
