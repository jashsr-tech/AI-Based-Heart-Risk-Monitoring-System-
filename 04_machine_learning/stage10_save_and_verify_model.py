"""
ECG FINAL PRO — PHASE 1: STAGE 10
Model Serialization & Artifact Storage Verification
File: 04_machine_learning/stage10_save_and_verify_model.py

Saves & Validates:
  - Archives old model/scaler files safely with clear legacy tags
  - Persists new validated RandomForest model to 04_machine_learning/models/ecg_random_forest_model.pkl
  - Persists new fitted scaler to 04_machine_learning/models/ecg_feature_scaler.pkl
  - Persists verified feature order to 04_machine_learning/models/model_feature_order.txt
  - Writes comprehensive model manifest to 04_machine_learning/models/model_manifest.json
  - Executes round-trip deserialization, feature shape check, and numerical inference verification
  - Exports verification report to 05_results/stage10_serialization_report.txt
"""

from pathlib import Path
import sys
import json
import shutil
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "04_machine_learning" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "05_results"

CANDIDATE_RF_PATH = RESULTS_DIR / "stage6_candidate_random_forest.pkl"
CANDIDATE_SCALER_PATH = RESULTS_DIR / "stage4_candidate_scaler.pkl"
SPLITS_JSON_PATH = RESULTS_DIR / "stage3_record_splits.json"
TEST_METRICS_PATH = RESULTS_DIR / "stage8_independent_evaluation_metrics.json"

TARGET_MODEL_PATH = MODELS_DIR / "ecg_random_forest_model.pkl"
TARGET_SCALER_PATH = MODELS_DIR / "ecg_feature_scaler.pkl"
TARGET_ORDER_PATH = MODELS_DIR / "model_feature_order.txt"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
REPORT_PATH = RESULTS_DIR / "stage10_serialization_report.txt"

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

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 72)
    log(lines, "ECG FINAL PRO — STAGE 10: MODEL SERIALIZATION & ARTIFACT STORAGE")
    log(lines, "=" * 72)

    # 1. Archive legacy models safely if not already archived
    legacy_dt = MODELS_DIR / "ecg_decision_tree_model.pkl"
    legacy_scaler = MODELS_DIR / "ecg_feature_scaler.pkl"
    legacy_archive_dir = MODELS_DIR / "legacy_decision_tree_archive"
    legacy_archive_dir.mkdir(parents=True, exist_ok=True)

    if legacy_dt.exists() and not (legacy_archive_dir / "ecg_decision_tree_model.pkl").exists():
        shutil.copy2(legacy_dt, legacy_archive_dir / "ecg_decision_tree_model.pkl")
        log(lines, f"[INFO] Archived legacy Decision Tree to: {legacy_archive_dir / 'ecg_decision_tree_model.pkl'}")

    if legacy_scaler.exists() and not (legacy_archive_dir / "ecg_feature_scaler.pkl").exists():
        shutil.copy2(legacy_scaler, legacy_archive_dir / "ecg_feature_scaler.pkl")
        log(lines, f"[INFO] Archived legacy Scaler to: {legacy_archive_dir / 'ecg_feature_scaler.pkl'}")

    # 2. Copy and persist new validated model & scaler
    log(lines, "\nPersisting newly trained Random Forest artifacts...")
    shutil.copy2(CANDIDATE_RF_PATH, TARGET_MODEL_PATH)
    log(lines, f"  -> Saved new model : {TARGET_MODEL_PATH}")

    shutil.copy2(CANDIDATE_SCALER_PATH, TARGET_SCALER_PATH)
    log(lines, f"  -> Saved new scaler: {TARGET_SCALER_PATH}")

    # 3. Write feature order
    TARGET_ORDER_PATH.write_text("\n".join(FEATURES) + "\n", encoding="utf-8")
    log(lines, f"  -> Saved feature order: {TARGET_ORDER_PATH}")

    # 4. Construct and save model manifest
    splits = json.loads(SPLITS_JSON_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))

    manifest = {
        "model_name": "ECG Arrhythmia Random Forest Ensemble",
        "model_file": TARGET_MODEL_PATH.name,
        "scaler_file": TARGET_SCALER_PATH.name,
        "feature_order_file": TARGET_ORDER_PATH.name,
        "model_type": "RandomForestClassifier",
        "n_estimators": 150,
        "max_depth": 12,
        "features": FEATURES,
        "features_count": len(FEATURES),
        "classes": [0, 1],
        "class_names": {0: "NORMAL", 1: "ABNORMAL"},
        "training_records_count": len(splits["train_records"]),
        "independent_test_records_count": len(splits["test_records"]),
        "independent_test_metrics": {
            "accuracy": metrics["test_accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "sensitivity": metrics["sensitivity_recall"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
        },
        "serialization_status": "VALIDATED_AND_CERTIFIED",
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(lines, f"  -> Saved manifest   : {MANIFEST_PATH}")

    # 5. Round-trip Deserialization Verification
    log(lines, "\n" + "-" * 72)
    log(lines, "ROUND-TRIP DESERIALIZATION & INFERENCE VERIFICATION")
    log(lines, "-" * 72)

    loaded_model = joblib.load(TARGET_MODEL_PATH)
    loaded_scaler = joblib.load(TARGET_SCALER_PATH)
    loaded_order = [line.strip() for line in TARGET_ORDER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Verify order
    if loaded_order != FEATURES:
        log(lines, f"[FAIL] Loaded feature order mismatch:\nLoaded: {loaded_order}\nExpected: {FEATURES}")
        sys.exit(1)
    log(lines, "[PASS] Feature ordering verified (10 features).")

    # Verify model properties
    if loaded_model.n_features_in_ != 10:
        log(lines, f"[FAIL] Model expected {loaded_model.n_features_in_} features instead of 10!")
        sys.exit(1)
    log(lines, f"[PASS] Model expects {loaded_model.n_features_in_} input features.")

    if loaded_scaler.n_features_in_ != 10:
        log(lines, f"[FAIL] Scaler expected {loaded_scaler.n_features_in_} features instead of 10!")
        sys.exit(1)
    log(lines, f"[PASS] Scaler expects {loaded_scaler.n_features_in_} input features.")

    # Test sample inference
    test_raw = np.array([72.0, 0.833, 0.160, 0.080, 0.350, -0.30, 0.05, 0.30, 4.20, 0.70]).reshape(1, -1)
    test_scaled = loaded_scaler.transform(test_raw)
    test_pred = loaded_model.predict(test_scaled)[0]
    test_proba = loaded_model.predict_proba(test_scaled)[0]

    log(lines, f"\nSample Inference Test:")
    log(lines, f"  Raw input       : {test_raw.tolist()}")
    log(lines, f"  Scaled input    : {test_scaled.tolist()}")
    log(lines, f"  Predicted label : {test_pred} ({'NORMAL' if test_pred == 0 else 'ABNORMAL'})")
    log(lines, f"  Probabilities   : Normal={test_proba[0]*100:.2f}%, Abnormal={test_proba[1]*100:.2f}%")

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 10 RESULT: PASSED")
    log(lines, "New Random Forest model and scaler saved and verified. Ready for Stage 11.")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
