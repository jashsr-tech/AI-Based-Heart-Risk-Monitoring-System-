
from pathlib import Path
from datetime import datetime
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# ============================================================
# STAGE 2M — MODEL DECISION DIAGNOSTIC
# Read-only analysis of the existing model and dataset.
# Does not retrain or overwrite the saved model/scaler.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(
    r"C:\Users\srjas\OneDrive\Desktop\proj code"
)

MODEL_FILE = MODEL_DIR / "ecg_decision_tree_model.pkl"
SCALER_FILE = MODEL_DIR / "ecg_feature_scaler.pkl"
ORDER_FILE = MODEL_DIR / "model_feature_order.txt"

DATASET_FILE = (
    PROJECT_DIR
    / "04_machine_learning"
    / "ecg_features_balanced_clean_labeled.csv"
)

VECTOR_FILE = (
    PROJECT_DIR
    / "03_feature_extraction"
    / "ecg_ml_features.npy"
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

LABEL_NAMES = {
    0: "NORMAL",
    1: "ABNORMAL",
}


def section(title, report):
    line = "\n" + "=" * 72
    message = f"{line}\n{title}\n{'=' * 72}"
    print(message)
    report.append(message)


def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def main():
    report = []

    def log(message=""):
        print(message)
        report.append(str(message))

    section("STAGE 2M — MODEL DECISION DIAGNOSTIC", report)

    log("Purpose: explain the saved model's prediction.")
    log("No training, model saving, or dataset modification is performed.")
    log(f"Project directory: {PROJECT_DIR}")
    log(f"Model directory:   {MODEL_DIR}")

    # --------------------------------------------------------
    # 1. Verify all files
    # --------------------------------------------------------
    section("1. FILE CHECK", report)

    required_files = {
        "Saved model": MODEL_FILE,
        "Saved scaler": SCALER_FILE,
        "Saved feature order": ORDER_FILE,
        "Training dataset": DATASET_FILE,
        "Stage 2E feature vector": VECTOR_FILE,
    }

    for name, path in required_files.items():
        if not path.is_file():
            stop(f"{name} not found:\n{path}")
        log(f"FOUND: {name}: {path}")

    # --------------------------------------------------------
    # 2. Validate feature order
    # --------------------------------------------------------
    section("2. FEATURE ORDER CHECK", report)

    saved_features = [
        line.strip()
        for line in ORDER_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    log(f"Expected number of features: {len(FEATURES)}")
    log(f"Saved number of features:    {len(saved_features)}")

    if saved_features != FEATURES:
        stop(
            "Feature order mismatch. Stop here; do not interpret "
            "the model's prediction until the order is resolved."
        )

    for i, name in enumerate(saved_features, start=1):
        log(f"{i:2}. {name}")

    log("PASS: Saved feature order matches.")

    # --------------------------------------------------------
    # 3. Load the synthetic feature vector
    # --------------------------------------------------------
    section("3. SYNTHETIC ECG FEATURE VECTOR", report)

    vector = np.load(VECTOR_FILE, allow_pickle=False)
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)

    if vector.size != len(FEATURES):
        stop(
            f"Expected {len(FEATURES)} values; found {vector.size}."
        )

    if not np.all(np.isfinite(vector)):
        stop("Feature vector contains NaN or infinite values.")

    # Use named columns to avoid the scaler's feature-name warning.
    X_raw = pd.DataFrame([vector], columns=FEATURES)

    for name, value in zip(FEATURES, vector):
        log(f"{name:15s}: {value:.8f}")

    # --------------------------------------------------------
    # 4. Load the existing model and scaler
    # --------------------------------------------------------
    section("4. LOAD EXISTING MODEL AND SCALER", report)

    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception as exc:
        stop(
            f"Loading failed: {type(exc).__name__}: {exc}\n"
            "Do not delete or overwrite the saved files."
        )

    if not hasattr(model, "tree_"):
        stop(
            "The loaded model does not expose tree_. "
            "This diagnostic expects a fitted decision tree."
        )

    log(f"Model type:  {type(model).__name__}")
    log(f"Scaler type: {type(scaler).__name__}")
    log(f"Model classes: {list(model.classes_)}")

    scaler_count = getattr(scaler, "n_features_in_", None)
    model_count = getattr(model, "n_features_in_", None)

    log(f"Scaler input feature count: {scaler_count}")
    log(f"Model input feature count:  {model_count}")

    if scaler_count != len(FEATURES):
        stop("Scaler does not expect exactly 10 features.")

    if model_count != len(FEATURES):
        stop("Model does not expect exactly 10 features.")

    # --------------------------------------------------------
    # 5. Transform the synthetic vector
    # --------------------------------------------------------
    section("5. SCALED FEATURE VALUES", report)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names"
            )
            X_scaled_array = scaler.transform(X_raw)

    except Exception as exc:
        stop(
            f"Scaling failed: {type(exc).__name__}: {exc}"
        )

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=FEATURES
    )

    for name in FEATURES:
        log(
            f"{name:15s}: raw={X_raw.iloc[0][name]: .6f}  "
            f"scaled={X_scaled.iloc[0][name]: .6f}"
        )

    # --------------------------------------------------------
    # 6. Reproduce the saved model prediction
    # --------------------------------------------------------
    section("6. REPRODUCE MODEL PREDICTION", report)

    try:
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
    except Exception as exc:
        stop(
            f"Prediction failed: {type(exc).__name__}: {exc}"
        )

    log(f"Predicted label: {prediction}")
    log(f"Predicted class: {LABEL_NAMES.get(prediction, 'UNKNOWN')}")

    for class_label, probability in zip(
        model.classes_, probabilities
    ):
        log(
            f"Class {class_label} "
            f"({LABEL_NAMES.get(int(class_label), 'UNKNOWN')}): "
            f"{probability * 100:.2f}%"
        )

    predicted_probability = float(np.max(probabilities))
    log(
        f"Highest model probability: "
        f"{predicted_probability * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 7. Feature importance
    # --------------------------------------------------------
    section("7. DECISION TREE FEATURE IMPORTANCE", report)

    importances = getattr(model, "feature_importances_", None)

    if importances is None or len(importances) != len(FEATURES):
        log("Feature importance is unavailable.")
    else:
        importance_order = np.argsort(importances)[::-1]

        log("Features ranked from highest to lowest importance:")
        for rank, index in enumerate(importance_order, start=1):
            log(
                f"{rank:2}. {FEATURES[index]:15s} "
                f"importance={importances[index]:.6f} "
                f"({importances[index] * 100:.2f}%)"
            )

        log(
            "Note: importance measures contribution to tree splits "
            "in training; it does not prove a feature causes a condition."
        )

    # --------------------------------------------------------
    # 8. Trace the exact decision path
    # --------------------------------------------------------
    section("8. EXACT DECISION TREE PATH", report)

    tree = model.tree_
    node_indicator = model.decision_path(X_scaled_array)
    leaf_id = int(model.apply(X_scaled_array)[0])

    visited_nodes = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    log("Each rule below is evaluated using SCALED feature values.")
    log("Raw-equivalent thresholds are approximate and assume StandardScaler.")

    scaler_mean = getattr(scaler, "mean_", None)
    scaler_scale = getattr(scaler, "scale_", None)

    for step, node_id in enumerate(visited_nodes, start=1):
        feature_index = int(tree.feature[node_id])

        if feature_index < 0:
            log(f"\nStep {step}: reached leaf node {node_id}.")
            continue

        feature_name = FEATURES[feature_index]
        threshold = float(tree.threshold[node_id])
        scaled_value = float(X_scaled_array[0, feature_index])

        if scaled_value <= threshold:
            decision = "<="
            branch = "LEFT"
        else:
            decision = ">"
            branch = "RIGHT"

        log(
            f"\nStep {step}: node {node_id}\n"
            f"  Rule: {feature_name} {decision} {threshold:.8f}\n"
            f"  Input scaled value: {scaled_value:.8f}\n"
            f"  Result: go {branch}"
        )

        if (
            scaler_mean is not None
            and scaler_scale is not None
            and scaler_scale[feature_index] != 0
        ):
            raw_threshold = (
                threshold * scaler_scale[feature_index]
                + scaler_mean[feature_index]
            )
            log(
                f"  Approximate raw threshold: "
                f"{raw_threshold:.8f}"
            )

    leaf_values = tree.value[leaf_id][0]

    log(f"\nFinal leaf node: {leaf_id}")
    log(f"Leaf class values: {leaf_values.tolist()}")

    if hasattr(tree, "n_node_samples"):
        log(
            f"Training samples reaching leaf: "
            f"{int(tree.n_node_samples[leaf_id])}"
        )

    # --------------------------------------------------------
    # 9. Load and inspect the training dataset
    # --------------------------------------------------------
    section("9. TRAINING DATASET CHECK", report)

    try:
        df = pd.read_csv(DATASET_FILE)
    except Exception as exc:
        stop(
            f"Dataset read failed: {type(exc).__name__}: {exc}"
        )

    required_columns = FEATURES + ["Label", "Record"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        stop(f"Dataset missing columns: {missing_columns}")

    log(f"Dataset rows: {len(df)}")
    log(f"Dataset columns: {len(df.columns)}")
    log(f"Unique records: {df['Record'].nunique()}")
    log(f"Missing feature values: {int(df[FEATURES].isna().sum().sum())}")

    log("\nLabel counts:")
    for label, count in df["Label"].value_counts().sort_index().items():
        log(
            f"Label {label} "
            f"({LABEL_NAMES.get(int(label), 'UNKNOWN')}): {count}"
        )

    log("\nLabel counts by record:")
    record_label_counts = (
        df.groupby("Record")["Label"].nunique()
    )
    mixed_records = int((record_label_counts > 1).sum())

    log(f"Records containing both labels: {mixed_records}")
    log(
        "If this value is zero, each record contains only one class label; "
        "record-level differences may influence predictions."
    )

    # --------------------------------------------------------
    # 10. Compare synthetic features with each class
    # --------------------------------------------------------
    section("10. SYNTHETIC FEATURES VS DATASET DISTRIBUTIONS", report)

    log(
        "Values below describe dataset rows, not independent patients. "
        "They do not establish clinical normal ranges."
    )

    for feature in FEATURES:
        log(f"\n--- {feature} ---")
        synthetic_value = float(X_raw.iloc[0][feature])
        log(f"Synthetic value: {synthetic_value:.8f}")

        for label in sorted(df["Label"].dropna().unique()):
            class_data = pd.to_numeric(
                df.loc[df["Label"] == label, feature],
                errors="coerce"
            ).dropna()

            if class_data.empty:
                log(f"Label {label}: no numeric values.")
                continue

            q01 = float(class_data.quantile(0.01))
            median = float(class_data.median())
            q99 = float(class_data.quantile(0.99))

            in_range = q01 <= synthetic_value <= q99

            log(
                f"Label {label} "
                f"({LABEL_NAMES.get(int(label), 'UNKNOWN')}): "
                f"median={median:.6f}, "
                f"1st-99th percentile=[{q01:.6f}, {q99:.6f}], "
                f"synthetic within range={in_range}"
            )

    # --------------------------------------------------------
    # 11. Check dataset label/record association
    # --------------------------------------------------------
    section("11. RECORD-LEVEL LABEL DIAGNOSTIC", report)

    record_summary = (
        df.groupby("Record")
        .agg(
            rows=("Label", "size"),
            labels=("Label", "nunique"),
            label=("Label", "first"),
        )
        .reset_index()
    )

    for label in sorted(df["Label"].dropna().unique()):
        records = record_summary.loc[
            record_summary["label"] == label, "Record"
        ].tolist()

        log(
            f"Label {label} "
            f"({LABEL_NAMES.get(int(label), 'UNKNOWN')}): "
            f"{len(records)} records -> {records}"
        )

    # --------------------------------------------------------
    # 12. Final interpretation and report
    # --------------------------------------------------------
    section("12. INTERPRETATION", report)

    log(f"Saved model prediction: {LABEL_NAMES.get(prediction, prediction)}")
    log(
        f"Highest model probability: "
        f"{predicted_probability * 100:.2f}%"
    )
    log(
        "The decision path identifies the exact split rules that led "
        "to this prediction."
    )
    log(
        "Feature importance is a global training statistic; the decision "
        "path explains this individual prediction."
    )
    log(
        "A feature lying inside a dataset percentile range does not prove "
        "that its definition or units match the training data."
    )
    log(
        "Because records may be associated with one label each, do not "
        "treat row-level balance or a single test split as proof of "
        "generalization to new patients."
    )
    log(
        "This output is not a medical diagnosis and must not be used "
        "to guide patient treatment."
    )

    # Timestamped output prevents overwriting an older report.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = (
        PROJECT_DIR
        / "04_machine_learning"
        / f"stage2M_diagnostic_report_{timestamp}.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("STAGE 2M COMPLETED")
    print("=" * 72)
    print(f"Diagnostic report saved to:\n{report_path}")
    print("\nNo existing model, scaler, dataset, or feature vector was changed.")


if __name__ == "__main__":
    main()
