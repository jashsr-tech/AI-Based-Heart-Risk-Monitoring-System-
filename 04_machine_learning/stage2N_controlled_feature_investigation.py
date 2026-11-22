
from pathlib import Path
from datetime import datetime
import sys

import joblib
import numpy as np
import pandas as pd

# ============================================================
# STAGE 2N — CONTROLLED FEATURE INVESTIGATION
#
# Changes one feature at a time and observes model output.
# Does not retrain or overwrite the existing model/scaler.
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


def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def main():
    print("=" * 72)
    print("STAGE 2N — CONTROLLED FEATURE INVESTIGATION")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Check required files
    # --------------------------------------------------------
    print("\n[1] Checking files...")

    required_files = [
        MODEL_FILE,
        SCALER_FILE,
        ORDER_FILE,
        DATASET_FILE,
        VECTOR_FILE,
    ]

    for path in required_files:
        if not path.is_file():
            stop(f"Required file not found:\n{path}")
        print(f"FOUND: {path}")

    # --------------------------------------------------------
    # 2. Validate feature order
    # --------------------------------------------------------
    saved_features = [
        line.strip()
        for line in ORDER_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if saved_features != FEATURES:
        stop(
            "Feature order does not match. "
            "No investigation was performed."
        )

    print("\n[2] Feature order verified.")

    # --------------------------------------------------------
    # 3. Load model, scaler, dataset, and baseline vector
    # --------------------------------------------------------
    print("\n[3] Loading existing files...")

    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        df = pd.read_csv(DATASET_FILE)
        baseline = np.asarray(
            np.load(VECTOR_FILE, allow_pickle=False),
            dtype=float,
        ).reshape(-1)
    except Exception as exc:
        stop(f"Could not load files: {type(exc).__name__}: {exc}")

    if baseline.size != len(FEATURES):
        stop(
            f"Expected {len(FEATURES)} baseline values, "
            f"found {baseline.size}."
        )

    if not np.all(np.isfinite(baseline)):
        stop("Baseline contains non-finite values.")

    missing = [
        name for name in FEATURES + ["Label"]
        if name not in df.columns
    ]

    if missing:
        stop(f"Dataset is missing columns: {missing}")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    df["Label"] = pd.to_numeric(
        df["Label"], errors="coerce"
    )

    df = df.dropna(subset=FEATURES + ["Label"])

    print(f"Dataset rows used: {len(df)}")
    print(f"Baseline features: {len(baseline)}")

    # --------------------------------------------------------
    # 4. Prediction helper
    # --------------------------------------------------------
    def predict_vector(values):
        # Keep feature names when passing data to the scaler.
        frame = pd.DataFrame([values], columns=FEATURES)

        scaled = scaler.transform(frame)

        # The saved tree was fitted without feature names;
        # pass an ndarray to avoid the feature-name warning.
        prediction = int(model.predict(scaled)[0])

        probabilities = model.predict_proba(scaled)[0]
        class_probabilities = {
            int(label): float(prob)
            for label, prob in zip(
                model.classes_, probabilities
            )
        }

        return prediction, class_probabilities

    # --------------------------------------------------------
    # 5. Baseline prediction
    # --------------------------------------------------------
    print("\n[4] Baseline prediction...")

    baseline_label, baseline_probs = predict_vector(baseline)

    print(
        f"Baseline result: "
        f"{LABEL_NAMES.get(baseline_label, baseline_label)}"
    )

    for label, probability in baseline_probs.items():
        print(
            f"Label {label} "
            f"({LABEL_NAMES.get(label, 'UNKNOWN')}): "
            f"{probability * 100:.2f}%"
        )

    rows = []

    def record_result(
        scenario,
        changed_feature,
        test_value,
        values,
        label,
        probabilities,
    ):
        rows.append({
            "Scenario": scenario,
            "ChangedFeature": changed_feature,
            "TestValue": test_value,
            "PredictedLabel": label,
            "PredictedClass": LABEL_NAMES.get(
                label, "UNKNOWN"
            ),
            "NormalProbabilityPercent": (
                probabilities.get(0, 0.0) * 100
            ),
            "AbnormalProbabilityPercent": (
                probabilities.get(1, 0.0) * 100
            ),
            "ChangedFromBaseline": (
                label != baseline_label
            ),
        })

    record_result(
        "Original synthetic baseline",
        "NONE",
        np.nan,
        baseline,
        baseline_label,
        baseline_probs,
    )

    # --------------------------------------------------------
    # 6. Change one feature at a time
    # --------------------------------------------------------
    print("\n[5] Running one-feature-at-a-time tests...")
    print(
        "Each test changes one feature only. "
        "All other features remain at baseline values."
    )

    for feature_index, feature in enumerate(FEATURES):
        feature_data = df[feature].dropna()

        if feature_data.empty:
            print(f"Skipping {feature}: no numeric data.")
            continue

        # Five dataset-wide reference percentiles.
        quantiles = {
            "dataset_p05": float(feature_data.quantile(0.05)),
            "dataset_p25": float(feature_data.quantile(0.25)),
            "dataset_p50": float(feature_data.quantile(0.50)),
            "dataset_p75": float(feature_data.quantile(0.75)),
            "dataset_p95": float(feature_data.quantile(0.95)),
        }

        # Add the median for each class where available.
        for label in (0, 1):
            class_values = df.loc[
                df["Label"] == label, feature
            ].dropna()

            if not class_values.empty:
                class_name = LABEL_NAMES[label].lower()
                quantiles[f"{class_name}_median"] = float(
                    class_values.median()
                )

        print(f"\nTesting feature: {feature}")

        for scenario, test_value in quantiles.items():
            changed = baseline.copy()
            changed[feature_index] = test_value

            try:
                label, probabilities = predict_vector(changed)
            except Exception as exc:
                stop(
                    f"Prediction failed for {feature} "
                    f"at {test_value}: {type(exc).__name__}: {exc}"
                )

            record_result(
                scenario,
                feature,
                test_value,
                changed,
                label,
                probabilities,
            )

            changed_text = (
                "YES" if label != baseline_label else "NO"
            )

            print(
                f"  {scenario:18s} "
                f"value={test_value: .6f} | "
                f"prediction={LABEL_NAMES.get(label, label):8s} | "
                f"abnormal={probabilities.get(1, 0.0) * 100:6.2f}% | "
                f"class changed={changed_text}"
            )

    # --------------------------------------------------------
    # 7. Summarize results
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("STAGE 2N — SUMMARY")
    print("=" * 72)

    results = pd.DataFrame(rows)

    tests = results[
        results["ChangedFeature"] != "NONE"
    ].copy()

    if tests.empty:
        stop("No feature tests were completed.")

    summary = (
        tests.groupby("ChangedFeature")
        .agg(
            Tests=("Scenario", "count"),
            ClassChanges=("ChangedFromBaseline", "sum"),
            MinimumAbnormalProbability=(
                "AbnormalProbabilityPercent", "min"
            ),
            MaximumAbnormalProbability=(
                "AbnormalProbabilityPercent", "max"
            ),
        )
        .sort_values(
            by=["ClassChanges", "MaximumAbnormalProbability"],
            ascending=[False, False],
        )
    )

    print("\nFeatures ranked by how often their changes switch the class:")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

    total_tests = len(tests)
    total_switches = int(tests["ChangedFromBaseline"].sum())

    print(f"\nOne-feature tests completed: {total_tests}")
    print(f"Tests that changed the predicted class: {total_switches}")

    # --------------------------------------------------------
    # 8. Save new reports; never overwrite previous reports
    # --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_csv = (
        PROJECT_DIR
        / "04_machine_learning"
        / f"stage2N_feature_test_results_{timestamp}.csv"
    )

    output_txt = (
        PROJECT_DIR
        / "04_machine_learning"
        / f"stage2N_feature_test_summary_{timestamp}.txt"
    )

    results.to_csv(output_csv, index=False)

    report_lines = [
        "STAGE 2N — CONTROLLED FEATURE INVESTIGATION",
        "=" * 60,
        f"Baseline class: {LABEL_NAMES.get(baseline_label, baseline_label)}",
        f"Baseline normal probability: "
        f"{baseline_probs.get(0, 0.0) * 100:.2f}%",
        f"Baseline abnormal probability: "
        f"{baseline_probs.get(1, 0.0) * 100:.2f}%",
        f"Tests completed: {total_tests}",
        f"Class-switching tests: {total_switches}",
        "",
        "Feature summary:",
        summary.to_string(float_format=lambda x: f"{x:.2f}"),
        "",
        "Interpretation limits:",
        "- These are model sensitivity tests, not medical tests.",
        "- Only one input feature is changed per scenario.",
        "- Some feature combinations may be physiologically inconsistent.",
        "- Model probabilities are not clinically calibrated risk estimates.",
        "- Dataset percentiles are not clinical reference ranges.",
        "- No model, scaler, dataset, or baseline vector was modified.",
    ]

    output_txt.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\nNew files saved:")
    print(output_csv)
    print(output_txt)

    print("\nStage 2N completed.")
    print("Existing model and source data were not overwritten.")


if __name__ == "__main__":
    main()
