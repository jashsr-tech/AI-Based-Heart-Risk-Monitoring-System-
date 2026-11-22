
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# STAGE 2I - RECORD-SEPARATED MODEL EVALUATION
# This script does NOT overwrite your saved model.
# ============================================================

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

TARGET = "Label"
GROUP = "Record"

print("=" * 65)
print("STAGE 2I - RECORD-SEPARATED EVALUATION")
print("=" * 65)

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

df = pd.read_csv(DATASET)

required = FEATURES + [TARGET, GROUP]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required].copy()

for column in FEATURES + [TARGET]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Expected binary labels 0 and 1.")

if df[TARGET].nunique() != 2:
    raise ValueError("Dataset must contain both classes.")

print(f"\nRows used: {len(df)}")
print(f"Unique records: {df[GROUP].nunique()}")
print("\nLabel counts:")
print(df[TARGET].value_counts().sort_index())

# Split entire records, not individual heartbeat rows.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df[FEATURES], df[TARGET], groups=df[GROUP])
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

train_records = set(train[GROUP])
test_records = set(test[GROUP])

print(f"\nTraining rows: {len(train)}")
print(f"Testing rows: {len(test)}")
print(f"Training records: {len(train_records)}")
print(f"Testing records: {len(test_records)}")

if train_records.intersection(test_records):
    raise RuntimeError("Record leakage detected!")

print("\nTraining label counts:")
print(train[TARGET].value_counts().sort_index())

print("\nTesting label counts:")
print(test[TARGET].value_counts().sort_index())

if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
    print(
        "\nSTOP: One split does not contain both classes. "
        "Do not interpret metrics. A better group split is needed."
    )
    raise SystemExit(1)

# Fit scaler only on training data.
scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
)
model.fit(X_train, train[TARGET])

predictions = model.predict(X_test)

print("\n" + "=" * 65)
print("EVALUATION RESULTS")
print("=" * 65)

print(f"Accuracy: {accuracy_score(test[TARGET], predictions):.4f}")
print(
    "Balanced accuracy: "
    f"{balanced_accuracy_score(test[TARGET], predictions):.4f}"
)

print("\nClassification report:")
print(
    classification_report(
        test[TARGET],
        predictions,
        labels=[0, 1],
        target_names=["Normal (0)", "Abnormal (1)"],
        zero_division=0,
    )
)

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(test[TARGET], predictions, labels=[0, 1]))

print("\nTraining and testing record IDs:")
print("Train:", sorted(train_records))
print("Test :", sorted(test_records))

print("\nNo saved model or scaler was overwritten.")
print("STAGE 2I COMPLETE")
