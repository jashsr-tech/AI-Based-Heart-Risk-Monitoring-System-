"""
ECG FINAL PRO — PHASE 1: STAGE 8
Patient-Grouped K-Fold Cross-Validation & Out-of-Fold Ensemble Evaluation
File: 04_machine_learning/stage8_patient_group_cv.py

Rigorous Invariants & Methodological Protocols:
  1. Strict Data Isolation:
     - Loads EXCLUSIVELY from 04_machine_learning/mitbih_development_dataset.csv.
     - Filters strictly to Partition == 'Train_Dev' (66,679 beats, 30 subjects, 31 records).
     - Validation partition (15,349 beats, 7 subjects) is held separate for out-of-sample comparison.
     - NEVER accesses or opens 04_machine_learning/mitbih_independent_test_dataset.csv.
  2. Patient Grouping Invariant:
     - Uses GroupKFold(n_splits=5) grouped strictly by Subject_ID.
     - All beats from a patient remain in the exact same fold.
     - Subject_201_202 (Records 201 & 202) is verified to reside wholly in Fold 1.
  3. Automated Pre-Training & Fold-Level Integrity Assertions:
     - Beat_ID uniqueness verified (Record + Sample_Idx).
     - GroupKFold subject assignment verified against documented fold census.
     - Zero subject overlap between training and held-out folds.
     - Bipolar class presence ({0, 1}) in both training and test subsets for every fold.
     - Zero duplicate beats across held-out folds.
     - Exactly 66,679 held-out predictions generated per model configuration.
  4. Zero-Leakage Fold-Level Preprocessing:
     - Fits a fresh SimpleImputer(strategy='median') inside each fold strictly on that fold's training data.
     - Fits a fresh StandardScaler(ddof=0) inside each fold strictly on the imputed fold training data.
     - Transforms training and held-out subsets without fold-level leakage.
     - Binary missingness indicators remain unscaled {0, 1}.
  5. Frozen Eight-Model Suite (Stage 7 Hyperparameters):
     - RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=12,
                              max_features='sqrt', min_samples_split=10, min_samples_leaf=4,
                              bootstrap=True, random_state=42, n_jobs=-1)
     - Primary Balanced (class_weight='balanced') and Comparison Unweighted (class_weight=None).
     - Four feature spaces: Exp A (2 feat), Exp B (7 feat), Exp C (9 feat), Exp D (15 feat).
  6. Dual Reporting Paradigm:
     - Paradigm A: Fold-Wise Mean ± Std (ddof=1) and Coefficient of Variation (CV = std/mean).
     - Paradigm B: Pooled Out-of-Fold (OOF) Global Evaluation (N=66,679 beats).
     - Metric Terminology: Average Precision (AP) strictly evaluated via average_precision_score.
  7. Dedicated Exp C vs Exp D Missingness Indicator Investigation:
     - Fold-by-fold delta analysis and pooled delta analysis.
  8. Comparison with Stage 7 Validation:
     - Clear disclosure that Stage 8 trains on ~24 subjects per fold vs 30 subjects in Stage 7.
"""

from pathlib import Path
import datetime
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "04_machine_learning"
RESULTS_DIR = PROJECT_ROOT / "05_results"

DEV_DATASET_PATH = ML_DIR / "mitbih_development_dataset.csv"
STAGE7_METRICS_PATH = RESULTS_DIR / "stage7_random_forest_metrics.csv"

# Target Stage 8 Outputs
FOLD_METRICS_CSV_PATH = RESULTS_DIR / "stage8_group_cv_fold_metrics.csv"
SUMMARY_METRICS_CSV_PATH = RESULTS_DIR / "stage8_group_cv_summary.csv"
POOLED_METRICS_CSV_PATH = RESULTS_DIR / "stage8_group_cv_pooled_metrics.csv"
OOF_PREDICTIONS_CSV_PATH = RESULTS_DIR / "stage8_group_cv_oof_predictions.csv"
EXP_CD_COMPARISON_CSV_PATH = RESULTS_DIR / "stage8_group_cv_exp_c_vs_d_comparison.csv"
STAGE7_COMPARISON_CSV_PATH = RESULTS_DIR / "stage8_group_cv_vs_stage7_comparison.csv"
REPORT_TXT_PATH = RESULTS_DIR / "stage8_group_cv_report.txt"

BOXPLOT_PNG_PATH = RESULTS_DIR / "stage8_group_cv_boxplots.png"
COMPARISON_PNG_PATH = RESULTS_DIR / "stage8_group_cv_comparison.png"

# Feature definitions
NUMERICAL_FEATURES = [
    "RRInterval",
    "HRV",
    "QRSDuration",
    "QTInterval",
    "STSegment",
    "PRInterval",
    "PAmplitude",
    "RAmplitude",
    "TAmplitude"
]

INDICATOR_FEATURES = [
    "P_Valid",
    "PR_Valid",
    "QRS_Valid",
    "ST_Valid",
    "QT_Valid",
    "Baseline_Valid"
]

TIMING_FEATURES = ["RRInterval", "HRV"]
MORPHOLOGY_FEATURES = [
    "QRSDuration",
    "QTInterval",
    "STSegment",
    "PRInterval",
    "PAmplitude",
    "RAmplitude",
    "TAmplitude"
]

EXPECTED_FOLDS = {
    1: {"Subject_105", "Subject_106", "Subject_118", "Subject_201_202", "Subject_232"},
    2: {"Subject_101", "Subject_111", "Subject_116", "Subject_200", "Subject_213", "Subject_231"},
    3: {"Subject_109", "Subject_113", "Subject_124", "Subject_209", "Subject_223", "Subject_230"},
    4: {"Subject_102", "Subject_107", "Subject_115", "Subject_122", "Subject_203", "Subject_210", "Subject_214"},
    5: {"Subject_100", "Subject_112", "Subject_117", "Subject_119", "Subject_212", "Subject_234"}
}

EXPERIMENT_SPECS = {
    "Exp_A_Timing_Only": {
        "name": "Exp A: Timing-Only",
        "num_feats": TIMING_FEATURES,
        "indicators": False,
        "feat_count": 2
    },
    "Exp_B_Morphology_Only": {
        "name": "Exp B: Morphology-Only",
        "num_feats": MORPHOLOGY_FEATURES,
        "indicators": False,
        "feat_count": 7
    },
    "Exp_C_Combined_No_Indicators": {
        "name": "Exp C: Combined No Indicators",
        "num_feats": NUMERICAL_FEATURES,
        "indicators": False,
        "feat_count": 9
    },
    "Exp_D_Combined_With_Indicators": {
        "name": "Exp D: Combined With Indicators",
        "num_feats": NUMERICAL_FEATURES,
        "indicators": True,
        "feat_count": 15
    }
}

WEIGHTING_SPECS = {
    "Balanced": {"class_weight": "balanced"},
    "Unweighted": {"class_weight": None}
}

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Accuracy": float(acc),
        "Balanced_Accuracy": float(bal_acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "Precision": float(prec),
        "F1_Score": float(f1),
        "ROC_AUC": float(roc_auc),
        "Average_Precision": float(ap)
    }

def main():
    lines = []
    log(lines, "=" * 85)
    log(lines, "ECG FINAL PRO — STAGE 8: PATIENT-GROUPED K-FOLD CROSS-VALIDATION")
    log(lines, "5-Fold GroupKFold | Zero Leakage Preprocessing | Dual Reporting | OOF Pooling")
    log(lines, f"Execution Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(lines, "=" * 85)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Dataset Loading & Strict Isolation
    # ---------------------------------------------------------
    log(lines, "\n1. DATASET LOADING & STRICT ISOLATION AUDIT:")
    log(lines, f"   Source Path: {DEV_DATASET_PATH}")
    assert DEV_DATASET_PATH.exists(), f"Missing development dataset: {DEV_DATASET_PATH}"
    df_raw = pd.read_csv(DEV_DATASET_PATH)
    log(lines, f"   Total rows loaded from development file: {len(df_raw):,} beats.")

    partitions_present = set(df_raw["Partition"].unique())
    log(lines, f"   Partitions present in development file: {sorted(list(partitions_present))}")
    assert partitions_present == {"Train_Dev", "Validation"}, f"Unexpected partitions: {partitions_present}"
    assert "Independent_Test" not in partitions_present, "CRITICAL ERROR: Independent_Test found in development file!"
    log(lines, "   [ISOLATION CERTIFIED] Independent_Test is physically absent from development dataset.")

    # Filter strictly to Train_Dev
    df_train = df_raw[df_raw["Partition"] == "Train_Dev"].copy().reset_index(drop=True)
    log(lines, f"   Train_Dev cohort filtered: {len(df_train):,} beats, {df_train['Subject_ID'].nunique()} subjects.")
    assert len(df_train) == 66679, f"Expected 66,679 beats, got {len(df_train)}"
    assert df_train["Subject_ID"].nunique() == 30, f"Expected 30 subjects, got {df_train['Subject_ID'].nunique()}"

    # Verify Validation remains untouched during CV
    df_val_held = df_raw[df_raw["Partition"] == "Validation"].copy().reset_index(drop=True)
    log(lines, f"   Validation partition held completely separate: {len(df_val_held):,} beats, {df_val_held['Subject_ID'].nunique()} subjects.")
    del df_raw, df_val_held

    # ---------------------------------------------------------
    # 2. Automated Pre-Training Integrity Verification Suite
    # ---------------------------------------------------------
    log(lines, "\n2. AUTOMATED PRE-TRAINING INTEGRITY VERIFICATION SUITE:")

    # Check 1: Unique Beat Identifier
    df_train["Beat_ID"] = df_train["Record"].astype(str) + "_" + df_train["Sample_Idx"].astype(str)
    assert df_train["Beat_ID"].nunique() == len(df_train) == 66679, "CRITICAL ERROR: Beat_ID is not globally unique in Train_Dev!"
    log(lines, "   [CHECK 1 PASSED] Unique beat identifier (Record + Sample_Idx) verified: 66,679 globally unique IDs.")

    # Check 2: Target classes
    y_all = df_train["Label"].values.astype(int)
    c0_count = int((y_all == 0).sum())
    c1_count = int((y_all == 1).sum())
    log(lines, f"   [CHECK 2 PASSED] Target labels verified: Class 0 (AAMI N) = {c0_count:,d} ({c0_count/len(y_all)*100:.2f}%), "
               f"Class 1 (S, V, F) = {c1_count:,d} ({c1_count/len(y_all)*100:.2f}%).")
    assert c0_count == 60034 and c1_count == 6645, "Class count mismatch on Train_Dev!"

    # Setup GroupKFold
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df_train, groups=df_train["Subject_ID"]))
    assert len(splits) == 5, "Expected 5 splits from GroupKFold"

    # Verify Subject assignments and Subject_201_202 lock
    log(lines, "\n   Verifying GroupKFold 5-Fold Partition Assignments:")
    seen_val_beat_ids = set()
    fold_assignments = np.zeros(len(df_train), dtype=int)

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_train_sub = set(df_train.iloc[train_idx]["Subject_ID"].unique())
        fold_val_sub = set(df_train.iloc[val_idx]["Subject_ID"].unique())
        fold_val_beats = set(df_train.iloc[val_idx]["Beat_ID"].values)

        # Subject overlap check
        assert len(fold_train_sub.intersection(fold_val_sub)) == 0, f"Subject overlap detected in Fold {fold_num}!"

        # Documented census match check
        assert fold_val_sub == EXPECTED_FOLDS[fold_num], (
            f"Fold {fold_num} subject mismatch! Expected {EXPECTED_FOLDS[fold_num]}, got {fold_val_sub}"
        )

        # Subject_201_202 lock check
        if "Subject_201_202" in fold_val_sub:
            records_in_fold = df_train.iloc[val_idx]["Record"].unique()
            assert 201 in records_in_fold and 202 in records_in_fold, (
                f"Subject_201_202 records split in Fold {fold_num}!"
            )
            log(lines, f"     * Fold {fold_num}: Confirmed Subject_201_202 intact with Records 201 and 202 co-assigned.")

        # Class presence check
        y_tr = y_all[train_idx]
        y_va = y_all[val_idx]
        assert set(np.unique(y_tr)) == {0, 1}, f"Fold {fold_num} training subset missing one class!"
        assert set(np.unique(y_va)) == {0, 1}, f"Fold {fold_num} held-out subset missing one class!"

        # Disjoint beat check across folds
        assert len(fold_val_beats.intersection(seen_val_beat_ids)) == 0, (
            f"Duplicate beat detected across held-out folds in Fold {fold_num}!"
        )
        seen_val_beat_ids.update(fold_val_beats)
        fold_assignments[val_idx] = fold_num

        log(lines, f"     * Fold {fold_num}: {len(fold_val_sub)} subjects, {len(val_idx):,} beats, "
                   f"Class 0 = {(y_va == 0).sum():,d}, Class 1 = {(y_va == 1).sum():,d} ({(y_va == 1).mean()*100:.2f}%).")

    # Complete held-out coverage check
    assert len(seen_val_beat_ids) == len(df_train) == 66679, (
        f"Coverage check failed! Saw {len(seen_val_beat_ids)} unique held-out beats, expected 66,679."
    )
    df_train["CV_Fold"] = fold_assignments
    log(lines, "   [CHECK 3 PASSED] All 5 folds disjoint, zero subject overlap, bipolar class presence verified.")
    log(lines, "   [CHECK 4 PASSED] Exactly 66,679 beats mapped to exactly one held-out fold.")
    log(lines, "   [CHECK 5 PASSED] Documented fold census verified with 100% fidelity.")

    # ---------------------------------------------------------
    # 3. Model Suite Setup & Out-of-Fold Arrays
    # ---------------------------------------------------------
    model_keys = []
    for exp_id in EXPERIMENT_SPECS.keys():
        for weight_id in WEIGHTING_SPECS.keys():
            model_keys.append(f"{exp_id}_{weight_id}")

    # Check if pre-computed OOF predictions and fold metrics exist and are complete
    can_resume = False
    if OOF_PREDICTIONS_CSV_PATH.exists() and FOLD_METRICS_CSV_PATH.exists():
        oof_df_check = pd.read_csv(OOF_PREDICTIONS_CSV_PATH)
        fold_metrics_check = pd.read_csv(FOLD_METRICS_CSV_PATH)
        if len(oof_df_check) == 66679 and oof_df_check.isna().sum().sum() == 0 and len(fold_metrics_check) == 40:
            log(lines, "\n   [RESUME CHECK] Complete out-of-fold predictions and fold metrics already verified on disk.")
            log(lines, "   [RESUME CHECK] Loading 40 completed model runs directly to finalize summary, comparisons, and plots.")
            oof_df = oof_df_check
            fold_metrics_records = fold_metrics_check.to_dict(orient="records")
            df_fold_metrics = fold_metrics_check
            can_resume = True

    if not can_resume:
        # Prepare DataFrame to collect out-of-fold predictions
        oof_df = pd.DataFrame({
            "Beat_ID": df_train["Beat_ID"].values,
            "Record": df_train["Record"].values,
            "Sample_Idx": df_train["Sample_Idx"].values,
            "Subject_ID": df_train["Subject_ID"].values,
            "Fold": df_train["CV_Fold"].values,
            "True_Label": y_all
        })

        for mk in model_keys:
            oof_df[f"{mk}_Pred"] = np.nan
            oof_df[f"{mk}_Prob"] = np.nan

        fold_metrics_records = []

        # ---------------------------------------------------------
        # 4. Zero-Leakage Cross-Validation Loop
        # ---------------------------------------------------------
        log(lines, "\n" + "=" * 85)
        log(lines, "3. EXECUTING 5-FOLD ZERO-LEAKAGE CROSS-VALIDATION LOOP (40 MODEL RUNS):")
        log(lines, "=" * 85)

        for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
            log(lines, f"\n--- [FOLD {fold_num} / 5] Training on {len(train_idx):,} beats ({len(EXPECTED_FOLDS[fold_num])} held-out subjects, {len(val_idx):,} test beats) ---")

            # Extract subsets
            df_tr_fold = df_train.iloc[train_idx]
            df_va_fold = df_train.iloc[val_idx]

            y_train_fold = y_all[train_idx]
            y_val_fold = y_all[val_idx]

            # Step 4.1: Fold-Specific Median Imputation strictly on fold training data
            imputer = SimpleImputer(strategy="median")
            X_tr_num_imp = imputer.fit_transform(df_tr_fold[NUMERICAL_FEATURES].values)
            X_va_num_imp = imputer.transform(df_va_fold[NUMERICAL_FEATURES].values)

            # Step 4.2: Fold-Specific Population StandardScaler (ddof=0) strictly on imputed fold training data
            scaler = StandardScaler()
            X_tr_num_scaled = scaler.fit_transform(X_tr_num_imp)
            X_va_num_scaled = scaler.transform(X_va_num_imp)

            # Convert back to structured format for feature subsets
            num_feat_idx = {feat: idx for idx, feat in enumerate(NUMERICAL_FEATURES)}

            # Extract indicators (unscaled binary {0, 1})
            X_tr_ind = df_tr_fold[INDICATOR_FEATURES].values.astype(float)
            X_va_ind = df_va_fold[INDICATOR_FEATURES].values.astype(float)

            # Feature matrix mappings
            timing_indices = [num_feat_idx[f] for f in TIMING_FEATURES]
            morph_indices = [num_feat_idx[f] for f in MORPHOLOGY_FEATURES]

            feature_matrices_tr = {
                "Exp_A_Timing_Only": X_tr_num_scaled[:, timing_indices],
                "Exp_B_Morphology_Only": X_tr_num_scaled[:, morph_indices],
                "Exp_C_Combined_No_Indicators": X_tr_num_scaled,
                "Exp_D_Combined_With_Indicators": np.hstack([X_tr_num_scaled, X_tr_ind])
            }

            feature_matrices_va = {
                "Exp_A_Timing_Only": X_va_num_scaled[:, timing_indices],
                "Exp_B_Morphology_Only": X_va_num_scaled[:, morph_indices],
                "Exp_C_Combined_No_Indicators": X_va_num_scaled,
                "Exp_D_Combined_With_Indicators": np.hstack([X_va_num_scaled, X_va_ind])
            }

            # Train and evaluate each configuration in this fold
            for exp_id, exp_spec in EXPERIMENT_SPECS.items():
                X_tr = feature_matrices_tr[exp_id]
                X_va = feature_matrices_va[exp_id]

                for weight_id, weight_spec in WEIGHTING_SPECS.items():
                    model_key = f"{exp_id}_{weight_id}"

                    rf = RandomForestClassifier(
                        n_estimators=150,
                        criterion="gini",
                        max_depth=12,
                        max_features="sqrt",
                        min_samples_split=10,
                        min_samples_leaf=4,
                        bootstrap=True,
                        class_weight=weight_spec["class_weight"],
                        random_state=42,
                        n_jobs=-1
                    )

                    rf.fit(X_tr, y_train_fold)

                    y_pred = rf.predict(X_va)
                    y_prob = rf.predict_proba(X_va)[:, 1]

                    # Store out-of-fold predictions
                    oof_df.loc[val_idx, f"{model_key}_Pred"] = y_pred
                    oof_df.loc[val_idx, f"{model_key}_Prob"] = y_prob

                    # Compute fold metrics
                    metrics = compute_metrics(y_val_fold, y_pred, y_prob)

                    fold_rec = {
                        "Model_Key": model_key,
                        "Experiment_ID": exp_id,
                        "Experiment_Name": exp_spec["name"],
                        "Weighting": weight_id,
                        "Num_Features": exp_spec["feat_count"],
                        "Fold": fold_num,
                        "Train_Beats": len(train_idx),
                        "Val_Beats": len(val_idx),
                        "Val_Class0": int((y_val_fold == 0).sum()),
                        "Val_Class1": int((y_val_fold == 1).sum()),
                        **metrics
                    }
                    fold_metrics_records.append(fold_rec)

                    log(lines, f"   [{model_key}] Fold {fold_num} -> Bal Acc: {metrics['Balanced_Accuracy']*100:.2f}%, "
                               f"Sens: {metrics['Sensitivity']*100:.2f}%, Spec: {metrics['Specificity']*100:.2f}%, "
                               f"Prec: {metrics['Precision']*100:.2f}%, F1: {metrics['F1_Score']*100:.2f}%, "
                               f"ROC-AUC: {metrics['ROC_AUC']:.4f}, AP: {metrics['Average_Precision']:.4f}")

        # Post-Loop Verification: Complete OOF coverage
        log(lines, "\n4. POST-CROSS-VALIDATION OUT-OF-FOLD INTEGRITY VERIFICATION:")
        for mk in model_keys:
            nan_preds = oof_df[f"{mk}_Pred"].isna().sum()
            nan_probs = oof_df[f"{mk}_Prob"].isna().sum()
            assert nan_preds == 0, f"Missing predictions for {mk}: {nan_preds} NaNs found!"
            assert nan_probs == 0, f"Missing probabilities for {mk}: {nan_probs} NaNs found!"
            log(lines, f"   [OOF VERIFIED] {mk}: Exactly 66,679 non-null held-out predictions and probabilities.")

        # Save OOF Predictions Dataset
        oof_df.to_csv(OOF_PREDICTIONS_CSV_PATH, index=False)
        log(lines, f"   Saved complete out-of-fold prediction dataset to: {OOF_PREDICTIONS_CSV_PATH}")

        # Convert fold metrics to DataFrame & Save
        df_fold_metrics = pd.DataFrame(fold_metrics_records)
        df_fold_metrics.to_csv(FOLD_METRICS_CSV_PATH, index=False)
        log(lines, f"   Saved fold-level metrics table to: {FOLD_METRICS_CSV_PATH}")

    # ---------------------------------------------------------
    # 5. Paradigm A: Fold-Wise Statistical Summary (Mean ± Std, CV)
    # ---------------------------------------------------------
    log(lines, "\n" + "=" * 85)
    log(lines, "5. PARADIGM A: FOLD-WISE STATISTICAL SUMMARY (N = 5 Folds, Equal Patient Weighting):")
    log(lines, "=" * 85)

    summary_records = []
    metric_cols = ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Score", "ROC_AUC", "Average_Precision"]

    for mk in model_keys:
        sub_df = df_fold_metrics[df_fold_metrics["Model_Key"] == mk]
        rec = {
            "Model_Key": mk,
            "Experiment_ID": sub_df["Experiment_ID"].iloc[0],
            "Experiment_Name": sub_df["Experiment_Name"].iloc[0],
            "Weighting": sub_df["Weighting"].iloc[0],
            "Num_Features": sub_df["Num_Features"].iloc[0]
        }
        for m in metric_cols:
            vals = sub_df[m].values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1))
            cv_val = float(std_val / mean_val) if mean_val > 0 else 0.0
            rec[f"{m}_Mean"] = mean_val
            rec[f"{m}_Std"] = std_val
            rec[f"{m}_CV"] = cv_val

        summary_records.append(rec)

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(SUMMARY_METRICS_CSV_PATH, index=False)
    log(lines, f"   Saved summary metrics table to: {SUMMARY_METRICS_CSV_PATH}\n")

    summary_header = f"{'Model Key':<35} | {'Bal Acc (Mean+/-Std)':<20} | {'Sens (Mean+/-Std)':<20} | {'Prec (Mean+/-Std)':<20} | {'F1 (Mean+/-Std)':<20} | {'AP (Mean+/-Std)':<18}"
    log(lines, summary_header)
    log(lines, "-" * len(summary_header))
    for _, r in df_summary.iterrows():
        log(lines, f"{r['Model_Key']:<35} | "
                   f"{r['Balanced_Accuracy_Mean']*100:.2f}+/-{r['Balanced_Accuracy_Std']*100:.2f}%{'':<6} | "
                   f"{r['Sensitivity_Mean']*100:.2f}+/-{r['Sensitivity_Std']*100:.2f}%{'':<6} | "
                   f"{r['Precision_Mean']*100:.2f}+/-{r['Precision_Std']*100:.2f}%{'':<6} | "
                   f"{r['F1_Score_Mean']*100:.2f}+/-{r['F1_Score_Std']*100:.2f}%{'':<6} | "
                   f"{r['Average_Precision_Mean']:.4f}+/-{r['Average_Precision_Std']:.4f}")

    # ---------------------------------------------------------
    # 6. Paradigm B: Pooled Out-of-Fold Global Evaluation (N = 66,679 beats)
    # ---------------------------------------------------------
    log(lines, "\n" + "=" * 85)
    log(lines, "6. PARADIGM B: POOLED OUT-OF-FOLD GLOBAL EVALUATION (N = 66,679 Beats):")
    log(lines, "=" * 85)

    pooled_records = []
    for mk in model_keys:
        exp_id = mk.rsplit("_", 1)[0]
        weight_id = mk.rsplit("_", 1)[1]
        exp_spec = EXPERIMENT_SPECS[exp_id]

        y_true_pool = oof_df["True_Label"].values.astype(int)
        y_pred_pool = oof_df[f"{mk}_Pred"].values.astype(int)
        y_prob_pool = oof_df[f"{mk}_Prob"].values.astype(float)

        m_pool = compute_metrics(y_true_pool, y_pred_pool, y_prob_pool)
        p_rec = {
            "Model_Key": mk,
            "Experiment_ID": exp_id,
            "Experiment_Name": exp_spec["name"],
            "Weighting": weight_id,
            "Num_Features": exp_spec["feat_count"],
            "Total_Beats": len(y_true_pool),
            "Total_Class0": int((y_true_pool == 0).sum()),
            "Total_Class1": int((y_true_pool == 1).sum()),
            **m_pool
        }
        pooled_records.append(p_rec)

    df_pooled = pd.DataFrame(pooled_records)
    df_pooled.to_csv(POOLED_METRICS_CSV_PATH, index=False)
    log(lines, f"   Saved pooled metrics table to: {POOLED_METRICS_CSV_PATH}\n")

    pooled_header = f"{'Model Key':<35} | {'TN':<6} {'FP':<6} {'FN':<6} {'TP':<6} | {'Bal Acc':<9} {'Sens':<9} {'Spec':<9} {'Prec':<9} {'F1':<9} {'ROC-AUC':<9} {'AP':<8}"
    log(lines, pooled_header)
    log(lines, "-" * len(pooled_header))
    for _, r in df_pooled.iterrows():
        log(lines, f"{r['Model_Key']:<35} | "
                   f"{r['TN']:<6d} {r['FP']:<6d} {r['FN']:<6d} {r['TP']:<6d} | "
                   f"{r['Balanced_Accuracy']*100:6.2f}% {r['Sensitivity']*100:6.2f}% {r['Specificity']*100:6.2f}% "
                   f"{r['Precision']*100:6.2f}% {r['F1_Score']*100:6.2f}% {r['ROC_AUC']:7.4f}  {r['Average_Precision']:6.4f}")

    # ---------------------------------------------------------
    # 7. Dedicated Investigation: Exp C vs Exp D (Missingness Indicators)
    # ---------------------------------------------------------
    log(lines, "\n" + "=" * 85)
    log(lines, "7. DEDICATED INVESTIGATION: EXPERIMENT C VS EXPERIMENT D (INDICATOR IMPACT):")
    log(lines, "=" * 85)

    comp_records = []
    for weight_id in ["Balanced", "Unweighted"]:
        key_c = f"Exp_C_Combined_No_Indicators_{weight_id}"
        key_d = f"Exp_D_Combined_With_Indicators_{weight_id}"

        # Fold by fold
        df_c = df_fold_metrics[df_fold_metrics["Model_Key"] == key_c].sort_values("Fold")
        df_d = df_fold_metrics[df_fold_metrics["Model_Key"] == key_d].sort_values("Fold")

        for f in range(1, 6):
            r_c = df_c[df_c["Fold"] == f].iloc[0]
            r_d = df_d[df_d["Fold"] == f].iloc[0]

            comp_rec = {
                "Weighting": weight_id,
                "Comparison_Scope": f"Fold_{f}",
                "Fold": f,
                "Delta_Bal_Acc": r_d["Balanced_Accuracy"] - r_c["Balanced_Accuracy"],
                "Delta_Sensitivity": r_d["Sensitivity"] - r_c["Sensitivity"],
                "Delta_Specificity": r_d["Specificity"] - r_c["Specificity"],
                "Delta_Precision": r_d["Precision"] - r_c["Precision"],
                "Delta_F1_Score": r_d["F1_Score"] - r_c["F1_Score"],
                "Delta_ROC_AUC": r_d["ROC_AUC"] - r_c["ROC_AUC"],
                "Delta_Average_Precision": r_d["Average_Precision"] - r_c["Average_Precision"]
            }
            comp_records.append(comp_rec)

        # Pooled OOF
        p_c = df_pooled[df_pooled["Model_Key"] == key_c].iloc[0]
        p_d = df_pooled[df_pooled["Model_Key"] == key_d].iloc[0]
        comp_records.append({
            "Weighting": weight_id,
            "Comparison_Scope": "Pooled_OOF",
            "Fold": 0,
            "Delta_Bal_Acc": p_d["Balanced_Accuracy"] - p_c["Balanced_Accuracy"],
            "Delta_Sensitivity": p_d["Sensitivity"] - p_c["Sensitivity"],
            "Delta_Specificity": p_d["Specificity"] - p_c["Specificity"],
            "Delta_Precision": p_d["Precision"] - p_c["Precision"],
            "Delta_F1_Score": p_d["F1_Score"] - p_c["F1_Score"],
            "Delta_ROC_AUC": p_d["ROC_AUC"] - p_c["ROC_AUC"],
            "Delta_Average_Precision": p_d["Average_Precision"] - p_c["Average_Precision"]
        })

    df_exp_cd = pd.DataFrame(comp_records)
    df_exp_cd.to_csv(EXP_CD_COMPARISON_CSV_PATH, index=False)
    log(lines, f"   Saved Exp C vs D comparison table to: {EXP_CD_COMPARISON_CSV_PATH}\n")

    log(lines, "   Paired Delta Summary: Experiment D (With Indicators) minus Experiment C (No Indicators):")
    cd_header = f"{'Weighting':<12} | {'Scope':<10} | {'Delta Bal Acc':<13} | {'Delta Sens':<11} | {'Delta Prec':<11} | {'Delta F1':<11} | {'Delta ROC-AUC':<13} | {'Delta AP':<10}"
    log(lines, cd_header)
    log(lines, "-" * len(cd_header))
    for _, r in df_exp_cd.iterrows():
        log(lines, f"{r['Weighting']:<12} | {r['Comparison_Scope']:<10} | "
                   f"{r['Delta_Bal_Acc']*100:+7.2f}%     | {r['Delta_Sensitivity']*100:+7.2f}%   | "
                   f"{r['Delta_Precision']*100:+7.2f}%   | {r['Delta_F1_Score']*100:+7.2f}%   | "
                   f"{r['Delta_ROC_AUC']:+8.4f}     | {r['Delta_Average_Precision']:+7.4f}")

    # ---------------------------------------------------------
    # 8. Comparison with Stage 7 Validation Metrics
    # ---------------------------------------------------------
    log(lines, "\n" + "=" * 85)
    log(lines, "8. COMPARATIVE ANALYSIS: STAGE 8 CROSS-VALIDATION VS STAGE 7 VALIDATION:")
    log(lines, "=" * 85)
    log(lines, "   [EXPERIMENTAL CONDITION DISCLOSURE]")
    log(lines, "   * Stage 7: Models trained on all 30 Train_Dev subjects (66,679 beats) and evaluated on 7 Validation subjects (15,349 beats).")
    log(lines, "   * Stage 8: Models trained on ~24 subjects (~53,000 beats) and evaluated on 5-7 held-out subjects (~13,000 beats) per fold.")

    stage7_records = []
    if STAGE7_METRICS_PATH.exists():
        df_s7 = pd.read_csv(STAGE7_METRICS_PATH)
        for _, s7_row in df_s7.iterrows():
            exp_id = s7_row["Experiment_ID"]
            var_name = s7_row["Model_Variant"]
            weight_id = "Balanced" if "Balanced" in var_name else "Unweighted"
            mk = f"{exp_id}_{weight_id}"

            # Match summary and pooled rows
            sum_row = df_summary[df_summary["Model_Key"] == mk].iloc[0]
            pool_row = df_pooled[df_pooled["Model_Key"] == mk].iloc[0]

            s7_rec = {
                "Model_Key": mk,
                "Experiment_ID": exp_id,
                "Weighting": weight_id,
                # Stage 7 Validation
                "Stage7_Val_Bal_Acc": s7_row["Balanced_Accuracy"],
                "Stage7_Val_Sensitivity": s7_row["Sensitivity_Recall_Class1"],
                "Stage7_Val_Precision": s7_row["Precision_Class1"],
                "Stage7_Val_F1_Score": s7_row["F1_Score_Class1"],
                "Stage7_Val_ROC_AUC": s7_row["ROC_AUC"],
                "Stage7_Val_AP": s7_row["Average_Precision_AP"],
                # Stage 8 CV Mean
                "Stage8_CV_Mean_Bal_Acc": sum_row["Balanced_Accuracy_Mean"],
                "Stage8_CV_Mean_Sensitivity": sum_row["Sensitivity_Mean"],
                "Stage8_CV_Mean_Precision": sum_row["Precision_Mean"],
                "Stage8_CV_Mean_F1_Score": sum_row["F1_Score_Mean"],
                "Stage8_CV_Mean_ROC_AUC": sum_row["ROC_AUC_Mean"],
                "Stage8_CV_Mean_AP": sum_row["Average_Precision_Mean"],
                # Stage 8 Pooled OOF
                "Stage8_OOF_Bal_Acc": pool_row["Balanced_Accuracy"],
                "Stage8_OOF_Sensitivity": pool_row["Sensitivity"],
                "Stage8_OOF_Precision": pool_row["Precision"],
                "Stage8_OOF_F1_Score": pool_row["F1_Score"],
                "Stage8_OOF_ROC_AUC": pool_row["ROC_AUC"],
                "Stage8_OOF_AP": pool_row["Average_Precision"],
                # Deltas: CV Mean - Stage 7 Val
                "Delta_CV_vs_Val_Bal_Acc": sum_row["Balanced_Accuracy_Mean"] - s7_row["Balanced_Accuracy"],
                "Delta_CV_vs_Val_Sens": sum_row["Sensitivity_Mean"] - s7_row["Sensitivity_Recall_Class1"],
                "Delta_CV_vs_Val_F1": sum_row["F1_Score_Mean"] - s7_row["F1_Score_Class1"],
                "Delta_CV_vs_Val_AP": sum_row["Average_Precision_Mean"] - s7_row["Average_Precision_AP"]
            }
            stage7_records.append(s7_rec)

        df_s7_comp = pd.DataFrame(stage7_records)
        df_s7_comp.to_csv(STAGE7_COMPARISON_CSV_PATH, index=False)
        log(lines, f"   Saved Stage 8 vs Stage 7 comparison table to: {STAGE7_COMPARISON_CSV_PATH}\n")

        s7_header = f"{'Model Key':<35} | {'Metric':<10} | {'Stage 7 Val':<12} | {'Stage 8 CV Mean':<16} | {'Stage 8 OOF Pool':<16} | {'Delta (CV - Val)':<15}"
        log(lines, s7_header)
        log(lines, "-" * len(s7_header))
        for _, r in df_s7_comp[df_s7_comp["Weighting"] == "Balanced"].iterrows():
            log(lines, f"{r['Model_Key']:<35} | {'Bal Acc':<10} | {r['Stage7_Val_Bal_Acc']*100:6.2f}%     | {r['Stage8_CV_Mean_Bal_Acc']*100:6.2f}%         | {r['Stage8_OOF_Bal_Acc']*100:6.2f}%         | {r['Delta_CV_vs_Val_Bal_Acc']*100:+6.2f}%")
            log(lines, f"{'':<35} | {'Sens':<10} | {r['Stage7_Val_Sensitivity']*100:6.2f}%     | {r['Stage8_CV_Mean_Sensitivity']*100:6.2f}%         | {r['Stage8_OOF_Sensitivity']*100:6.2f}%         | {r['Delta_CV_vs_Val_Sens']*100:+6.2f}%")
            log(lines, f"{'':<35} | {'F1':<10} | {r['Stage7_Val_F1_Score']*100:6.2f}%     | {r['Stage8_CV_Mean_F1_Score']*100:6.2f}%         | {r['Stage8_OOF_F1_Score']*100:6.2f}%         | {r['Delta_CV_vs_Val_F1']*100:+6.2f}%")
            log(lines, f"{'':<35} | {'AP':<10} | {r['Stage7_Val_AP']:6.4f}      | {r['Stage8_CV_Mean_AP']:6.4f}          | {r['Stage8_OOF_AP']:6.4f}          | {r['Delta_CV_vs_Val_AP']:+6.4f}")
            log(lines, "-" * len(s7_header))
    else:
        log(lines, f"   [WARNING] Stage 7 metrics not found at: {STAGE7_METRICS_PATH}")

    # ---------------------------------------------------------
    # 9. Visualizations
    # ---------------------------------------------------------
    log(lines, "\n9. GENERATING DIAGNOSTIC VISUALIZATIONS:")

    # Plot 1: Boxplots across folds
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    metrics_to_plot = [
        ("Balanced_Accuracy", "Balanced Accuracy", axes[0, 0]),
        ("Sensitivity", "Sensitivity (Class 1 Recall)", axes[0, 1]),
        ("Precision", "Precision (Class 1)", axes[0, 2]),
        ("F1_Score", "F1-Score (Class 1)", axes[1, 0]),
        ("ROC_AUC", "ROC-AUC", axes[1, 1]),
        ("Average_Precision", "Average Precision (AP)", axes[1, 2])
    ]

    balanced_keys = [mk for mk in model_keys if "Balanced" in mk]
    short_labels = ["Exp A\n(Timing)", "Exp B\n(Morph)", "Exp C\n(Comb No Ind)", "Exp D\n(Comb Ind)"]
    palette = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]

    for m_col, m_title, ax in metrics_to_plot:
        data_to_plot = [df_fold_metrics[df_fold_metrics["Model_Key"] == k][m_col].values for k in balanced_keys]
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=short_labels, widths=0.5)
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.5)

        # Overlay scatter points for individual folds
        for i, fold_data in enumerate(data_to_plot, start=1):
            x = np.random.normal(i, 0.04, size=len(fold_data))
            ax.scatter(x, fold_data, color="black", s=30, alpha=0.8, zorder=4)

        ax.set_title(m_title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Stage 8: Patient-Grouped 5-Fold Cross-Validation Distributions (Balanced Models)\n"
                 "Inter-Patient Generalization Across 30 Train_Dev Subjects", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(BOXPLOT_PNG_PATH, dpi=300)
    plt.close()
    log(lines, f"   Saved fold-distribution boxplots to: {BOXPLOT_PNG_PATH}")

    # Plot 2: CV Mean vs Stage 7 Validation
    if STAGE7_METRICS_PATH.exists():
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        plot_comp_metrics = [
            ("Balanced_Accuracy", "Stage7_Val_Bal_Acc", "Stage8_CV_Mean_Bal_Acc", "Stage8_OOF_Bal_Acc", "Balanced Accuracy", axes[0]),
            ("Sensitivity", "Stage7_Val_Sensitivity", "Stage8_CV_Mean_Sensitivity", "Stage8_OOF_Sensitivity", "Sensitivity (Class 1)", axes[1]),
            ("F1_Score", "Stage7_Val_F1_Score", "Stage8_CV_Mean_F1_Score", "Stage8_OOF_F1_Score", "F1-Score (Class 1)", axes[2]),
            ("Average_Precision", "Stage7_Val_AP", "Stage8_CV_Mean_AP", "Stage8_OOF_AP", "Average Precision (AP)", axes[3])
        ]

        x_pos = np.arange(len(balanced_keys))
        width = 0.25

        for col_name, s7_col, cv_col, oof_col, title_str, ax in plot_comp_metrics:
            s7_vals = [df_s7_comp[df_s7_comp["Model_Key"] == k][s7_col].iloc[0] for k in balanced_keys]
            cv_vals = [df_s7_comp[df_s7_comp["Model_Key"] == k][cv_col].iloc[0] for k in balanced_keys]
            cv_stds = [df_summary[df_summary["Model_Key"] == k][f"{col_name}_Std"].iloc[0] for k in balanced_keys]
            oof_vals = [df_s7_comp[df_s7_comp["Model_Key"] == k][oof_col].iloc[0] for k in balanced_keys]

            ax.bar(x_pos - width, s7_vals, width, label="Stage 7 Validation (30 subj train)", color="#34495e", alpha=0.85)
            ax.bar(x_pos, cv_vals, width, yerr=cv_stds, capsize=4, label="Stage 8 CV Mean (24 subj train)", color="#3498db", alpha=0.85)
            ax.bar(x_pos + width, oof_vals, width, label="Stage 8 Pooled OOF (N=66,679)", color="#2ecc71", alpha=0.85)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(["Exp A\nTiming", "Exp B\nMorph", "Exp C\nComb No Ind", "Exp D\nComb Ind"], fontsize=9)
            ax.set_title(title_str, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            if ax == axes[0]:
                ax.legend(loc="lower right", fontsize=8)

        fig.suptitle("Stage 8 Comparative Benchmark: Stage 7 Validation vs Stage 8 Cross-Validation\n"
                     "Demonstrating Cohort Sample Size & Inter-Patient Variance Effects", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        plt.savefig(COMPARISON_PNG_PATH, dpi=300)
        plt.close()
        log(lines, f"   Saved benchmark comparison chart to: {COMPARISON_PNG_PATH}")

    # ---------------------------------------------------------
    # 10. Write Comprehensive Technical Report
    # ---------------------------------------------------------
    log(lines, "\n10. WRITING COMPREHENSIVE TECHNICAL REPORT...")
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n" + "=" * 85 + "\n")
        f.write("END OF STAGE 8 TECHNICAL AUDIT & CROSS-VALIDATION REPORT\n")
        f.write("=" * 85 + "\n")

    log(lines, f"   Technical report written successfully to: {REPORT_TXT_PATH}")
    log(lines, "\nSTAGE 8 EXECUTION COMPLETE. Ready for review.\n")

if __name__ == "__main__":
    main()
