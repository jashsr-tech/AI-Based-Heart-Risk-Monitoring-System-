"""
ECG FINAL PRO — PHASE 1: STAGE 7
Random Forest Architecture, Training & Explainability Benchmarking
File: 04_machine_learning/stage7_random_forest_training.py

Rigorous Implementation Details:
  1. Strict Data Isolation:
     - Loads EXCLUSIVELY from 04_machine_learning/mitbih_development_dataset.csv (82,028 rows).
     - NEVER accesses or opens 04_machine_learning/mitbih_independent_test_dataset.csv.
     - Trains exclusively on Train_Dev (66,679 beats: 60,034 Class 0 vs 6,645 Class 1).
     - Evaluates out-of-sample on Validation (15,349 beats: 13,309 Class 0 vs 2,040 Class 1).
  2. Four Standardized Feature Spaces:
     - Exp A: Timing-Only (2 features: RRInterval, HRV)
     - Exp B: Morphology-Only (7 features: QRSDuration, QTInterval, STSegment, PRInterval, PAmplitude, RAmplitude, TAmplitude)
     - Exp C: Combined No Indicators (9 numerical features)
     - Exp D: Combined With Indicators (15 features: 9 numerical + 6 indicators)
  3. Preprocessing Preservation:
     - Loads fitted Stage 5 imputer (models/ecg_feature_imputer.pkl) and scaler (models/ecg_feature_scaler.pkl).
     - No new imputer or scaler is fitted. Indicators remain unscaled binary {0, 1}.
  4. Hyperparameter Specifications (Frozen Eight-Model Suite):
     - Primary Balanced: n_estimators=150, criterion='gini', max_depth=12, max_features='sqrt',
       min_samples_split=10, min_samples_leaf=4, bootstrap=True, class_weight='balanced', random_state=42, n_jobs=-1
     - Comparison Unweighted: Same parameters with class_weight=None
  5. Dual Feature Importance Suite:
     - MDI: Accumulated Gini reduction during training.
     - PFI: Out-of-sample Average Precision drop on Validation (10 independent shuffles, seed 42).
       Preserves negative values; reports mean and std.
  6. Visualizations & Reports:
     - ROC curves, PR curves, MDI vs PFI bar charts, confusion matrices, RF vs DT comparison charts.
     - Full CSV tables and comprehensive Stage 7 report.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "04_machine_learning"
RESULTS_DIR = PROJECT_ROOT / "05_results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR_ALT = ML_DIR / "models"

DEV_DATASET_PATH = ML_DIR / "mitbih_development_dataset.csv"
IMPUTER_PATH = MODELS_DIR / "ecg_feature_imputer.pkl"
SCALER_PATH = MODELS_DIR / "ecg_feature_scaler.pkl"
DT_METRICS_PATH = RESULTS_DIR / "stage6_baseline_decision_tree_metrics.csv"

METRICS_CSV_PATH = RESULTS_DIR / "stage7_random_forest_metrics.csv"
CONFUSION_CSV_PATH = RESULTS_DIR / "stage7_random_forest_confusion_matrices.csv"
COMPARISON_CSV_PATH = RESULTS_DIR / "stage7_rf_vs_dt_comparison.csv"
MDI_CSV_PATH = RESULTS_DIR / "stage7_feature_importance_mdi.csv"
PFI_CSV_PATH = RESULTS_DIR / "stage7_feature_importance_pfi.csv"
REPORT_TXT_PATH = RESULTS_DIR / "stage7_random_forest_report.txt"

FIG_ROC_PATH = RESULTS_DIR / "stage7_roc_curves.png"
FIG_PR_PATH = RESULTS_DIR / "stage7_precision_recall_curves.png"
FIG_IMPORTANCE_PATH = RESULTS_DIR / "stage7_feature_importance_mdi_vs_pfi.png"
FIG_CONFUSION_PATH = RESULTS_DIR / "stage7_confusion_matrices.png"
FIG_COMPARISON_PATH = RESULTS_DIR / "stage7_rf_vs_dt_comparison.png"

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

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 80)
    log(lines, "ECG FINAL PRO — STAGE 7: RANDOM FOREST ARCHITECTURE & TRAINING")
    log(lines, "Strict Development-Dataset Isolation | 150 Decorrelated Trees | Dual Feature Importance")
    log(lines, "=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR_ALT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading & Strict Physical Isolation
    log(lines, f"\n1. DATASET LOADING & PHYSICAL ISOLATION VERIFICATION:")
    log(lines, f"   Input Source: {DEV_DATASET_PATH}")
    df_dev = pd.read_csv(DEV_DATASET_PATH)
    log(lines, f"   Total rows loaded from development file: {len(df_dev):,} beats.")

    # Certify zero test presence
    partitions_present = set(df_dev["Partition"].unique())
    log(lines, f"   Partitions present in development file: {sorted(list(partitions_present))}")
    assert partitions_present == {"Train_Dev", "Validation"}, f"Unexpected partitions: {partitions_present}"
    assert "Independent_Test" not in partitions_present, "CRITICAL ERROR: Independent_Test found in development file!"
    log(lines, "   [ISOLATION CERTIFIED] Independent_Test is physically absent from the loaded file.")

    df_train = df_dev[df_dev["Partition"] == "Train_Dev"].copy().reset_index(drop=True)
    df_val = df_dev[df_dev["Partition"] == "Validation"].copy().reset_index(drop=True)

    y_train = df_train["Label"].values.astype(int)
    y_val = df_val["Label"].values.astype(int)

    log(lines, f"\n   Verified Class Distributions:")
    log(lines, f"     * Train_Dev  ({len(df_train):,} beats, {df_train['Subject_ID'].nunique()} subjects):")
    log(lines, f"       - Class 0 (AAMI N)                     : {(y_train == 0).sum():,d} ({(y_train == 0).mean()*100:.2f}%)")
    log(lines, f"       - Class 1 (Selected Non-Normal S, V, F): {(y_train == 1).sum():,d} ({(y_train == 1).mean()*100:.2f}%)")
    log(lines, f"     * Validation ({len(df_val):,} beats, {df_val['Subject_ID'].nunique()} subjects):")
    log(lines, f"       - Class 0 (AAMI N)                     : {(y_val == 0).sum():,d} ({(y_val == 0).mean()*100:.2f}%)")
    log(lines, f"       - Class 1 (Selected Non-Normal S, V, F): {(y_val == 1).sum():,d} ({(y_val == 1).mean()*100:.2f}%)")

    # 2. Preprocessing with Stage 5 Artifacts
    log(lines, f"\n2. PREPROCESSING APPLICATION (Stage 5 Artifacts):")
    log(lines, f"   Loading imputer: {IMPUTER_PATH}")
    log(lines, f"   Loading scaler : {SCALER_PATH}")
    with open(IMPUTER_PATH, "rb") as f:
        imputer = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_train_num_imp = imputer.transform(df_train[NUMERICAL_FEATURES])
    X_val_num_imp = imputer.transform(df_val[NUMERICAL_FEATURES])

    Z_train_num = scaler.transform(X_train_num_imp)
    Z_val_num = scaler.transform(X_val_num_imp)

    X_train_ind = df_train[INDICATOR_FEATURES].values.astype(int)
    X_val_ind = df_val[INDICATOR_FEATURES].values.astype(int)

    # 3. Four Feature Configurations
    feature_configs = {
        "Exp_A_Timing_Only": {
            "name": "Exp A: Timing-Only (2 features: RRInterval, HRV)",
            "short_name": "Timing_Only",
            "feature_names": ["RRInterval", "HRV"],
            "X_train": Z_train_num[:, [0, 1]],
            "X_val": Z_val_num[:, [0, 1]],
            "file_suffix": "timing_only"
        },
        "Exp_B_Morphology_Only": {
            "name": "Exp B: Morphology-Only (7 features: QRS, QT, ST, PR, P_amp, R_amp, T_amp)",
            "short_name": "Morphology_Only",
            "feature_names": NUMERICAL_FEATURES[2:],
            "X_train": Z_train_num[:, 2:9],
            "X_val": Z_val_num[:, 2:9],
            "file_suffix": "morphology_only"
        },
        "Exp_C_Combined_No_Indicators": {
            "name": "Exp C: Combined WITHOUT Missingness Indicators (9 numerical features)",
            "short_name": "Combined_No_Indicators",
            "feature_names": NUMERICAL_FEATURES,
            "X_train": Z_train_num,
            "X_val": Z_val_num,
            "file_suffix": "combined_no_ind"
        },
        "Exp_D_Combined_With_Indicators": {
            "name": "Exp D: Combined WITH Missingness Indicators (15 features: 9 numerical + 6 indicators)",
            "short_name": "Combined_With_Indicators",
            "feature_names": NUMERICAL_FEATURES + INDICATOR_FEATURES,
            "X_train": np.hstack([Z_train_num, X_train_ind]),
            "X_val": np.hstack([Z_val_num, X_val_ind]),
            "file_suffix": "combined_with_ind"
        }
    }

    # 4. Model Training & Evaluation Loop
    log(lines, "\n" + "-" * 80)
    log(lines, "3. EIGHT-MODEL RANDOM FOREST BENCHMARKING (150 TREES, MAX_DEPTH=12, MAX_FEATURES='SQRT')")
    log(lines, "-" * 80)

    metrics_records = []
    confusion_records = []
    trained_models = {}
    val_probs = {}
    val_preds = {}

    for exp_key, cfg in feature_configs.items():
        log(lines, f"\n=== {cfg['name']} (Train: {cfg['X_train'].shape}, Val: {cfg['X_val'].shape}) ===")

        variants = [
            ("Balanced", "balanced", f"stage7_rf_{cfg['file_suffix']}.pkl"),
            ("Unweighted", None, f"stage7_rf_unweighted_{cfg['file_suffix']}.pkl")
        ]

        for var_name, weight_param, model_fname in variants:
            rf = RandomForestClassifier(
                n_estimators=150,
                criterion="gini",
                max_depth=12,
                max_features="sqrt",
                min_samples_split=10,
                min_samples_leaf=4,
                bootstrap=True,
                class_weight=weight_param,
                random_state=42,
                n_jobs=-1
            )

            # Fit strictly on Train_Dev
            rf.fit(cfg["X_train"], y_train)

            # Evaluate on Validation
            y_pred = rf.predict(cfg["X_val"])
            y_prob = rf.predict_proba(cfg["X_val"])[:, 1]

            model_tag = f"{exp_key}_{var_name}"
            trained_models[model_tag] = rf
            val_probs[model_tag] = y_prob
            val_preds[model_tag] = y_pred

            # Metrics
            acc = accuracy_score(y_val, y_pred)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            sens = recall_score(y_val, y_pred, pos_label=1)
            spec = recall_score(y_val, y_pred, pos_label=0)
            prec_c1 = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
            prec_c0 = precision_score(y_val, y_pred, pos_label=0, zero_division=0)
            f1_c1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            f1_c0 = f1_score(y_val, y_pred, pos_label=0, zero_division=0)
            macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            roc_auc = roc_auc_score(y_val, y_prob)
            ap = average_precision_score(y_val, y_prob)

            cm = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel()

            log(lines, f"  [{var_name:<11}] Acc: {acc*100:6.2f}% | BalAcc: {bal_acc*100:6.2f}% | Sens (C1): {sens*100:6.2f}% | Spec (C0): {spec*100:6.2f}% | Prec (C1): {prec_c1*100:6.2f}% | F1 (C1): {f1_c1*100:6.2f}% | Macro F1: {macro_f1*100:6.2f}% | ROC-AUC: {roc_auc:.4f} | AP: {ap:.4f}")
            log(lines, f"                Confusion Matrix: TN={tn:,d}, FP={fp:,d}, FN={fn:,d}, TP={tp:,d} (Total Class 1: {tp+fn:,d})")

            # Save models
            for target_dir in [MODELS_DIR, MODELS_DIR_ALT]:
                with open(target_dir / model_fname, "wb") as f:
                    pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)

            metrics_records.append({
                "Experiment_ID": exp_key,
                "Experiment_Name": cfg["short_name"],
                "Model_Variant": var_name,
                "Class_Weight": str(weight_param),
                "Num_Features": cfg["X_train"].shape[1],
                "N_Estimators": 150,
                "Max_Depth": 12,
                "Max_Features": "sqrt",
                "Min_Samples_Split": 10,
                "Min_Samples_Leaf": 4,
                "Accuracy": round(acc, 4),
                "Balanced_Accuracy": round(bal_acc, 4),
                "Sensitivity_Recall_Class1": round(sens, 4),
                "Specificity_Recall_Class0": round(spec, 4),
                "Precision_Class1": round(prec_c1, 4),
                "Precision_Class0": round(prec_c0, 4),
                "F1_Score_Class1": round(f1_c1, 4),
                "F1_Score_Class0": round(f1_c0, 4),
                "Macro_F1_Score": round(macro_f1, 4),
                "ROC_AUC": round(roc_auc, 4),
                "Average_Precision_AP": round(ap, 4),
                "True_Negatives_TN": int(tn),
                "False_Positives_FP": int(fp),
                "False_Negatives_FN": int(fn),
                "True_Positives_TP": int(tp),
                "Model_File": model_fname
            })

            confusion_records.append({
                "Experiment_ID": exp_key,
                "Model_Variant": var_name,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "Class0_Total": int(tn + fp),
                "Class1_Total": int(fn + tp),
                "Class0_Accuracy_Pct": round(tn / (tn + fp) * 100.0, 2),
                "Class1_Accuracy_Pct": round(tp / (fn + tp) * 100.0, 2),
                "False_Positive_Rate_Pct": round(fp / (tn + fp) * 100.0, 2),
                "False_Negative_Rate_Pct": round(fn / (fn + tp) * 100.0, 2)
            })

    # Save metrics and confusion tables
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(METRICS_CSV_PATH, index=False)
    log(lines, f"\nSaved Random Forest metrics table: {METRICS_CSV_PATH}")

    df_cm = pd.DataFrame(confusion_records)
    df_cm.to_csv(CONFUSION_CSV_PATH, index=False)
    log(lines, f"Saved confusion matrices table: {CONFUSION_CSV_PATH}")

    # 5. Direct Delta Comparison against Stage 6 Decision Tree
    log(lines, "\n" + "-" * 80)
    log(lines, "4. RANDOM FOREST VS. DECISION TREE DELTA BENCHMARKING")
    log(lines, "-" * 80)
    df_dt = pd.read_csv(DT_METRICS_PATH)
    dt_bal_map = df_dt[df_dt["Model_Variant"] == "Balanced_Regenerated_V2"].set_index("Experiment_ID")

    comparison_records = []
    log(lines, f"{'Experiment':<26} | {'Metric':<18} | {'Decision Tree':<14} | {'Random Forest':<14} | {'Ensemble Delta (RF - DT)'}")
    log(lines, "-" * 90)

    for exp_key, cfg in feature_configs.items():
        rf_row = df_metrics[(df_metrics["Experiment_ID"] == exp_key) & (df_metrics["Model_Variant"] == "Balanced")].iloc[0]
        dt_row = dt_bal_map.loc[exp_key]

        comp_metrics = [
            ("Accuracy", "Accuracy", True),
            ("Balanced_Accuracy", "Balanced Acc", True),
            ("Sensitivity_Recall_Class1", "Sens (C1)", True),
            ("Precision_Class1", "Prec (C1)", True),
            ("F1_Score_Class1", "F1 (C1)", True),
            ("Average_Precision_AP", "Average Prec (AP)", False),
            ("ROC_AUC", "ROC-AUC", False)
        ]

        for m_col, m_label, is_pct in comp_metrics:
            dt_val = dt_row[m_col]
            rf_val = rf_row[m_col]
            delta = rf_val - dt_val

            comparison_records.append({
                "Experiment_ID": exp_key,
                "Experiment_Name": cfg["short_name"],
                "Metric_Name": m_label,
                "Decision_Tree_Value": dt_val,
                "Random_Forest_Value": rf_val,
                "Delta_RF_minus_DT": delta
            })

            if is_pct:
                log(lines, f"{cfg['short_name']:<26} | {m_label:<18} | {dt_val*100:6.2f}%       | {rf_val*100:6.2f}%       | {delta*100:+6.2f}%")
            else:
                log(lines, f"{cfg['short_name']:<26} | {m_label:<18} | {dt_val:8.4f}       | {rf_val:8.4f}       | {delta:+8.4f}")
        log(lines, "-" * 90)

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(COMPARISON_CSV_PATH, index=False)
    log(lines, f"Saved RF vs DT comparison table: {COMPARISON_CSV_PATH}")

    # 6. Dual Feature Importance Suite (MDI & PFI)
    log(lines, "\n" + "-" * 80)
    log(lines, "5. DUAL FEATURE IMPORTANCE ANALYSIS (MDI & PERMUTATION IMPORTANCE)")
    log(lines, "-" * 80)
    log(lines, "   Evaluating Experiment D (15 features) and Experiment C (9 features):")
    log(lines, "   - MDI: Accumulated Gini reduction during training.")
    log(lines, "   - PFI: Out-of-sample Average Precision drop across 10 repeats on Validation.")

    mdi_records = []
    pfi_records = []

    for target_exp in ["Exp_C_Combined_No_Indicators", "Exp_D_Combined_With_Indicators"]:
        rf_model = trained_models[f"{target_exp}_Balanced"]
        cfg = feature_configs[target_exp]
        feat_names = cfg["feature_names"]
        X_val_data = cfg["X_val"]

        # 1. MDI
        mdi_vals = rf_model.feature_importances_
        for f_name, mdi_score in zip(feat_names, mdi_vals):
            mdi_records.append({
                "Experiment_ID": target_exp,
                "Feature_Name": f_name,
                "MDI_Gini_Importance": round(float(mdi_score), 6)
            })

        # 2. PFI (using Validation Average Precision drop)
        log(lines, f"\n   Computing Permutation Importance for {cfg['short_name']} (10 repeats)...")
        pfi_res = permutation_importance(
            rf_model,
            X_val_data,
            y_val,
            scoring="average_precision",
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        for f_name, m_drop, s_drop in zip(feat_names, pfi_res.importances_mean, pfi_res.importances_std):
            pfi_records.append({
                "Experiment_ID": target_exp,
                "Feature_Name": f_name,
                "PFI_Mean_AP_Drop": round(float(m_drop), 6),
                "PFI_Std_AP_Drop": round(float(s_drop), 6)
            })

    df_mdi = pd.DataFrame(mdi_records)
    df_mdi.to_csv(MDI_CSV_PATH, index=False)
    df_pfi = pd.DataFrame(pfi_records)
    df_pfi.to_csv(PFI_CSV_PATH, index=False)
    log(lines, f"Saved MDI importances to: {MDI_CSV_PATH}")
    log(lines, f"Saved PFI importances to: {PFI_CSV_PATH}")

    # Print Experiment D Feature Importance Rankings
    pfi_d = df_pfi[df_pfi["Experiment_ID"] == "Exp_D_Combined_With_Indicators"].sort_values(by="PFI_Mean_AP_Drop", ascending=False)
    mdi_d = df_mdi[df_mdi["Experiment_ID"] == "Exp_D_Combined_With_Indicators"].set_index("Feature_Name")

    log(lines, "\n   Experiment D (15 Features) Importance Ranking (Sorted by Out-of-Sample PFI AP Drop):")
    log(lines, f"   {'Rank':<5} | {'Feature Name':<16} | {'PFI AP Drop (mean +- std)':<26} | {'MDI Gini Importance':<20} | Type")
    log(lines, "-" * 85)
    for rank, (_, row) in enumerate(pfi_d.iterrows(), 1):
        f_name = row["Feature_Name"]
        p_mean = row["PFI_Mean_AP_Drop"]
        p_std = row["PFI_Std_AP_Drop"]
        mdi_val = mdi_d.loc[f_name, "MDI_Gini_Importance"]
        f_type = "Missingness Indicator" if f_name in INDICATOR_FEATURES else "Numerical Feature"
        log(lines, f"   {rank:<5} | {f_name:<16} | {p_mean:+8.4f} +- {p_std:6.4f}          | {mdi_val:12.4f}         | {f_type}")

    # 7. Generate Visualizations (Matplotlib)
    log(lines, "\n" + "-" * 80)
    log(lines, "6. GENERATING EVALUATION FIGURES")
    log(lines, "-" * 80)

    # Figure 1: ROC Curves
    plt.figure(figsize=(9, 7))
    for exp_key, cfg in feature_configs.items():
        y_prob = val_probs[f"{exp_key}_Balanced"]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc_val = roc_auc_score(y_val, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{cfg['short_name']} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Stage 7 Random Forest: ROC Curves on Validation Partition", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_ROC_PATH, dpi=300)
    plt.close()
    log(lines, f"   Saved ROC curves plot: {FIG_ROC_PATH}")

    # Figure 2: Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    baseline_ap = (y_val == 1).mean()
    for exp_key, cfg in feature_configs.items():
        y_prob = val_probs[f"{exp_key}_Balanced"]
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        ap_val = average_precision_score(y_val, y_prob)
        plt.plot(rec, prec, lw=2, label=f"{cfg['short_name']} (AP = {ap_val:.4f})")
    plt.axhline(y=baseline_ap, color="k", linestyle="--", lw=1.5, label=f"Prevalence Baseline (AP = {baseline_ap:.4f})")
    plt.xlabel("Recall / Sensitivity (Class 1)", fontsize=11)
    plt.ylabel("Precision (Class 1)", fontsize=11)
    plt.title("Stage 7 Random Forest: Precision-Recall Curves on Validation Partition", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PR_PATH, dpi=300)
    plt.close()
    log(lines, f"   Saved PR curves plot: {FIG_PR_PATH}")

    # Figure 3: MDI vs PFI Horizontal Bar Charts (Exp D)
    plt.figure(figsize=(12, 6))
    pfi_d_sorted = df_pfi[df_pfi["Experiment_ID"] == "Exp_D_Combined_With_Indicators"].sort_values(by="PFI_Mean_AP_Drop")
    y_pos = np.arange(len(pfi_d_sorted))

    plt.subplot(1, 2, 1)
    plt.barh(y_pos, pfi_d_sorted["PFI_Mean_AP_Drop"], xerr=pfi_d_sorted["PFI_Std_AP_Drop"], color="#1f77b4", capsize=3)
    plt.yticks(y_pos, pfi_d_sorted["Feature_Name"], fontsize=9)
    plt.xlabel("Mean Drop in Validation Average Precision", fontsize=10)
    plt.title("Permutation Feature Importance (PFI)", fontsize=11, fontweight="bold")
    plt.grid(True, axis="x", alpha=0.3)

    plt.subplot(1, 2, 2)
    mdi_d_matched = df_mdi[df_mdi["Experiment_ID"] == "Exp_D_Combined_With_Indicators"].set_index("Feature_Name").loc[pfi_d_sorted["Feature_Name"]]
    plt.barh(y_pos, mdi_d_matched["MDI_Gini_Importance"], color="#ff7f0e")
    plt.yticks(y_pos, [""] * len(y_pos))
    plt.xlabel("MDI Gini Importance (Training)", fontsize=10)
    plt.title("Mean Decrease in Impurity (MDI)", fontsize=11, fontweight="bold")
    plt.grid(True, axis="x", alpha=0.3)

    plt.suptitle("Experiment D (15 Features): Out-of-Sample PFI vs. In-Sample MDI Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_IMPORTANCE_PATH, dpi=300)
    plt.close()
    log(lines, f"   Saved feature importance comparison plot: {FIG_IMPORTANCE_PATH}")

    # Figure 4: Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()
    for idx, (exp_key, cfg) in enumerate(feature_configs.items()):
        y_pred = val_preds[f"{exp_key}_Balanced"]
        cm = confusion_matrix(y_val, y_pred)
        ax = axes[idx]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(cfg["short_name"], fontsize=12, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["AAMI N (0)", "Non-Normal (1)"], fontsize=10)
        ax.set_yticklabels(["AAMI N (0)", "Non-Normal (1)"], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

        # Annotate text
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val_cnt = cm[i, j]
                pct = val_cnt / cm[i].sum() * 100.0
                ax.text(j, i, f"{val_cnt:,d}\n({pct:.1f}%)",
                        ha="center", va="center",
                        color="white" if val_cnt > thresh else "black",
                        fontsize=10)
    plt.suptitle("Stage 7 Balanced Random Forest: Confusion Matrices on Validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_CONFUSION_PATH, dpi=300)
    plt.close()
    log(lines, f"   Saved confusion matrices grid: {FIG_CONFUSION_PATH}")

    # Figure 5: RF vs DT Metric Comparison Bar Chart
    plt.figure(figsize=(11, 6))
    dt_f1 = [df_dt[(df_dt["Experiment_ID"] == k) & (df_dt["Model_Variant"] == "Balanced_Regenerated_V2")]["F1_Score_Class1"].values[0] for k in feature_configs.keys()]
    rf_f1 = [df_metrics[(df_metrics["Experiment_ID"] == k) & (df_metrics["Model_Variant"] == "Balanced")]["F1_Score_Class1"].values[0] for k in feature_configs.keys()]
    dt_ap = [df_dt[(df_dt["Experiment_ID"] == k) & (df_dt["Model_Variant"] == "Balanced_Regenerated_V2")]["Average_Precision_AP"].values[0] for k in feature_configs.keys()]
    rf_ap = [df_metrics[(df_metrics["Experiment_ID"] == k) & (df_metrics["Model_Variant"] == "Balanced")]["Average_Precision_AP"].values[0] for k in feature_configs.keys()]

    x = np.arange(len(feature_configs))
    width = 0.20

    plt.bar(x - 1.5*width, dt_f1, width, label="Decision Tree F1 (Class 1)", color="#aec7e8")
    plt.bar(x - 0.5*width, rf_f1, width, label="Random Forest F1 (Class 1)", color="#1f77b4")
    plt.bar(x + 0.5*width, dt_ap, width, label="Decision Tree AP", color="#ffbb78")
    plt.bar(x + 1.5*width, rf_ap, width, label="Random Forest AP", color="#ff7f0e")

    plt.xticks(x, [cfg["short_name"] for cfg in feature_configs.values()], fontsize=10)
    plt.ylabel("Score", fontsize=11)
    plt.title("Ensemble Gain: Stage 7 Random Forest vs. Stage 6 Decision Tree", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_COMPARISON_PATH, dpi=300)
    plt.close()
    log(lines, f"   Saved RF vs DT comparison plot: {FIG_COMPARISON_PATH}")

    # 8. Report Summary & Diagnostic Insights
    log(lines, "\n" + "-" * 80)
    log(lines, "7. SUMMARY OF STAGE 7 FINDINGS")
    log(lines, "-" * 80)
    row_A = df_metrics[(df_metrics["Experiment_ID"] == "Exp_A_Timing_Only") & (df_metrics["Model_Variant"] == "Balanced")].iloc[0]
    row_B = df_metrics[(df_metrics["Experiment_ID"] == "Exp_B_Morphology_Only") & (df_metrics["Model_Variant"] == "Balanced")].iloc[0]
    row_C = df_metrics[(df_metrics["Experiment_ID"] == "Exp_C_Combined_No_Indicators") & (df_metrics["Model_Variant"] == "Balanced")].iloc[0]
    row_D = df_metrics[(df_metrics["Experiment_ID"] == "Exp_D_Combined_With_Indicators") & (df_metrics["Model_Variant"] == "Balanced")].iloc[0]

    log(lines, f"   A. Ensemble Improvement over Decision Tree Baseline:")
    for exp_k, exp_label in [("Exp_A_Timing_Only", "Exp A (Timing)"), ("Exp_B_Morphology_Only", "Exp B (Morphology)"),
                             ("Exp_C_Combined_No_Indicators", "Exp C (Combined No Ind)"), ("Exp_D_Combined_With_Indicators", "Exp D (Combined With Ind)")]:
        dt_ap_val = df_dt[(df_dt["Experiment_ID"] == exp_k) & (df_dt["Model_Variant"] == "Balanced_Regenerated_V2")]["Average_Precision_AP"].values[0]
        rf_ap_val = df_metrics[(df_metrics["Experiment_ID"] == exp_k) & (df_metrics["Model_Variant"] == "Balanced")]["Average_Precision_AP"].values[0]
        log(lines, f"      * {exp_label:<28}: Decision Tree AP = {dt_ap_val:.4f} -> Random Forest AP = {rf_ap_val:.4f} (Delta: {rf_ap_val - dt_ap_val:+.4f})")

    log(lines, f"\n   B. Missingness Shortcut Analysis (Exp D vs. Exp C):")
    log(lines, f"      * Exp C (9 Numerical Features)   : AP = {row_C['Average_Precision_AP']:.4f} | F1 (C1) = {row_C['F1_Score_Class1']*100:.2f}% | Sens = {row_C['Sensitivity_Recall_Class1']*100:.2f}%")
    log(lines, f"      * Exp D (15 Features w/ Ind)     : AP = {row_D['Average_Precision_AP']:.4f} | F1 (C1) = {row_D['F1_Score_Class1']*100:.2f}% | Sens = {row_D['Sensitivity_Recall_Class1']*100:.2f}%")
    log(lines, f"      * Average Precision Delta (D - C): {row_D['Average_Precision_AP'] - row_C['Average_Precision_AP']:+.4f}")
    log(lines, f"      * F1-Score Delta (D - C)         : {(row_D['F1_Score_Class1'] - row_C['F1_Score_Class1'])*100:+.2f}%")

    log(lines, "\n   C. Physiological vs. Indicator Importance in Experiment D:")
    top_3_pfi = pfi_d.head(3)["Feature_Name"].tolist()
    log(lines, f"      * Top 3 Features by Out-of-Sample PFI Drop: {top_3_pfi}")
    log(lines, "      * Important Caveat: Positive AP drop indicates predictive dependency within this model architecture,")
    log(lines, "        not proof of biological causality or clinical pathogenesis.")

    # Write report
    REPORT_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"\nSaved comprehensive Stage 7 validation report: {REPORT_TXT_PATH}")

    log(lines, "\n" + "=" * 80)
    log(lines, "STAGE 7 EXECUTION COMPLETE: PASSED.")
    log(lines, "Strict Stop: Awaiting user review before Stage 8.")
    log(lines, "=" * 80)

if __name__ == "__main__":
    main()
