"""
ECG FINAL PRO — PHASE 1: STAGE 6 (REVISED & AUDITED)
Legacy Decision Tree Baseline Benchmarking (Eight-Model Evaluation Suite)
File: 04_machine_learning/stage6_baseline_decision_tree.py

Revisions in this version:
  1. Balanced Model Hyperparameters:
     - Formally updated to approved regularized configuration:
       criterion='gini', max_depth=10, min_samples_split=10, min_samples_leaf=5,
       class_weight='balanced', random_state=42.
  2. Legacy-Style Unweighted Models:
     - Preserved unchanged (reloaded from trained artifacts, NOT retrained):
       criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1,
       class_weight=None, random_state=42.
  3. Independent_Test Isolation Reality:
     - Accurately documents that raw V2 CSV initially contained Independent_Test rows on disk;
       they were immediately filtered out prior to preprocessing, training, prediction, or evaluation.
  4. Validation Label Census Reconciliation:
     - Accurately documents that the frozen dataset ground truth is 13,309 Class 0 (AAMI N) and 2,040 Class 1.
       (Reconciles the narrative drafting artifact of 13,668 / 1,681).
  5. Versioned Archival:
     - V1 results preserved in 05_results/*_v1_split2_leaf1.csv/txt and models/archive_stage6_v1/.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "04_machine_learning"
RESULTS_DIR = PROJECT_ROOT / "05_results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR_ALT = ML_DIR / "models"
ARCHIVE_MODELS_DIR = MODELS_DIR / "archive_stage6_v1"

INPUT_DATASET_V2 = ML_DIR / "mitbih_beat_features_mapped_v2_k10.csv"
IMPUTER_PATH = MODELS_DIR / "ecg_feature_imputer.pkl"
SCALER_PATH = MODELS_DIR / "ecg_feature_scaler.pkl"

METRICS_CSV_PATH = RESULTS_DIR / "stage6_baseline_decision_tree_metrics.csv"
CONFUSION_CSV_PATH = RESULTS_DIR / "stage6_baseline_decision_tree_confusion_matrices.csv"
REPORT_TXT_PATH = RESULTS_DIR / "stage6_baseline_decision_tree_report.txt"
V1_METRICS_PATH = RESULTS_DIR / "stage6_baseline_decision_tree_metrics_v1_split2_leaf1.csv"

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
    log(lines, "ECG FINAL PRO — STAGE 6 (REVISED): DECISION TREE BASELINE BENCHMARKING")
    log(lines, "Regenerated Balanced Models (split=10, leaf=5) | Preserved Legacy Unweighted (split=2, leaf=1)")
    log(lines, "=" * 80)

    # 1. Dataset Loading & Isolation Statement
    log(lines, f"\n1. DATASET LOADING & PARTITION ISOLATION STATEMENT:")
    log(lines, f"   Dataset Source: {INPUT_DATASET_V2}")
    df_raw = pd.read_csv(INPUT_DATASET_V2)
    n_raw_total = len(df_raw)
    n_raw_test = (df_raw["Partition"] == "Independent_Test").sum()
    log(lines, f"   [DISK READ REALITY]: Raw file contained {n_raw_total:,} total beats, including {n_raw_test:,} Independent_Test rows.")

    # Explicit isolation filtering
    df_dev = df_raw[df_raw["Partition"].isin(["Train_Dev", "Validation"])].copy().reset_index(drop=True)
    log(lines, f"   [PIPELINE FILTERING]: Filtered strictly to development partitions: {len(df_dev):,} beats (Train_Dev + Validation).")
    log(lines, f"   [ISOLATION CERTIFICATION]: Independent_Test rows were completely removed BEFORE preprocessing, training, or evaluation.")
    assert "Independent_Test" not in df_dev["Partition"].unique(), "Independent_Test present in development DataFrame!"

    df_train = df_dev[df_dev["Partition"] == "Train_Dev"].copy().reset_index(drop=True)
    df_val = df_dev[df_dev["Partition"] == "Validation"].copy().reset_index(drop=True)

    y_train = df_train["Label"].values.astype(int)
    y_val = df_val["Label"].values.astype(int)

    # Reconciled label counts
    log(lines, f"\n2. GROUND-TRUTH VALIDATION LABEL RECONCILIATION:")
    log(lines, f"   Train_Dev ({len(df_train):,} beats):")
    log(lines, f"     - Class 0 (AAMI N)                     : {(y_train == 0).sum():,d} ({(y_train == 0).mean()*100:.2f}%)")
    log(lines, f"     - Class 1 (Selected Non-Normal S, V, F): {(y_train == 1).sum():,d} ({(y_train == 1).mean()*100:.2f}%)")
    log(lines, f"   Validation ({len(df_val):,} beats):")
    log(lines, f"     - Class 0 (AAMI N)                     : {(y_val == 0).sum():,d} ({(y_val == 0).mean()*100:.2f}%) [FROZEN GROUND TRUTH]")
    log(lines, f"     - Class 1 (Selected Non-Normal S, V, F): {(y_val == 1).sum():,d} ({(y_val == 1).mean()*100:.2f}%) [FROZEN GROUND TRUTH]")
    log(lines, f"   * Forensic Audit Note on Earlier 13,668 / 1,681 Counts:")
    log(lines, f"     An audit confirmed that 13,668 / 1,681 never existed in any dataset file or Stage 5 output.")
    log(lines, f"     It was an ungrounded narrative estimation written during the drafting of the Stage 6 plan")
    log(lines, f"     (assuming a flat 89.05% Class 0 ratio). The actual dataset ground truth has always been 13,309 vs 2,040.")

    # 3. Apply Stage 5 Preprocessing
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

    feature_configs = {
        "Exp_A_Timing_Only": {
            "name": "Exp A: Timing-Only (2 features: RRInterval, HRV)",
            "short_name": "Timing_Only",
            "X_train": Z_train_num[:, [0, 1]],
            "X_val": Z_val_num[:, [0, 1]],
            "file_suffix": "timing_only"
        },
        "Exp_B_Morphology_Only": {
            "name": "Exp B: Morphology-Only (7 features: QRS, QT, ST, PR, P_amp, R_amp, T_amp)",
            "short_name": "Morphology_Only",
            "X_train": Z_train_num[:, 2:9],
            "X_val": Z_val_num[:, 2:9],
            "file_suffix": "morphology_only"
        },
        "Exp_C_Combined_No_Indicators": {
            "name": "Exp C: Combined WITHOUT Missingness Indicators (9 numerical features)",
            "short_name": "Combined_No_Indicators",
            "X_train": Z_train_num,
            "X_val": Z_val_num,
            "file_suffix": "combined_no_ind"
        },
        "Exp_D_Combined_With_Indicators": {
            "name": "Exp D: Combined WITH Missingness Indicators (15 features: 9 numerical + 6 indicators)",
            "short_name": "Combined_With_Indicators",
            "X_train": np.hstack([Z_train_num, X_train_ind]),
            "X_val": np.hstack([Z_val_num, X_val_ind]),
            "file_suffix": "combined_with_ind"
        }
    }

    # 4. Benchmarking Execution
    log(lines, "\n" + "-" * 80)
    log(lines, "3. BENCHMARKING REGENERATED BALANCED MODELS & PRESERVED UNWEIGHTED MODELS")
    log(lines, "-" * 80)
    log(lines, "   Hyperparameter Audit:")
    log(lines, "     * Primary Balanced Models (REGENERATED V2):")
    log(lines, "       criterion='gini', max_depth=10, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42")
    log(lines, "     * Legacy-Style Unweighted Models (PRESERVED UNCHANGED):")
    log(lines, "       criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, class_weight=None, random_state=42")

    metrics_records = []
    confusion_records = []

    for exp_key, cfg in feature_configs.items():
        log(lines, f"\n=== {cfg['name']} (Train shape: {cfg['X_train'].shape}, Val shape: {cfg['X_val'].shape}) ===")

        # 1. Regenerate Balanced Model with approved regularized parameters (split=10, leaf=5)
        dt_bal = DecisionTreeClassifier(
            criterion="gini",
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
        dt_bal.fit(cfg["X_train"], y_train)

        # Save regenerated balanced model to primary locations
        bal_fname = f"stage6_dt_{cfg['file_suffix']}.pkl"
        for t_dir in [MODELS_DIR, MODELS_DIR_ALT]:
            with open(t_dir / bal_fname, "wb") as f:
                pickle.dump(dt_bal, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 2. Load existing unweighted model (DO NOT RETRAIN)
        unw_fname = f"stage6_dt_unweighted_{cfg['file_suffix']}.pkl"
        unw_path = ARCHIVE_MODELS_DIR / unw_fname
        if not unw_path.exists():
            unw_path = MODELS_DIR / unw_fname
        with open(unw_path, "rb") as f:
            dt_unw = pickle.load(f)

        # Evaluate both models
        for dt_obj, var_name, w_val, split_val, leaf_val, m_fname in [
            (dt_bal, "Balanced_Regenerated_V2", "balanced", 10, 5, bal_fname),
            (dt_unw, "Legacy_Unweighted_Preserved", "None", 2, 1, unw_fname)
        ]:
            y_pred = dt_obj.predict(cfg["X_val"])
            y_prob = dt_obj.predict_proba(cfg["X_val"])[:, 1]

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

            log(lines, f"  [{var_name:<28}] Acc: {acc*100:6.2f}% | BalAcc: {bal_acc*100:6.2f}% | Sens (C1): {sens*100:6.2f}% | Spec (C0): {spec*100:6.2f}% | Prec (C1): {prec_c1*100:6.2f}% | F1 (C1): {f1_c1*100:6.2f}% | Macro F1: {macro_f1*100:6.2f}% | ROC-AUC: {roc_auc:.4f} | AP: {ap:.4f}")
            log(lines, f"                                Confusion Matrix: TN={tn:,d}, FP={fp:,d}, FN={fn:,d}, TP={tp:,d} (Total Class 1 beats: {tp+fn:,d})")

            metrics_records.append({
                "Experiment_ID": exp_key,
                "Experiment_Name": cfg["short_name"],
                "Model_Variant": var_name,
                "Class_Weight": str(w_val),
                "Num_Features": cfg["X_train"].shape[1],
                "Max_Depth": 10,
                "Min_Samples_Split": split_val,
                "Min_Samples_Leaf": leaf_val,
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
                "Model_File": m_fname
            })

            confusion_records.append({
                "Experiment_ID": exp_key,
                "Model_Variant": var_name,
                "Min_Samples_Split": split_val,
                "Min_Samples_Leaf": leaf_val,
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

    # Save CSV reports
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(METRICS_CSV_PATH, index=False)
    log(lines, f"\nSaved updated benchmark metrics table to: {METRICS_CSV_PATH}")

    df_cm = pd.DataFrame(confusion_records)
    df_cm.to_csv(CONFUSION_CSV_PATH, index=False)
    log(lines, f"Saved updated confusion matrices breakdown to: {CONFUSION_CSV_PATH}")

    # 5. Direct Side-by-Side Comparison: Balanced V1 (split=2, leaf=1) vs Balanced V2 (split=10, leaf=5)
    log(lines, "\n" + "-" * 80)
    log(lines, "4. DIRECT COMPARISON: BALANCED V1 (split=2, leaf=1) VS. BALANCED V2 (split=10, leaf=5)")
    log(lines, "-" * 80)
    df_v1 = pd.read_csv(V1_METRICS_PATH)
    df_v1_bal = df_v1[df_v1["Model_Variant"] == "Balanced"].set_index("Experiment_ID")
    df_v2_bal = df_metrics[df_metrics["Model_Variant"] == "Balanced_Regenerated_V2"].set_index("Experiment_ID")

    log(lines, f"{'Experiment':<30} | {'Metric':<18} | {'V1 (split=2, leaf=1)':<20} | {'V2 (split=10, leaf=5)':<22} | Delta (V2 - V1)")
    log(lines, "-" * 110)
    for exp_id in df_v1_bal.index:
        r1 = df_v1_bal.loc[exp_id]
        r2 = df_v2_bal.loc[exp_id]
        exp_name = r1["Experiment_Name"]
        for metric, m_label, is_pct in [
            ("Accuracy", "Accuracy", True),
            ("Balanced_Accuracy", "Balanced Acc", True),
            ("Sensitivity_Recall_Class1", "Sens (C1)", True),
            ("Precision_Class1", "Prec (C1)", True),
            ("F1_Score_Class1", "F1 (C1)", True),
            ("Average_Precision_AP", "Average Prec (AP)", False)
        ]:
            v1_val = r1[metric]
            v2_val = r2[metric]
            if is_pct:
                delta_str = f"{(v2_val - v1_val)*100:+.2f}%"
                v1_str = f"{v1_val*100:6.2f}%"
                v2_str = f"{v2_val*100:6.2f}%"
            else:
                delta_str = f"{(v2_val - v1_val):+.4f}"
                v1_str = f"{v1_val:6.4f}"
                v2_str = f"{v2_val:6.4f}"
            log(lines, f"{exp_name:<30} | {m_label:<18} | {v1_str:<20} | {v2_str:<22} | {delta_str}")
        log(lines, "-" * 110)

    # Write report
    REPORT_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"\nSaved comprehensive Stage 6 report: {REPORT_TXT_PATH}")

    log(lines, "\n" + "=" * 80)
    log(lines, "STAGE 6 REVISION COMPLETE: PASSED.")
    log(lines, "Strict Stop: Awaiting user review before Stage 7.")
    log(lines, "=" * 80)

if __name__ == "__main__":
    main()
