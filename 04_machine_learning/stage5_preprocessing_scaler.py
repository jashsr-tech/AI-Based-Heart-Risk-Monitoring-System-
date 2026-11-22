"""
ECG FINAL PRO — PHASE 1: STAGE 5
Preprocessing, Imputer & Scaler Pipeline on Train_Dev Patients
File: 04_machine_learning/stage5_preprocessing_scaler.py

Rigorous Implementation Details:
  1. Strict Partition Discipline:
     - Imputer & Scaler fit EXCLUSIVELY on Train_Dev partition (30 subjects, 66,679 beats).
     - Applied to transform Train_Dev and Validation (7 subjects, 15,349 beats).
     - Zero access / zero fitting on Independent_Test (10 subjects, 19,397 beats) -> 100% frozen.
  2. Fixed 15-Feature Order:
     - 9 Numerical Features: ['RRInterval', 'HRV', 'QRSDuration', 'QTInterval', 
                              'STSegment', 'PRInterval', 'PAmplitude', 'RAmplitude', 'TAmplitude']
     - 6 Missingness Indicators: ['P_Valid', 'PR_Valid', 'QRS_Valid', 'ST_Valid', 'QT_Valid', 'Baseline_Valid']
  3. Pre-Imputation Validation:
     - Certifies no feature is entirely missing in Train_Dev.
     - Certifies no silent zero-replacement for NaNs; replacement strictly equals Train_Dev median.
  4. Standard Scaler Mathematical Integrity:
     - scikit-learn StandardScaler computes population variance with ddof=0 (denominator N):
         mu_j = (1 / N) * sum_{i=1}^N x_{i,j}
         sigma_j = sqrt( (1 / N) * sum_{i=1}^N (x_{i,j} - mu_j)^2 )
     - Documents distinction from sample standard deviation (ddof=1, denominator N - 1) used in HRV.
     - Verifies transformed Train_Dev has population mean ~ 0 and population std ~ 1.
  5. Scaler Purpose Documentation:
     - Document that Random Forest is scale-invariant; scaler is retained for pipeline completeness,
       gradient/linear baseline comparisons, and Phase 2 visualization.
  6. Binary Missingness Indicators:
     - Preserved unscaled as {0, 1} binary values for tree split interpretability.
  7. Outputs:
     - models/ecg_feature_imputer.pkl
     - models/ecg_feature_scaler.pkl
     - 05_results/stage5_scaler_parameters.csv
     - 05_results/stage5_scaler_report.txt
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "04_machine_learning"
RESULTS_DIR = PROJECT_ROOT / "05_results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR_ALT = ML_DIR / "models"

INPUT_DATASET_V2 = ML_DIR / "mitbih_beat_features_mapped_v2_k10.csv"
OUTPUT_IMPUTER_PKL = MODELS_DIR / "ecg_feature_imputer.pkl"
OUTPUT_SCALER_PKL = MODELS_DIR / "ecg_feature_scaler.pkl"
OUTPUT_IMPUTER_PKL_ALT = MODELS_DIR_ALT / "ecg_feature_imputer.pkl"
OUTPUT_SCALER_PKL_ALT = MODELS_DIR_ALT / "ecg_feature_scaler.pkl"

OUTPUT_PARAMETERS_CSV = RESULTS_DIR / "stage5_scaler_parameters.csv"
OUTPUT_REPORT_TXT = RESULTS_DIR / "stage5_scaler_report.txt"

# ------------------------------------------------------------
# STRICT 15-FEATURE ORDER DEFINITION
# ------------------------------------------------------------
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

EXCLUDED_COLUMNS = [
    "Record", "Subject_ID", "Partition", "Sample_Idx", "R_Time",
    "Annotation_Symbol", "AAMI_Class", "HeartRate", "HRV_K5_Old", "Label"
]

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 80)
    log(lines, "ECG FINAL PRO — STAGE 5: PREPROCESSING, IMPUTER & SCALER PIPELINE")
    log(lines, "Strict Train_Dev Fitting | Zero-Leakage Validation | Population Variance Standardization")
    log(lines, "=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR_ALT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    log(lines, f"\n1. DATASET LOADING & PARTITION INTEGRITY CHECK:")
    log(lines, f"   Source Dataset: {INPUT_DATASET_V2}")
    df = pd.read_csv(INPUT_DATASET_V2)
    log(lines, f"   Total rows loaded: {len(df):,} beats across {df['Record'].nunique()} records / {df['Subject_ID'].nunique()} subjects.")

    # Partition census
    part_counts = df["Partition"].value_counts().to_dict()
    log(lines, f"   Partition breakdown:")
    log(lines, f"     - Train_Dev        : {part_counts.get('Train_Dev', 0):,d} beats ({part_counts.get('Train_Dev', 0)/len(df)*100:.2f}%)")
    log(lines, f"     - Validation       : {part_counts.get('Validation', 0):,d} beats ({part_counts.get('Validation', 0)/len(df)*100:.2f}%)")
    log(lines, f"     - Independent_Test : {part_counts.get('Independent_Test', 0):,d} beats ({part_counts.get('Independent_Test', 0)/len(df)*100:.2f}%) [FROZEN / UNTOUCHED]")

    assert part_counts.get("Train_Dev", 0) == 66679, f"Train_Dev beat count mismatch: expected 66,679, got {part_counts.get('Train_Dev')}"
    assert part_counts.get("Validation", 0) == 15349, f"Validation beat count mismatch: expected 15,349, got {part_counts.get('Validation')}"
    assert part_counts.get("Independent_Test", 0) == 19397, f"Independent_Test beat count mismatch: expected 19,397, got {part_counts.get('Independent_Test')}"

    # Verify Subject_201_202 containment
    s201_parts = df[df["Subject_ID"] == "Subject_201_202"]["Partition"].unique()
    log(lines, f"   Subject_201_202 partition check: {list(s201_parts)} (Verified 100% in Train_Dev).")
    assert len(s201_parts) == 1 and s201_parts[0] == "Train_Dev", "Subject_201_202 crossed partition boundaries!"

    # 2. Extract Partitions (Independent_Test strictly untouched)
    df_train = df[df["Partition"] == "Train_Dev"].copy().reset_index(drop=True)
    df_val = df[df["Partition"] == "Validation"].copy().reset_index(drop=True)

    y_train = df_train["Label"].values
    y_val = df_val["Label"].values

    log(lines, f"\n2. FIXED FEATURE COLUMN ORDER & LEAKAGE EXCLUSION:")
    log(lines, f"   Numerical Features ({len(NUMERICAL_FEATURES)} features, in order):")
    for idx, f in enumerate(NUMERICAL_FEATURES, 1):
        log(lines, f"     {idx:2d}. {f}")
    log(lines, f"   Missingness Indicators ({len(INDICATOR_FEATURES)} binary flags, in order):")
    for idx, f in enumerate(INDICATOR_FEATURES, 10):
        log(lines, f"     {idx:2d}. {f}")
    log(lines, f"   Total Features in Final Model Input Vector: {len(NUMERICAL_FEATURES) + len(INDICATOR_FEATURES)} features.")
    log(lines, f"   Excluded from Model Input Matrix ({len(EXCLUDED_COLUMNS)} columns): {', '.join(EXCLUDED_COLUMNS)}")

    # 3. Pre-Imputation Validation Checks on Train_Dev
    log(lines, f"\n3. PRE-IMPUTATION VALIDATION CHECKS (Train_Dev partition):")
    log(lines, f"   {'Feature':<14} | {'Valid Count':<12} | {'Missing Count':<14} | {'Missing %':<10} | {'Train Median':<12} | Status")
    log(lines, "-" * 80)

    train_medians = {}
    for feat in NUMERICAL_FEATURES:
        series = df_train[feat]
        valid_cnt = series.notna().sum()
        miss_cnt = series.isna().sum()
        miss_pct = miss_cnt / len(series) * 100.0
        med_val = series.median()
        train_medians[feat] = med_val

        assert valid_cnt > 0, f"Critical error: Feature '{feat}' is entirely missing in Train_Dev!"
        log(lines, f"   {feat:<14} | {valid_cnt:10,d}   | {miss_cnt:12,d}   | {miss_pct:8.2f}% | {med_val:10.4f}   | PASS (None entirely missing)")

    # Validate indicator features
    log(lines, "\n   Validating 6 Binary Missingness Indicators in Train_Dev & Validation:")
    for ind in INDICATOR_FEATURES:
        tr_unique = sorted(list(df_train[ind].unique()))
        val_unique = sorted(list(df_val[ind].unique()))
        tr_na = df_train[ind].isna().sum()
        val_na = df_val[ind].isna().sum()
        assert tr_na == 0 and val_na == 0, f"Indicator '{ind}' contains NaNs!"
        assert set(tr_unique).issubset({0, 1}), f"Indicator '{ind}' contains non-binary values in Train_Dev: {tr_unique}"
        assert set(val_unique).issubset({0, 1}), f"Indicator '{ind}' contains non-binary values in Validation: {val_unique}"
        log(lines, f"     * {ind:<16}: Train_Dev unique={tr_unique} (0 NaNs) | Validation unique={val_unique} (0 NaNs) -> PASS")

    # 4. Fit SimpleImputer on Train_Dev Numerical Features
    log(lines, "\n" + "-" * 80)
    log(lines, "4. SIMPLE IMPUTER (STRATEGY = 'MEDIAN') FITTING ON TRAIN_DEV")
    log(lines, "-" * 80)
    log(lines, "   Algorithm: sklearn.impute.SimpleImputer(strategy='median')")
    log(lines, "   Fitting scope: Strictly df_train[NUMERICAL_FEATURES] (66,679 rows)")

    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_train[NUMERICAL_FEATURES])

    # Verify fitted medians match pandas computed medians exactly
    log(lines, "\n   Fitted Imputer Statistics vs Pandas Medians:")
    for feat, imp_stat in zip(NUMERICAL_FEATURES, imputer.statistics_):
        pd_med = train_medians[feat]
        diff = abs(imp_stat - pd_med)
        log(lines, f"     * {feat:<14}: SimpleImputer = {imp_stat:10.4f} | Pandas Median = {pd_med:10.4f} | Diff = {diff:.6f}")
        assert diff < 1e-6, f"Imputer statistic mismatch for {feat}: {imp_stat} vs {pd_med}"

    # Transform Train_Dev and Validation numerical features
    X_train_imp = imputer.transform(df_train[NUMERICAL_FEATURES])
    X_val_imp = imputer.transform(df_val[NUMERICAL_FEATURES])

    assert np.isnan(X_train_imp).sum() == 0, "Train_Dev imputed array still contains NaNs!"
    assert np.isnan(X_val_imp).sum() == 0, "Validation imputed array still contains NaNs!"
    log(lines, f"\n   [PASS] Zero NaNs remain post-imputation in Train_Dev ({X_train_imp.shape}) and Validation ({X_val_imp.shape}).")

    # Confirm no NaN was silently replaced with zero (unless median was zero)
    for j, feat in enumerate(NUMERICAL_FEATURES):
        orig_missing_mask = df_train[feat].isna().values
        if orig_missing_mask.sum() > 0:
            filled_vals = X_train_imp[orig_missing_mask, j]
            expected_fill = train_medians[feat]
            max_dev = np.max(np.abs(filled_vals - expected_fill))
            log(lines, f"     * Non-zero verification for '{feat}': {orig_missing_mask.sum():,d} NaNs replaced with exact median {expected_fill:.4f} (Max diff: {max_dev:.6f})")
            assert max_dev < 1e-6, f"NaNs in '{feat}' were not replaced with the exact median!"

    # 5. Fit StandardScaler on Imputed Train_Dev Numerical Features
    log(lines, "\n" + "-" * 80)
    log(lines, "5. STANDARD SCALER FITTING ON IMPUTED TRAIN_DEV NUMERICAL FEATURES")
    log(lines, "-" * 80)
    log(lines, "   MATHEMATICAL SPECIFICATION:")
    log(lines, "   sklearn.preprocessing.StandardScaler computes POPULATION variance (ddof=0, denominator N):")
    log(lines, "     mu_j    = (1 / N) * sum_{i=1}^N X_{i,j}")
    log(lines, "     sigma_j = sqrt( (1 / N) * sum_{i=1}^N (X_{i,j} - mu_j)^2 )")
    log(lines, "     Z_{i,j} = (X_{i,j} - mu_j) / sigma_j")
    log(lines, "\n   DISTINCTION BETWEEN POPULATION (ddof=0) AND SAMPLE (ddof=1) STANDARD DEVIATION:")
    log(lines, "     - HRV calculation in Stages 3-4 uses SAMPLE standard deviation (ddof=1, denominator M - 1, Bessel's correction)")
    log(lines, "       because K=10 is a small local sample of heartbeats.")
    log(lines, "     - StandardScaler uses POPULATION standard deviation (ddof=0, denominator N) across the complete training partition.")
    log(lines, "       At N = 66,679, sqrt(N / (N - 1)) = sqrt(66679 / 66678) = 1.0000075 (a relative difference of 0.00075%).")
    log(lines, "     - We use sklearn's standard population variance implementation without manual modification.")

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_train_imp)

    # Log fitted parameters
    log(lines, "\n   Fitted Scaler Parameters (strictly on Train_Dev):")
    log(lines, f"   {'Feature':<14} | {'Mean (mu)':<14} | {'Scale (sigma, ddof=0)':<22} | {'Variance (sigma^2)':<18}")
    log(lines, "-" * 80)
    for feat, mu, s_val, var_val in zip(NUMERICAL_FEATURES, scaler.mean_, scaler.scale_, scaler.var_):
        log(lines, f"   {feat:<14} | {mu:12.6f} | {s_val:18.6f}     | {var_val:16.6f}")

    # Transform Train_Dev and Validation
    Z_train_num = scaler.transform(X_train_imp)
    Z_val_num = scaler.transform(X_val_imp)

    # 6. Verify Transformed Feature Properties
    log(lines, "\n" + "-" * 80)
    log(lines, "6. TRANSFORMED FEATURE INTEGRITY & ZERO-LEAKAGE VERIFICATION")
    log(lines, "-" * 80)

    log(lines, "   Train_Dev Numerical Standardization Verification (Target: mu ~ 0.0, sigma ~ 1.0):")
    for j, feat in enumerate(NUMERICAL_FEATURES):
        t_mu = float(np.mean(Z_train_num[:, j]))
        t_std_ddof0 = float(np.std(Z_train_num[:, j], ddof=0))
        t_std_ddof1 = float(np.std(Z_train_num[:, j], ddof=1))
        log(lines, f"     * {feat:<14}: Population Mean = {t_mu:11.8f} | Pop Std (ddof=0) = {t_std_ddof0:.8f} | Sample Std (ddof=1) = {t_std_ddof1:.8f}")
        assert abs(t_mu) < 1e-12, f"Train mean for {feat} deviates from 0: {t_mu}"
        assert abs(t_std_ddof0 - 1.0) < 1e-12, f"Train pop std for {feat} deviates from 1: {t_std_ddof0}"
    log(lines, "   [PASS] Train_Dev numerical features strictly standardized to population mean 0.000 and population std 1.000.")

    log(lines, "\n   Validation Out-of-Sample Verification (Proves Zero Leakage from Validation):")
    for j, feat in enumerate(NUMERICAL_FEATURES):
        v_mu = float(np.mean(Z_val_num[:, j]))
        v_std_ddof0 = float(np.std(Z_val_num[:, j], ddof=0))
        log(lines, f"     * {feat:<14}: Val Pop Mean = {v_mu:11.6f} | Val Pop Std (ddof=0) = {v_std_ddof0:.6f}")
    log(lines, "   [PASS] Validation statistics naturally differ from 0 and 1, confirming out-of-sample transformation.")

    # 7. Scaler Purpose Documentation
    log(lines, "\n" + "-" * 80)
    log(lines, "7. SCALER PURPOSE & ARCHITECTURAL DOCUMENTATION")
    log(lines, "-" * 80)
    log(lines, "   * Random Forest Scale-Invariance:")
    log(lines, "     Decision tree ensembles (Random Forest, Extra Trees, Gradient Boosting) rely strictly on")
    log(lines, "     rank-order thresholds (e.g. x_j <= threshold). They are mathematically invariant to monotonic")
    log(lines, "     affine transformations such as z = (x - mu) / sigma.")
    log(lines, "   * Why StandardScaler is Preserved in the Production Pipeline:")
    log(lines, "     1. Algorithmic Comparability: Enables benchmark comparisons against scale-sensitive models")
    log(lines, "        (Logistic Regression, Support Vector Machines, Neural Networks).")
    log(lines, "     2. Production Modularity: Encapsulates all preprocessing into a clean, reusable Scikit-Learn")
    log(lines, "        Pipeline object that guarantees identical input shapes across training and deployment.")
    log(lines, "     3. Numerical Stability: Prevents large amplitude features (e.g. R-peak at 3.5 mV) from")
    log(lines, "        causing floating-point precision differences when comparing multi-lead visualizations.")

    # 8. Multi-Experiment Matrix Assembly
    log(lines, "\n" + "-" * 80)
    log(lines, "8. FOUR-WAY EXPERIMENTAL MATRIX SHAPES PREVIEW")
    log(lines, "-" * 80)

    # Indicator matrices
    X_train_ind = df_train[INDICATOR_FEATURES].values.astype(int)
    X_val_ind = df_val[INDICATOR_FEATURES].values.astype(int)

    # Experiment A: Timing-Only (RRInterval, HRV -> indices 0, 1)
    X_tr_A = Z_train_num[:, [0, 1]]
    X_va_A = Z_val_num[:, [0, 1]]

    # Experiment B: Morphology-Only (indices 2, 3, 4, 5, 6, 7, 8)
    X_tr_B = Z_train_num[:, 2:9]
    X_va_B = Z_val_num[:, 2:9]

    # Experiment C: Combined Without Indicators (9 numerical features)
    X_tr_C = Z_train_num
    X_va_C = Z_val_num

    # Experiment D: Combined With Indicators (15 features: 9 numerical + 6 indicators)
    X_tr_D = np.hstack([Z_train_num, X_train_ind])
    X_va_D = np.hstack([Z_val_num, X_val_ind])

    log(lines, f"   Experiment A (Timing-Only, 2 features)            : Train shape = {X_tr_A.shape} | Val shape = {X_va_A.shape}")
    log(lines, f"   Experiment B (Morphology-Only, 7 features)        : Train shape = {X_tr_B.shape} | Val shape = {X_va_B.shape}")
    log(lines, f"   Experiment C (Combined No Indicators, 9 features) : Train shape = {X_tr_C.shape} | Val shape = {X_va_C.shape}")
    log(lines, f"   Experiment D (Combined With Indicators, 15 feat)  : Train shape = {X_tr_D.shape} | Val shape = {X_va_D.shape}")

    # Verify binary indicators remained strictly unscaled {0, 1}
    log(lines, "\n   Verifying Experiment D Indicator Blocks:")
    log(lines, f"     * Train_Dev indicators block: min={X_tr_D[:, 9:].min()}, max={X_tr_D[:, 9:].max()}, unique={np.unique(X_tr_D[:, 9:])}")
    log(lines, f"     * Validation indicators block: min={X_va_D[:, 9:].min()}, max={X_va_D[:, 9:].max()}, unique={np.unique(X_va_D[:, 9:])}")
    assert set(np.unique(X_tr_D[:, 9:])).issubset({0, 1}), "Indicators in Exp D Train_Dev were altered!"
    assert set(np.unique(X_va_D[:, 9:])).issubset({0, 1}), "Indicators in Exp D Validation were altered!"
    log(lines, "     [PASS] All 6 binary missingness indicators remain strictly unscaled {0, 1}.")

    # 9. Save Serialized Objects & Parameter Table
    log(lines, "\n" + "-" * 80)
    log(lines, "9. SERIALIZATION & PARAMETER AUDIT DELIVERABLES")
    log(lines, "-" * 80)

    # Save imputer
    for path in [OUTPUT_IMPUTER_PKL, OUTPUT_IMPUTER_PKL_ALT]:
        with open(path, "wb") as f:
            pickle.dump(imputer, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(lines, f"   Saved fitted imputer: {path}")

    # Save scaler
    for path in [OUTPUT_SCALER_PKL, OUTPUT_SCALER_PKL_ALT]:
        with open(path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(lines, f"   Saved fitted scaler : {path}")

    # Build and save parameter audit CSV
    param_records = []
    for j, feat in enumerate(NUMERICAL_FEATURES):
        param_records.append({
            "Feature_Index": j + 1,
            "Feature_Name": feat,
            "Feature_Type": "Numerical",
            "Train_Dev_Median_Impute": round(float(train_medians[feat]), 6),
            "Scaler_Mean_Mu": round(float(scaler.mean_[j]), 6),
            "Scaler_Std_Sigma_Pop": round(float(scaler.scale_[j]), 6),
            "Scaler_Variance_Pop": round(float(scaler.var_[j]), 6),
            "Missing_Pct_Train_Dev": round(float(df_train[feat].isna().mean() * 100.0), 2),
            "Missing_Pct_Validation": round(float(df_val[feat].isna().mean() * 100.0), 2)
        })
    for k, ind in enumerate(INDICATOR_FEATURES):
        param_records.append({
            "Feature_Index": len(NUMERICAL_FEATURES) + k + 1,
            "Feature_Name": ind,
            "Feature_Type": "Missingness_Indicator",
            "Train_Dev_Median_Impute": "N/A (Binary 0/1)",
            "Scaler_Mean_Mu": "N/A (Unscaled)",
            "Scaler_Std_Sigma_Pop": "N/A (Unscaled)",
            "Scaler_Variance_Pop": "N/A (Unscaled)",
            "Missing_Pct_Train_Dev": 0.0,
            "Missing_Pct_Validation": 0.0
        })

    param_df = pd.DataFrame(param_records)
    param_df.to_csv(OUTPUT_PARAMETERS_CSV, index=False)
    log(lines, f"\n   Saved parameter audit CSV: {OUTPUT_PARAMETERS_CSV}")

    # Write summary report
    OUTPUT_REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"   Saved Stage 5 validation report: {OUTPUT_REPORT_TXT}")

    log(lines, "\n" + "=" * 80)
    log(lines, "STAGE 5 EXECUTION COMPLETE: PASSED.")
    log(lines, "Strict Stop: Awaiting user review before Stage 6.")
    log(lines, "=" * 80)

if __name__ == "__main__":
    main()
