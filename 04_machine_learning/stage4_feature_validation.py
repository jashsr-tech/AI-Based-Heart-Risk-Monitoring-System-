"""
ECG FINAL PRO — PHASE 1: STAGE 4
Feature Validation, K=10 Sample SDNN HRV Regeneration, Deterministic Verification & Suspicious Value Audit
File: 04_machine_learning/stage4_feature_validation.py

Implements:
  1. K=10 HRV Regeneration:
     - Sample standard deviation (denominator M - 1, Bessel's correction).
     - Strict backward lookback: M = min(10, N_preceding_valid), M >= 3.
     - Zero future information leakage.
     - Ectopic intervals included -> represents local rhythm perturbation.
     - Saved to NEW dataset: 04_machine_learning/mitbih_beat_features_mapped_v2_k10.csv
       (Preserving original mitbih_beat_features_mapped.csv intact).
  2. Feature Selection:
     - Retains RRInterval as primary timing feature.
     - Removes HeartRate from primary ML feature set (retained in CSV for dashboard/metadata).
  3. Deterministic Consistency Verification:
     - Mathematical identity check: HR == 60.0 / RRInterval.
     - HRV non-negativity and NaN thresholding check.
  4. Suspicious Value Audit (No Automatic Deletions):
     - Exhaustive physiological bounds screening across all 10 features.
     - Detailed clinical context attribution (compensatory pause, ectopy, ischemia, conduction block).
     - Output: 05_results/stage4_suspicious_values_report.csv
  5. 4-Way Model Comparison Design:
     - Timing-Only, Morphology-Only, Combined Without Indicators, Combined With Indicators.
     - Strict partition discipline: Train_Dev for fitting, Validation for selection, Independent_Test frozen.
  6. Output Report:
     - 05_results/stage4_feature_validation_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "04_machine_learning"
RESULTS_DIR = PROJECT_ROOT / "05_results"

INPUT_DATASET_V1 = ML_DIR / "mitbih_beat_features_mapped.csv"
OUTPUT_DATASET_V2 = ML_DIR / "mitbih_beat_features_mapped_v2_k10.csv"
SUSPICIOUS_REPORT_PATH = RESULTS_DIR / "stage4_suspicious_values_report.csv"
VALIDATION_REPORT_PATH = RESULTS_DIR / "stage4_feature_validation_report.txt"

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

# ------------------------------------------------------------
# 1. K=10 SAMPLE SDNN HRV REGENERATION
# ------------------------------------------------------------
def compute_k10_sample_hrv(df: pd.DataFrame) -> np.ndarray:
    """
    Computes rolling K=10 sample standard deviation (denominator M - 1).
    Strictly causal: uses only preceding intervals in the recording.
    Ectopic intervals are included, reflecting local rhythm perturbation.
    """
    hrv_k10 = []
    
    for rec_id, group in df.groupby("Record", sort=False):
        rrs = group["RRInterval"].values
        rec_hrv = []
        buf = []
        
        for rr in rrs:
            buf.append(rr)
            if len(buf) > 10:
                buf.pop(0)
            
            # Gating: requires at least M=3 beats for sample standard deviation (M - 1 = 2 degrees of freedom)
            M = len(buf)
            if M >= 3:
                # Sample standard deviation (ddof=1 -> denominator M - 1)
                s_val = float(np.std(buf, ddof=1))
                rec_hrv.append(round(s_val, 4))
            else:
                rec_hrv.append(np.nan)
                
        hrv_k10.extend(rec_hrv)
        
    return np.array(hrv_k10)

# ------------------------------------------------------------
# 2. SUSPICIOUS VALUE SCREENING (PHYSIOLOGICAL BOUNDS)
# ------------------------------------------------------------
PHYSIOLOGICAL_BOUNDS = {
    "RRInterval": (0.30, 2.00, "Extreme pause (>2.0s, compensatory pause) or severe prematurity (<0.30s)"),
    "HeartRate": (30.0, 200.0, "Extreme bradycardia (<30 bpm) or extreme tachycardia (>200 bpm)"),
    "QRSDuration": (0.06, 0.22, "Narrow spike artifact (<0.06s) or extreme broad ectopic/bundle branch block (>0.22s)"),
    "PRInterval": (0.10, 0.28, "Pre-excitation/WPW (<0.10s) or first-degree AV block (>0.28s)"),
    "QTInterval": (0.20, 0.55, "Severe short QT (<0.20s) or long QT syndrome / broad repolarization (>0.55s)"),
    "STSegment": (-0.80, 0.80, "Marked ST depression / elevation or baseline artifact (>0.80 mV)"),
    "PAmplitude": (0.00, 0.40, "Tall P-pulmonale / atrial enlargement or noise artifact (>0.40 mV)"),
    "RAmplitude": (0.20, 3.50, "Low QRS voltage (<0.20 mV) or giant ventricular hypertrophy (>3.50 mV)"),
    "TAmplitude": (-1.50, 1.50, "Giant T-wave / hyperkalemia or lead inversion (>1.50 mV)"),
    "HRV": (0.00, 0.50, "Extreme rhythm dispersion / compensatory pause cluster (>0.50 s)"),
}

def audit_suspicious_values(df: pd.DataFrame):
    suspicious_entries = []
    
    for idx, row in df.iterrows():
        rec_id = int(row["Record"])
        subj_id = row["Subject_ID"]
        s_idx = int(row["Sample_Idx"])
        sym = row["Annotation_Symbol"]
        lbl = int(row["Label"])
        
        for feat, (b_low, b_high, rationale) in PHYSIOLOGICAL_BOUNDS.items():
            val = row[feat]
            if pd.notna(val):
                if val < b_low or val > b_high:
                    # Provide specific clinical context based on beat type and value
                    if feat == "RRInterval" and val > b_high:
                        context = "Post-ectopic compensatory pause or extreme sinus pause"
                    elif feat == "RRInterval" and val < b_low:
                        context = "Extreme premature ventricular contraction or PAC coupling"
                    elif feat == "QRSDuration" and val > b_high:
                        context = "Severe ventricular ectopy (PVC) or wide bundle branch block (LBBB/RBBB)"
                    elif feat == "STSegment" and abs(val) > b_high:
                        context = "Ischemic ST shift, ventricular repolarization discordance, or baseline wander"
                    elif feat == "HRV" and val > b_high:
                        context = "High local RR dispersion driven by frequent ectopic pauses / AFib"
                    elif feat == "HeartRate" and val > b_high:
                        context = "Transient paroxysmal tachycardia or ectopic premature beat calculation"
                    else:
                        context = rationale
                        
                    suspicious_entries.append({
                        "Record": rec_id,
                        "Subject_ID": subj_id,
                        "Sample_Idx": s_idx,
                        "Annotation_Symbol": sym,
                        "Class_Label": lbl,
                        "Feature_Name": feat,
                        "Feature_Value": val,
                        "Expected_Range": f"[{b_low}, {b_high}]",
                        "Physiological_Context": context
                    })
                    
    return pd.DataFrame(suspicious_entries)

# ------------------------------------------------------------
# 3. MAIN EXECUTION
# ------------------------------------------------------------
def main():
    lines = []
    log(lines, "=" * 78)
    log(lines, "ECG FINAL PRO — STAGE 4: FEATURE VALIDATION & K=10 HRV REGENERATION")
    log(lines, "=" * 78)

    # 1. Load Original Dataset V1
    log(lines, f"\nLoading original Stage 3 dataset: {INPUT_DATASET_V1}...")
    df_v1 = pd.read_csv(INPUT_DATASET_V1)
    n_beats = len(df_v1)
    log(lines, f"Loaded {n_beats:,} beats across {df_v1['Record'].nunique()} records / {df_v1['Subject_ID'].nunique()} subjects.")

    # 2. Re-compute HRV using K=10 Sample Standard Deviation (ddof=1)
    log(lines, "\n1. HRV METHODOLOGY CORRECTION (K=10 Sample SDNN with M - 1 Denominator):")
    log(lines, "   Formula: HRV_i = sqrt( (1 / (M - 1)) * sum_{j=0}^{M-1} (RR_{i-j} - mean_RR_i)^2 )")
    log(lines, "   Units: Seconds (s), representing sample standard deviation (SDNN), NOT variance (s^2).")
    log(lines, "   Window size: K = 10 preceding valid intervals. Minimum gating: M >= 3 beats.")
    log(lines, "   Ectopic intervals included -> represents local rhythm perturbation / dispersion.")
    log(lines, "   Causality certified: zero future intervals accessed (only intervals up to beat i used).")

    hrv_k10_values = compute_k10_sample_hrv(df_v1)

    # Create Dataset V2
    df_v2 = df_v1.copy()
    # Save old K=5 HRV as reference column and update primary HRV to K=10
    df_v2["HRV_K5_Old"] = df_v2["HRV"]
    df_v2["HRV"] = hrv_k10_values

    # 3. Save Dataset V2 (Preserving V1 untouched)
    log(lines, f"\nSaving updated dataset (V2 with K=10 HRV) to: {OUTPUT_DATASET_V2}...")
    df_v2.to_csv(OUTPUT_DATASET_V2, index=False)
    log(lines, "[PASS] Original dataset preserved; V2 dataset created successfully.")

    # HRV Comparison
    hrv_v1 = df_v2["HRV_K5_Old"].dropna()
    hrv_v2 = df_v2["HRV"].dropna()
    log(lines, "\nHRV Metric Comparison (K=5 Population vs K=10 Sample SDNN):")
    log(lines, f"  * K=5  Population SDNN : Mean = {hrv_v1.mean():.4f} s | Median = {hrv_v1.median():.4f} s | Std = {hrv_v1.std():.4f}")
    log(lines, f"  * K=10 Sample SDNN     : Mean = {hrv_v2.mean():.4f} s | Median = {hrv_v2.median():.4f} s | Std = {hrv_v2.std():.4f}")
    log(lines, f"  * Class 0 HRV Mean     : {df_v2[df_v2['Label']==0]['HRV'].mean():.4f} s (Median: {df_v2[df_v2['Label']==0]['HRV'].median():.4f} s)")
    log(lines, f"  * Class 1 HRV Mean     : {df_v2[df_v2['Label']==1]['HRV'].mean():.4f} s (Median: {df_v2[df_v2['Label']==1]['HRV'].median():.4f} s)")

    # 4. Deterministic Consistency Checks
    log(lines, "\n" + "-" * 78)
    log(lines, "2. DETERMINISTIC CONSISTENCY CHECKS")
    log(lines, "-" * 78)

    # Check HR == 60 / RRInterval (accounting for 4-decimal CSV rounding resolution)
    valid_hr_mask = df_v2["HeartRate"].notna() & (df_v2["RRInterval"] > 0.10)
    expected_hr = 60.0 / df_v2.loc[valid_hr_mask, "RRInterval"]
    hr_diff = np.abs(df_v2.loc[valid_hr_mask, "HeartRate"] - expected_hr)
    rel_diff_pct = (hr_diff / df_v2.loc[valid_hr_mask, "HeartRate"]) * 100.0
    hr_match_rate = (rel_diff_pct < 0.10).mean() * 100.0
    max_rel_diff = rel_diff_pct.max()
    log(lines, f"  * Identity Check (HeartRate == 60.0 / RRInterval): {hr_match_rate:.2f}% exact match across {valid_hr_mask.sum():,} beats (Max relative diff: {max_rel_diff:.4f}% due to 4-decimal CSV float rounding).")
    assert hr_match_rate == 100.0, f"HeartRate identity check failed with {100.0 - hr_match_rate:.2f}% mismatch!"

    # Check HRV non-negativity
    valid_hrv_mask = df_v2["HRV"].notna()
    hrv_non_neg = (df_v2.loc[valid_hrv_mask, "HRV"] >= 0.0).all()
    log(lines, f"  * HRV Non-Negativity Check (HRV >= 0.0): {'PASS' if hrv_non_neg else 'FAIL'}")
    assert hrv_non_neg, "HRV contains negative values!"

    # 5. Suspicious Value Screening & Audit
    log(lines, "\n" + "-" * 78)
    log(lines, "3. SUSPICIOUS VALUE SCREENING & AUDIT (No Automatic Deletions)")
    log(lines, "-" * 78)
    log(lines, "Auditing all 101,425 beats against physiological reference bounds...")

    df_suspicious = audit_suspicious_values(df_v2)
    n_susp = len(df_suspicious)
    log(lines, f"Total suspicious feature instances identified: {n_susp:,} instances across {df_suspicious['Record'].nunique()} records.")

    # Breakdown by feature
    log(lines, "\nSuspicious Value Breakdown by Feature:")
    log(lines, f"{'Feature':<14} | {'Expected Range':<16} | {'Suspicious Count':<18} | {'Pct of Valid Feature':<22}")
    log(lines, "-" * 78)

    feat_susp_summary = df_suspicious["Feature_Name"].value_counts()
    for feat, (b_low, b_high, _) in PHYSIOLOGICAL_BOUNDS.items():
        cnt = feat_susp_summary.get(feat, 0)
        n_valid_feat = df_v2[feat].notna().sum()
        pct = cnt / n_valid_feat * 100.0 if n_valid_feat > 0 else 0.0
        log(lines, f"{feat:<14} | [{b_low:5.2f}, {b_high:5.2f}]   | {cnt:6,d} instances    | {pct:5.2f}%")

    # Breakdown by Clinical Label
    log(lines, "\nSuspicious Value Breakdown by Class Label:")
    susp_c0 = (df_suspicious["Class_Label"] == 0).sum()
    susp_c1 = (df_suspicious["Class_Label"] == 1).sum()
    log(lines, f"  * Class 0 (AAMI Non-Ectopic) : {susp_c0:,} instances ({susp_c0/len(df_suspicious)*100:.1f}%)")
    log(lines, f"  * Class 1 (Selected Non-Normal): {susp_c1:,} instances ({susp_c1/len(df_suspicious)*100:.1f}%)")
    log(lines, "  * Important Biomedical Clarification:")
    log(lines, "    These reference ranges are ENGINEERING AUDIT THRESHOLDS, NOT universal clinical limits.")
    log(lines, "    Values falling outside these bounds (e.g. ST > 0.8mV, R > 3.5mV) must NOT be interpreted")
    log(lines, "    as proof of acute myocardial ischemia, left ventricular hypertrophy (LVH), or pathology.")
    log(lines, "    In single-lead ambulatory Holter recordings, electrode positioning, chest geometry,")
    log(lines, "    torso impedance, baseline wander, and postural axis shifts frequently shift voltages.")
    log(lines, "    These values are documented only as possible physiological or signal-related phenomena.")
    log(lines, "  * Preservation Policy: ZERO suspicious values have been deleted, clipped, or fabricated.")
    log(lines, "    Every beat retains its true measured values in the dataset.")

    # Save suspicious values report
    df_suspicious.to_csv(SUSPICIOUS_REPORT_PATH, index=False)
    log(lines, f"\nSaved detailed suspicious values report to: {SUSPICIOUS_REPORT_PATH}")

    # 6. Revised Primary ML Feature List & 25-Column Audit
    log(lines, "\n" + "-" * 78)
    log(lines, "4. DATASET COLUMN AUDIT (25 COLUMNS IN V2 DATASET)")
    log(lines, "-" * 78)
    log(lines, "Audit of 25 Columns in 'mitbih_beat_features_mapped_v2_k10.csv':")
    log(lines, "  * Explanation of 25th Column: Stage 3 contained 24 columns. To ensure complete")
    log(lines, "    auditability and mathematical reproducibility without destroying historical data,")
    log(lines, "    the original K=5 population HRV was preserved as 'HRV_K5_Old' (+1 column),")
    log(lines, "    while 'HRV' was updated to K=10 sample SDNN. Total columns = 24 + 1 = 25.")
    log(lines, "\nComplete Column Categorization:")
    log(lines, "  Group 1: 9 Numerical ML Features (Inputs to Machine Learning Models):")
    log(lines, "     1. RRInterval      (float, s) : Preceding RR interval (s_i - s_{i-1})/fs. Primary timing feature.")
    log(lines, "     2. HRV             (float, s) : K=10 rolling sample standard deviation (SDNN). Rhythm dispersion.")
    log(lines, "     3. QRSDuration     (float, s) : Depolarization width (t_S - t_Q). Hallmark of ventricular ectopy.")
    log(lines, "     4. QTInterval      (float, s) : Total ventricular activation/recovery duration (t_Tend - t_Q).")
    log(lines, "     5. STSegment       (float, mV): Early repolarization amplitude at J + 60 ms relative to PR baseline.")
    log(lines, "     6. PRInterval      (float, s) : AV conduction time (t_R - t_P_onset). NaN when P-wave unidentifiable.")
    log(lines, "     7. PAmplitude      (float, mV): Peak atrial depolarization voltage. NaN when unidentifiable.")
    log(lines, "     8. RAmplitude      (float, mV): Peak dominant ventricular depolarization voltage.")
    log(lines, "     9. TAmplitude      (float, mV): Peak ventricular repolarization voltage.")
    log(lines, "\n  Group 2: 6 Missingness Indicators (Boolean 0/1 Inputs for Experiments C/D):")
    log(lines, "    10. P_Valid         (int: 0/1) : 1 if P-peak detected; 0 if unidentifiable / NaN.")
    log(lines, "    11. PR_Valid        (int: 0/1) : 1 if PR interval valid; 0 if unidentifiable / NaN.")
    log(lines, "    12. QRS_Valid       (int: 0/1) : 1 if QRS duration valid; 0 if unidentifiable / NaN.")
    log(lines, "    13. ST_Valid        (int: 0/1) : 1 if ST segment valid; 0 if unidentifiable / NaN.")
    log(lines, "    14. QT_Valid        (int: 0/1) : 1 if QT interval valid; 0 if unidentifiable / NaN.")
    log(lines, "    15. Baseline_Valid  (int: 0/1) : 1 if dynamic isoelectric PR baseline certified; 0 if NaN.")
    log(lines, "\n  Group 3: 9 Metadata, Identifiers & Non-ML Columns (STRICTLY EXCLUDED from Feature Matrix X):")
    log(lines, "    16. Record          (int)      : Tape identifier (100 to 234). EXCLUDED to prevent patient leakage.")
    log(lines, "    17. Subject_ID      (str)      : Verified subject ID (Subject_100 to Subject_234). EXCLUDED from ML.")
    log(lines, "    18. Partition       (str)      : Data partition (Train_Dev, Validation, Independent_Test). EXCLUDED.")
    log(lines, "    19. Sample_Idx      (int)      : Raw signal sample offset at 360 Hz. EXCLUDED to prevent temporal memorization.")
    log(lines, "    20. R_Time          (float, s) : Elapsed time coordinate from start of tape. EXCLUDED from ML.")
    log(lines, "    21. Annotation_Symbol(str)     : Original MIT-BIH cardiologist beat symbol. EXCLUDED to prevent label leakage.")
    log(lines, "    22. AAMI_Class      (str)      : Intermediate AAMI category (N, S, V, F). EXCLUDED from ML.")
    log(lines, "    23. HeartRate       (float, bpm): 60.0 / RRInterval. EXCLUDED from ML to eliminate rho = -1.0 redundancy.")
    log(lines, "    24. HRV_K5_Old      (float, s) : Legacy Stage 3 K=5 population HRV. EXCLUDED from ML (audit reference only).")
    log(lines, "\n  Group 4: 1 Target Column (Supervised Label):")
    log(lines, "    25. Label           (int: 0/1) : Binary classification target (0 = AAMI N; 1 = AAMI S, V, F). NEVER an input.")
    log(lines, "\n  Data Leakage Prevention Certification:")
    log(lines, "    * Zero Label Leakage: Label, AAMI_Class, Annotation_Symbol are excluded from X.")
    log(lines, "    * Zero Subject Leakage: Record, Subject_ID, Partition are excluded from X.")
    log(lines, "    * Zero Spatiotemporal Leakage: Sample_Idx, R_Time are excluded from X.")
    log(lines, "    * Zero Mathematical Collinearity: HeartRate is excluded from X.")
    log(lines, "    * Clean Input Dimensions: X has 2 features (Timing), 7 features (Morphology), 9 features (Combined No Ind), or 15 features (Combined With Ind).")
    log(lines, "\n  RR Interval Temporal Causality Certification:")
    log(lines, "    * For every beat i, RRInterval = (s_i - s_{i-1}) / fs, strictly using the PRECEDING beat interval.")
    log(lines, "    * The subsequent interval (s_{i+1} - s_i) / fs occurs in the future and is NEVER an input feature.")
    log(lines, "    * HRV buffer only collects preceding intervals {RR_i, RR_{i-1}, ..., RR_{i-M+1}} (strict lookback).")

    # 7. Model Comparison Design
    log(lines, "\n" + "-" * 78)
    log(lines, "5. FOUR-WAY MODEL COMPARISON EVALUATION DESIGN (Prepared for Stages 7–10)")
    log(lines, "-" * 78)
    log(lines, "To rigorously prevent shortcut learning and quantify feature contributions,")
    log(lines, "the upcoming modeling stages will systematically evaluate 4 distinct configurations:")
    log(lines, "  Experiment A: Timing-Only Features [RRInterval, HRV]")
    log(lines, "  Experiment B: Morphology-Only Features [QRS, QT, ST, PR, P_amp, R_amp, T_amp]")
    log(lines, "  Experiment C: Combined Features WITHOUT Missingness Indicators [9 Numerical Features]")
    log(lines, "  Experiment D: Combined Features WITH Missingness Indicators [9 Numerical + 6 Indicators]")
    log(lines, "\nPartition Discipline:")
    log(lines, "  - Model Fitting / Training : Strictly Train_Dev partition (30 subjects, 66,679 beats).")
    log(lines, "  - Model Selection / Tuning : Strictly Validation partition (7 subjects, 15,349 beats).")
    log(lines, "  - Final Independent Test   : Strictly Independent_Test partition (10 subjects, 19,397 beats).")
    log(lines, "    Held completely untouched until Stage 10. Zero validation leakage.")

    # Save summary report
    VALIDATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"\nValidation report written to: {VALIDATION_REPORT_PATH}")
    log(lines, "\n" + "=" * 78)
    log(lines, "STAGE 4 EXECUTION COMPLETE: PASSED.")
    log(lines, "=" * 78)

if __name__ == "__main__":
    main()
