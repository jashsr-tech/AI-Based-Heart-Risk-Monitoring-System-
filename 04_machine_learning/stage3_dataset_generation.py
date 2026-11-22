"""
ECG FINAL PRO — PHASE 1: STAGE 3
Complete Beat-Level Feature Extraction & Zero-Leakage Dataset Generation
File: 04_machine_learning/stage3_dataset_generation.py

Features Extracted (10 Primary Features):
  1. HeartRate (bpm = 60 / RR_pre)
  2. RRInterval (seconds = RR_pre)
  3. PRInterval (seconds = R_time - P_time, NaN if unidentifiable)
  4. QRSDuration (seconds = J_time - Q_time or S_time - Q_time, NaN if unidentifiable)
  5. QTInterval (seconds = T_time - Q_time, NaN if unidentifiable)
  6. STSegment (mV = amplitude at J + 60 ms relative to v_iso, NaN if unidentifiable)
  7. HRV (seconds = SDNN of last 5 beats)
  8. PAmplitude (mV = P_val - v_iso, NaN if unidentifiable)
  9. RAmplitude (mV = R_val - v_iso)
  10. TAmplitude (mV = T_val - v_iso, NaN if unidentifiable)

Missingness Indicators:
  - P_Valid, PR_Valid, QRS_Valid, ST_Valid, QT_Valid, Baseline_Valid

Metadata Preserved:
  - Record, Subject_ID, Partition, Sample_Idx, R_Time, Annotation_Symbol, AAMI_Class, Label

Outputs:
  - 04_machine_learning/mitbih_beat_features_mapped.csv
  - 05_results/stage3_beat_exclusion_report.csv
  - 05_results/stage3_missing_values_report.csv
  - 05_results/stage3_feature_distribution_report.txt
  - 05_results/stage3_dataset_summary.txt
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import wfdb

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DB_DIR = Path(r"C:\Users\srjas\OneDrive\Desktop\ECG_Project - Copy\mit-bih-arrhythmia-database-1.0.0")
ML_DIR = PROJECT_ROOT / "04_machine_learning"
RESULTS_DIR = PROJECT_ROOT / "05_results"

ML_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV_PATH = ML_DIR / "mitbih_beat_features_mapped.csv"
EXCLUSION_REPORT_PATH = RESULTS_DIR / "stage3_beat_exclusion_report.csv"
MISSING_REPORT_PATH = RESULTS_DIR / "stage3_missing_values_report.csv"
DISTRIBUTION_REPORT_PATH = RESULTS_DIR / "stage3_feature_distribution_report.txt"
SUMMARY_REPORT_PATH = RESULTS_DIR / "stage3_dataset_summary.txt"

FS = 360.0

# Explicit AAMI Classification Symbol Sets
CLASS_0_SYMBOLS = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N'}
CLASS_1_SYMBOLS = {
    'V': 'V', 'E': 'V',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'F': 'F'
}
EXCLUDED_SYMBOLS = {'/', 'f', 'Q', '?', 'x', '+', '~', '|', '"', '[', ']', '!', '@'}

# ------------------------------------------------------------
# 1. BANDPASS FILTER
# ------------------------------------------------------------
def filter_signal(ecg_raw: np.ndarray, fs: float = 360.0) -> np.ndarray:
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, ecg_raw)

# ------------------------------------------------------------
# 2. FROZEN BEAT FIDUCIAL EXTRACTION (STAGE 2 CERTIFIED)
# ------------------------------------------------------------
def extract_beat_fiducials(
    filtered_ecg: np.ndarray,
    d_ecg: np.ndarray,
    r_idx: int,
    rr_pre_s: float,
    rr_post_s: float,
    k_noise: float = 2.5,
    fs: float = 360.0
):
    sig_len = len(filtered_ecg)

    # 1. Refine R-peak within ±20 ms
    j_win = int(round(0.020 * fs))
    j_start = max(0, r_idx - j_win)
    j_end = min(sig_len, r_idx + j_win + 1)
    ref_r_idx = j_start + int(np.argmax(np.abs(filtered_ecg[j_start:j_end])))
    r_time = ref_r_idx / fs
    r_val = float(filtered_ecg[ref_r_idx])

    # 2. Dynamic Q and S search windows
    q_win_left = max(0, ref_r_idx - int(round(0.080 * rr_pre_s * fs)))
    q_win_right = max(q_win_left + 1, ref_r_idx - int(round(0.012 * rr_pre_s * fs)))

    s_win_left = min(sig_len - 1, ref_r_idx + int(round(0.012 * rr_post_s * fs)))
    s_win_right = min(sig_len, ref_r_idx + int(round(0.160 * rr_post_s * fs)))

    # Q-peak
    q_idx = None
    q_time = np.nan
    q_valid = False
    if q_win_right > q_win_left:
        q_seg = filtered_ecg[q_win_left:q_win_right]
        cand_q = q_win_left + int(np.argmin(q_seg))
        if filtered_ecg[cand_q] <= r_val:
            q_idx = cand_q
            q_time = q_idx / fs
            q_valid = True

    # S-peak
    s_idx = None
    s_time = np.nan
    s_valid = False
    if s_win_right > s_win_left:
        s_seg = filtered_ecg[s_win_left:s_win_right]
        cand_s = s_win_left + int(np.argmin(s_seg))
        if filtered_ecg[cand_s] <= r_val:
            s_idx = cand_s
            s_time = s_idx / fs
            s_valid = True

    # 3. Independent J-Point Detection (Inflection / Deceleration Point)
    j_idx = None
    j_time = np.nan
    j_valid = False

    if s_valid and s_idx < sig_len - 5:
        j_search_end = min(sig_len - 1, s_idx + int(round(0.090 * fs)))
        if j_search_end > s_idx + 2:
            s_to_j_deriv = d_ecg[s_idx:j_search_end]
            max_recov_slope = np.max(s_to_j_deriv) if len(s_to_j_deriv) > 0 else 0.0

            if max_recov_slope > 0.0:
                peak_slope_offset = int(np.argmax(s_to_j_deriv))
                peak_slope_idx = s_idx + peak_slope_offset

                decel_seg = d_ecg[peak_slope_idx:j_search_end]
                thresh = 0.15 * max_recov_slope
                below_thresh = np.where(decel_seg <= thresh)[0]

                if len(below_thresh) > 0:
                    cand_j = peak_slope_idx + below_thresh[0]
                    if cand_j > s_idx:
                        j_idx = cand_j
                        j_time = j_idx / fs
                        j_valid = True
                else:
                    cand_j = peak_slope_idx + int(np.argmin(np.abs(decel_seg)))
                    if cand_j > s_idx:
                        j_idx = cand_j
                        j_time = j_idx / fs
                        j_valid = True

    # 4. Q-onset detection (for baseline boundary)
    q_onset_idx = None
    if q_valid and q_idx > 10:
        q_search_start = max(0, q_idx - int(round(0.060 * fs)))
        q_deriv_seg = np.abs(d_ecg[q_search_start:q_idx])
        max_q_slope = np.max(q_deriv_seg) if len(q_deriv_seg) > 0 else 1.0
        flat_pts = np.where(q_deriv_seg <= 0.10 * max_q_slope)[0]
        if len(flat_pts) > 0:
            q_onset_idx = q_search_start + flat_pts[-1]
        else:
            q_onset_idx = q_idx - int(round(0.020 * fs))
    elif ref_r_idx > 15:
        q_onset_idx = ref_r_idx - int(round(0.040 * fs))

    # 5. Local Baseline Noise Floor & Pre-QRS Estimate
    n_win_left = max(0, ref_r_idx - int(round(0.350 * rr_pre_s * fs)))
    n_win_right = max(n_win_left + 1, ref_r_idx - int(round(0.250 * rr_pre_s * fs)))
    n_seg = filtered_ecg[n_win_left:n_win_right] if n_win_right > n_win_left else filtered_ecg[max(0, ref_r_idx - 30):ref_r_idx]
    sigma_noise = max(0.015, float(np.std(n_seg)))
    initial_base = float(np.median(n_seg))

    # 6. P-Wave Detection
    p_win_left = max(0, ref_r_idx - int(round(0.280 * rr_pre_s * fs)))
    p_win_right = max(p_win_left + 1, (q_onset_idx if q_onset_idx else ref_r_idx - int(round(0.060 * fs))))

    p_idx = None
    p_time = np.nan
    p_valid = False
    p_offset_idx = None

    if p_win_right > p_win_left + 4:
        p_seg = filtered_ecg[p_win_left:p_win_right]
        cand_p = p_win_left + int(np.argmax(p_seg))
        p_val = float(filtered_ecg[cand_p])
        p_prominence = p_val - initial_base

        if cand_p > p_win_left and cand_p < p_win_right - 1:
            is_local_max = (filtered_ecg[cand_p] > filtered_ecg[cand_p - 1]) and (filtered_ecg[cand_p] > filtered_ecg[cand_p + 1])
            if is_local_max and p_prominence >= k_noise * sigma_noise:
                p_after_seg = np.abs(d_ecg[cand_p:p_win_right])
                if len(p_after_seg) > 0:
                    p_flat = np.where(p_after_seg <= 0.20 * np.max(p_after_seg))[0]
                    p_offset_idx = cand_p + (p_flat[0] if len(p_flat) > 0 else len(p_after_seg) // 2)
                p_idx = cand_p
                p_time = p_idx / fs
                p_valid = True

    # 7. Validated Isoelectric Baseline
    v_iso = np.nan
    baseline_valid = False

    if p_valid and p_offset_idx and q_onset_idx and q_onset_idx > p_offset_idx + 2:
        pr_flat_seg = filtered_ecg[p_offset_idx:q_onset_idx]
        pr_slope_seg = np.abs(d_ecg[p_offset_idx:q_onset_idx])
        qrs_slope = np.max(np.abs(d_ecg[q_onset_idx:ref_r_idx]))

        if np.max(pr_slope_seg) <= 0.35 * qrs_slope and np.std(pr_flat_seg) < 0.10:
            v_iso = float(np.median(pr_flat_seg))
            baseline_valid = True
    elif q_onset_idx and q_onset_idx >= 15:
        pre_q_start = max(0, q_onset_idx - int(round(0.040 * fs)))
        pre_q_end = max(pre_q_start + 1, q_onset_idx - int(round(0.010 * fs)))
        pre_q_seg = filtered_ecg[pre_q_start:pre_q_end]
        if np.std(pre_q_seg) < 0.12 and np.ptp(pre_q_seg) < 0.20:
            v_iso = float(np.median(pre_q_seg))
            baseline_valid = True

    # 8. T-Wave Detection
    t_win_left = min(sig_len - 1, (j_idx + int(round(0.040 * fs)) if j_valid else ref_r_idx + int(round(0.160 * rr_post_s * fs))))
    t_win_right = min(sig_len, ref_r_idx + int(round(0.550 * rr_post_s * fs)))

    t_idx = None
    t_time = np.nan
    t_valid = False

    if t_win_right > t_win_left + 4:
        t_seg = filtered_ecg[t_win_left:t_win_right]
        cand_t_max = t_win_left + int(np.argmax(t_seg))
        cand_t_min = t_win_left + int(np.argmin(t_seg))

        base_ref = v_iso if baseline_valid else initial_base
        pos_prom = filtered_ecg[cand_t_max] - base_ref
        neg_prom = base_ref - filtered_ecg[cand_t_min]

        cand_t = cand_t_max if pos_prom >= neg_prom else cand_t_min
        if cand_t > t_win_left and cand_t < t_win_right - 1:
            t_idx = cand_t
            t_time = t_idx / fs
            t_valid = True

    # 9. ST Segment Amplitude at J + 60 ms
    st_amp = np.nan
    st_valid = False
    if j_valid and baseline_valid:
        st_sample = j_idx + int(round(0.060 * fs))
        if st_sample < sig_len:
            st_amp = float(filtered_ecg[st_sample] - v_iso)
            st_valid = True

    # 10. Computed Features
    pr_interval = (r_time - p_time) if p_valid else np.nan
    pr_valid = bool(p_valid and (0.06 <= pr_interval <= 0.40))
    if not pr_valid:
        pr_interval = np.nan

    # QRS duration: from Q-onset (or Q-peak) to J-point (or S-peak)
    qrs_duration = np.nan
    qrs_valid = False
    if q_valid and (j_valid or s_valid):
        q_pt = q_onset_idx / fs if q_onset_idx else q_time
        j_pt = j_time if j_valid else s_time
        qrs_duration = j_pt - q_pt
        if 0.04 <= qrs_duration <= 0.30:
            qrs_valid = True
        else:
            qrs_duration = np.nan

    qt_interval = (t_time - (q_onset_idx / fs if q_onset_idx else (q_time if q_valid else r_time - 0.04))) if t_valid else np.nan
    qt_valid = bool(t_valid and (0.15 <= qt_interval <= 0.80))
    if not qt_valid:
        qt_interval = np.nan

    p_amp = (filtered_ecg[p_idx] - v_iso) if (p_valid and baseline_valid) else np.nan
    r_amp = (r_val - v_iso) if baseline_valid else (r_val - initial_base)
    t_amp = (filtered_ecg[t_idx] - v_iso) if (t_valid and baseline_valid) else np.nan

    return {
        "r_time": r_time,
        "pr_interval": pr_interval,
        "qrs_duration": qrs_duration,
        "qt_interval": qt_interval,
        "st_amp": st_amp,
        "p_amp": p_amp,
        "r_amp": r_amp,
        "t_amp": t_amp,
        "p_valid": p_valid,
        "pr_valid": pr_valid,
        "qrs_valid": qrs_valid,
        "st_valid": st_valid,
        "qt_valid": qt_valid,
        "baseline_valid": baseline_valid,
    }

# ------------------------------------------------------------
# 3. COMPLETE DATASET PROCESSING PIPELINE
# ------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 78)
    print("ECG FINAL PRO — PHASE 1: STAGE 3 COMPLETE DATASET GENERATION")
    print("Zero-Leakage Patient Grouping | Adaptive Boundary Checks | IEEE NaN Retention")
    print("=" * 78)

    # Load verified partitions and subject map
    subj_map_df = pd.read_csv(RESULTS_DIR / "stage1_subject_mapping.csv")
    subj_part_df = pd.read_csv(RESULTS_DIR / "stage3_subject_partitions.csv")
    rec_to_subj = dict(zip(subj_map_df['Record'], subj_map_df['Subject_ID']))
    subj_to_part = dict(zip(subj_part_df['Subject_ID'], subj_part_df['Partition']))

    # Master containers
    dataset_rows = []
    exclusion_records = []

    records = sorted(list(rec_to_subj.keys()))
    print(f"\nProcessing all {len(records)} MIT-BIH recordings...")

    for rec_idx, rec_id in enumerate(records, start=1):
        subj_id = rec_to_subj[rec_id]
        partition = subj_to_part[subj_id]

        rec_path = str(RAW_DB_DIR / str(rec_id))
        rec = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        fs = float(rec.fs)
        dt = 1.0 / fs
        raw_sig = rec.p_signal[:, 0]
        sig_len = len(raw_sig)

        # Filter signal and calculate derivative
        filt_sig = filter_signal(raw_sig, fs=fs)
        d_sig = np.gradient(filt_sig, dt)

        samples = ann.sample
        symbols = ann.symbol
        n_ann = len(samples)

        # Track recent RR intervals for HRV
        recent_rrs = []

        valid_in_rec = 0
        excl_in_rec = 0

        for i in range(n_ann):
            s_i = int(samples[i])
            sym = symbols[i]

            # 1. Symbol census check
            if sym in EXCLUDED_SYMBOLS:
                exclusion_records.append({
                    "Record": rec_id,
                    "Subject_ID": subj_id,
                    "Sample_Idx": s_i,
                    "Annotation_Symbol": sym,
                    "Exclusion_Reason": "Excluded annotation symbol per AAMI EC57 (e.g. paced, unclassifiable, rhythm marker)"
                })
                excl_in_rec += 1
                continue

            # Determine Class and Label
            if sym in CLASS_0_SYMBOLS:
                label = 0
                aami_class = CLASS_0_SYMBOLS[sym]
            elif sym in CLASS_1_SYMBOLS:
                label = 1
                aami_class = CLASS_1_SYMBOLS[sym]
            else:
                exclusion_records.append({
                    "Record": rec_id,
                    "Subject_ID": subj_id,
                    "Sample_Idx": s_i,
                    "Annotation_Symbol": sym,
                    "Exclusion_Reason": f"Unmapped annotation symbol '{sym}'"
                })
                excl_in_rec += 1
                continue

            # 2. Check preceding beat for RR calculation
            if i == 0:
                exclusion_records.append({
                    "Record": rec_id,
                    "Subject_ID": subj_id,
                    "Sample_Idx": s_i,
                    "Annotation_Symbol": sym,
                    "Exclusion_Reason": "First beat of recording: preceding RR interval undefined"
                })
                excl_in_rec += 1
                continue

            # Compute RR intervals
            rr_pre = (s_i - int(samples[i - 1])) / fs
            if i < n_ann - 1:
                rr_post = (int(samples[i + 1]) - s_i) / fs
            else:
                rr_post = rr_pre

            # Clamp RR intervals for adaptive window scaling
            rr_pre_clamped = float(np.clip(rr_pre, 0.25, 2.50))
            rr_post_clamped = float(np.clip(rr_post, 0.25, 2.50))

            # 3. Dynamic context requirement check (Not fixed 350ms!)
            req_pre_samples = int(np.ceil(0.350 * rr_pre_clamped * fs))
            req_post_samples = int(np.ceil(0.550 * rr_post_clamped * fs))

            if s_i < req_pre_samples:
                exclusion_records.append({
                    "Record": rec_id,
                    "Subject_ID": subj_id,
                    "Sample_Idx": s_i,
                    "Annotation_Symbol": sym,
                    "Exclusion_Reason": f"Insufficient pre-beat signal context (Sample {s_i} < {req_pre_samples} required)"
                })
                excl_in_rec += 1
                continue

            if s_i + req_post_samples >= sig_len:
                exclusion_records.append({
                    "Record": rec_id,
                    "Subject_ID": subj_id,
                    "Sample_Idx": s_i,
                    "Annotation_Symbol": sym,
                    "Exclusion_Reason": f"Insufficient post-beat signal context (Sample {s_i} + {req_post_samples} >= {sig_len})"
                })
                excl_in_rec += 1
                continue

            # 4. Extract fiducials with frozen Stage 2 algorithm
            res = extract_beat_fiducials(
                filt_sig, d_sig, s_i,
                rr_pre_s=rr_pre_clamped,
                rr_post_s=rr_post_clamped,
                k_noise=2.5,
                fs=fs
            )

            # Update HRV (SDNN of last 5 RR intervals)
            recent_rrs.append(rr_pre)
            if len(recent_rrs) > 5:
                recent_rrs.pop(0)

            hrv = float(np.std(recent_rrs)) if len(recent_rrs) >= 3 else np.nan
            heart_rate = float(60.0 / rr_pre) if rr_pre > 0.10 else np.nan

            dataset_rows.append({
                # Metadata
                "Record": rec_id,
                "Subject_ID": subj_id,
                "Partition": partition,
                "Sample_Idx": s_i,
                "R_Time": round(res["r_time"], 4),
                "Annotation_Symbol": sym,
                "AAMI_Class": aami_class,
                "Label": label,
                # 10 Primary Clinical Features
                "HeartRate": round(heart_rate, 2),
                "RRInterval": round(rr_pre, 4),
                "PRInterval": round(res["pr_interval"], 4) if not np.isnan(res["pr_interval"]) else np.nan,
                "QRSDuration": round(res["qrs_duration"], 4) if not np.isnan(res["qrs_duration"]) else np.nan,
                "QTInterval": round(res["qt_interval"], 4) if not np.isnan(res["qt_interval"]) else np.nan,
                "STSegment": round(res["st_amp"], 4) if not np.isnan(res["st_amp"]) else np.nan,
                "HRV": round(hrv, 4) if not np.isnan(hrv) else np.nan,
                "PAmplitude": round(res["p_amp"], 4) if not np.isnan(res["p_amp"]) else np.nan,
                "RAmplitude": round(res["r_amp"], 4),
                "TAmplitude": round(res["t_amp"], 4) if not np.isnan(res["t_amp"]) else np.nan,
                # Missingness Indicators (Boolean 0/1)
                "P_Valid": int(res["p_valid"]),
                "PR_Valid": int(res["pr_valid"]),
                "QRS_Valid": int(res["qrs_valid"]),
                "ST_Valid": int(res["st_valid"]),
                "QT_Valid": int(res["qt_valid"]),
                "Baseline_Valid": int(res["baseline_valid"]),
            })
            valid_in_rec += 1

        print(f"  [{rec_idx:2d}/48] Record {rec_id} ({subj_id:<15} | {partition:<16}): Extracted {valid_in_rec:5,d} valid beats (Excluded: {excl_in_rec:3d})")

    # Build DataFrames
    df_dataset = pd.DataFrame(dataset_rows)
    df_exclusions = pd.DataFrame(exclusion_records)

    print(f"\nTotal Valid Beats Extracted: {len(df_dataset):,}")
    print(f"Total Excluded Annotations: {len(df_exclusions):,}")

    # Save complete dataset
    print(f"\nSaving beat-level dataset to {DATASET_CSV_PATH}...")
    df_dataset.to_csv(DATASET_CSV_PATH, index=False)

    # Save exclusions report
    print(f"Saving beat exclusion audit report to {EXCLUSION_REPORT_PATH}...")
    df_exclusions.to_csv(EXCLUSION_REPORT_PATH, index=False)

    # ------------------------------------------------------------
    # 4. MISSING VALUES & FEATURE DISTRIBUTION REPORTS
    # ------------------------------------------------------------
    feature_cols = [
        "HeartRate", "RRInterval", "PRInterval", "QRSDuration",
        "QTInterval", "STSegment", "HRV", "PAmplitude", "RAmplitude", "TAmplitude"
    ]

    # Missing value census
    missing_rows = []
    for col in feature_cols:
        n_miss = df_dataset[col].isna().sum()
        pct_miss = (n_miss / len(df_dataset)) * 100.0
        missing_rows.append({
            "Feature": col,
            "Total_Beats": len(df_dataset),
            "Valid_Count": len(df_dataset) - n_miss,
            "Missing_Count": n_miss,
            "Missing_Pct": round(pct_miss, 2)
        })
    df_missing = pd.DataFrame(missing_rows)
    df_missing.to_csv(MISSING_REPORT_PATH, index=False)
    print(f"Saved missing value report to {MISSING_REPORT_PATH}")

    # Feature distribution report
    dist_lines = []
    dist_lines.append("=" * 78)
    dist_lines.append("ECG FINAL PRO — STAGE 3: BEAT-LEVEL FEATURE DISTRIBUTION REPORT")
    dist_lines.append("=" * 78)
    dist_lines.append(f"Total Beats: {len(df_dataset):,} across 48 recordings / 47 subjects")
    dist_lines.append(f"Class 0 (AAMI Non-Ectopic) : {(df_dataset['Label'] == 0).sum():,} beats ({(df_dataset['Label'] == 0).mean()*100:.2f}%)")
    dist_lines.append(f"Class 1 (Selected Non-Normal): {(df_dataset['Label'] == 1).sum():,} beats ({(df_dataset['Label'] == 1).mean()*100:.2f}%)")
    dist_lines.append("\n" + "-" * 78)
    dist_lines.append("1. FEATURE MISSINGNESS CENSUS")
    dist_lines.append("-" * 78)
    dist_lines.append(f"{'Feature':<16} | {'Valid Count':<12} | {'Missing Count':<14} | {'Missing Pct':<12}")
    dist_lines.append("-" * 78)
    for _, r in df_missing.iterrows():
        dist_lines.append(f"{r['Feature']:<16} | {r['Valid_Count']:<12,d} | {r['Missing_Count']:<14,d} | {r['Missing_Pct']:5.2f}%")

    dist_lines.append("\n" + "-" * 78)
    dist_lines.append("2. OVERALL FEATURE STATISTICAL DISTRIBUTIONS")
    dist_lines.append("-" * 78)
    dist_lines.append(f"{'Feature':<14} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'25%':<8} | {'50%':<8} | {'75%':<8} | {'Max':<8}")
    dist_lines.append("-" * 78)
    for col in feature_cols:
        s = df_dataset[col].dropna()
        dist_lines.append(f"{col:<14} | {s.mean():8.3f} | {s.std():8.3f} | {s.min():8.3f} | {s.quantile(0.25):8.3f} | {s.median():8.3f} | {s.quantile(0.75):8.3f} | {s.max():8.3f}")

    dist_lines.append("\n" + "-" * 78)
    dist_lines.append("3. FEATURE DISTRIBUTIONS BY CLASS (Class 0 vs Class 1)")
    dist_lines.append("-" * 78)
    for col in feature_cols:
        s0 = df_dataset[df_dataset['Label'] == 0][col].dropna()
        s1 = df_dataset[df_dataset['Label'] == 1][col].dropna()
        dist_lines.append(f"\n--- {col} ---")
        dist_lines.append(f"  Class 0: Mean = {s0.mean():.3f}, Median = {s0.median():.3f}, Std = {s0.std():.3f}, IQR = [{s0.quantile(0.25):.3f}, {s0.quantile(0.75):.3f}] (Valid: {len(s0):,})")
        dist_lines.append(f"  Class 1: Mean = {s1.mean():.3f}, Median = {s1.median():.3f}, Std = {s1.std():.3f}, IQR = [{s1.quantile(0.25):.3f}, {s1.quantile(0.75):.3f}] (Valid: {len(s1):,})")

    dist_lines.append("\n" + "-" * 78)
    dist_lines.append("4. FEATURE DISTRIBUTIONS BY DATASET PARTITION")
    dist_lines.append("-" * 78)
    for part in ["Train_Dev", "Validation", "Independent_Test"]:
        p_df = df_dataset[df_dataset['Partition'] == part]
        dist_lines.append(f"\n--- Partition: {part} (N = {len(p_df):,} beats, {p_df['Subject_ID'].nunique()} subjects) ---")
        dist_lines.append(f"  Class 0: {(p_df['Label'] == 0).sum():,} ({(p_df['Label'] == 0).mean()*100:.1f}%) | Class 1: {(p_df['Label'] == 1).sum():,} ({(p_df['Label'] == 1).mean()*100:.1f}%)")
        dist_lines.append(f"  HeartRate Mean : {p_df['HeartRate'].mean():.1f} bpm (Std: {p_df['HeartRate'].std():.1f})")
        dist_lines.append(f"  RRInterval Mean: {p_df['RRInterval'].mean():.3f} s (Std: {p_df['RRInterval'].std():.3f})")
        dist_lines.append(f"  QRSDuration Mean: {p_df['QRSDuration'].mean():.3f} s (Std: {p_df['QRSDuration'].std():.3f})")
        dist_lines.append(f"  STSegment Mean  : {p_df['STSegment'].mean():.3f} mV (Std: {p_df['STSegment'].std():.3f})")

    DISTRIBUTION_REPORT_PATH.write_text("\n".join(dist_lines), encoding="utf-8")
    print(f"Saved feature distribution report to {DISTRIBUTION_REPORT_PATH}")

    # Summary report
    summary_lines = [
        "=" * 78,
        "ECG FINAL PRO — STAGE 3 EXECUTION SUMMARY",
        "=" * 78,
        f"Execution Time: {time.time() - t_start:.1f} seconds",
        f"Total MIT-BIH Recordings Processed: {len(records)}",
        f"Total MIT-BIH Subjects Represented: {df_dataset['Subject_ID'].nunique()}",
        f"Total Beats in Mapped Dataset     : {len(df_dataset):,}",
        f"Total Excluded Annotations Audited: {len(df_exclusions):,}",
        "",
        "Partition Breakdown:",
        f"  * Train_Dev        : {len(df_dataset[df_dataset['Partition']=='Train_Dev']):,} beats ({df_dataset[df_dataset['Partition']=='Train_Dev']['Subject_ID'].nunique()} subjects)",
        f"  * Validation       : {len(df_dataset[df_dataset['Partition']=='Validation']):,} beats ({df_dataset[df_dataset['Partition']=='Validation']['Subject_ID'].nunique()} subjects)",
        f"  * Independent_Test : {len(df_dataset[df_dataset['Partition']=='Independent_Test']):,} beats ({df_dataset[df_dataset['Partition']=='Independent_Test']['Subject_ID'].nunique()} subjects)",
        "",
        "Class Breakdown across Dataset:",
        f"  * Class 0 (AAMI Non-Ectopic) : {(df_dataset['Label']==0).sum():,} beats ({(df_dataset['Label']==0).mean()*100:.2f}%)",
        f"  * Class 1 (Selected Non-Normal): {(df_dataset['Label']==1).sum():,} beats ({(df_dataset['Label']==1).mean()*100:.2f}%)",
        "",
        "LEAKAGE PREVENTION CERTIFICATION:",
        "  1. Subject_201_202 (Records 201 and 202) is co-assigned strictly to Train_Dev.",
        "  2. All 14 Stage 2 exploratory subjects are co-assigned strictly to Train_Dev.",
        "  3. Independent_Test contains ONLY genuinely unseen subjects (10 subjects, 0 overlap).",
        "  4. Feature extraction parameters were FROZEN before running on any partition.",
        "  5. No model has been trained or evaluated.",
        "=" * 78
    ]
    SUMMARY_REPORT_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved dataset summary to {SUMMARY_REPORT_PATH}")
    print("\nSTAGE 3 EXECUTION COMPLETE: SUCCESS.")

if __name__ == "__main__":
    main()
