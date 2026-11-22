"""
ECG FINAL PRO — Test Cases Evaluation Runner
Runs both demonstration test cases through the complete DSP, Feature Extraction,
and Machine Learning inference pipeline using the copied model in:
  04_machine_learning/models/

Preserves all original files and records results in:
  01_signal_generation/test_cases/
"""

from pathlib import Path
import csv
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, sosfiltfilt, find_peaks

# ============================================================
# CONFIGURATION & PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_CASES_DIR = ROOT_DIR / "01_signal_generation" / "test_cases"
MODELS_DIR = ROOT_DIR / "04_machine_learning" / "models"

MODEL_FILE = MODELS_DIR / "ecg_decision_tree_model.pkl"
SCALER_FILE = MODELS_DIR / "ecg_feature_scaler.pkl"
ORDER_FILE = MODELS_DIR / "model_feature_order.txt"

REPORT_FILE = TEST_CASES_DIR / "test_case_evaluation_report.txt"
CSV_FILE = TEST_CASES_DIR / "test_case_features_and_predictions.csv"

FS = 360.0

# PQRST Search windows relative to R-peak (seconds) — exact match with detect_pqrst.py
P_WINDOW = (-0.205, -0.105)
Q_WINDOW = (-0.075, -0.015)
S_WINDOW = (0.015, 0.085)
T_WINDOW = (0.200, 0.390)

# ============================================================
# SIGNAL PROCESSING FUNCTIONS
# ============================================================

def filter_signal(ecg_raw):
    """Fourth-order Butterworth bandpass (0.5 - 40 Hz) using zero-phase SOS."""
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, ecg_raw)


def detect_r_peaks(filtered_ecg, min_distance_sec=0.40, min_prominence=1.0):
    """Detect R-peaks using find_peaks."""
    min_dist = int(round(min_distance_sec * FS))
    peak_indices, _ = find_peaks(filtered_ecg, distance=min_dist, prominence=min_prominence)
    return peak_indices, peak_indices / FS


def detect_pqrst_peaks(filtered_ecg, r_indices):
    """Locate P, Q, R, S, T extrema in designated time windows around each R."""
    pqrst = []
    for r_idx in r_indices:
        r_time = r_idx / FS

        def local_peak(win, mode):
            left = max(0, r_idx + int(round(win[0] * FS)))
            right = min(len(filtered_ecg), r_idx + int(round(win[1] * FS)) + 1)
            if right <= left:
                return np.nan
            seg = filtered_ecg[left:right]
            if len(seg) == 0:
                return np.nan
            if mode == "max":
                return (left + int(np.argmax(seg))) / FS
            else:
                return (left + int(np.argmin(seg))) / FS

        p_time = local_peak(P_WINDOW, "max")
        q_time = local_peak(Q_WINDOW, "min")
        s_time = local_peak(S_WINDOW, "min")
        t_time = local_peak(T_WINDOW, "max")
        pqrst.append([p_time, q_time, r_time, s_time, t_time])

    return np.asarray(pqrst, dtype=float)


def extract_10_features(filtered_ecg, pqrst):
    """Extract the 10 standard features matching Stage 2E."""
    p_times = pqrst[:, 0]
    q_times = pqrst[:, 1]
    r_times = pqrst[:, 2]
    s_times = pqrst[:, 3]
    t_times = pqrst[:, 4]

    valid_rr = np.diff(r_times)
    valid_rr = valid_rr[valid_rr > 0]
    if len(valid_rr) < 2:
        raise ValueError("Insufficient valid RR intervals.")

    mean_rr = float(np.mean(valid_rr))
    heart_rate = float(60.0 / mean_rr)
    sdnn_s = float(np.std(valid_rr, ddof=1))

    pr_peak_s = r_times - p_times
    qrs_peak_s = s_times - q_times
    qt_peak_s = t_times - q_times

    def amplitude_at(times):
        indices = np.rint(times * FS).astype(int)
        indices = np.clip(indices, 0, len(filtered_ecg) - 1)
        return filtered_ecg[indices]

    p_amp = amplitude_at(p_times)
    r_amp = amplitude_at(r_times)
    t_amp = amplitude_at(t_times)
    st_amp = amplitude_at(s_times + 0.060)

    features = {
        "HeartRate": float(heart_rate),
        "RRInterval": float(mean_rr),
        "PRInterval": float(np.nanmean(pr_peak_s)),
        "QRSDuration": float(np.nanmean(qrs_peak_s)),
        "QTInterval": float(np.nanmean(qt_peak_s)),
        "STSegment": float(np.nanmean(st_amp)),
        "HRV": float(sdnn_s),
        "PAmplitude": float(np.nanmean(p_amp)),
        "RAmplitude": float(np.nanmean(r_amp)),
        "TAmplitude": float(np.nanmean(t_amp)),
    }
    return features


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 72)
    print("ECG FINAL PRO — TEST CASES EVALUATION RUNNER")
    print("=" * 72)

    # 1. Load model, scaler, and feature order
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_order = [line.strip() for line in ORDER_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"Loaded model : {MODEL_FILE.name} ({type(model).__name__})")
    print(f"Loaded scaler: {SCALER_FILE.name} ({type(scaler).__name__})")
    print(f"Feature order: {feature_order}\n")

    cases = [
        ("Case A: Normal Synthetic", TEST_CASES_DIR / "normal_ecg.npy", 0.40),
        ("Case B: Simulated PVC (Adaptive Peak Detector)", TEST_CASES_DIR / "pvc_ecg.npy", 0.40),
        ("Case B: Simulated PVC (Strict 0.60s Detector)", TEST_CASES_DIR / "pvc_ecg.npy", 0.60),
    ]

    results_table = []
    report_lines = [
        "========================================================================",
        "ECG FINAL PRO — DEMONSTRATION TEST CASES EVALUATION REPORT",
        "========================================================================",
        f"Evaluation Date: {pd.Timestamp.now()}",
        f"Models Location: {MODELS_DIR}",
        "",
        "NOTE: All signals are synthetic test waveforms for technical demonstration.",
        "They are not clinical diagnostic evaluations.",
        "========================================================================",
        "",
    ]

    for name, path, min_dist in cases:
        if not path.is_file():
            print(f"ERROR: {path} not found.")
            continue

        raw_ecg = np.load(path)
        filtered = filter_signal(raw_ecg)
        r_indices, r_times = detect_r_peaks(filtered, min_distance_sec=min_dist)
        pqrst = detect_pqrst_peaks(filtered, r_indices)
        feats = extract_10_features(filtered, pqrst)

        vec = np.array([feats[col] for col in feature_order], dtype=float).reshape(1, -1)
        scaled_vec = scaler.transform(vec)
        pred = int(model.predict(scaled_vec)[0])
        probs = model.predict_proba(scaled_vec)[0]

        label_str = "NORMAL" if pred == 0 else "ABNORMAL"
        prob_norm = probs[0] * 100
        prob_abnorm = probs[1] * 100

        print(f"--- {name} (min_dist={min_dist}s) ---")
        print(f"  Detected beats      : {len(r_indices)}")
        print(f"  Heart Rate          : {feats['HeartRate']:.2f} BPM")
        print(f"  RR Interval         : {feats['RRInterval']:.4f} s")
        print(f"  QRS Duration        : {feats['QRSDuration']*1000:.1f} ms")
        print(f"  HRV (SDNN)          : {feats['HRV']*1000:.1f} ms")
        print(f"  Prediction          : {label_str} (Class {pred})")
        print(f"  Model Confidence    : Normal={prob_norm:.2f}%, Abnormal={prob_abnorm:.2f}%\n")

        row = {
            "TestCase": name,
            "MinDistanceSec": min_dist,
            "DetectedBeats": len(r_indices),
            "Prediction": label_str,
            "NormalProbPercent": round(prob_norm, 2),
            "AbnormalProbPercent": round(prob_abnorm, 2),
        }
        for col in feature_order:
            row[col] = feats[col]
        results_table.append(row)

        report_lines.extend([
            f"TEST CASE: {name}",
            f"R-peak min distance: {min_dist} s",
            f"Detected beats: {len(r_indices)}",
            f"Model Prediction: {label_str} (Class {pred})",
            f"Class Probabilities: Normal={prob_norm:.2f}%, Abnormal={prob_abnorm:.2f}%",
            "Features:",
        ])
        for col in feature_order:
            report_lines.append(f"  {col:15s}: {feats[col]:.6f}")
        report_lines.append("")

    # Save CSV and Report
    df_res = pd.DataFrame(results_table)
    df_res.to_csv(CSV_FILE, index=False)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 72)
    print("TEST CASES SUMMARY")
    print("=" * 72)
    print(df_res[["TestCase", "DetectedBeats", "QRSDuration", "HRV", "Prediction", "NormalProbPercent", "AbnormalProbPercent"]].to_string(index=False))
    print(f"\nSaved CSV    : {CSV_FILE}")
    print(f"Saved Report : {REPORT_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
