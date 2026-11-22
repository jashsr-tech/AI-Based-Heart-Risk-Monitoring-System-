
from pathlib import Path
import csv
import numpy as np

# ============================================================
# ECG FINAL PRO — STAGE 2E: FEATURE EXTRACTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ECG_FILE = BASE_DIR / "02_signal_processing" / "filtered_ecg.npy"
PQRST_FILE = BASE_DIR / "02_signal_processing" / "detected_pqrst.npy"
RR_FILE = BASE_DIR / "02_signal_processing" / "rr_intervals.npy"

OUT_NPY = BASE_DIR / "03_feature_extraction" / "ecg_ml_features.npy"
OUT_CSV = BASE_DIR / "03_feature_extraction" / "ecg_features.csv"
OUT_REPORT = BASE_DIR / "05_results" / "stage2E_feature_report.txt"

FS = 360.0

print("=" * 68)
print("ECG FINAL PRO — STAGE 2E: FEATURE EXTRACTION")
print("=" * 68)

for path in [ECG_FILE, PQRST_FILE, RR_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

ecg = np.load(ECG_FILE)
pqrst = np.load(PQRST_FILE)
rr = np.load(RR_FILE)

if pqrst.ndim != 2 or pqrst.shape[1] != 5:
    raise ValueError("PQRST data must have five columns: P,Q,R,S,T.")

# PQRST columns contain peak times in seconds.
p_times = pqrst[:, 0]
q_times = pqrst[:, 1]
r_times = pqrst[:, 2]
s_times = pqrst[:, 3]
t_times = pqrst[:, 4]

# Use consecutive R peaks for RR and HR.
valid_rr = np.diff(r_times)
valid_rr = valid_rr[valid_rr > 0]

if len(valid_rr) < 2:
    raise ValueError("Not enough valid RR intervals to extract features.")

mean_rr = float(np.mean(valid_rr))
heart_rate = float(60.0 / mean_rr)

# SDNN and RMSSD are calculated in seconds first.
sdnn_s = float(np.std(valid_rr, ddof=1))
rmssd_s = float(np.sqrt(np.mean(np.diff(valid_rr) ** 2)))

# Peak-to-peak measurements:
# These are explicitly not clinical onset/offset intervals.
pr_peak_s = r_times - p_times
qrs_peak_s = s_times - q_times
qt_peak_s = t_times - q_times

# Amplitudes sampled from the filtered signal at detected peak times.
def amplitude_at(times):
    indices = np.rint(times * FS).astype(int)
    indices = np.clip(indices, 0, len(ecg) - 1)
    return ecg[indices]

p_amp = amplitude_at(p_times)
r_amp = amplitude_at(r_times)
t_amp = amplitude_at(t_times)

# ST-level proxy: filtered amplitude 60 ms after each S peak.
st_times = s_times + 0.060
st_amp = amplitude_at(st_times)

# Calculate averages across detected beats.
features = {
    "HeartRate": float(heart_rate),
    "RRInterval": float(mean_rr),
    "PRInterval": float(np.mean(pr_peak_s)),
    "QRSDuration": float(np.mean(qrs_peak_s)),
    "QTInterval": float(np.mean(qt_peak_s)),
    "STSegment": float(np.mean(st_amp)),
    "HRV": float(sdnn_s),
    "PAmplitude": float(np.mean(p_amp)),
    "RAmplitude": float(np.mean(r_amp)),
    "TAmplitude": float(np.mean(t_amp)),
}

# Required ML feature order.
feature_names = [
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

feature_vector = np.array(
    [features[name] for name in feature_names],
    dtype=float
)

np.save(OUT_NPY, feature_vector)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Feature", "Value", "Unit_or_definition"])

    definitions = {
        "HeartRate": "BPM",
        "RRInterval": "seconds",
        "PRInterval": "P peak to R peak, seconds",
        "QRSDuration": "Q peak to S peak, seconds",
        "QTInterval": "Q peak to T peak, seconds",
        "STSegment": "Filtered amplitude 60 ms after S, arbitrary units",
        "HRV": "SDNN of detected RR intervals, seconds",
        "PAmplitude": "Filtered amplitude, arbitrary units",
        "RAmplitude": "Filtered amplitude, arbitrary units",
        "TAmplitude": "Filtered amplitude, arbitrary units",
    }

    for name in feature_names:
        writer.writerow([name, features[name], definitions[name]])

report_lines = [
    "ECG FINAL PRO — STAGE 2E FEATURE REPORT",
    "",
    f"Sampling rate: {FS:.1f} Hz",
    f"Filtered samples: {len(ecg)}",
    f"Detected beats: {len(pqrst)}",
    f"RR intervals used: {len(valid_rr)}",
    "",
    "FEATURES",
]

for name in feature_names:
    report_lines.append(f"{name}: {features[name]:.8f}")

report_lines.extend([
    "",
    "IMPORTANT DEFINITIONS",
    "PRInterval currently means P-peak to R-peak, not clinical PR.",
    "QRSDuration currently means Q-peak to S-peak.",
    "QTInterval currently means Q-peak to T-peak.",
    "STSegment is an amplitude proxy, not a clinical ST measurement.",
    "Amplitude values are arbitrary synthetic units, not mV.",
])

OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

print("\nEXTRACTED FEATURES")
for name in feature_names:
    print(f"{name:14s}: {features[name]:.8f}")

print("\nML FEATURE VECTOR")
print(feature_vector)

print("\nSaved:")
print(OUT_NPY)
print(OUT_CSV)
print(OUT_REPORT)

print("\nSTAGE 2E COMPLETE")
print("Next: verify feature definitions and compatibility before ML prediction.")
