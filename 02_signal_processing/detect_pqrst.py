
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================================================
# ECG FINAL PRO — STAGE 2D: P-Q-R-S-T PEAK DETECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

ECG_FILE = BASE_DIR / "02_signal_processing" / "filtered_ecg.npy"
R_PEAKS_FILE = BASE_DIR / "02_signal_processing" / "r_peaks.npy"
GT_FILE = BASE_DIR / "01_signal_generation" / "ecg_ground_truth.npy"

OUT_FILE = BASE_DIR / "02_signal_processing" / "detected_pqrst.npy"
CSV_FILE = BASE_DIR / "03_feature_extraction" / "pqrst_measurements.csv"
PLOT_FILE = BASE_DIR / "05_results" / "stage2D_pqrst_detection.png"

FS = 360.0

# Search windows relative to the detected R peak, in seconds.
# They are appropriate for this controlled synthetic morphology.
P_WINDOW = (-0.205, -0.105)
Q_WINDOW = (-0.075, -0.015)
S_WINDOW = (0.015, 0.085)
T_WINDOW = (0.200, 0.390)

print("=" * 68)
print("ECG FINAL PRO — STAGE 2D: P-Q-R-S-T DETECTION")
print("=" * 68)

for path in [ECG_FILE, R_PEAKS_FILE, GT_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")

ecg = np.load(ECG_FILE)
r_indices = np.load(R_PEAKS_FILE).astype(int)
ground_truth = np.load(GT_FILE)

duration = len(ecg) / FS
true_r_times = ground_truth[:, 2]
valid_gt = (true_r_times >= 0) & (true_r_times < duration)
ground_truth = ground_truth[valid_gt]

def local_peak(signal, r_idx, window, mode):
    """Find local max/min in a time window around an R index."""
    left = max(0, r_idx + int(round(window[0] * FS)))
    right = min(len(signal), r_idx + int(round(window[1] * FS)) + 1)

    if right <= left:
        return np.nan

    segment = signal[left:right]

    if mode == "max":
        idx = left + int(np.argmax(segment))
    else:
        idx = left + int(np.argmin(segment))

    return idx / FS

detected = []

for r_idx in r_indices:
    r_time = r_idx / FS

    p_time = local_peak(ecg, r_idx, P_WINDOW, "max")
    q_time = local_peak(ecg, r_idx, Q_WINDOW, "min")
    s_time = local_peak(ecg, r_idx, S_WINDOW, "min")
    t_time = local_peak(ecg, r_idx, T_WINDOW, "max")

    detected.append([p_time, q_time, r_time, s_time, t_time])

detected = np.asarray(detected, dtype=float)
np.save(OUT_FILE, detected)

# Match each detected R to the closest ground-truth R.
# Ground-truth rows contain [P, Q, R, S, T].
true_rows = []
for row in ground_truth:
    true_rows.append(row)

matched_true = []
matched_detected = []
used = set()

for i, row in enumerate(detected):
    r_time = row[2]
    if len(true_rows) == 0:
        continue

    distances = np.array([abs(gt[2] - r_time) for gt in true_rows])
    for j in np.argsort(distances):
        if j not in used and distances[j] <= 0.030:
            matched_true.append(true_rows[j])
            matched_detected.append(row)
            used.add(int(j))
            break

true_matched = np.asarray(matched_true)
det_matched = np.asarray(matched_detected)

print(f"Filtered samples       : {len(ecg)}")
print(f"Detected R peaks      : {len(r_indices)}")
print(f"Matched PQRST beats   : {len(det_matched)}")

if len(det_matched):
    names = ["P", "Q", "R", "S", "T"]
    errors_ms = (det_matched - true_matched) * 1000

    print("\nPEAK TIMING VALIDATION")
    for col, name in enumerate(names):
        print(
            f"{name} peak: mean absolute error = "
            f"{np.mean(np.abs(errors_ms[:, col])):.3f} ms; "
            f"maximum absolute error = "
            f"{np.max(np.abs(errors_ms[:, col])):.3f} ms"
        )

    # Peak-to-peak intervals, not clinical onset-to-offset intervals.
    pr_peak = (det_matched[:, 2] - det_matched[:, 0]) * 1000
    qrs_peak = (det_matched[:, 3] - det_matched[:, 1]) * 1000
    qt_peak = (det_matched[:, 4] - det_matched[:, 1]) * 1000

    print("\nMEASURED PEAK-TO-PEAK INTERVALS")
    print(f"P peak to R peak : {np.mean(pr_peak):.3f} ms")
    print(f"Q peak to S peak : {np.mean(qrs_peak):.3f} ms")
    print(f"Q peak to T peak : {np.mean(qt_peak):.3f} ms")

# Save a CSV for subsequent feature extraction.
header = "P_peak_s,Q_peak_s,R_peak_s,S_peak_s,T_peak_s"
np.savetxt(
    CSV_FILE,
    detected,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.8f"
)

# Plot the first 5 seconds.
t = np.arange(len(ecg)) / FS
mask = t <= 5.0

plt.figure(figsize=(13, 5))
plt.plot(t[mask], ecg[mask], linewidth=0.8, label="Filtered ECG")

labels = ["P", "Q", "R", "S", "T"]
for col, label in enumerate(labels):
    times = detected[:, col]
    valid = (times >= 0) & (times <= 5.0)
    indices = np.rint(times[valid] * FS).astype(int)
    plt.scatter(
        times[valid],
        ecg[indices],
        s=18,
        label=f"{label} detected"
    )

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (arbitrary units)")
plt.title("Stage 2D — P-Q-R-S-T Peak Detection")
plt.grid(True)
plt.legend(ncol=3)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()

print("\nSaved:")
print(OUT_FILE)
print(CSV_FILE)
print(PLOT_FILE)
print("\nSTAGE 2D COMPLETE — review errors before Stage 2E.")
