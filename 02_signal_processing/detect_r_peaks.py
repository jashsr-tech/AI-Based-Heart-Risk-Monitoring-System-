
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================================================
# ECG FINAL PRO — STAGE 2C: R-PEAK DETECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

ECG_FILE = BASE_DIR / "02_signal_processing" / "filtered_ecg.npy"
GT_FILE = BASE_DIR / "01_signal_generation" / "ecg_ground_truth.npy"

OUT_PEAKS = BASE_DIR / "02_signal_processing" / "r_peaks.npy"
OUT_TIMES = BASE_DIR / "02_signal_processing" / "r_peak_times.npy"
OUT_RR = BASE_DIR / "02_signal_processing" / "rr_intervals.npy"
PLOT_FILE = BASE_DIR / "05_results" / "stage2C_r_peak_detection.png"

FS = 360.0

# Detection settings for this controlled synthetic signal
MIN_DISTANCE_SEC = 0.60
MIN_PROMINENCE = 1.0
MATCH_TOLERANCE_SEC = 0.030

print("=" * 65)
print("ECG FINAL PRO — STAGE 2C: R-PEAK DETECTION")
print("=" * 65)

if not ECG_FILE.exists():
    raise FileNotFoundError(f"Filtered ECG not found: {ECG_FILE}")

if not GT_FILE.exists():
    raise FileNotFoundError(f"Ground truth not found: {GT_FILE}")

ecg = np.load(ECG_FILE)
ground_truth = np.load(GT_FILE)

if ecg.ndim != 1:
    raise ValueError("Filtered ECG must be one-dimensional.")

if len(ecg) != int(60 * FS):
    raise ValueError(
        f"Expected {int(60 * FS)} samples, received {len(ecg)}."
    )

# Ground-truth columns are [P, Q, R, S, T].
true_r_times = ground_truth[:, 2]
true_r_times = true_r_times[
    (true_r_times >= 0) &
    (true_r_times < len(ecg) / FS)
]

# Minimum distance prevents multiple detections in one QRS complex.
min_distance = int(MIN_DISTANCE_SEC * FS)

peak_indices, properties = find_peaks(
    ecg,
    distance=min_distance,
    prominence=MIN_PROMINENCE
)

detected_times = peak_indices / FS
rr_intervals = np.diff(detected_times)

np.save(OUT_PEAKS, peak_indices)
np.save(OUT_TIMES, detected_times)
np.save(OUT_RR, rr_intervals)

# One-to-one matching: each ground-truth beat can match only once.
errors = []
matched_true = set()
matched_detected = set()

for i, true_time in enumerate(true_r_times):
    if len(detected_times) == 0:
        continue

    distances = np.abs(detected_times - true_time)
    order = np.argsort(distances)

    for j in order:
        if j in matched_detected:
            continue

        if distances[j] <= MATCH_TOLERANCE_SEC:
            matched_true.add(i)
            matched_detected.add(int(j))
            errors.append(detected_times[j] - true_time)
            break

matched = len(matched_true)
missed = len(true_r_times) - matched
false_positives = len(detected_times) - matched

print(f"Samples                 : {len(ecg)}")
print(f"Sampling rate           : {FS:.1f} Hz")
print(f"Ground-truth R peaks    : {len(true_r_times)}")
print(f"Detected R peaks        : {len(detected_times)}")
print(f"Matched peaks           : {matched}")
print(f"Missed peaks            : {missed}")
print(f"False positives         : {false_positives}")
print(f"Match tolerance         : {MATCH_TOLERANCE_SEC * 1000:.1f} ms")

if errors:
    errors_ms = np.asarray(errors) * 1000
    print(f"Mean signed timing error: {np.mean(errors_ms):.3f} ms")
    print(f"Mean absolute error     : {np.mean(np.abs(errors_ms)):.3f} ms")
    print(f"Maximum absolute error  : {np.max(np.abs(errors_ms)):.3f} ms")

if len(rr_intervals):
    mean_rr = np.mean(rr_intervals)
    print(f"Mean RR                 : {mean_rr:.6f} s")
    print(f"Mean heart rate         : {60 / mean_rr:.3f} BPM")
    print(
        f"RR SDNN                 : "
        f"{np.std(rr_intervals, ddof=1) * 1000:.3f} ms"
        if len(rr_intervals) > 1 else
        "RR SDNN                 : unavailable"
    )

# Plot first 10 seconds
time = np.arange(len(ecg)) / FS
mask = time <= 10

plt.figure(figsize=(13, 5))
plt.plot(time[mask], ecg[mask], linewidth=0.8, label="Filtered ECG")
plt.plot(
    detected_times[detected_times <= 10],
    ecg[peak_indices[detected_times <= 10]],
    "x",
    label="Detected R peaks"
)
plt.plot(
    true_r_times[true_r_times <= 10],
    ecg[np.rint(true_r_times[true_r_times <= 10] * FS).astype(int)],
    "o",
    markersize=4,
    fillstyle="none",
    label="Ground-truth R times"
)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (arbitrary units)")
plt.title("Stage 2C — R-Peak Detection")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()

print("\nSaved:")
print(OUT_PEAKS)
print(OUT_TIMES)
print(OUT_RR)
print(PLOT_FILE)
print("\nSTAGE 2C COMPLETE — inspect validation before Stage 2D.")
