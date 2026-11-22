
"""
ECG FINAL PRO — Stage 1
Controlled synthetic ECG generator and validation.

Outputs:
  synthetic_ecg.npy
  synthetic_ecg.csv
  ecg_ground_truth.npy
  ecg_ground_truth.csv
  ecg_generator_validation.png

This is a synthetic software-test signal, not a real patient ECG.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

FS = 360                       # Samples per second
DURATION = 60.0                # Seconds
TARGET_HR = 72.0               # Approximate average BPM

OUTPUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)

# Amplitudes are arbitrary synthetic units, NOT calibrated mV.
P_AMP = 0.60
Q_AMP = -0.80
R_AMP = 4.50
S_AMP = -1.00
T_AMP = 1.00

# Gaussian wave widths, seconds
P_WIDTH = 0.025
Q_WIDTH = 0.012
R_WIDTH = 0.014
S_WIDTH = 0.014
T_WIDTH = 0.050

# Wave peak locations relative to each R peak, seconds.
# These are peak-to-peak design targets.
P_OFFSET = -0.160
Q_OFFSET = -0.045
R_OFFSET = 0.000
S_OFFSET = 0.045
T_OFFSET = 0.305

BASE_RR = 60.0 / TARGET_HR

# Controlled RR variation pattern, seconds.
# Repeated with a fixed seed for reproducibility.
RR_VARIATION = np.array([
    -0.080, -0.045, -0.020,  0.010,
     0.035,  0.065,  0.090,  0.055,
     0.025, -0.005, -0.035, -0.070,
     0.080,  0.045,  0.015, -0.015,
    -0.050, -0.085, -0.040,  0.005
])

# Small noise and baseline wander
BASELINE_AMPLITUDE = 0.010
BASELINE_FREQUENCY = 0.20
NOISE_STD = 0.003

# ============================================================
# 2. CREATE TIME AXIS
# ============================================================

sample_count = int(round(FS * DURATION))
time = np.arange(sample_count, dtype=float) / FS

# ============================================================
# 3. CREATE R-PEAK TIMES
# ============================================================

r_times = [0.850]
rr_index = 0

while True:
    variation = RR_VARIATION[rr_index % len(RR_VARIATION)]
    rr = BASE_RR + variation

    # Safety bounds for the controlled test signal.
    rr = float(np.clip(rr, 0.65, 1.05))

    next_r = r_times[-1] + rr

    # Leave enough time for the final T wave.
    if next_r + T_OFFSET >= DURATION:
        break

    r_times.append(next_r)
    rr_index += 1

r_times = np.asarray(r_times)

# Ground truth: each row is [P, Q, R, S, T] in seconds.
offsets = np.array([
    P_OFFSET, Q_OFFSET, R_OFFSET, S_OFFSET, T_OFFSET
])

ground_truth = r_times[:, None] + offsets[None, :]

# ============================================================
# 4. GENERATE ECG MORPHOLOGY
# ============================================================

def gaussian_wave(t, center, width, amplitude):
    z = (t - center) / width
    return amplitude * np.exp(-0.5 * z * z)


ecg = np.zeros(sample_count, dtype=float)

for r_time in r_times:
    ecg += gaussian_wave(
        time, r_time + P_OFFSET, P_WIDTH, P_AMP
    )
    ecg += gaussian_wave(
        time, r_time + Q_OFFSET, Q_WIDTH, Q_AMP
    )
    ecg += gaussian_wave(
        time, r_time + R_OFFSET, R_WIDTH, R_AMP
    )
    ecg += gaussian_wave(
        time, r_time + S_OFFSET, S_WIDTH, S_AMP
    )
    ecg += gaussian_wave(
        time, r_time + T_OFFSET, T_WIDTH, T_AMP
    )

# Add small baseline wander and reproducible noise.
ecg += BASELINE_AMPLITUDE * np.sin(
    2 * np.pi * BASELINE_FREQUENCY * time
)
ecg += RNG.normal(0.0, NOISE_STD, sample_count)

# ============================================================
# 5. MEASURE WAVE PEAKS FROM THE GENERATED SIGNAL
# ============================================================

def find_extremum(target_time, radius, kind):
    """
    Find a local maximum or minimum near a known design time.
    This validates the generated waveform, not a clinical ECG.
    """
    center = int(round(target_time * FS))
    half_width = int(round(radius * FS))

    left = max(0, center - half_width)
    right = min(sample_count, center + half_width + 1)

    segment = ecg[left:right]

    if len(segment) == 0:
        return np.nan

    if kind == "max":
        index = left + int(np.argmax(segment))
    else:
        index = left + int(np.argmin(segment))

    return index / FS


measured = np.empty_like(ground_truth)

for i, row in enumerate(ground_truth):
    p, q, r, s, t = row

    measured[i] = [
        find_extremum(p, 0.025, "max"),
        find_extremum(q, 0.018, "min"),
        find_extremum(r, 0.020, "max"),
        find_extremum(s, 0.020, "min"),
        find_extremum(t, 0.045, "max"),
    ]

# ============================================================
# 6. CALCULATE VALIDATION METRICS
# ============================================================

rr_intervals = np.diff(r_times)

measured_pr_peak = measured[:, 2] - measured[:, 0]
measured_qrs = measured[:, 3] - measured[:, 1]
measured_qt = measured[:, 4] - measured[:, 1]

timing_errors_ms = (measured - ground_truth) * 1000

# RR statistics use the generated beat schedule.
mean_rr = float(np.mean(rr_intervals))
mean_hr = 60.0 / mean_rr
sdnn = float(np.std(rr_intervals, ddof=1)) if len(rr_intervals) > 1 else 0.0

# ============================================================
# 7. SAVE OUTPUT FILES
# ============================================================

np.save(OUTPUT_DIR / "synthetic_ecg.npy", ecg)
np.save(OUTPUT_DIR / "ecg_ground_truth.npy", ground_truth)

np.savetxt(
    OUTPUT_DIR / "synthetic_ecg.csv",
    np.column_stack((time, ecg)),
    delimiter=",",
    header="Time_s,ECG_amplitude",
    comments="",
    fmt="%.8f"
)

with open(
    OUTPUT_DIR / "ecg_ground_truth.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "P_peak_s", "Q_peak_s", "R_peak_s",
        "S_peak_s", "T_peak_s"
    ])
    writer.writerows(ground_truth)

# ============================================================
# 8. PLOT THE SIGNAL
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(13, 7))

# Show the first 5 seconds.
mask = time <= 5.0
axes[0].plot(time[mask], ecg[mask], linewidth=0.8)
axes[0].set_title("Synthetic ECG — First 5 Seconds")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude (arbitrary units)")
axes[0].grid(True)

# Show one representative beat.
beat_idx = min(5, len(r_times) - 1)
beat_center = r_times[beat_idx]
beat_mask = (
    (time >= beat_center - 0.25) &
    (time <= beat_center + 0.45)
)

axes[1].plot(time[beat_mask], ecg[beat_mask], linewidth=1.0)

for label, col in zip(["P", "Q", "R", "S", "T"], range(5)):
    target = ground_truth[beat_idx, col]
    axes[1].axvline(target, linestyle="--", alpha=0.5, label=f"{label} target")

axes[1].set_title("One Beat — Ground-Truth Peak Positions")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude (arbitrary units)")
axes[1].grid(True)
axes[1].legend(ncol=5)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "ecg_generator_validation.png",
    dpi=150
)
plt.show()

# ============================================================
# 9. PRINT VALIDATION REPORT
# ============================================================

print("=" * 68)
print("ECG FINAL PRO — STAGE 1: GENERATION AND VALIDATION")
print("=" * 68)

print("\nSIGNAL")
print(f"Sampling rate       : {FS} Hz")
print(f"Duration            : {DURATION:.2f} s")
print(f"Expected samples    : {sample_count}")
print(f"Actual samples      : {len(ecg)}")
print(f"Sample spacing      : {1 / FS:.8f} s")
print(f"Generated beats     : {len(r_times)}")
print(f"Mean RR             : {mean_rr:.6f} s")
print(f"Mean heart rate     : {mean_hr:.3f} BPM")
print(f"RR SDNN             : {sdnn * 1000:.3f} ms")
print(f"Amplitude min/max   : {ecg.min():.4f} / {ecg.max():.4f}")

print("\nDESIGN TARGETS")
print(f"P peak to R peak    : {-P_OFFSET * 1000:.1f} ms")
print(f"Q peak to S peak    : {(S_OFFSET - Q_OFFSET) * 1000:.1f} ms")
print(f"Q peak to T peak    : {(T_OFFSET - Q_OFFSET) * 1000:.1f} ms")

print("\nMEASURED FROM GENERATED WAVEFORM")
print(f"Mean P peak to R    : {np.mean(measured_pr_peak) * 1000:.3f} ms")
print(f"Mean Q peak to S    : {np.mean(measured_qrs) * 1000:.3f} ms")
print(f"Mean Q peak to T    : {np.mean(measured_qt) * 1000:.3f} ms")

print("\nMAXIMUM ABSOLUTE PEAK-TIMING ERROR")
for i, label in enumerate(["P", "Q", "R", "S", "T"]):
    print(
        f"{label} peak           : "
        f"{np.max(np.abs(timing_errors_ms[:, i])):.3f} ms"
    )

print("\nOUTPUT FILES")
for name in [
    "synthetic_ecg.npy",
    "synthetic_ecg.csv",
    "ecg_ground_truth.npy",
    "ecg_ground_truth.csv",
    "ecg_generator_validation.png",
]:
    print(f"  {name}")

print("\nNOTE")
print("PR peak-to-peak here means P peak to R peak.")
print("Clinical PR interval requires P onset to QRS onset.")
print("Amplitude is synthetic and is not calibrated in millivolts.")
print("=" * 68)
