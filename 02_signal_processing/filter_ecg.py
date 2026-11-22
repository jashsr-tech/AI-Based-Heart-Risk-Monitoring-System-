
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ============================================================
# ECG FINAL PRO — STAGE 2B: OFFLINE FILTERING
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "01_signal_generation" / "synthetic_ecg.npy"
OUTPUT_FILE = BASE_DIR / "02_signal_processing" / "filtered_ecg.npy"
PLOT_FILE = BASE_DIR / "05_results" / "stage2B_filter_comparison.png"

FS = 360.0
LOWCUT = 0.5
HIGHCUT = 40.0
ORDER = 4

print("=" * 65)
print("ECG FINAL PRO — STAGE 2B: FILTERING")
print("=" * 65)

# Load signal
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input signal not found: {INPUT_FILE}")

ecg = np.load(INPUT_FILE)

if ecg.ndim != 1:
    raise ValueError("Expected a one-dimensional ECG signal.")

if len(ecg) != int(FS * 60):
    raise ValueError(
        f"Unexpected sample count: {len(ecg)}. "
        "Check Stage 1 output and sampling rate."
    )

# Fourth-order Butterworth bandpass in second-order sections
sos = butter(
    ORDER,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=FS,
    output="sos"
)

# Zero-phase filtering; intended for offline processing
filtered = sosfiltfilt(sos, ecg)

# Save output
np.save(OUTPUT_FILE, filtered)

# Statistics
print(f"Input samples       : {len(ecg)}")
print(f"Sampling rate       : {FS:.1f} Hz")
print(f"Duration            : {len(ecg) / FS:.2f} s")
print(f"Filter              : Butterworth bandpass")
print(f"Filter order        : {ORDER}")
print(f"Passband            : {LOWCUT}–{HIGHCUT} Hz")
print(f"Original min/max    : {ecg.min():.4f} / {ecg.max():.4f}")
print(f"Filtered min/max    : {filtered.min():.4f} / {filtered.max():.4f}")
print(f"Output              : {OUTPUT_FILE}")

# Plot first 5 seconds for comparison
t = np.arange(len(ecg)) / FS
mask = t <= 5.0

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(t[mask], ecg[mask], linewidth=0.8)
axes[0].set_title("Original synthetic ECG")
axes[0].set_ylabel("Amplitude (arbitrary units)")
axes[0].grid(True)

axes[1].plot(t[mask], filtered[mask], linewidth=0.8)
axes[1].set_title("Filtered ECG — 0.5–40 Hz")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude (arbitrary units)")
axes[1].grid(True)

fig.tight_layout()
fig.savefig(PLOT_FILE, dpi=150)
plt.show()

print(f"Comparison plot     : {PLOT_FILE}")
print("\nSTAGE 2B COMPLETE")
print("Next: inspect filtering before implementing R-peak detection.")
