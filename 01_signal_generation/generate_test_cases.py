"""
ECG FINAL PRO — Demonstration Test Cases Generator
Generates two controlled synthetic demonstration signals:
  1. Normal-looking synthetic ECG (~72 BPM, Q-to-S peak width ~75 ms)
  2. Simulated PVC-like ECG (premature beat timing, wide QRS, discordant T wave)

Outputs are saved in:
  01_signal_generation/test_cases/

NOTE:
These are synthetic software-test signals for demonstration purposes.
They are not real clinical patient recordings.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

FS = 360.0                     # Samples per second
DURATION = 60.0                # Seconds (21,600 samples)
TARGET_HR = 72.0               # Nominal BPM
BASE_RR = 60.0 / TARGET_HR     # ~0.8333 seconds

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "test_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# Normal morphology design targets (Q-to-S ~75 ms)
NORM_P_AMP = 0.60
NORM_Q_AMP = -0.80
NORM_R_AMP = 4.50
NORM_S_AMP = -1.00
NORM_T_AMP = 1.00

NORM_P_WIDTH = 0.025
NORM_Q_WIDTH = 0.010
NORM_R_WIDTH = 0.011
NORM_S_WIDTH = 0.011
NORM_T_WIDTH = 0.050

NORM_P_OFFSET = -0.160
NORM_Q_OFFSET = -0.035
NORM_R_OFFSET = 0.000
NORM_S_OFFSET = 0.040          # 0.040 - (-0.035) = 0.075 s (75 ms)
NORM_T_OFFSET = 0.300

# Controlled natural RR variation for normal rhythm (seconds)
RR_VARIATION = np.array([
    -0.080, -0.045, -0.020,  0.010,
     0.035,  0.065,  0.090,  0.055,
     0.025, -0.005, -0.035, -0.070,
     0.080,  0.045,  0.015, -0.015,
    -0.050, -0.085, -0.040,  0.005
])

def gaussian_wave(t, center, width, amplitude):
    z = (t - center) / width
    return amplitude * np.exp(-0.5 * z * z)


# ============================================================
# 2. GENERATE CASE A: NORMAL SYNTHETIC ECG
# ============================================================

def generate_normal_ecg():
    sample_count = int(round(FS * DURATION))
    time = np.arange(sample_count, dtype=float) / FS

    r_times = [0.850]
    rr_index = 0

    while True:
        variation = RR_VARIATION[rr_index % len(RR_VARIATION)]
        rr = float(np.clip(BASE_RR + variation, 0.65, 1.05))
        next_r = r_times[-1] + rr
        if next_r + NORM_T_OFFSET >= DURATION:
            break
        r_times.append(next_r)
        rr_index += 1

    r_times = np.asarray(r_times)

    ground_truth = np.column_stack([
        r_times + NORM_P_OFFSET,
        r_times + NORM_Q_OFFSET,
        r_times + NORM_R_OFFSET,
        r_times + NORM_S_OFFSET,
        r_times + NORM_T_OFFSET,
    ])

    ecg = np.zeros(sample_count, dtype=float)
    for r_time in r_times:
        ecg += gaussian_wave(time, r_time + NORM_P_OFFSET, NORM_P_WIDTH, NORM_P_AMP)
        ecg += gaussian_wave(time, r_time + NORM_Q_OFFSET, NORM_Q_WIDTH, NORM_Q_AMP)
        ecg += gaussian_wave(time, r_time + NORM_R_OFFSET, NORM_R_WIDTH, NORM_R_AMP)
        ecg += gaussian_wave(time, r_time + NORM_S_OFFSET, NORM_S_WIDTH, NORM_S_AMP)
        ecg += gaussian_wave(time, r_time + NORM_T_OFFSET, NORM_T_WIDTH, NORM_T_AMP)

    # Baseline wander and small noise
    ecg += 0.010 * np.sin(2 * np.pi * 0.20 * time)
    rng = np.random.default_rng(42)
    ecg += rng.normal(0.0, 0.003, sample_count)

    # Save data
    np.save(OUTPUT_DIR / "normal_ecg.npy", ecg)
    np.save(OUTPUT_DIR / "normal_ground_truth.npy", ground_truth)
    np.savetxt(
        OUTPUT_DIR / "normal_ecg.csv",
        np.column_stack((time, ecg)),
        delimiter=",",
        header="Time_s,ECG_amplitude",
        comments="",
        fmt="%.8f"
    )
    np.savetxt(
        OUTPUT_DIR / "normal_ground_truth.csv",
        ground_truth,
        delimiter=",",
        header="P_peak_s,Q_peak_s,R_peak_s,S_peak_s,T_peak_s",
        comments="",
        fmt="%.8f"
    )

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    mask5 = time <= 5.0
    axes[0].plot(time[mask5], ecg[mask5], color="#1f77b4", linewidth=0.9)
    axes[0].set_title("Case A: Normal Synthetic ECG (~72 BPM, Q-to-S ~75 ms) — First 5 Seconds")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (arb. units)")
    axes[0].grid(True, alpha=0.4)

    beat_idx = 4
    beat_center = r_times[beat_idx]
    beat_mask = (time >= beat_center - 0.25) & (time <= beat_center + 0.45)
    axes[1].plot(time[beat_mask], ecg[beat_mask], color="#1f77b4", linewidth=1.2)

    for label, col, color in zip(["P", "Q", "R", "S", "T"], range(5), ["purple", "red", "green", "orange", "brown"]):
        t_target = ground_truth[beat_idx, col]
        axes[1].axvline(t_target, linestyle="--", alpha=0.7, color=color, label=f"{label} target ({t_target:.3f} s)")

    qs_ms = (ground_truth[beat_idx, 3] - ground_truth[beat_idx, 1]) * 1000
    axes[1].set_title(f"One Representative Normal Beat (Design Q-to-S = {qs_ms:.1f} ms)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (arb. units)")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(loc="upper right", ncol=5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "normal_validation.png", dpi=150)
    plt.close(fig)

    print(f"[Case A] Saved normal synthetic signal ({len(ecg)} samples, {len(r_times)} beats, Q-S={qs_ms:.1f} ms)")
    return ecg, ground_truth, r_times


# ============================================================
# 3. GENERATE CASE B: SIMULATED PVC-LIKE SYNTHETIC ECG
# ============================================================

def generate_pvc_ecg():
    sample_count = int(round(FS * DURATION))
    time = np.arange(sample_count, dtype=float) / FS

    # Beat schedule: Normal beats with periodic PVCs (every 5th beat)
    # The PVC arrives early (0.55 * BASE_RR) followed by a compensatory pause (1.45 * BASE_RR)
    r_times = []
    is_pvc = []
    current_time = 0.850
    beat_counter = 1

    while True:
        if current_time + 0.450 >= DURATION:
            break
        r_times.append(current_time)

        if beat_counter % 5 == 0:
            is_pvc.append(True)
            # Full compensatory pause follows the PVC
            next_interval = BASE_RR * 1.45
        elif (beat_counter + 1) % 5 == 0:
            # Beat preceding PVC: premature beat arrives soon!
            is_pvc.append(False)
            next_interval = BASE_RR * 0.55
        else:
            is_pvc.append(False)
            next_interval = BASE_RR

        current_time += next_interval
        beat_counter += 1

    r_times = np.asarray(r_times)
    is_pvc = np.asarray(is_pvc, dtype=bool)

    ecg = np.zeros(sample_count, dtype=float)
    ground_truth_rows = []

    # PVC Morphology parameters:
    # No P wave; broad Q, large R, deep wide S (QRS duration ~135 ms); discordant inverted T
    PVC_Q_OFFSET = -0.065
    PVC_R_OFFSET = 0.000
    PVC_S_OFFSET = 0.070           # 0.070 - (-0.065) = 0.135 s (135 ms)
    PVC_T_OFFSET = 0.280

    for r_time, pvc in zip(r_times, is_pvc):
        if not pvc:
            # Normal beat
            ecg += gaussian_wave(time, r_time + NORM_P_OFFSET, NORM_P_WIDTH, NORM_P_AMP)
            ecg += gaussian_wave(time, r_time + NORM_Q_OFFSET, NORM_Q_WIDTH, NORM_Q_AMP)
            ecg += gaussian_wave(time, r_time + NORM_R_OFFSET, NORM_R_WIDTH, NORM_R_AMP)
            ecg += gaussian_wave(time, r_time + NORM_S_OFFSET, NORM_S_WIDTH, NORM_S_AMP)
            ecg += gaussian_wave(time, r_time + NORM_T_OFFSET, NORM_T_WIDTH, NORM_T_AMP)
            ground_truth_rows.append([
                r_time + NORM_P_OFFSET,
                r_time + NORM_Q_OFFSET,
                r_time + NORM_R_OFFSET,
                r_time + NORM_S_OFFSET,
                r_time + NORM_T_OFFSET,
            ])
        else:
            # PVC Ectopic Beat (no P wave, wide QRS, inverted T)
            ecg += gaussian_wave(time, r_time + PVC_Q_OFFSET, 0.025, -1.20)
            ecg += gaussian_wave(time, r_time + PVC_R_OFFSET, 0.030, 4.20)
            ecg += gaussian_wave(time, r_time + PVC_S_OFFSET, 0.035, -2.50)
            ecg += gaussian_wave(time, r_time + PVC_T_OFFSET, 0.070, -1.80)
            ground_truth_rows.append([
                np.nan,                      # Absent P wave
                r_time + PVC_Q_OFFSET,
                r_time + PVC_R_OFFSET,
                r_time + PVC_S_OFFSET,
                r_time + PVC_T_OFFSET,
            ])

    # Baseline wander and noise
    ecg += 0.010 * np.sin(2 * np.pi * 0.20 * time)
    rng = np.random.default_rng(42)
    ecg += rng.normal(0.0, 0.003, sample_count)

    ground_truth = np.asarray(ground_truth_rows, dtype=float)

    # Save data
    np.save(OUTPUT_DIR / "pvc_ecg.npy", ecg)
    np.save(OUTPUT_DIR / "pvc_ground_truth.npy", ground_truth)
    np.savetxt(
        OUTPUT_DIR / "pvc_ecg.csv",
        np.column_stack((time, ecg)),
        delimiter=",",
        header="Time_s,ECG_amplitude",
        comments="",
        fmt="%.8f"
    )
    np.savetxt(
        OUTPUT_DIR / "pvc_ground_truth.csv",
        ground_truth,
        delimiter=",",
        header="P_peak_s,Q_peak_s,R_peak_s,S_peak_s,T_peak_s",
        comments="",
        fmt="%.8f"
    )

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    mask6 = time <= 6.0
    axes[0].plot(time[mask6], ecg[mask6], color="#d62728", linewidth=0.9)
    axes[0].set_title("Case B: Simulated PVC-Like ECG (Premature Beat, Wide QRS, Inverted T) — First 6 Seconds")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (arb. units)")
    axes[0].grid(True, alpha=0.4)

    # Annotate premature beats in the first 6 seconds
    for r_time, pvc in zip(r_times[r_times <= 6.0], is_pvc[r_times <= 6.0]):
        if pvc:
            axes[0].annotate(
                "Simulated PVC\n(Ectopic Beat)",
                xy=(r_time, 4.2),
                xytext=(r_time, 5.2),
                arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.5),
                ha="center", fontsize=9, fontweight="bold", color="red"
            )

    # Bottom plot: comparison of one Normal beat vs one PVC beat
    norm_idx = 3
    pvc_idx = 4
    norm_r = r_times[norm_idx]
    pvc_r = r_times[pvc_idx]

    window = 0.35
    t_norm_rel = np.linspace(-window, window, 300)
    t_norm_abs = norm_r + t_norm_rel
    t_pvc_abs = pvc_r + t_norm_rel

    axes[1].plot(t_norm_rel * 1000, np.interp(t_norm_abs, time, ecg),
                 color="#1f77b4", linewidth=1.4, label="Normal Sinus Beat (Narrow QRS ~75 ms, Upright T)")
    axes[1].plot(t_norm_rel * 1000, np.interp(t_pvc_abs, time, ecg),
                 color="#d62728", linewidth=1.4, linestyle="--", label="Simulated PVC Beat (Wide QRS ~135 ms, Inverted T, No P)")

    axes[1].set_title("Direct Morphological Comparison: Normal Beat vs. Simulated PVC Beat")
    axes[1].set_xlabel("Time Relative to R-Peak (ms)")
    axes[1].set_ylabel("Amplitude (arb. units)")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pvc_validation.png", dpi=150)
    plt.close(fig)

    pvc_count = int(np.sum(is_pvc))
    print(f"[Case B] Saved simulated PVC signal ({len(ecg)} samples, {len(r_times)} beats, {pvc_count} PVC beats)")
    return ecg, ground_truth, r_times


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main():
    print("=" * 68)
    print("ECG FINAL PRO — STAGE 1: DEMONSTRATION TEST CASES GENERATOR")
    print("=" * 68)
    print(f"Sampling frequency : {FS} Hz")
    print(f"Signal duration    : {DURATION} seconds ({int(round(FS * DURATION))} samples)")
    print(f"Output directory   : {OUTPUT_DIR}\n")

    generate_normal_ecg()
    generate_pvc_ecg()

    print("\nGeneration completed successfully.")
    print("Files created in test_cases/:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    print("=" * 68)


if __name__ == "__main__":
    main()
