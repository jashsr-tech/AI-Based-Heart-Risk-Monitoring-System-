"""
Dashboard Visual and Functional Verification Script
File: 04_machine_learning/verify_dashboard.py

Tests the dashboard's rendering components and saves verification preview images
to 05_results/pipeline_tests/ for visual inspection.
"""

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_pipeline import ECGPipeline

TEST_CASES_DIR = PROJECT_ROOT / "01_signal_generation" / "test_cases"
OUTPUT_DIR = PROJECT_ROOT / "05_results" / "pipeline_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def render_preview(case_name, file_path, output_png):
    pipeline = ECGPipeline()
    res = pipeline.process(file_path)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    fs = res.sampling_rate
    t_max = 6.0
    time_axis = np.arange(len(res.filtered_ecg)) / fs
    mask = time_axis <= t_max

    ax.plot(time_axis[mask], res.filtered_ecg[mask], color="#2b6cb0", lw=1.0, label="Filtered ECG (0.5–40 Hz)")

    # PQRST markers
    pqrst = res.pqrst_matrix
    p_t = pqrst[:, 0]
    q_t = pqrst[:, 1]
    r_t = pqrst[:, 2]
    s_t = pqrst[:, 3]
    t_t = pqrst[:, 4]

    def plot_pts(times, col, mark, lbl):
        v = (~np.isnan(times)) & (times <= t_max)
        if np.any(v):
            pts = times[v]
            idx = np.clip(np.rint(pts * fs).astype(int), 0, len(res.filtered_ecg) - 1)
            ax.scatter(pts, res.filtered_ecg[idx], color=col, marker=mark, s=30, zorder=5, label=lbl)

    plot_pts(p_t, "#805ad5", "o", "P peak")
    plot_pts(q_t, "#e53e3e", "v", "Q peak")
    plot_pts(r_t, "#38a169", "^", "R peak")
    plot_pts(s_t, "#dd6b20", "v", "S peak")
    plot_pts(t_t, "#b7791f", "o", "T peak")

    hr = res.features.get("HeartRate", 0.0)
    pred_cls = res.predicted_class
    norm_p = res.class_probabilities.get(0, 0.0) * 100
    abn_p = res.class_probabilities.get(1, 0.0) * 100

    ax.set_title(
        f"Dashboard Waveform Preview: {case_name}\n"
        f"Result: {pred_cls} (Normal: {norm_p:.1f}%, Abnormal: {abn_p:.1f}%) | HR: {hr:.1f} BPM | SpO2: Not connected",
        fontsize=10,
        fontweight="bold"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (arb. units)")
    ax.set_xlim(0, t_max)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", ncol=6, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
    print(f"Saved preview: {output_png.name}")

if __name__ == "__main__":
    print("Generating Dashboard preview plots...")
    render_preview("Case A (Normal Synthetic)", TEST_CASES_DIR / "normal_ecg.npy", OUTPUT_DIR / "dashboard_preview_case_a.png")
    render_preview("Case B (Simulated PVC)", TEST_CASES_DIR / "pvc_ecg.npy", OUTPUT_DIR / "dashboard_preview_case_b.png")
    print("Preview generation complete.")
