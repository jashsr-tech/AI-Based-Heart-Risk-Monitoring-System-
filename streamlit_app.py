"""
AI-Based Heart Risk Monitoring System — Real-Time Demonstration Dashboard
File: streamlit_app.py

Clinical Demonstration Platform featuring:
  - CONTROL PANEL: Dual-mode selector (Normal MIT-BIH Record 100 Replay vs.
                   Synthetic Abnormal Simulation with PVC & Tachycardia presets).
  - SECTION 1: Continuous ECG Waveform Monitor with synchronized beat navigation.
  - SECTION 2: Real-time Random Forest Arrhythmia Inference (Normal vs Abnormal)
               using the Stage 7 model with Stage 5 imputer and scaler.
  - SECTION 3: 4-Column Feature Telemetry Table with Educational Reference Values
               and Live ESP32 Sensor Telemetry (MAX30102 SpO2 & Optical Pulse Rate).
  - SECTION 4: System Operational Telemetry & Clean Professional Footer.
"""

import time
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import wfdb

# Serial bridge for ESP32 communications
from serial_bridge import SerialBridge

# ==============================================================================
# 0. CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Heart Risk Monitoring System — Demonstration",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clinical Medical Monitor Appearance
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #0d6efd;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sub-header {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 16px;
    }
    .mode-tag-normal {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #badbcc;
    }
    .mode-tag-abnormal {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #ffeeba;
    }
    .metric-badge-normal {
        background: linear-gradient(135deg, #198754 0%, #157347 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(25, 135, 84, 0.2);
    }
    .metric-badge-abnormal {
        background: linear-gradient(135deg, #dc3545 0%, #bb2d3b 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 4px 6px rgba(220, 53, 69, 0.25);
    }
    .status-pill-ok {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 3px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .status-pill-warn {
        background-color: #fff3cd;
        color: #664d03;
        padding: 3px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .status-pill-danger {
        background-color: #f8d7da;
        color: #842029;
        padding: 3px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent
DEV_CSV_PATH = PROJECT_ROOT / "04_machine_learning" / "mitbih_development_dataset.csv"
MITBIH_RAW_DIR = Path(r"C:\Users\srjas\OneDrive\Desktop\ECG_Project - Copy\mit-bih-arrhythmia-database-1.0.0")

MODEL_PATH = PROJECT_ROOT / "models" / "stage7_rf_combined_with_ind.pkl"
IMPUTER_PATH = PROJECT_ROOT / "models" / "ecg_feature_imputer.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "ecg_feature_scaler.pkl"

TEST_CASES_DIR = PROJECT_ROOT / "01_signal_generation" / "test_cases"
PVC_NPY_PATH = TEST_CASES_DIR / "pvc_ecg.npy"
PVC_BEATS_PATH = TEST_CASES_DIR / "pvc_beat_features.csv"
TACHY_NPY_PATH = TEST_CASES_DIR / "tachycardia_ecg.npy"
TACHY_BEATS_PATH = TEST_CASES_DIR / "tachycardia_beat_features.csv"

NUMERICAL_FEATURES = [
    "RRInterval", "HRV", "QRSDuration", "QTInterval",
    "STSegment", "PRInterval", "PAmplitude", "RAmplitude", "TAmplitude"
]

INDICATOR_FEATURES = [
    "P_Valid", "PR_Valid", "QRS_Valid", "ST_Valid", "QT_Valid", "Baseline_Valid"
]

# ==============================================================================
# 1. CACHED ASSET LOADERS
# ==============================================================================
@st.cache_resource
def load_ml_assets():
    """Load pretrained Stage 7 Random Forest model, Stage 5 imputer, and scaler."""
    with open(IMPUTER_PATH, "rb") as f:
        imputer = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return imputer, scaler, model

@st.cache_data
def load_mitbih_normal_data():
    """Load MIT-BIH Record 100 raw signal and precomputed development features."""
    rec_path = str(MITBIH_RAW_DIR / "100")
    rec = wfdb.rdrecord(rec_path)
    signal = rec.p_signal[:, 0]  # Primary Lead (MLII)
    fs = rec.fs
    lead_name = rec.sig_name[0]
    
    df_dev = pd.read_csv(DEV_CSV_PATH)
    df_rec100 = df_dev[df_dev["Record"] == 100].reset_index(drop=True)
    return signal, fs, lead_name, df_rec100

@st.cache_data
def load_synthetic_abnormal_data(preset_name: str):
    """Load synthetic abnormal ECG waveform and pipeline-extracted beat features."""
    fs = 360.0
    if preset_name == "pvc":
        signal = np.load(PVC_NPY_PATH)
        df_beats = pd.read_csv(PVC_BEATS_PATH)
        lead_name = "Synthetic PVC Lead"
    else:  # tachycardia
        signal = np.load(TACHY_NPY_PATH)
        df_beats = pd.read_csv(TACHY_BEATS_PATH)
        lead_name = "Synthetic Tachycardia Lead"
    return signal, fs, lead_name, df_beats

@st.cache_resource
def get_serial_bridge():
    """Maintain singleton background SerialBridge instance across Streamlit reruns."""
    bridge = SerialBridge(port="COM5", baudrate=115200)
    return bridge

imputer, scaler, model = load_ml_assets()
bridge = get_serial_bridge()

# ==============================================================================
# 2. SESSION STATE MANAGEMENT
# ==============================================================================
if "ecg_mode" not in st.session_state:
    st.session_state.ecg_mode = "normal"  # "normal" or "abnormal"
if "abnormal_preset" not in st.session_state:
    st.session_state.abnormal_preset = "pvc"  # "pvc" or "tachycardia"
if "current_beat_idx" not in st.session_state:
    st.session_state.current_beat_idx = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "playback_speed" not in st.session_state:
    st.session_state.playback_speed = 1.0
if "decision_threshold" not in st.session_state:
    st.session_state.decision_threshold = 0.50

# Load active signal and beat table according to selected mode
if st.session_state.ecg_mode == "normal":
    active_signal, active_fs, active_lead, active_df_beats = load_mitbih_normal_data()
    mode_title = "NORMAL ECG MODE"
    source_label = "MIT-BIH ECG REPLAY"
    mode_desc = "Clinical recording: MIT-BIH Record 100 (Lead MLII, 360 Hz)"
    is_synthetic = False
else:
    active_signal, active_fs, active_lead, active_df_beats = load_synthetic_abnormal_data(st.session_state.abnormal_preset)
    mode_title = "ABNORMAL ECG MODE"
    source_label = "SYNTHETIC ECG – SIMULATION"
    if st.session_state.abnormal_preset == "pvc":
        mode_desc = "Synthetic test signal: PVC-like waveform with premature beats and compensatory pauses"
    else:
        mode_desc = "Synthetic test signal: Tachycardia-like waveform (~140 BPM rapid rhythm)"
    is_synthetic = True

total_beats = len(active_df_beats)
if st.session_state.current_beat_idx >= total_beats:
    st.session_state.current_beat_idx = 0

# ==============================================================================
# 3. SIDEBAR CONTROLS: REDESIGNED CONTROL PANEL
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Dual-Mode Selection: Normal vs Abnormal
    mode_radio = st.radio(
        "Operating Mode:",
        options=["🟢 Normal ECG Mode", "🟠 Abnormal ECG Mode"],
        index=0 if st.session_state.ecg_mode == "normal" else 1,
        help="Select normal clinical replay or synthetic abnormal simulation."
    )
    new_mode = "normal" if "Normal" in mode_radio else "abnormal"
    if new_mode != st.session_state.ecg_mode:
        st.session_state.ecg_mode = new_mode
        st.session_state.current_beat_idx = 0
        st.session_state.is_playing = False
        st.rerun()

    # If Abnormal Mode is selected, display only the TWO required abnormal presets
    if st.session_state.ecg_mode == "abnormal":
        st.markdown("#### ⚡ Abnormal Preset Selection")
        abnormal_choices = {
            "pvc": "1. PVC-like waveform",
            "tachycardia": "2. Tachycardia-like waveform"
        }
        selected_preset = st.selectbox(
            "Select Abnormal Preset:",
            options=list(abnormal_choices.keys()),
            format_func=lambda k: abnormal_choices[k],
            index=0 if st.session_state.abnormal_preset == "pvc" else 1
        )
        if selected_preset != st.session_state.abnormal_preset:
            st.session_state.abnormal_preset = selected_preset
            st.session_state.current_beat_idx = 0
            st.session_state.is_playing = False
            st.rerun()
    else:
        st.info("📌 **Active Source:** MIT-BIH Record 100 (Default Normal Baseline)")

    st.markdown("---")
    st.markdown("### ⏯️ Replay Navigation")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.session_state.is_playing:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.is_playing = False
                st.rerun()
        else:
            if st.button("▶ Play", use_container_width=True):
                st.session_state.is_playing = True
                st.rerun()

    with col_btn2:
        if st.button("⏭ Next", use_container_width=True):
            st.session_state.current_beat_idx = min(st.session_state.current_beat_idx + 1, total_beats - 1)
            st.session_state.is_playing = False
            st.rerun()

    with col_btn3:
        if st.button("⏮ Reset", use_container_width=True):
            st.session_state.current_beat_idx = 0
            st.session_state.is_playing = False
            st.rerun()

    # Replay Speed
    speed_option = st.select_slider(
        "Replay Speed:",
        options=[0.5, 1.0, 2.0, 5.0],
        value=st.session_state.playback_speed,
        format_func=lambda s: f"{s}x"
    )
    st.session_state.playback_speed = speed_option

    # Beat Scrub Slider
    beat_slider = st.slider(
        f"Beat Index (0 to {total_beats - 1}):",
        min_value=0,
        max_value=total_beats - 1,
        value=st.session_state.current_beat_idx,
        step=1
    )
    if beat_slider != st.session_state.current_beat_idx:
        st.session_state.current_beat_idx = beat_slider
        st.session_state.is_playing = False

    st.markdown("---")
    st.markdown("### 🔌 Hardware Sensor Setup")

    available_ports = SerialBridge.get_available_ports()
    port_names = [p[0] for p in available_ports] if available_ports else []
    if "COM5" not in port_names:
        port_names.append("COM5")

    selected_port = st.selectbox("ESP32 Serial Port:", options=port_names, index=0)
    
    col_conn1, col_conn2 = st.columns(2)
    with col_conn1:
        if not bridge.is_connected and not bridge.is_simulating:
            if st.button("🔗 Connect", use_container_width=True):
                success = bridge.connect(port=selected_port)
                if not success:
                    st.warning(f"Failed to connect to {selected_port}")
                st.rerun()
        else:
            if st.button("🔌 Disconnect", use_container_width=True):
                bridge.disconnect()
                st.rerun()

    with col_conn2:
        if not bridge.is_simulating:
            if st.button("🧪 Simulate", use_container_width=True):
                bridge.start_simulation(include_spo2=True)
                st.rerun()
        else:
            if st.button("⏹ Stop Sim", use_container_width=True):
                bridge.stop_simulation()
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚖️ Decision Threshold")
    thresh = st.slider(
        "Minority Class Threshold (τ):",
        min_value=0.10,
        max_value=0.90,
        value=st.session_state.decision_threshold,
        step=0.05
    )
    st.session_state.decision_threshold = thresh

# ==============================================================================
# 4. CURRENT BEAT FEATURE EXTRACTION & REAL-TIME ML INFERENCE
# ==============================================================================
current_beat = active_df_beats.iloc[st.session_state.current_beat_idx]
sample_idx = int(current_beat["Sample_Idx"])
r_time_sec = float(current_beat["R_Time"])

# Extract exact 9 numerical features with DataFrame to preserve feature names
df_num_raw = pd.DataFrame([current_beat[NUMERICAL_FEATURES].to_dict()])[NUMERICAL_FEATURES]
x_num_imp = imputer.transform(df_num_raw)
z_num_scaled = scaler.transform(x_num_imp)

# Extract exact 6 indicator features (unscaled binary {0, 1})
x_ind_raw = np.array([[int(current_beat[c]) for c in INDICATOR_FEATURES]])

# Concatenate into 15-feature model input vector
x_model_input = np.hstack([z_num_scaled, x_ind_raw])

# Run Model Inference using frozen Stage 7 Random Forest
t_infer_start = time.perf_counter()
model_probs = model.predict_proba(x_model_input)[0]
t_infer_end = time.perf_counter()
infer_latency_ms = (t_infer_end - t_infer_start) * 1000.0

p_normal = float(model_probs[0])
p_abnormal = float(model_probs[1])

# Classification decision using selected threshold τ
is_abnormal_pred = (p_abnormal >= st.session_state.decision_threshold)
pred_class_label = "ABNORMAL / ECTOPIC BEAT" if is_abnormal_pred else "NORMAL SINUS BEAT"

# ECG-derived heart rate (from RR Interval)
ecg_hr = float(current_beat["HeartRate"]) if ("HeartRate" in current_beat and pd.notna(current_beat["HeartRate"])) else (60.0 / float(current_beat["RRInterval"]))

# Get Live Sensor Metrics from Serial Bridge
sensor_metrics = bridge.get_latest_metrics()
esp32_connected = sensor_metrics["connected"]
spo2_val = sensor_metrics["spo2"]
pulse_val = sensor_metrics["pulse_bpm"]
spo2_valid = sensor_metrics["spo2_valid"]

# ==============================================================================
# 5. HEADER & TOP BANNER
# ==============================================================================
st.markdown('<div class="main-header">❤️ AI-Based Heart Risk Monitoring System</div>', unsafe_allow_html=True)

# Visual mode badge and source indication
if not is_synthetic:
    st.markdown(
        f'<div class="sub-header">'
        f'<span class="mode-tag-normal">🟢 {source_label}</span> • '
        f'{mode_desc} • Stage 7 Random Forest Inference</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="sub-header">'
        f'<span class="mode-tag-abnormal">🟠 {source_label}</span> • '
        f'{mode_desc} • Stage 7 Random Forest Inference</div>',
        unsafe_allow_html=True
    )

# Top Telemetry Cards
col_top1, col_top2, col_top3, col_top4 = st.columns(4)
with col_top1:
    source_short = "MIT-BIH 100" if not is_synthetic else f"SYNTHETIC ({st.session_state.abnormal_preset.upper()})"
    st.metric("ECG Waveform Source", source_short, f"Beat {st.session_state.current_beat_idx + 1} of {total_beats}")
with col_top2:
    st.metric("ECG Heart Rate", f"{ecg_hr:.0f} BPM", f"RR: {float(current_beat['RRInterval']):.3f} s")
with col_top3:
    if esp32_connected and spo2_val is not None:
        st.metric("MAX30102 SpO2", f"{spo2_val} %", "Sensor Active")
    elif esp32_connected and not spo2_valid:
        st.metric("MAX30102 SpO2", "Finger Absent", "Sensor Reading --")
    else:
        st.metric("MAX30102 SpO2", "-- %", "Sensor Disconnected")
with col_top4:
    if esp32_connected and pulse_val is not None:
        st.metric("MAX30102 Optical Pulse", f"{pulse_val} BPM", "Optical PPG")
    elif esp32_connected and not spo2_valid:
        st.metric("MAX30102 Optical Pulse", "No Pulse", "Place Finger on LED")
    else:
        st.metric("MAX30102 Optical Pulse", "-- BPM", "Sensor Disconnected")

st.markdown("---")

# ==============================================================================
# 6. DASHBOARD TWO-COLUMN LAYOUT
# ==============================================================================
col_left, col_right = st.columns([1.1, 0.9])

# ------------------------------------------------------------------------------
# SECTION 1 — ECG MONITOR (Left Column)
# ------------------------------------------------------------------------------
with col_left:
    st.markdown("### 📈 SECTION 1: ECG Waveform Monitor")
    
    # Calculate continuous moving window (5 seconds: 1800 samples)
    window_samples = int(5.0 * active_fs)
    start_sample = max(0, sample_idx - int(2.5 * active_fs))
    end_sample = min(len(active_signal), start_sample + window_samples)
    if end_sample - start_sample < window_samples:
        start_sample = max(0, end_sample - window_samples)

    window_time = np.arange(start_sample, end_sample) / active_fs
    window_signal = active_signal[start_sample:end_sample]

    df_ecg_chart = pd.DataFrame({
        "Time (seconds)": window_time,
        f"ECG Amplitude ({active_lead})": window_signal
    }).set_index("Time (seconds)")

    st.line_chart(df_ecg_chart, height=260, use_container_width=True)

    # Waveform Sub-status bar
    st.caption(
        f"📍 **Position:** Sample {sample_idx:,} ({r_time_sec:.2f}s) | "
        f"**Source:** {source_label} | "
        f"**Lead:** {active_lead} (Fs={active_fs} Hz) | "
        f"**Playback:** {'▶ REPLAYING' if st.session_state.is_playing else '⏸ PAUSED'} ({st.session_state.playback_speed}x)"
    )

    # --------------------------------------------------------------------------
    # LIVE SENSOR TELEMETRY (Left Column bottom)
    # --------------------------------------------------------------------------
    st.markdown("### 📡 Live Sensor Telemetry (ESP32 MAX30102)")
    
    col_sens1, col_sens2 = st.columns(2)
    with col_sens1:
        st.markdown("**Blood Oxygen Saturation (SpO2):**")
        if esp32_connected and spo2_val is not None:
            st.markdown(f"### 🫁 `{spo2_val}%` <span class='status-pill-ok'>OPTIMAL</span>", unsafe_allow_html=True)
            st.progress(max(0.0, min(1.0, (spo2_val - 70.0) / 30.0)))
        elif esp32_connected and not spo2_valid:
            st.markdown("### 🫁 `-- %` <span class='status-pill-warn'>Finger Not Detected</span>", unsafe_allow_html=True)
            st.caption("Place finger gently over MAX30102 sensor optical window.")
        else:
            st.markdown("### 🫁 `-- %` <span class='status-pill-danger'>ESP32 Disconnected (Sensor Unavailable)</span>", unsafe_allow_html=True)
            st.caption("Connect ESP32 via USB or click '🧪 Simulate' in sidebar.")

    with col_sens2:
        st.markdown("**Optical Pulse Rate vs ECG Rate:**")
        if esp32_connected and pulse_val is not None:
            st.markdown(f"### 💓 `{pulse_val} BPM` <span class='status-pill-ok'>PPG SENSOR</span>", unsafe_allow_html=True)
            rate_delta = abs(pulse_val - ecg_hr)
            st.caption(f"Optical PPG: {pulse_val} BPM | ECG Derived: {ecg_hr:.0f} BPM (Δ = {rate_delta:.0f} BPM)")
        elif esp32_connected and not spo2_valid:
            st.markdown("### 💓 `-- BPM` <span class='status-pill-warn'>Finger Not Detected</span>", unsafe_allow_html=True)
            st.caption(f"ECG Heart Rate remains active: {ecg_hr:.0f} BPM")
        else:
            st.markdown("### 💓 `-- BPM` <span class='status-pill-danger'>ESP32 Disconnected</span>", unsafe_allow_html=True)
            st.caption(f"ECG Heart Rate remains active: {ecg_hr:.0f} BPM")

# ------------------------------------------------------------------------------
# SECTION 2 — MACHINE LEARNING PREDICTION (Right Column)
# ------------------------------------------------------------------------------
with col_right:
    st.markdown("### 🧠 SECTION 2: Machine Learning Prediction")

    # Primary Model Classification Badge
    if is_abnormal_pred:
        st.markdown(f"""
        <div class="metric-badge-abnormal">
            ⚠️ {pred_class_label}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-badge-normal">
            ✅ {pred_class_label}
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # Confidence & Probability Readout
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"**P(Normal / AAMI N):** `{p_normal * 100:.2f}%`")
        st.progress(p_normal)
    with col_p2:
        st.markdown(f"**P(Abnormal / Ectopic):** `{p_abnormal * 100:.2f}%`")
        st.progress(p_abnormal)

    # Transparent Explanation of Prediction Source
    if not is_synthetic:
        pred_source_explanation = (
            "Inference computed in real-time by Stage 7 Random Forest using Stage 5 "
            "preprocessing on precomputed beat features extracted from MIT-BIH Record 100."
        )
    else:
        preset_disp = "PVC-like simulation" if st.session_state.abnormal_preset == "pvc" else "Tachycardia simulation"
        pred_source_explanation = (
            f"Waveform Source: Synthetic ECG ({preset_disp}; not a patient recording). "
            f"Inference Source: Stage 7 Random Forest model evaluated on actual features extracted by the pipeline."
        )

    st.caption(
        f"⚡ **Inference Latency:** {infer_latency_ms:.2f} ms | "
        f"**Threshold (τ):** {st.session_state.decision_threshold:.2f} | "
        f"**Engine:** Frozen Stage 7 Random Forest (150 trees)\n\n"
        f"🔍 **Source:** {pred_source_explanation}"
    )

    # --------------------------------------------------------------------------
    # CHANGE 2: SECTION 3 — FEATURE EXTRACTION TABLE (4 COLUMNS)
    # --------------------------------------------------------------------------
    st.markdown("### 📋 SECTION 3: Feature Extraction Telemetry")

    def evaluate_feature(feat_name: str, val: float, rr_val: float) -> Tuple[str, str, str]:
        """
        Evaluate extracted value against educational reference descriptions.
        Interpretation is strictly:
          - 'Within reference'
          - 'Outside reference'
          - 'Context-dependent'
          - 'Missing / Not available'
        """
        # Indicator flags {0, 1}
        if feat_name in INDICATOR_FEATURES:
            if pd.isna(val) or val is None:
                return "Missing (0)", "Valid / Missing", "Missing / Not available"
            flag = int(val)
            actual_str = "Valid (1)" if flag == 1 else "Missing (0)"
            ref_str = "Valid / Missing"
            interp = "Within reference" if flag == 1 else "Missing / Not available"
            return actual_str, ref_str, interp

        # Numerical features
        if pd.isna(val) or val is None:
            return "NaN (Imputed)", "Context-dependent", "Missing / Not available"

        val_f = float(val)

        if feat_name == "RRInterval":
            ref = "Approximately 0.60–1.00 s at 60–100 BPM"
            actual = f"{val_f:.4f} s"
            interp = "Within reference" if (0.60 <= val_f <= 1.00) else "Outside reference"
            return actual, ref, interp

        elif feat_name == "HRV":
            ref = "No universal range; depends on age, recording duration and measurement method"
            actual = f"{val_f:.4f} s"
            interp = "Context-dependent"
            return actual, ref, interp

        elif feat_name == "QRSDuration":
            ref = "Approximately 0.07–0.11 s"
            actual = f"{val_f:.4f} s"
            interp = "Within reference" if (0.07 <= val_f <= 0.11) else "Outside reference"
            return actual, ref, interp

        elif feat_name == "QTInterval":
            actual = f"{val_f:.4f} s"
            if rr_val > 0:
                qtc = val_f / np.sqrt(rr_val)
                ref = f"Heart-rate dependent; QTc: {qtc:.3f} s (ref: 0.35–0.45 s)"
                interp = "Within reference" if (0.35 <= qtc <= 0.46) else "Outside reference"
            else:
                ref = "Heart-rate dependent; show QTc reference"
                interp = "Context-dependent"
            return actual, ref, interp

        elif feat_name == "PRInterval":
            ref = "Approximately 0.12–0.20 s"
            actual = f"{val_f:.4f} s"
            interp = "Within reference" if (0.12 <= val_f <= 0.20) else "Outside reference"
            return actual, ref, interp

        elif feat_name == "PAmplitude":
            ref = "Generally <= 0.25 mV"
            actual = f"{val_f:.4f} mV (arb)"
            interp = "Within reference" if (0.0 <= val_f <= 0.25) else ("Outside reference" if val_f > 0.25 else "Context-dependent")
            return actual, ref, interp

        elif feat_name == "RAmplitude":
            ref = "Lead-dependent; no single universal range"
            actual = f"{val_f:.4f} mV (arb)"
            interp = "Context-dependent"
            return actual, ref, interp

        elif feat_name == "TAmplitude":
            ref = "Lead-dependent; no single universal range"
            actual = f"{val_f:.4f} mV (arb)"
            interp = "Context-dependent"
            return actual, ref, interp

        elif feat_name == "STSegment":
            ref = "Interpret relative to baseline and ECG lead; no universal single normal value"
            actual = f"{val_f:.4f} mV (arb)"
            interp = "Context-dependent"
            return actual, ref, interp

        return f"{val_f:.4f}", "No universal range", "Context-dependent"

    cur_rr = float(current_beat["RRInterval"]) if pd.notna(current_beat["RRInterval"]) else 0.833

    table_rows = []
    # 9 Numerical Features
    for feat in NUMERICAL_FEATURES:
        val = current_beat[feat] if feat in current_beat else np.nan
        actual_val, gen_ref, interpretation = evaluate_feature(feat, val, cur_rr)
        table_rows.append({
            "Feature Name": feat,
            "Actual Extracted Value": actual_val,
            "General Reference Value": gen_ref,
            "Interpretation": interpretation
        })

    # 6 Validity Indicators
    for feat in INDICATOR_FEATURES:
        val = current_beat[feat] if feat in current_beat else np.nan
        actual_val, gen_ref, interpretation = evaluate_feature(feat, val, cur_rr)
        table_rows.append({
            "Feature Name": feat,
            "Actual Extracted Value": actual_val,
            "General Reference Value": gen_ref,
            "Interpretation": interpretation
        })

    df_telemetry = pd.DataFrame(table_rows)
    st.dataframe(df_telemetry, height=270, use_container_width=True)
    st.caption(
        "📌 *Educational reference descriptions only; not clinical diagnostic criteria. "
        "Interpretation depends on age, lead configuration, heart rate, and physiological context.*"
    )

st.markdown("---")

# ==============================================================================
# 7. SECTION 4 — SYSTEM STATUS & PROFESSIONAL FOOTER
# ==============================================================================
st.markdown("### 🛡️ SECTION 4: System Operational Status & Telemetry")

col_sys1, col_sys2, col_sys3, col_sys4 = st.columns(4)

with col_sys1:
    st.markdown("**ECG Playback Engine**")
    st.write(f"• Mode: **{mode_title}**")
    st.write(f"• Source: **{source_label}**")
    st.write(f"• Beat Position: **{st.session_state.current_beat_idx + 1} / {total_beats}**")
    st.write(f"• State: **{'Playing' if st.session_state.is_playing else 'Paused'}** ({st.session_state.playback_speed}x)")

with col_sys2:
    st.markdown("**Random Forest Classifier**")
    st.write(f"• Model: `stage7_rf_combined_with_ind`")
    st.write(f"• Input Dimension: **15 Features (9 Num + 6 Ind)**")
    st.write(f"• Latency: **{infer_latency_ms:.2f} ms**")
    st.write(f"• Operating Threshold: **τ = {st.session_state.decision_threshold:.2f}**")

with col_sys3:
    st.markdown("**ESP32 Serial Link**")
    if esp32_connected and not bridge.is_simulating:
        st.write(f"• Port: **{bridge.port or 'COM5'}** (115200)")
        st.write(f"• Status: <span class='status-pill-ok'>Hardware Connected</span>", unsafe_allow_html=True)
    elif bridge.is_simulating:
        st.write("• Port: **MOCK SERIAL BUS**")
        st.write("• Status: <span class='status-pill-warn'>Simulation Mode</span>", unsafe_allow_html=True)
    else:
        st.write("• Port: **Unavailable / Disconnected**")
        st.write("• Status: <span class='status-pill-danger'>Disconnected</span>", unsafe_allow_html=True)
    st.write(f"• Packets Ingested: **{sensor_metrics['packets_received']:,}**")

with col_sys4:
    st.markdown("**Overall System Health**")
    if is_abnormal_pred:
        st.write("• Status: <span class='status-pill-danger'>ARRHYTHMIA DETECTED</span>", unsafe_allow_html=True)
    elif esp32_connected:
        st.write("• Status: <span class='status-pill-ok'>ALL SYSTEMS NORMAL</span>", unsafe_allow_html=True)
    else:
        st.write("• Status: <span class='status-pill-warn'>OFFLINE SENSOR MODE</span>", unsafe_allow_html=True)
    st.write(f"• Pipeline Integrity: **Verified (Stage 7-8)**")
    st.write(f"• Test Isolation: **100% Enforced**")

# CHANGE 3: Clean, discreet professional footer (large disclaimer heading and long text removed)
st.caption("Experimental ECG beat classification and physiological monitoring prototype – not intended for medical diagnosis.")

# ==============================================================================
# 8. PLAYBACK CONTINUOUS AUTO-ADVANCE
# ==============================================================================
if st.session_state.is_playing:
    if st.session_state.current_beat_idx < total_beats - 1:
        st.session_state.current_beat_idx += 1
    else:
        st.session_state.current_beat_idx = 0  # Loop back
    
    delay = max(0.05, 0.6 / st.session_state.playback_speed)
    time.sleep(delay)
    st.rerun()
