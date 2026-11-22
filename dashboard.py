"""
AI-Based Heart Risk Monitoring System — Desktop Demonstration Dashboard
File: dashboard.py

A non-blocking Tkinter + Matplotlib desktop graphical interface providing:
  - Input selection:
      * Case A: Normal Synthetic ECG (~72 BPM)
      * Case B: Simulated PVC-Like ECG (Ectopic)
      * Load Custom File (.npy / .csv)
      * ESP32 Live Mode (AD8232 ECG + MAX30102 SpO2 via USB Serial or Live Simulation)
  - Real-time rolling oscilloscope display during live streaming
  - Filtered ECG waveform visualization with detected P-Q-R-S-T peak markers
  - Interactive navigation (Pan, Zoom, Time-range presets)
  - Heart Rate metric card (ECG-derived) and MAX30102 SpO2 / Optical Pulse card
  - Honest missing-value handling ("Not connected" / "Sensor unavailable")
  - AD8232 Lead-Off detection alert banner
  - Model classification badge and class probability readout
  - 10-Feature vector table with documented units and clinical distinction notes
  - Decision tree path explainability panel
  - Beat-by-beat measurement table (resolving normal vs PVC beat morphology)
  - Clear academic demonstration and medical disclaimers

Uses ecg_pipeline.py and serial_bridge.py with frozen models in 04_machine_learning/models/.
"""

import sys
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_pipeline import ECGPipeline, ECGPipelineResult
from serial_bridge import SerialBridge

# Paths to default test cases
TEST_CASES_DIR = PROJECT_ROOT / "01_signal_generation" / "test_cases"
NORMAL_CASE_PATH = TEST_CASES_DIR / "normal_ecg.npy"
PVC_CASE_PATH = TEST_CASES_DIR / "pvc_ecg.npy"

# Feature definitions and documentation
FEATURE_METADATA = {
    "HeartRate": ("BPM", "Average heart rate derived from mean RR interval (60 / mean_RR)."),
    "RRInterval": ("seconds", "Mean duration between consecutive detected R-peaks."),
    "PRInterval": ("seconds", "Peak-to-peak interval: P-peak to R-peak (Note: not clinical P-onset to QRS-onset)."),
    "QRSDuration": ("seconds", "Peak-to-peak width: Q-peak to S-peak (Note: proxy width, clinical is QRS onset-to-offset)."),
    "QTInterval": ("seconds", "Peak-to-peak interval: Q-peak to T-peak (Note: clinical is QRS onset to T-offset)."),
    "STSegment": ("arb. units", "Filtered signal amplitude sampled at S-peak + 60 ms."),
    "HRV": ("seconds", "Standard deviation of normal RR intervals (SDNN)."),
    "PAmplitude": ("arb. units", "Filtered signal amplitude sampled at detected P-peak."),
    "RAmplitude": ("arb. units", "Filtered signal amplitude sampled at detected R-peak."),
    "TAmplitude": ("arb. units", "Filtered signal amplitude sampled at detected T-peak."),
}


class ECGDashboard(tk.Tk):
    """Main Application Window for AI-Based Heart Risk Monitoring System."""

    def __init__(self, run_automated_test: bool = False):
        super().__init__()

        self.title("AI-Based Heart Risk Monitoring System — Demonstration Dashboard")
        self.geometry("1300x840")
        self.minsize(1080, 720)

        # Initialize pipeline & serial bridge
        try:
            self.pipeline = ECGPipeline()
        except Exception as exc:
            messagebox.showerror("Pipeline Initialization Error", f"Failed to load ECG pipeline:\n{exc}")
            self.destroy()
            return

        self.bridge = SerialBridge(fs=360.0, buffer_seconds=15.0)

        self.current_result: Optional[ECGPipelineResult] = None
        self.selected_file_path: Optional[Path] = None
        self.is_processing = False
        self.live_stream_active = False

        # Build UI layout
        self._setup_styles()
        self._build_layout()

        # Set default selection to Case A
        self.mode_var.set("case_a")
        self._on_mode_change()

        # Start periodic serial polling loop for live streaming (50 ms interval)
        self._poll_serial_state()

        # If launched in automated test mode, run verification sequence and exit
        if run_automated_test:
            self.after(250, self._run_automated_verification)

    # ========================================================
    # UI SETUP & STYLES
    # ========================================================
    def _setup_styles(self):
        self.style = ttk.Style(self)
        available = self.style.theme_names()
        if "clam" in available:
            self.style.theme_use("clam")

        self.style.configure(".", font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground="#1a365d")
        self.style.configure("Disclaimer.TLabel", font=("Segoe UI", 8, "italic"), foreground="#744210", background="#fefcbf")
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("Connect.TButton", font=("Segoe UI", 8, "bold"))

    def _build_layout(self):
        # 1. Top Banner (Title & Medical Disclaimer)
        top_frame = ttk.Frame(self, padding=(10, 8, 10, 4))
        top_frame.pack(side=tk.TOP, fill=tk.X)

        title_lbl = ttk.Label(
            top_frame,
            text="AI-Based Heart Risk Monitoring System",
            style="Header.TLabel"
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = ttk.Label(
            top_frame,
            text=" | ECE Final-Year Project Demonstration (ESP32 / AD8232 / MAX30102)",
            font=("Segoe UI", 10)
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # Disclaimer bar directly below title
        disclaimer_frame = tk.Frame(self, bg="#fefcbf", padx=8, pady=3, relief=tk.SOLID, bd=1)
        disclaimer_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(2, 4))

        disclaimer_text = (
            "⚠ ACADEMIC DEMONSTRATION ONLY: This prototype evaluates mathematical beat patterns for software validation. "
            "It is NOT a medical device and does NOT diagnose heart attacks, ischemia, or clinical cardiovascular conditions."
        )
        disclaimer_lbl = tk.Label(
            disclaimer_frame,
            text=disclaimer_text,
            bg="#fefcbf",
            fg="#744210",
            font=("Segoe UI", 8, "bold")
        )
        disclaimer_lbl.pack(side=tk.LEFT)

        # Lead-Off Alert Banner (Hidden by default, shown when AD8232 electrodes disconnect)
        self.lead_off_banner = tk.Frame(self, bg="#fed7d7", padx=8, pady=3, relief=tk.SOLID, bd=1)
        self.lead_off_lbl = tk.Label(
            self.lead_off_banner,
            text="⚠ ELECTRODE LEAD-OFF DETECTED: AD8232 leads are disconnected. Check RA / LA / RL electrode contacts.",
            bg="#fed7d7",
            fg="#742a2a",
            font=("Segoe UI", 8, "bold")
        )
        self.lead_off_lbl.pack(side=tk.LEFT)

        # 2. Main Horizontal Container (Left Controls + Right Panels)
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)

        # --- LEFT PANEL (Controls & Cards) ---
        left_container = ttk.Frame(main_pane, padding=6, width=330)
        main_pane.add(left_container, weight=0)

        # --- RIGHT PANEL (Waveform Plot + Tabs) ---
        right_container = ttk.Frame(main_pane, padding=6)
        main_pane.add(right_container, weight=1)

        self._build_left_panel(left_container)
        self._build_right_panel(right_container)

        # 3. Status Bar at bottom
        self.status_var = tk.StringVar(value="Ready. Select an input mode and click 'Process & Analyze Signal'.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(6, 3))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ========================================================
    # LEFT PANEL: CONTROLS & VITAL SIGNS CARDS
    # ========================================================
    def _build_left_panel(self, parent):
        # A. Input Selection Frame
        input_group = ttk.LabelFrame(parent, text=" 1. Input Signal Mode ", padding=8)
        input_group.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.mode_var = tk.StringVar(value="case_a")

        r1 = ttk.Radiobutton(
            input_group,
            text="Case A: Normal Synthetic ECG (~72 BPM)",
            variable=self.mode_var,
            value="case_a",
            command=self._on_mode_change
        )
        r1.pack(anchor=tk.W, pady=2)

        r2 = ttk.Radiobutton(
            input_group,
            text="Case B: Simulated PVC-Like ECG (Ectopic)",
            variable=self.mode_var,
            value="case_b",
            command=self._on_mode_change
        )
        r2.pack(anchor=tk.W, pady=2)

        r3 = ttk.Radiobutton(
            input_group,
            text="Load Custom File (.npy / .csv)...",
            variable=self.mode_var,
            value="custom",
            command=self._on_mode_change
        )
        r3.pack(anchor=tk.W, pady=2)

        r4 = ttk.Radiobutton(
            input_group,
            text="ESP32 Live Mode (Hardware / Stream)",
            variable=self.mode_var,
            value="esp32",
            command=self._on_mode_change
        )
        r4.pack(anchor=tk.W, pady=2)

        # File selector frame (for Custom mode)
        self.file_frame = ttk.Frame(input_group)
        self.file_frame.pack(fill=tk.X, pady=(4, 2))

        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, font=("Segoe UI", 8))
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.browse_btn = ttk.Button(self.file_frame, text="Browse", width=7, command=self._browse_custom_file)
        self.browse_btn.pack(side=tk.RIGHT)

        # ESP32 Serial Controls Frame (for ESP32 mode)
        self.serial_controls_frame = ttk.Frame(input_group)
        # Packed conditionally when in esp32 mode

        serial_top = ttk.Frame(self.serial_controls_frame)
        serial_top.pack(fill=tk.X, pady=2)

        ttk.Label(serial_top, text="Port:").pack(side=tk.LEFT, padx=(0, 2))
        self.port_combo = ttk.Combobox(serial_top, width=9, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(0, 2))

        self.refresh_ports_btn = ttk.Button(serial_top, text="↻", width=3, command=self._refresh_serial_ports)
        self.refresh_ports_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.connect_btn = ttk.Button(serial_top, text="Connect", style="Connect.TButton", width=8, command=self._toggle_serial_connection)
        self.connect_btn.pack(side=tk.RIGHT)

        # Simulation stream button (for testing when hardware USB is unplugged)
        sim_frame = ttk.Frame(self.serial_controls_frame)
        sim_frame.pack(fill=tk.X, pady=(4, 2))

        self.sim_stream_btn = ttk.Button(
            sim_frame,
            text="⚡ Start Mock ESP32 Stream (Demo)",
            command=self._toggle_simulation_stream
        )
        self.sim_stream_btn.pack(fill=tk.X)

        # Sampling rate entry
        fs_frame = ttk.Frame(input_group)
        fs_frame.pack(fill=tk.X, pady=(4, 4))

        ttk.Label(fs_frame, text="Sampling Rate (Hz):").pack(side=tk.LEFT)
        self.fs_var = tk.StringVar(value="360.0")
        fs_entry = ttk.Entry(fs_frame, textvariable=self.fs_var, width=8)
        fs_entry.pack(side=tk.RIGHT)

        # Main Action Button
        self.process_btn = ttk.Button(
            input_group,
            text="▶  Process & Analyze Signal",
            style="Action.TButton",
            command=self.start_processing
        )
        self.process_btn.pack(fill=tk.X, pady=(4, 2))

        # B. Vital Signs & Results Cards Frame
        vitals_group = ttk.LabelFrame(parent, text=" 2. Vitals & Classifier Result ", padding=8)
        vitals_group.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 4))

        # Heart Rate Card (ECG derived)
        hr_card = tk.Frame(vitals_group, bg="#ebf8ff", relief=tk.GROOVE, bd=1, padx=6, pady=6)
        hr_card.pack(fill=tk.X, pady=3)
        tk.Label(hr_card, text="HEART RATE (ECG R-PEAK DERIVED)", bg="#ebf8ff", fg="#2b6cb0", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)
        self.hr_val_lbl = tk.Label(hr_card, text="-- BPM", bg="#ebf8ff", fg="#2c5282", font=("Segoe UI", 16, "bold"))
        self.hr_val_lbl.pack(anchor=tk.W)

        # SpO2 Card (MAX30102 Optical PPG)
        self.spo2_card = tk.Frame(vitals_group, bg="#fffaf0", relief=tk.GROOVE, bd=1, padx=6, pady=6)
        self.spo2_card.pack(fill=tk.X, pady=3)
        tk.Label(self.spo2_card, text="PULSE OXIMETRY (MAX30102 SpO₂)", bg="#fffaf0", fg="#c05621", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)

        self.spo2_val_lbl = tk.Label(self.spo2_card, text="Not connected", bg="#fffaf0", fg="#9c4221", font=("Segoe UI", 14, "bold"))
        self.spo2_val_lbl.pack(anchor=tk.W)

        self.pulse_val_lbl = tk.Label(self.spo2_card, text="Optical Pulse: -- BPM", bg="#fffaf0", fg="#7b341e", font=("Segoe UI", 8))
        self.pulse_val_lbl.pack(anchor=tk.W)

        self.spo2_subtext_lbl = tk.Label(
            self.spo2_card,
            text="Awaiting live MAX30102 data.\nSynthetic values are not fabricated.",
            bg="#fffaf0",
            fg="#7b341e",
            font=("Segoe UI", 7, "italic"),
            justify=tk.LEFT
        )
        self.spo2_subtext_lbl.pack(anchor=tk.W, pady=(2, 0))

        # Classification Badge Card
        self.class_card = tk.Frame(vitals_group, bg="#edf2f7", relief=tk.GROOVE, bd=1, padx=6, pady=6)
        self.class_card.pack(fill=tk.X, pady=3)
        tk.Label(self.class_card, text="RANDOM FOREST ENSEMBLE CLASSIFICATION", bg="#edf2f7", fg="#4a5568", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)

        self.class_val_lbl = tk.Label(self.class_card, text="Awaiting Input", bg="#edf2f7", fg="#4a5568", font=("Segoe UI", 13, "bold"))
        self.class_val_lbl.pack(anchor=tk.W)

        self.class_prob_lbl = tk.Label(self.class_card, text="Normal: --% | Abnormal: --%", bg="#edf2f7", fg="#4a5568", font=("Segoe UI", 8))
        self.class_prob_lbl.pack(anchor=tk.W)

        # Progress bar during processing
        self.progress_bar = ttk.Progressbar(vitals_group, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, pady=(6, 2))

    # ========================================================
    # RIGHT PANEL: WAVEFORM PLOT & TABS
    # ========================================================
    def _build_right_panel(self, parent):
        split_pane = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        split_pane.pack(fill=tk.BOTH, expand=True)

        # A. Matplotlib Waveform Frame
        plot_frame = ttk.LabelFrame(split_pane, text=" Filtered ECG Waveform & Detected P-Q-R-S-T Markers ", padding=4)
        split_pane.add(plot_frame, weight=3)

        ctrl_bar = ttk.Frame(plot_frame)
        ctrl_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

        ttk.Label(ctrl_bar, text="View Span:").pack(side=tk.LEFT, padx=(2, 4))
        self.view_span_var = tk.StringVar(value="6s")
        ttk.Radiobutton(ctrl_bar, text="0–6 s (Morphology)", variable=self.view_span_var, value="6s", command=self._update_plot_view).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(ctrl_bar, text="0–15 s (Rhythm)", variable=self.view_span_var, value="15s", command=self._update_plot_view).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(ctrl_bar, text="Full Recording", variable=self.view_span_var, value="all", command=self._update_plot_view).pack(side=tk.LEFT, padx=3)

        self.live_indicator_lbl = tk.Label(ctrl_bar, text="", font=("Segoe UI", 8, "bold"))
        self.live_indicator_lbl.pack(side=tk.RIGHT, padx=6)

        # Create Matplotlib Figure & Canvas
        self.fig = Figure(figsize=(7, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self._draw_placeholder_plot()

        # B. Bottom Notebook Tabs Frame
        tabs_frame = ttk.Frame(split_pane)
        split_pane.add(tabs_frame, weight=2)

        self.notebook = ttk.Notebook(tabs_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Extracted Features (10-Feature Vector)
        tab_features = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab_features, text=" 10 Extracted Features ")
        self._build_features_tab(tab_features)

        # Tab 2: Model Explainability
        tab_explain = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab_explain, text=" Model Explainability ")
        self._build_explain_tab(tab_explain)

        # Tab 3: Beat-by-Beat Measurements
        tab_beats = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab_beats, text=" Beat-by-Beat Measurements ")
        self._build_beats_tab(tab_beats)

        # Tab 4: System Log & Disclaimers
        tab_log = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab_log, text=" System Log & Limitations ")
        self._build_log_tab(tab_log)

    def _build_features_tab(self, parent):
        cols = ("Feature", "Measured Value", "Units", "Methodological Description")
        self.features_tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)

        self.features_tree.heading("Feature", text="Feature Name")
        self.features_tree.heading("Measured Value", text="Measured Value")
        self.features_tree.heading("Units", text="Units")
        self.features_tree.heading("Methodological Description", text="Definition & Clinical Disclaimers")

        self.features_tree.column("Feature", width=120, anchor=tk.W)
        self.features_tree.column("Measured Value", width=120, anchor=tk.E)
        self.features_tree.column("Units", width=110, anchor=tk.W)
        self.features_tree.column("Methodological Description", width=420, anchor=tk.W)

        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.features_tree.yview)
        self.features_tree.configure(yscrollcommand=scrollbar.set)

        self.features_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_explain_tab(self, parent):
        self.explain_text = tk.Text(parent, wrap=tk.WORD, font=("Consolas", 9), relief=tk.FLAT, padx=6, pady=6)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.explain_text.yview)
        self.explain_text.configure(yscrollcommand=scroll.set)

        self.explain_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.explain_text.insert(tk.END, "Process a signal to inspect the Random Forest ensemble consensus and top feature contributions.")
        self.explain_text.configure(state=tk.DISABLED)

    def _build_beats_tab(self, parent):
        cols = ("Beat", "R_Time", "RR_Interval", "PR_Interval", "QRS_Duration", "QT_Interval", "R_Amp", "ST_Segment")
        self.beats_tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)

        for c in cols:
            self.beats_tree.heading(c, text=c.replace("_", " "))
            self.beats_tree.column(c, width=80, anchor=tk.CENTER)
        self.beats_tree.column("Beat", width=50)

        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.beats_tree.yview)
        self.beats_tree.configure(yscrollcommand=scroll.set)

        self.beats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_log_tab(self, parent):
        self.log_text = tk.Text(parent, wrap=tk.WORD, font=("Consolas", 8), relief=tk.FLAT, padx=6, pady=6)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.insert(
            tk.END,
            "ECG FINAL PRO — PIPELINE LOG\n"
            "Academic Prototype Demonstration Dashboard.\n"
            "Status: System loaded. Frozen DecisionTreeClassifier & StandardScaler ready.\n"
        )
        self.log_text.configure(state=tk.DISABLED)

    # ========================================================
    # EVENT HANDLERS & MODE SWITCHING
    # ========================================================
    def _on_mode_change(self):
        mode = self.mode_var.get()

        # If switching away from live mode, stop streaming cleanly
        if mode != "esp32" and (self.bridge.is_connected or self.bridge.is_simulating):
            self.bridge.disconnect()
            self._update_serial_ui_state()

        if mode == "case_a":
            self.file_frame.pack(fill=tk.X, pady=(4, 2))
            self.serial_controls_frame.pack_forget()
            self.selected_file_path = NORMAL_CASE_PATH
            self.file_path_var.set(str(NORMAL_CASE_PATH))
            self.file_entry.configure(state="readonly")
            self.browse_btn.configure(state=tk.DISABLED)
            self.process_btn.configure(text="▶  Process & Analyze Signal")
            self._reset_spo2_to_unconnected()
            self._hide_lead_off()
            self.status_var.set("Selected: Case A (Normal Synthetic ECG). Ready to process.")

        elif mode == "case_b":
            self.file_frame.pack(fill=tk.X, pady=(4, 2))
            self.serial_controls_frame.pack_forget()
            self.selected_file_path = PVC_CASE_PATH
            self.file_path_var.set(str(PVC_CASE_PATH))
            self.file_entry.configure(state="readonly")
            self.browse_btn.configure(state=tk.DISABLED)
            self.process_btn.configure(text="▶  Process & Analyze Signal")
            self._reset_spo2_to_unconnected()
            self._hide_lead_off()
            self.status_var.set("Selected: Case B (Simulated PVC-like ECG). Ready to process.")

        elif mode == "custom":
            self.file_frame.pack(fill=tk.X, pady=(4, 2))
            self.serial_controls_frame.pack_forget()
            self.file_entry.configure(state=tk.NORMAL)
            self.browse_btn.configure(state=tk.NORMAL)
            self.process_btn.configure(text="▶  Process & Analyze Signal")
            self._reset_spo2_to_unconnected()
            self._hide_lead_off()
            self.status_var.set("Please browse and select a custom .npy or .csv ECG recording.")

        elif mode == "esp32":
            self.file_frame.pack_forget()
            self.serial_controls_frame.pack(fill=tk.X, pady=(4, 2))
            self.process_btn.configure(text="▶  Analyze Live Buffer Window")
            self._refresh_serial_ports()
            self._update_serial_ui_state()

    def _browse_custom_file(self):
        path = filedialog.askopenfilename(
            title="Select ECG Recording File",
            filetypes=[("ECG Data Files", "*.npy *.csv"), ("NumPy Array", "*.npy"), ("CSV Spreadsheet", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            self.selected_file_path = Path(path)
            self.file_path_var.set(str(self.selected_file_path))
            self.status_var.set(f"Selected file: {self.selected_file_path.name}. Ready to process.")

    # ========================================================
    # SERIAL BRIDGE & HARDWARE MANAGEMENT
    # ========================================================
    def _refresh_serial_ports(self):
        ports = SerialBridge.get_available_ports()
        port_names = [p[0] for p in ports]
        self.port_combo["values"] = port_names
        if port_names:
            self.port_combo.current(0)
            self.status_var.set(f"Detected {len(port_names)} COM port(s): {', '.join(port_names)}")
        else:
            self.port_combo.set("No Ports")
            self.status_var.set("No hardware COM ports detected. Connect ESP32 via USB or use Simulation Mode.")

    def _toggle_serial_connection(self):
        if self.bridge.is_connected:
            self.bridge.disconnect()
            self._update_serial_ui_state()
            self.status_var.set("ESP32 disconnected.")
        else:
            port = self.port_combo.get().strip()
            if not port or port == "No Ports":
                messagebox.showwarning(
                    "No Port Selected",
                    "No serial port detected. Please plug in the ESP32 via USB and click '↻' to refresh,\n"
                    "or click '⚡ Start Mock ESP32 Stream' to test the interface without physical hardware."
                )
                return

            success = self.bridge.connect(port)
            self._update_serial_ui_state()
            if not success:
                messagebox.showerror("Connection Failed", f"Could not open serial port {port}.\n{self.bridge.status_message}")

    def _toggle_simulation_stream(self):
        if self.bridge.is_simulating:
            self.bridge.stop_simulation()
            self._update_serial_ui_state()
            self.status_var.set("Mock ESP32 simulation stopped.")
        else:
            # Load normal test signal as the mock source
            mock_data = None
            if NORMAL_CASE_PATH.is_file():
                mock_data = np.load(NORMAL_CASE_PATH)
            self.bridge.start_simulation(signal_array=mock_data, include_spo2=True)
            self._update_serial_ui_state()
            self.status_var.set("Mock ESP32 stream active (360 Hz packets with SpO2 and pulse).")

    def _update_serial_ui_state(self):
        if self.bridge.is_connected:
            self.connect_btn.configure(text="Disconnect")
            self.sim_stream_btn.configure(state=tk.DISABLED)
            self.live_indicator_lbl.configure(text="● LIVE (ESP32 USB)", fg="#22543d")
        elif self.bridge.is_simulating:
            self.connect_btn.configure(state=tk.DISABLED)
            self.sim_stream_btn.configure(text="⏹ Stop Mock Stream")
            self.live_indicator_lbl.configure(text="● LIVE (SIMULATED STREAM)", fg="#2b6cb0")
        else:
            self.connect_btn.configure(text="Connect", state=tk.NORMAL)
            self.sim_stream_btn.configure(text="⚡ Start Mock ESP32 Stream (Demo)", state=tk.NORMAL)
            self.live_indicator_lbl.configure(text="")

    def _poll_serial_state(self):
        """Timer callback running every 50 ms to update live oscilloscope and sensors."""
        mode = self.mode_var.get()

        if mode == "esp32" and (self.bridge.is_connected or self.bridge.is_simulating):
            metrics = self.bridge.get_latest_metrics()

            # 1. Update SpO2 & Optical Pulse independently
            spo2_val = metrics["spo2"]
            bpm_val = metrics["pulse_bpm"]

            if metrics["spo2_valid"] and spo2_val is not None:
                self.spo2_val_lbl.configure(text=f"{spo2_val}% SpO₂", fg="#22543d")
            else:
                self.spo2_val_lbl.configure(text="SpO₂ unavailable", fg="#9c4221")

            if bpm_val is not None:
                self.pulse_val_lbl.configure(text=f"MAX30102 Optical Pulse: {bpm_val} BPM", fg="#22543d")
            else:
                self.pulse_val_lbl.configure(text="Optical Pulse unavailable", fg="#7b341e")

            if (metrics["spo2_valid"] and spo2_val is not None) or (bpm_val is not None):
                self.spo2_subtext_lbl.configure(text="Live optical plethysmogram reading active.", fg="#22543d")
            else:
                self.spo2_subtext_lbl.configure(text="Ensure finger is placed firmly on sensor.", fg="#7b341e")

            # 2. Update Lead-Off Alert
            if metrics["lead_off"]:
                self._show_lead_off()
            else:
                self._hide_lead_off()

            # 3. Rolling live ECG display (last 5 seconds)
            live_ecg = self.bridge.get_recent_ecg(duration_sec=5.0)
            if len(live_ecg) > 36:
                self._draw_live_oscilloscope(live_ecg)

        # Schedule next poll
        self.after(50, self._poll_serial_state)

    def _draw_live_oscilloscope(self, live_samples: np.ndarray):
        """Draw rolling real-time ECG oscilloscope trace."""
        self.ax.clear()
        t = np.arange(len(live_samples)) / self.bridge.fs
        self.ax.plot(t, live_samples, color="#2b6cb0", lw=1.0)
        self.ax.set_title("Live ECG Stream — Oscilloscope View (Last 5.0 Seconds)", fontsize=9, fontweight="bold")
        self.ax.set_xlabel("Time (seconds)", fontsize=8)
        self.ax.set_ylabel("Raw ADC / Amplitude", fontsize=8)
        self.ax.set_xlim(0, t[-1])
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw_idle()

    def _show_lead_off(self):
        self.lead_off_banner.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 4), before=self.winfo_children()[2])

    def _hide_lead_off(self):
        self.lead_off_banner.pack_forget()

    def _reset_spo2_to_unconnected(self):
        self.spo2_val_lbl.configure(text="Not connected", fg="#9c4221")
        self.pulse_val_lbl.configure(text="Optical Pulse: -- BPM", fg="#7b341e")
        self.spo2_subtext_lbl.configure(
            text="Hardware integration pending Phase 4.\nSynthetic values are not fabricated.",
            fg="#7b341e"
        )

    # ========================================================
    # SIGNAL PROCESSING & PIPELINE TRIGGER
    # ========================================================
    def start_processing(self):
        """Execute pipeline analysis on selected file or live buffer window."""
        if self.is_processing:
            return

        mode = self.mode_var.get()
        target_input = None

        if mode == "esp32":
            if not (self.bridge.is_connected or self.bridge.is_simulating):
                messagebox.showwarning(
                    "Stream Not Active",
                    "ESP32 stream is not currently connected.\n"
                    "Connect an ESP32 or click '⚡ Start Mock ESP32 Stream' before analyzing live data."
                )
                return

            live_buf = self.bridge.get_recent_ecg(duration_sec=10.0)
            if len(live_buf) < int(self.bridge.fs * 2.5):
                messagebox.showwarning(
                    "Insufficient Data",
                    f"Only {len(live_buf)} live samples accumulated ({len(live_buf)/self.bridge.fs:.1f} s).\n"
                    "Please wait a few seconds to accumulate at least 3 seconds of continuous signal."
                )
                return
            target_input = live_buf
            file_label = f"Live Buffer ({len(live_buf)/self.bridge.fs:.1f} s)"
        else:
            target_file = self.file_path_var.get().strip()
            if not target_file or not Path(target_file).is_file():
                messagebox.showerror("File Error", f"The specified ECG file does not exist:\n{target_file}")
                return
            target_input = target_file
            file_label = Path(target_file).name

        try:
            fs = float(self.fs_var.get())
            if fs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Sampling Rate", "Please specify a positive numeric sampling rate (e.g., 360.0).")
            return

        self.is_processing = True
        self.process_btn.configure(state=tk.DISABLED)
        self.progress_bar.start(10)
        self.status_var.set(f"Processing '{file_label}' through Butterworth filter, peak detector, and ML classifier...")

        # Run pipeline in background thread
        thread = threading.Thread(
            target=self._worker_process,
            args=(target_input, fs),
            daemon=True
        )
        thread.start()

    def _worker_process(self, source_input, fs: float):
        """Worker thread executing the real pipeline."""
        try:
            result = self.pipeline.process(source_input, sampling_rate=fs)
        except Exception as exc:
            self.after(0, self._on_processing_error, str(exc))
            return

        self.after(0, self._on_processing_complete, result)

    def _on_processing_error(self, err_msg: str):
        self.is_processing = False
        self.progress_bar.stop()
        self.process_btn.configure(state=tk.NORMAL)
        self.status_var.set("Processing failed.")
        messagebox.showerror("Pipeline Processing Failure", f"An error occurred during pipeline execution:\n{err_msg}")

    def _on_processing_complete(self, result: ECGPipelineResult):
        self.is_processing = False
        self.progress_bar.stop()
        self.process_btn.configure(state=tk.NORMAL)
        self.current_result = result

        if not result.success:
            err_str = "\n".join(result.errors)
            self.status_var.set("Pipeline completed with errors.")
            messagebox.showwarning("Pipeline Issues", f"Processing completed with warnings/errors:\n{err_str}")
        else:
            self.status_var.set(f"Analysis complete: {result.duration_sec:.1f} s signal, {len(result.r_peak_indices)} beats detected.")

        # Update all UI panels with real data
        self._update_vital_cards(result)
        self._update_waveform_plot(result)
        self._update_features_table(result)
        self._update_explainability(result)
        self._update_beats_table(result)
        self._update_log_tab(result)

    # ========================================================
    # UI COMPONENT UPDATES
    # ========================================================
    def _update_vital_cards(self, result: ECGPipelineResult):
        # 1. Heart Rate
        if len(result.rr_intervals) > 0 and not np.isnan(result.features.get("HeartRate", np.nan)):
            hr_val = result.features["HeartRate"]
            self.hr_val_lbl.configure(text=f"{hr_val:.1f} BPM", fg="#2b6cb0")
        else:
            self.hr_val_lbl.configure(text="-- BPM", fg="#a0aec0")

        # 2. Model Classification Badge
        if result.predicted_label is not None:
            pred_lbl = result.predicted_label
            norm_prob = result.class_probabilities.get(0, 0.0) * 100
            abn_prob = result.class_probabilities.get(1, 0.0) * 100

            if pred_lbl == 0:
                bg_color = "#c6f6d5"
                fg_color = "#22543d"
                badge_text = "NORMAL PATTERN (Label 0)"
            else:
                bg_color = "#fed7d7"
                fg_color = "#742a2a"
                badge_text = "ABNORMAL PATTERN (Label 1)"

            self.class_card.configure(bg=bg_color)
            for widget in self.class_card.winfo_children():
                widget.configure(bg=bg_color)

            self.class_val_lbl.configure(text=badge_text, fg=fg_color)
            self.class_prob_lbl.configure(
                text=f"Normal Confidence: {norm_prob:.1f}% | Abnormal Confidence: {abn_prob:.1f}%",
                fg=fg_color
            )
        else:
            self.class_val_lbl.configure(text="Inference Unavailable", fg="#718096")
            self.class_prob_lbl.configure(text="Check pipeline log for details.", fg="#718096")

    def _draw_placeholder_plot(self):
        self.ax.clear()
        self.ax.set_title("ECG Waveform Display — Awaiting Input", fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude (arbitrary units)")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw_idle()

    def _update_waveform_plot(self, result: ECGPipelineResult):
        self.ax.clear()

        fs = result.sampling_rate
        signal = result.filtered_ecg if len(result.filtered_ecg) > 0 else result.raw_ecg

        if len(signal) == 0:
            self._draw_placeholder_plot()
            return

        time_axis = np.arange(len(signal)) / fs

        span_mode = self.view_span_var.get()
        if span_mode == "6s":
            t_max = min(6.0, time_axis[-1])
        elif span_mode == "15s":
            t_max = min(15.0, time_axis[-1])
        else:
            t_max = time_axis[-1]

        mask = time_axis <= t_max
        self.ax.plot(time_axis[mask], signal[mask], color="#2b6cb0", linewidth=0.9, label="Filtered ECG (0.5–40 Hz)")

        # Plot P-Q-R-S-T markers if available
        if len(result.pqrst_matrix) > 0:
            pqrst = result.pqrst_matrix
            p_times = pqrst[:, 0]
            q_times = pqrst[:, 1]
            r_times = pqrst[:, 2]
            s_times = pqrst[:, 3]
            t_times = pqrst[:, 4]

            def plot_markers(times, color, marker, label_name):
                valid = (~np.isnan(times)) & (times <= t_max)
                if np.any(valid):
                    t_pts = times[valid]
                    idx = np.clip(np.rint(t_pts * fs).astype(int), 0, len(signal) - 1)
                    self.ax.scatter(t_pts, signal[idx], color=color, marker=marker, s=26, zorder=4, label=label_name)

            plot_markers(p_times, "#805ad5", "o", "P wave")
            plot_markers(q_times, "#e53e3e", "v", "Q wave")
            plot_markers(r_times, "#38a169", "^", "R peak")
            plot_markers(s_times, "#dd6b20", "v", "S wave")
            plot_markers(t_times, "#b7791f", "o", "T wave")

        self.ax.set_title(
            f"Filtered ECG Signal — Showing 0.0 to {t_max:.1f} s ({len(result.r_peak_indices)} Total Beats Detected)",
            fontsize=9,
            fontweight="bold"
        )
        self.ax.set_xlabel("Time (seconds)", fontsize=8)
        self.ax.set_ylabel("Amplitude (arb. units)", fontsize=8)
        self.ax.set_xlim(0, t_max)
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.legend(loc="upper right", ncol=6, fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _update_plot_view(self):
        if self.current_result is not None:
            self._update_waveform_plot(self.current_result)

    def _update_features_table(self, result: ECGPipelineResult):
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)

        for name in result.feature_names:
            val = result.features.get(name, np.nan)
            unit_str, desc = FEATURE_METADATA.get(name, ("", ""))
            val_str = f"{val:.6f}" if np.isfinite(val) else "N/A"
            self.features_tree.insert("", tk.END, values=(name, val_str, unit_str, desc))

    def _update_explainability(self, result: ECGPipelineResult):
        self.explain_text.configure(state=tk.NORMAL)
        self.explain_text.delete("1.0", tk.END)

        self.explain_text.insert(tk.END, "RANDOM FOREST ENSEMBLE EXPLAINABILITY & CONSENSUS\n")
        self.explain_text.insert(tk.END, "=" * 60 + "\n")
        self.explain_text.insert(
            tk.END,
            f"Classifier: {self.pipeline.model_type}\n"
            f"Predicted Label: {result.predicted_label} ({result.predicted_class})\n"
            f"Normal Probability  : {result.class_probabilities.get(0, 0.0)*100:.2f}%\n"
            f"Abnormal Probability: {result.class_probabilities.get(1, 0.0)*100:.2f}%\n"
            + "-" * 60 + "\n\n"
        )

        if result.decision_explanation:
            self.explain_text.insert(tk.END, "Ensemble Breakdown & Key Influences:\n")
            for step in result.decision_explanation:
                self.explain_text.insert(tk.END, f"  {step}\n")
        else:
            self.explain_text.insert(tk.END, "No decision explanation available.\n")

        self.explain_text.insert(
            tk.END,
            "\nINTERPRETATION NOTE:\n"
            "- Feature rankings reflect ensemble decision contributions weighted by feature importance.\n"
            "- A high abnormal probability indicates morphological/variability patterns characteristic of arrhythmias in MIT-BIH recordings,\n"
            "  not an etiology or clinical diagnosis.\n"
        )
        self.explain_text.configure(state=tk.DISABLED)

    def _update_beats_table(self, result: ECGPipelineResult):
        for item in self.beats_tree.get_children():
            self.beats_tree.delete(item)

        if result.beat_table.empty:
            return

        for _, row in result.beat_table.head(100).iterrows():
            beat_idx = int(row["Beat"])
            r_time = f"{row['R_Time_s']:.3f}"
            rr = f"{row['RR_Interval_s']:.3f}" if not np.isnan(row['RR_Interval_s']) else "--"
            pr = f"{row['PR_Interval_s']*1000:.1f} ms" if not np.isnan(row['PR_Interval_s']) else "--"
            qrs = f"{row['QRS_Duration_s']*1000:.1f} ms" if not np.isnan(row['QRS_Duration_s']) else "--"
            qt = f"{row['QT_Interval_s']*1000:.1f} ms" if not np.isnan(row['QT_Interval_s']) else "--"
            r_amp = f"{row['R_Amplitude']:.2f}" if not np.isnan(row['R_Amplitude']) else "--"
            st = f"{row['ST_Segment']:.2f}" if not np.isnan(row['ST_Segment']) else "--"

            self.beats_tree.insert("", tk.END, values=(beat_idx, r_time, rr, pr, qrs, qt, r_amp, st))

    def _update_log_tab(self, result: ECGPipelineResult):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, result.summary())
        self.log_text.configure(state=tk.DISABLED)

    # ========================================================
    # AUTOMATED HEADLESS / TEST MODE
    # ========================================================
    def _run_automated_verification(self):
        """Run programmatic self-test sequence to verify dashboard logic."""
        print(">>> STARTING DASHBOARD AUTOMATED VERIFICATION SEQUENCE...")

        # Test Case A
        print("[1/4] Testing Case A (Normal)...")
        self.mode_var.set("case_a")
        self._on_mode_change()
        res_a = self.pipeline.process(NORMAL_CASE_PATH)
        self._on_processing_complete(res_a)
        assert res_a.predicted_class == "NORMAL", f"Expected NORMAL, got {res_a.predicted_class}"
        assert self.hr_val_lbl.cget("text") == f"{res_a.features['HeartRate']:.1f} BPM"
        assert self.spo2_val_lbl.cget("text") == "Not connected"
        print(f"      Passed Case A: Prediction = {res_a.predicted_class}, HR = {self.hr_val_lbl.cget('text')}")

        # Test Case B
        print("[2/4] Testing Case B (PVC)...")
        self.mode_var.set("case_b")
        self._on_mode_change()
        res_b = self.pipeline.process(PVC_CASE_PATH)
        self._on_processing_complete(res_b)
        assert res_b.predicted_class == "ABNORMAL", f"Expected ABNORMAL, got {res_b.predicted_class}"
        assert self.spo2_val_lbl.cget("text") == "Not connected"
        print(f"      Passed Case B: Prediction = {res_b.predicted_class}, Conf = {res_b.class_probabilities[1]*100:.1f}%")

        # Test Custom CSV
        print("[3/4] Testing Custom CSV loading...")
        csv_path = TEST_CASES_DIR / "normal_ecg.csv"
        self.mode_var.set("custom")
        self.file_path_var.set(str(csv_path))
        res_csv = self.pipeline.process(csv_path)
        self._on_processing_complete(res_csv)
        assert res_csv.predicted_class == "NORMAL"
        print(f"      Passed CSV Loading: {csv_path.name}")

        # Test Live ESP32 Simulation Mode
        print("[4/4] Testing ESP32 Live Mode & Simulation Stream...")
        self.mode_var.set("esp32")
        self._on_mode_change()
        self._toggle_simulation_stream()
        import time
        time.sleep(1.0)
        metrics = self.bridge.get_latest_metrics()
        assert metrics["connected"] is True, "Expected simulation stream to be active"
        assert metrics["spo2"] == 98, f"Expected SpO2=98, got {metrics['spo2']}"
        print(f"      Passed Live Mode Stream: SpO2 = {metrics['spo2']}%, Buffer = {metrics['buffer_len']} samples")

        # Switch back to Case A and confirm SpO2 resets
        self.mode_var.set("case_a")
        self._on_mode_change()
        assert self.bridge.is_simulating is False, "Expected stream to stop on mode switch"
        assert self.spo2_val_lbl.cget("text") == "Not connected", "Expected SpO2 to reset to Not connected"
        print("      Passed Mode Switch Fallback: Serial stream stopped and SpO2 reset to 'Not connected'.")

        print(">>> ALL DASHBOARD VERIFICATION TESTS PASSED SUCCESSFULLY.")
        self.destroy()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    is_test = "--test" in sys.argv
    app = ECGDashboard(run_automated_test=is_test)
    app.mainloop()
