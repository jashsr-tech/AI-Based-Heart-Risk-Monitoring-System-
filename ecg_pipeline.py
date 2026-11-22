"""
ECG FINAL PRO — Unified Non-Destructive ECG Processing Pipeline
File: ecg_pipeline.py

Connects:
  - Input ingestion (NumPy arrays, .npy files, .csv files)
  - 4th-order Butterworth bandpass filtering (0.5–40 Hz)
  - R-peak detection (adaptive / configurable minimum distance)
  - P-Q-R-S-T morphological peak detection
  - 10-feature extraction matching Stage 2E definitions
  - Per-beat measurements and summary statistics
  - Exploratory inference using frozen DecisionTreeClassifier & StandardScaler
  - Decision tree path explainability

IMPORTANT NOTICE & DISCLAIMERS:
  1. This software is an engineering prototype for academic demonstration.
     It is NOT a medical device and must NOT be used for clinical diagnosis.
  2. The underlying ML model was trained on beat-level MIT-BIH rows, whereas
     pipeline features are aggregated across the recording window.
     Predictions on aggregated features represent software pattern-scoring,
     not a clinically validated risk assessment.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict, Any, Tuple
import csv
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, sosfiltfilt, find_peaks

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_FS = 360.0                 # Samples per second
MIN_SIGNAL_SECONDS = 2.5           # Minimum length for meaningful RR analysis
DEFAULT_MIN_DISTANCE_SEC = 0.40    # Default minimum distance between R peaks (~150 BPM)
DEFAULT_PROMINENCE = 1.0           # Peak prominence threshold for filtered ECG

# P-Q-R-S-T search windows relative to R-peak (seconds)
DEFAULT_WINDOWS = {
    "P": (-0.205, -0.105),
    "Q": (-0.075, -0.015),
    "S": (0.015, 0.085),
    "T": (0.200, 0.390),
}

# ============================================================
# PIPELINE RESULT CONTAINER
# ============================================================

@dataclass
class ECGPipelineResult:
    """Structured container holding all outputs from the ECG pipeline."""
    success: bool
    sampling_rate: float
    duration_sec: float
    sample_count: int
    raw_ecg: np.ndarray
    filtered_ecg: np.ndarray
    r_peak_indices: np.ndarray
    r_peak_times: np.ndarray
    rr_intervals: np.ndarray
    pqrst_matrix: np.ndarray
    beat_table: pd.DataFrame
    features: Dict[str, float]
    feature_vector: np.ndarray
    feature_names: List[str]
    predicted_label: Optional[int] = None
    predicted_class: Optional[str] = None
    class_probabilities: Dict[int, float] = field(default_factory=dict)
    decision_explanation: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted text summary of the pipeline results."""
        lines = [
            "=" * 72,
            "ECG FINAL PRO — PIPELINE ANALYSIS REPORT",
            "=" * 72,
            f"Status               : {'SUCCESS' if self.success else 'FAILED'}",
            f"Sampling Rate        : {self.sampling_rate:.1f} Hz",
            f"Signal Duration      : {self.duration_sec:.2f} s ({self.sample_count} samples)",
            f"Detected R-Peaks     : {len(self.r_peak_indices)} beats",
        ]

        if len(self.rr_intervals) > 0:
            lines.extend([
                f"Mean RR Interval     : {np.mean(self.rr_intervals):.4f} s",
                f"Estimated Heart Rate : {60.0 / np.mean(self.rr_intervals):.2f} BPM",
                f"Heart Rate Variability (SDNN) : {np.std(self.rr_intervals, ddof=1)*1000:.2f} ms"
                if len(self.rr_intervals) > 1 else "Heart Rate Variability: N/A (< 2 beats)",
            ])

        lines.extend([
            "",
            "-" * 72,
            "EXTRACTED 10-FEATURE VECTOR (STAGE 2E COMPLIANT)",
            "-" * 72,
        ])
        for name in self.feature_names:
            val = self.features.get(name, np.nan)
            unit_str = {
                "HeartRate": "BPM",
                "RRInterval": "s",
                "PRInterval": "s",
                "QRSDuration": "s",
                "QTInterval": "s",
                "STSegment": "arb. amp (S+60ms)",
                "HRV": "s (SDNN)",
                "PAmplitude": "arb. amp",
                "RAmplitude": "arb. amp",
                "TAmplitude": "arb. amp",
            }.get(name, "")
            lines.append(f"  {name:15s}: {val:12.6f}  [{unit_str}]")

        lines.extend([
            "",
            "-" * 72,
            "EXPLORATORY MACHINE LEARNING CLASSIFICATION",
            "-" * 72,
            f"Predicted Class      : {self.predicted_class} (Label {self.predicted_label})",
        ])
        if self.class_probabilities:
            lines.append("Class Probabilities  :")
            for cls_lbl, prob in self.class_probabilities.items():
                lbl_name = "NORMAL" if cls_lbl == 0 else "ABNORMAL"
                lines.append(f"  - Label {cls_lbl} ({lbl_name:8s}): {prob*100:6.2f}%")

        if self.decision_explanation:
            lines.extend([
                "",
                "-" * 72,
                "DECISION TREE PATH EXPLANATION (KEY RULES)",
                "-" * 72,
            ])
            for step in self.decision_explanation:
                lines.append(f"  * {step}")

        if self.warnings:
            lines.extend(["", "-" * 72, "WARNINGS", "-" * 72])
            for w in self.warnings:
                lines.append(f"  [!] {w}")

        if self.errors:
            lines.extend(["", "-" * 72, "ERRORS", "-" * 72])
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")

        if self.limitations:
            lines.extend(["", "-" * 72, "METHODOLOGICAL LIMITATIONS & DISCLAIMERS", "-" * 72])
            for lim in self.limitations:
                lines.append(f"  - {lim}")

        lines.append("=" * 72)
        return "\n".join(lines)

    def compare_ground_truth(self, ground_truth: np.ndarray, tolerance_sec: float = 0.030) -> Dict[str, Any]:
        """Compare detected P-Q-R-S-T peaks to ground truth matrix [N x 5]."""
        if len(self.pqrst_matrix) == 0 or len(ground_truth) == 0:
            return {"error": "Insufficient peaks or ground truth rows for comparison."}

        true_r = ground_truth[:, 2]
        det_r = self.r_peak_times

        matched_true = []
        matched_det = []
        used_gt = set()

        for det_idx, r_time in enumerate(det_r):
            distances = np.abs(true_r - r_time)
            order = np.argsort(distances)
            for j in order:
                if j not in used_gt and distances[j] <= tolerance_sec:
                    matched_true.append(ground_truth[j])
                    matched_det.append(self.pqrst_matrix[det_idx])
                    used_gt.add(int(j))
                    break

        matched_count = len(matched_det)
        missed_count = len(true_r) - matched_count
        false_positives = len(det_r) - matched_count

        res = {
            "ground_truth_beats": len(true_r),
            "detected_beats": len(det_r),
            "matched_beats": matched_count,
            "missed_beats": missed_count,
            "false_positives": false_positives,
            "match_tolerance_sec": tolerance_sec,
        }

        if matched_count > 0:
            gt_arr = np.asarray(matched_true)
            det_arr = np.asarray(matched_det)
            err_ms = (det_arr - gt_arr) * 1000.0

            labels = ["P", "Q", "R", "S", "T"]
            timing_stats = {}
            for col, lbl in enumerate(labels):
                valid_mask = ~np.isnan(err_ms[:, col])
                if np.any(valid_mask):
                    col_err = err_ms[valid_mask, col]
                    timing_stats[f"{lbl}_mae_ms"] = float(np.mean(np.abs(col_err)))
                    timing_stats[f"{lbl}_max_ae_ms"] = float(np.max(np.abs(col_err)))
                else:
                    timing_stats[f"{lbl}_mae_ms"] = np.nan
                    timing_stats[f"{lbl}_max_ae_ms"] = np.nan
            res["timing_errors"] = timing_stats

        return res


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

class ECGPipeline:
    """
    Unified, non-destructive pipeline for ECG signal processing,
    feature extraction, and exploratory classification.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        fs: float = DEFAULT_FS,
        min_distance_sec: float = DEFAULT_MIN_DISTANCE_SEC,
        prominence: float = DEFAULT_PROMINENCE,
        search_windows: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.fs = float(fs)
        self.min_distance_sec = float(min_distance_sec)
        self.prominence = float(prominence)
        self.search_windows = search_windows or DEFAULT_WINDOWS.copy()

        # Resolve model directory
        if model_dir is None:
            # Default location: 04_machine_learning/models relative to project root
            self.model_dir = Path(__file__).resolve().parent / "04_machine_learning" / "models"
        else:
            self.model_dir = Path(model_dir).resolve()

        # Prioritize the validated Random Forest model
        rf_path = self.model_dir / "ecg_random_forest_model.pkl"
        dt_path = self.model_dir / "ecg_decision_tree_model.pkl"

        if rf_path.is_file():
            self.model_file = rf_path
            self.model_type = "RandomForestClassifier"
        elif dt_path.is_file():
            self.model_file = dt_path
            self.model_type = "DecisionTreeClassifier"
        else:
            self.model_file = rf_path  # will trigger informative error in _load_models

        self.scaler_file = self.model_dir / "ecg_feature_scaler.pkl"
        self.order_file = self.model_dir / "model_feature_order.txt"

        self.model = None
        self.scaler = None
        self.feature_order = []

        self._load_models()

    def _load_models(self) -> None:
        """Load frozen classifier, scaler, and expected feature order."""
        if not self.model_file.is_file():
            raise FileNotFoundError(f"Saved model not found: {self.model_file}")
        if not self.scaler_file.is_file():
            raise FileNotFoundError(f"Saved scaler not found: {self.scaler_file}")
        if not self.order_file.is_file():
            raise FileNotFoundError(f"Saved feature order not found: {self.order_file}")

        self.model = joblib.load(self.model_file)
        self.scaler = joblib.load(self.scaler_file)
        self.feature_order = [
            line.strip()
            for line in self.order_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # --------------------------------------------------------
    # 1. INPUT LOADING & VALIDATION
    # --------------------------------------------------------
    def load_signal(self, source: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Load ECG data from array, .npy file, or .csv file.
        Accepts variable signal lengths.
        """
        if isinstance(source, (str, Path)):
            src_path = Path(source).resolve()
            if not src_path.is_file():
                raise FileNotFoundError(f"ECG file not found: {src_path}")

            if src_path.suffix.lower() == ".npy":
                data = np.load(src_path)
            elif src_path.suffix.lower() == ".csv":
                df = pd.read_csv(src_path)
                # If 2 columns, assume Time and ECG
                if df.shape[1] >= 2:
                    # Prefer column named 'ecg' or 'ecg_amplitude' or second column
                    col_match = [c for c in df.columns if "ecg" in c.lower() or "amp" in c.lower()]
                    data = df[col_match[0]].values if col_match else df.iloc[:, 1].values
                else:
                    data = df.iloc[:, 0].values
            else:
                raise ValueError(f"Unsupported file format: {src_path.suffix}. Expected .npy or .csv.")
        elif isinstance(source, np.ndarray):
            data = source.copy()
        elif hasattr(source, "__iter__"):
            data = np.asarray(source, dtype=float)
        else:
            raise TypeError(f"Unsupported signal source type: {type(source).__name__}")

        data = np.asarray(data, dtype=float).flatten()

        # Sanity validation
        if len(data) == 0:
            raise ValueError("Input ECG signal is empty.")

        if not np.all(np.isfinite(data)):
            # If NaNs exist, drop trailing or interpolate only if minor
            nan_count = np.sum(~np.isfinite(data))
            if nan_count > 0.05 * len(data):
                raise ValueError(f"Signal contains excessive NaNs or Infs ({nan_count} samples).")
            # Fill small internal NaNs with nearest finite value
            nans = np.isnan(data)
            data[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), data[~nans])

        std_val = float(np.std(data))
        if std_val < 1e-6:
            raise ValueError("Input signal is a flatline (near-zero variance).")

        return data

    # --------------------------------------------------------
    # 2. FILTERING
    # --------------------------------------------------------
    def filter(self, ecg_raw: np.ndarray) -> np.ndarray:
        """
        Apply 4th-order Butterworth bandpass filter (0.5 to 40.0 Hz)
        using zero-phase second-order sections (sosfiltfilt).
        Safe for short and long signals.
        """
        nyquist = 0.5 * self.fs
        lowcut = 0.5
        highcut = 40.0

        if highcut >= nyquist:
            highcut = nyquist - 2.0

        order = 4
        sos = butter(order, [lowcut, highcut], btype="bandpass", fs=self.fs, output="sos")

        # Adjust padlen if signal is short
        min_pad = 3 * (2 * order + 1)
        if len(ecg_raw) <= min_pad:
            raise ValueError(f"Signal too short for filtering ({len(ecg_raw)} samples, min is {min_pad}).")

        padlen = min(min_pad, len(ecg_raw) - 1)
        return sosfiltfilt(sos, ecg_raw, padlen=padlen)

    # --------------------------------------------------------
    # 3. R-PEAK DETECTION
    # --------------------------------------------------------
    def detect_r_peaks(self, filtered_ecg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect R peaks using find_peaks with adaptive minimum distance."""
        min_dist_samples = max(1, int(round(self.min_distance_sec * self.fs)))
        peak_indices, _ = find_peaks(
            filtered_ecg,
            distance=min_dist_samples,
            prominence=self.prominence
        )
        peak_times = peak_indices / self.fs
        return peak_indices, peak_times

    # --------------------------------------------------------
    # 4. P-Q-R-S-T WAVE DETECTION
    # --------------------------------------------------------
    def detect_pqrst(self, filtered_ecg: np.ndarray, r_indices: np.ndarray) -> np.ndarray:
        """Locate P, Q, R, S, T peaks around each detected R-peak."""
        pqrst = []
        sig_len = len(filtered_ecg)

        for r_idx in r_indices:
            r_time = r_idx / self.fs

            def local_peak(win_tuple: Tuple[float, float], mode: str) -> float:
                left = max(0, r_idx + int(round(win_tuple[0] * self.fs)))
                right = min(sig_len, r_idx + int(round(win_tuple[1] * self.fs)) + 1)
                if right <= left:
                    return np.nan
                seg = filtered_ecg[left:right]
                if len(seg) == 0:
                    return np.nan
                if mode == "max":
                    return (left + int(np.argmax(seg))) / self.fs
                else:
                    return (left + int(np.argmin(seg))) / self.fs

            p_time = local_peak(self.search_windows["P"], "max")
            q_time = local_peak(self.search_windows["Q"], "min")
            s_time = local_peak(self.search_windows["S"], "min")
            t_time = local_peak(self.search_windows["T"], "max")

            pqrst.append([p_time, q_time, r_time, s_time, t_time])

        return np.asarray(pqrst, dtype=float)

    # --------------------------------------------------------
    # 5. FEATURE EXTRACTION & PER-BEAT TABLE
    # --------------------------------------------------------
    def extract_features(
        self,
        filtered_ecg: np.ndarray,
        pqrst: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, pd.DataFrame]:
        """
        Extract the 10 features matching Stage 2E definitions
        and construct a detailed beat-by-beat measurement table.
        """
        p_times = pqrst[:, 0]
        q_times = pqrst[:, 1]
        r_times = pqrst[:, 2]
        s_times = pqrst[:, 3]
        t_times = pqrst[:, 4]

        valid_rr = np.diff(r_times)
        valid_rr = valid_rr[valid_rr > 0]

        if len(valid_rr) < 1:
            raise ValueError("Less than 2 valid beats detected; cannot compute RR intervals.")

        mean_rr = float(np.mean(valid_rr))
        heart_rate = float(60.0 / mean_rr)
        sdnn_s = float(np.std(valid_rr, ddof=1)) if len(valid_rr) > 1 else 0.0

        pr_peak_s = r_times - p_times
        qrs_peak_s = s_times - q_times
        qt_peak_s = t_times - q_times

        def amplitude_at(times: np.ndarray) -> np.ndarray:
            valid = ~np.isnan(times)
            amps = np.full_like(times, np.nan, dtype=float)
            idx = np.rint(times[valid] * self.fs).astype(int)
            idx = np.clip(idx, 0, len(filtered_ecg) - 1)
            amps[valid] = filtered_ecg[idx]
            return amps

        p_amp = amplitude_at(p_times)
        r_amp = amplitude_at(r_times)
        t_amp = amplitude_at(t_times)
        st_times = s_times + 0.060
        st_amp = amplitude_at(st_times)

        # Per-beat DataFrame
        beat_rows = []
        for i in range(len(pqrst)):
            prev_rr = (r_times[i] - r_times[i-1]) if i > 0 else np.nan
            beat_rows.append({
                "Beat": i + 1,
                "R_Time_s": r_times[i],
                "RR_Interval_s": prev_rr,
                "PR_Interval_s": pr_peak_s[i],
                "QRS_Duration_s": qrs_peak_s[i],
                "QT_Interval_s": qt_peak_s[i],
                "P_Amplitude": p_amp[i],
                "R_Amplitude": r_amp[i],
                "T_Amplitude": t_amp[i],
                "ST_Segment": st_amp[i],
            })
        beat_df = pd.DataFrame(beat_rows)

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

        vector = np.array([features[name] for name in self.feature_order], dtype=float)
        return features, vector, beat_df

    # --------------------------------------------------------
    # 6. MODEL DECISION EXPLAINABILITY
    # --------------------------------------------------------
    def _trace_tree_explanation(self, scaled_vector: np.ndarray) -> List[str]:
        """Generate explainability traces for Random Forest or Decision Tree."""
        scaled_x = scaled_vector.reshape(1, -1)
        explanation = []

        # Case A: Ensemble Random Forest
        if hasattr(self.model, "estimators_"):
            n_trees = len(self.model.estimators_)
            tree_preds = [tree.predict(scaled_x)[0] for tree in self.model.estimators_]
            norm_votes = int(np.sum(np.array(tree_preds) == 0))
            abn_votes = int(np.sum(np.array(tree_preds) == 1))

            explanation.append(
                f"Ensemble Consensus: {n_trees} trees evaluated ({norm_votes} voted Normal [0], {abn_votes} voted Abnormal [1])."
            )

            # Feature influence ranking (combination of absolute scaled deviation and feature importance)
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                deviations = np.abs(scaled_x[0])
                impact_scores = deviations * importances
                top_indices = np.argsort(impact_scores)[::-1]

                explanation.append("Top Predictive Feature Influences for this Signal:")
                for rank, f_idx in enumerate(top_indices[:4], 1):
                    feat_name = self.feature_order[f_idx]
                    raw_val = self.scaler.mean_[f_idx] + scaled_x[0, f_idx] * self.scaler.scale_[f_idx]
                    explanation.append(
                        f"  * {feat_name:12s}: raw = {raw_val:8.4f} (scaled = {scaled_x[0, f_idx]:+6.2f}, global importance = {importances[f_idx]*100:5.2f}%)"
                    )

            return explanation

        # Case B: Legacy Decision Tree
        if hasattr(self.model, "tree_"):
            tree = self.model.tree_
            feature_names = self.feature_order

            node_id = 0
            step = 1
            while tree.children_left[node_id] != tree.children_right[node_id]:
                feature_idx = tree.feature[node_id]
                feat_name = feature_names[feature_idx]
                threshold = tree.threshold[node_id]
                input_val = scaled_x[0, feature_idx]

                # Approximate unscaled threshold
                if hasattr(self.scaler, "mean_") and hasattr(self.scaler, "scale_"):
                    raw_thresh = self.scaler.mean_[feature_idx] + threshold * self.scaler.scale_[feature_idx]
                    thresh_str = f"{raw_thresh:.4f} (scaled: {threshold:.4f})"
                else:
                    thresh_str = f"{threshold:.4f}"

                if input_val <= threshold:
                    next_node = tree.children_left[node_id]
                    direction = "<="
                else:
                    next_node = tree.children_right[node_id]
                    direction = ">"

                explanation.append(
                    f"Rule {step}: {feat_name} {direction} {thresh_str} "
                    f"(Input raw: {self.scaler.mean_[feature_idx] + input_val * self.scaler.scale_[feature_idx]:.4f})"
                )
                node_id = next_node
                step += 1

            leaf_values = tree.value[node_id][0]
            total_samples = np.sum(leaf_values)
            if total_samples > 0:
                norm_pct = (leaf_values[0] / total_samples) * 100
                abn_pct = (leaf_values[1] / total_samples) * 100
                explanation.append(
                    f"Final Leaf {node_id}: {total_samples:.0f} training beats reached leaf "
                    f"({norm_pct:.1f}% Normal, {abn_pct:.1f}% Abnormal)."
                )

            return explanation

        return ["Model does not expose decision structure."]

    # --------------------------------------------------------
    # 7. MAIN END-TO-END PROCESSING METHOD
    # --------------------------------------------------------
    def process(
        self,
        source: Union[np.ndarray, str, Path],
        sampling_rate: Optional[float] = None
    ) -> ECGPipelineResult:
        """
        Process an ECG recording end-to-end:
        validate -> filter -> detect peaks -> extract features -> classify.
        """
        if sampling_rate is not None:
            self.fs = float(sampling_rate)

        warnings_list = []
        errors_list = []
        limitations_list = [
            "This exploratory analysis is for engineering demonstration, not medical diagnosis.",
            "The classifier was trained on beat-level rows; features here are aggregated window averages.",
            "Aggregating normal beats with premature beats dilutes morphological metrics.",
        ]

        # 1. Load signal
        try:
            raw_ecg = self.load_signal(source)
        except Exception as exc:
            return ECGPipelineResult(
                success=False,
                sampling_rate=self.fs,
                duration_sec=0.0,
                sample_count=0,
                raw_ecg=np.array([]),
                filtered_ecg=np.array([]),
                r_peak_indices=np.array([]),
                r_peak_times=np.array([]),
                rr_intervals=np.array([]),
                pqrst_matrix=np.empty((0, 5)),
                beat_table=pd.DataFrame(),
                features={},
                feature_vector=np.array([]),
                feature_names=self.feature_order,
                errors=[f"Signal ingestion failure: {exc}"],
                limitations=limitations_list
            )

        duration_sec = len(raw_ecg) / self.fs
        sample_count = len(raw_ecg)

        if duration_sec < MIN_SIGNAL_SECONDS:
            warnings_list.append(
                f"Short signal ({duration_sec:.2f} s < {MIN_SIGNAL_SECONDS} s). "
                "HRV/SDNN metrics may have higher variance."
            )

        # 2. Filter signal
        try:
            filtered_ecg = self.filter(raw_ecg)
        except Exception as exc:
            errors_list.append(f"Bandpass filtering failure: {exc}")
            return ECGPipelineResult(
                success=False,
                sampling_rate=self.fs,
                duration_sec=duration_sec,
                sample_count=sample_count,
                raw_ecg=raw_ecg,
                filtered_ecg=np.array([]),
                r_peak_indices=np.array([]),
                r_peak_times=np.array([]),
                rr_intervals=np.array([]),
                pqrst_matrix=np.empty((0, 5)),
                beat_table=pd.DataFrame(),
                features={},
                feature_vector=np.array([]),
                feature_names=self.feature_order,
                errors=errors_list,
                warnings=warnings_list,
                limitations=limitations_list
            )

        # 3. Detect R-peaks
        r_indices, r_times = self.detect_r_peaks(filtered_ecg)
        if len(r_indices) < 2:
            errors_list.append(
                f"Too few R-peaks detected ({len(r_indices)}). "
                "Cannot perform beat interval or HRV calculations."
            )
            return ECGPipelineResult(
                success=False,
                sampling_rate=self.fs,
                duration_sec=duration_sec,
                sample_count=sample_count,
                raw_ecg=raw_ecg,
                filtered_ecg=filtered_ecg,
                r_peak_indices=r_indices,
                r_peak_times=r_times,
                rr_intervals=np.array([]),
                pqrst_matrix=np.empty((0, 5)),
                beat_table=pd.DataFrame(),
                features={},
                feature_vector=np.array([]),
                feature_names=self.feature_order,
                errors=errors_list,
                warnings=warnings_list,
                limitations=limitations_list
            )

        rr_intervals = np.diff(r_times)

        # 4. Detect PQRST
        pqrst_matrix = self.detect_pqrst(filtered_ecg, r_indices)

        # 5. Extract Features
        try:
            features, feature_vector, beat_table = self.extract_features(filtered_ecg, pqrst_matrix)
        except Exception as exc:
            errors_list.append(f"Feature extraction failure: {exc}")
            return ECGPipelineResult(
                success=False,
                sampling_rate=self.fs,
                duration_sec=duration_sec,
                sample_count=sample_count,
                raw_ecg=raw_ecg,
                filtered_ecg=filtered_ecg,
                r_peak_indices=r_indices,
                r_peak_times=r_times,
                rr_intervals=rr_intervals,
                pqrst_matrix=pqrst_matrix,
                beat_table=pd.DataFrame(),
                features={},
                feature_vector=np.array([]),
                feature_names=self.feature_order,
                errors=errors_list,
                warnings=warnings_list,
                limitations=limitations_list
            )

        # 6. Exploratory Model Inference
        predicted_label = None
        predicted_class = None
        class_probabilities = {}
        decision_explanation = []

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scaled_x = self.scaler.transform(feature_vector.reshape(1, -1))
                pred = int(self.model.predict(scaled_x)[0])
                probs = self.model.predict_proba(scaled_x)[0]

            predicted_label = pred
            predicted_class = "NORMAL" if pred == 0 else "ABNORMAL"
            for cls_idx, prob in zip(self.model.classes_, probs):
                class_probabilities[int(cls_idx)] = float(prob)

            decision_explanation = self._trace_tree_explanation(scaled_x)

        except Exception as exc:
            errors_list.append(f"Model inference failure: {exc}")

        return ECGPipelineResult(
            success=len(errors_list) == 0,
            sampling_rate=self.fs,
            duration_sec=duration_sec,
            sample_count=sample_count,
            raw_ecg=raw_ecg,
            filtered_ecg=filtered_ecg,
            r_peak_indices=r_indices,
            r_peak_times=r_times,
            rr_intervals=rr_intervals,
            pqrst_matrix=pqrst_matrix,
            beat_table=beat_table,
            features=features,
            feature_vector=feature_vector,
            feature_names=self.feature_order,
            predicted_label=predicted_label,
            predicted_class=predicted_class,
            class_probabilities=class_probabilities,
            decision_explanation=decision_explanation,
            warnings=warnings_list,
            errors=errors_list,
            limitations=limitations_list
        )


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    print("=" * 72)
    print("ECG FINAL PRO — UNIFIED PIPELINE CLI")
    print("=" * 72)

    if len(sys.argv) < 2:
        print("Usage: py -3.12 ecg_pipeline.py <path_to_ecg.npy_or_csv>")
        print("Example: py -3.12 ecg_pipeline.py 01_signal_generation/test_cases/normal_ecg.npy")
        sys.exit(0)

    input_path = sys.argv[1]
    pipeline = ECGPipeline()
    result = pipeline.process(input_path)
    print(result.summary())
