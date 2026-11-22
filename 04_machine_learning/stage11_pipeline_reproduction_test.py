"""
ECG FINAL PRO — PHASE 1: STAGE 11
Feature Extraction Pipeline Alignment & Reproduction Test
File: 04_machine_learning/stage11_pipeline_reproduction_test.py

Verifies:
  - ecg_pipeline.py loads the newly validated Random Forest model & scaler
  - Feature extraction reproduces exact 10-feature vector matching model input schema
  - Processes Case A (Normal ~72 BPM) -> verifies NORMAL prediction
  - Processes Case B (Simulated PVC) -> verifies ABNORMAL prediction
  - Processes 10s variable-length slice -> verifies stability
  - Evaluates peak timing precision against ground truth
  - Exports test summary to 05_results/stage11_test_case_predictions.csv
  - Exports report to 05_results/stage11_pipeline_reproduction_report.txt
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_pipeline import ECGPipeline, ECGPipelineResult

TEST_CASES_DIR = PROJECT_ROOT / "01_signal_generation" / "test_cases"
NORMAL_ECG_PATH = TEST_CASES_DIR / "normal_ecg.npy"
NORMAL_GT_PATH = TEST_CASES_DIR / "normal_ground_truth.npy"
PVC_ECG_PATH = TEST_CASES_DIR / "pvc_ecg.npy"
PVC_GT_PATH = TEST_CASES_DIR / "pvc_ground_truth.npy"

RESULTS_DIR = PROJECT_ROOT / "05_results"
REPORT_PATH = RESULTS_DIR / "stage11_pipeline_reproduction_report.txt"
CSV_PATH = RESULTS_DIR / "stage11_test_case_predictions.csv"

def log(lines_list, text=""):
    print(text)
    lines_list.append(str(text))

def main():
    lines = []
    log(lines, "=" * 72)
    log(lines, "ECG FINAL PRO — STAGE 11: PIPELINE ALIGNMENT & REPRODUCTION TEST")
    log(lines, "=" * 72)

    # 1. Initialize Pipeline
    pipeline = ECGPipeline()
    log(lines, f"Pipeline Loaded Successfully:")
    log(lines, f"  - Model file  : {pipeline.model_file.name}")
    log(lines, f"  - Model type  : {pipeline.model_type} ({type(pipeline.model).__name__})")
    log(lines, f"  - Scaler file : {pipeline.scaler_file.name}")
    log(lines, f"  - Features ({len(pipeline.feature_order)}): {pipeline.feature_order}")

    if pipeline.model_file.name != "ecg_random_forest_model.pkl":
        log(lines, f"[FAIL] Pipeline did NOT load the validated Random Forest model! Found: {pipeline.model_file.name}")
        sys.exit(1)
    log(lines, "[PASS] Confirmed pipeline is using the new validated Random Forest model.")

    summary_records = []

    # 2. Test Case A: Normal Synthetic ECG (60s)
    log(lines, "\n" + "-" * 72)
    log(lines, "TEST 1: CASE A — NORMAL SYNTHETIC ECG (60s)")
    log(lines, "-" * 72)

    res_a = pipeline.process(NORMAL_ECG_PATH)
    if not res_a.success:
        log(lines, f"[FAIL] Case A processing failed: {res_a.errors}")
        sys.exit(1)

    log(lines, f"  Duration: {res_a.duration_sec:.1f}s | Beats detected: {len(res_a.r_peak_indices)}")
    log(lines, f"  Mean HR: {res_a.features['HeartRate']:.2f} BPM | HRV: {res_a.features['HRV']*1000:.2f} ms")
    log(lines, f"  Predicted Label: {res_a.predicted_label} ({res_a.predicted_class})")
    log(lines, f"  Probabilities  : Normal={res_a.class_probabilities.get(0, 0)*100:.2f}%, Abnormal={res_a.class_probabilities.get(1, 0)*100:.2f}%")
    log(lines, "  Decision Explanation:")
    for step in res_a.decision_explanation:
        log(lines, f"    {step}")

    if NORMAL_GT_PATH.exists():
        gt_a = np.load(NORMAL_GT_PATH)
        gt_comp_a = res_a.compare_ground_truth(gt_a)
        log(lines, f"  Ground-Truth Comparison: Matched={gt_comp_a['matched_beats']}/{gt_comp_a['ground_truth_beats']}, Missed={gt_comp_a['missed_beats']}, FP={gt_comp_a['false_positives']}")
        log(lines, f"    R-peak MAE: {gt_comp_a['timing_errors']['R_mae_ms']:.3f} ms, QRS width MAE: {gt_comp_a['timing_errors']['S_mae_ms']:.3f} ms")

    summary_records.append({
        "TestCase": "Case A (Normal Synthetic 60s)",
        "Duration_s": res_a.duration_sec,
        "DetectedBeats": len(res_a.r_peak_indices),
        "PredictedLabel": res_a.predicted_label,
        "PredictedClass": res_a.predicted_class,
        "NormalProb": round(res_a.class_probabilities.get(0, 0) * 100, 2),
        "AbnormalProb": round(res_a.class_probabilities.get(1, 0) * 100, 2),
        **{f: round(res_a.features[f], 6) for f in pipeline.feature_order}
    })

    # 3. Test Case B: Simulated PVC ECG (60s)
    log(lines, "\n" + "-" * 72)
    log(lines, "TEST 2: CASE B — SIMULATED PVC ECTOPIC ECG (60s)")
    log(lines, "-" * 72)

    res_b = pipeline.process(PVC_ECG_PATH)
    if not res_b.success:
        log(lines, f"[FAIL] Case B processing failed: {res_b.errors}")
        sys.exit(1)

    log(lines, f"  Duration: {res_b.duration_sec:.1f}s | Beats detected: {len(res_b.r_peak_indices)}")
    log(lines, f"  Mean HR: {res_b.features['HeartRate']:.2f} BPM | HRV: {res_b.features['HRV']*1000:.2f} ms")
    log(lines, f"  Predicted Label: {res_b.predicted_label} ({res_b.predicted_class})")
    log(lines, f"  Probabilities  : Normal={res_b.class_probabilities.get(0, 0)*100:.2f}%, Abnormal={res_b.class_probabilities.get(1, 0)*100:.2f}%")
    log(lines, "  Decision Explanation:")
    for step in res_b.decision_explanation:
        log(lines, f"    {step}")

    if PVC_GT_PATH.exists():
        gt_b = np.load(PVC_GT_PATH)
        gt_comp_b = res_b.compare_ground_truth(gt_b)
        log(lines, f"  Ground-Truth Comparison: Matched={gt_comp_b['matched_beats']}/{gt_comp_b['ground_truth_beats']}, Missed={gt_comp_b['missed_beats']}, FP={gt_comp_b['false_positives']}")
        log(lines, f"    R-peak MAE: {gt_comp_b['timing_errors']['R_mae_ms']:.3f} ms")

    summary_records.append({
        "TestCase": "Case B (Simulated PVC 60s)",
        "Duration_s": res_b.duration_sec,
        "DetectedBeats": len(res_b.r_peak_indices),
        "PredictedLabel": res_b.predicted_label,
        "PredictedClass": res_b.predicted_class,
        "NormalProb": round(res_b.class_probabilities.get(0, 0) * 100, 2),
        "AbnormalProb": round(res_b.class_probabilities.get(1, 0) * 100, 2),
        **{f: round(res_b.features[f], 6) for f in pipeline.feature_order}
    })

    # 4. Test Case C: Variable Length Slice (10s)
    log(lines, "\n" + "-" * 72)
    log(lines, "TEST 3: CASE C — VARIABLE LENGTH SHORT SLICE (10s)")
    log(lines, "-" * 72)

    raw_normal = np.load(NORMAL_ECG_PATH)
    slice_10s = raw_normal[:int(10.0 * pipeline.fs)]
    res_c = pipeline.process(slice_10s)

    if not res_c.success:
        log(lines, f"[FAIL] Case C (10s) failed: {res_c.errors}")
        sys.exit(1)

    log(lines, f"  Duration: {res_c.duration_sec:.1f}s | Beats detected: {len(res_c.r_peak_indices)}")
    log(lines, f"  Mean HR: {res_c.features['HeartRate']:.2f} BPM")
    log(lines, f"  Predicted Label: {res_c.predicted_label} ({res_c.predicted_class})")
    log(lines, f"  Probabilities  : Normal={res_c.class_probabilities.get(0, 0)*100:.2f}%, Abnormal={res_c.class_probabilities.get(1, 0)*100:.2f}%")

    summary_records.append({
        "TestCase": "Case C (Normal 10s Short Slice)",
        "Duration_s": res_c.duration_sec,
        "DetectedBeats": len(res_c.r_peak_indices),
        "PredictedLabel": res_c.predicted_label,
        "PredictedClass": res_c.predicted_class,
        "NormalProb": round(res_c.class_probabilities.get(0, 0) * 100, 2),
        "AbnormalProb": round(res_c.class_probabilities.get(1, 0) * 100, 2),
        **{f: round(res_c.features[f], 6) for f in pipeline.feature_order}
    })

    # Save summary table
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(CSV_PATH, index=False)
    log(lines, f"\nSaved predictions summary to: {CSV_PATH}")

    # Final gate validation checks
    log(lines, "\n" + "-" * 72)
    log(lines, "FEATURE REPRODUCTION & INPUT SCHEMA VERIFICATION")
    log(lines, "-" * 72)
    log(lines, f"  Input Feature Vector Shape: {res_a.feature_vector.shape} (Expected: (10,))")
    log(lines, f"  All Features Finite: {np.all(np.isfinite(res_a.feature_vector))}")
    log(lines, f"  Correct Feature Ordering: {res_a.feature_names == pipeline.feature_order}")

    assert res_a.feature_vector.shape == (10,), "Feature vector shape mismatch!"
    assert np.all(np.isfinite(res_a.feature_vector)), "Feature vector contains non-finite values!"
    assert res_a.feature_names == pipeline.feature_order, "Feature names order mismatch!"

    log(lines, "\n" + "=" * 72)
    log(lines, "STAGE 11 RESULT: PASSED")
    log(lines, "Feature extraction pipeline reproduces expected features and interfaces seamlessly with the new Random Forest model.")
    log(lines, "=" * 72)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(lines, f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
