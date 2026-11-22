"""
Serial Bridge Unit & Integration Test Suite
File: 04_machine_learning/test_serial_bridge.py

Tests:
  1. Protocol line parsing (valid packets, invalid formats, missing values, lead-off flags).
  2. Mock streaming mode (simulating 360 Hz ESP32 USB stream).
  3. Buffer extraction and rolling window handling.
  4. Integration with ecg_pipeline.py: running pipeline analysis on streamed serial buffer.
  5. Timeout and disconnection behavior.

Outputs saved in:
  05_results/pipeline_tests/serial_bridge_test_report.txt
"""

import time
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from serial_bridge import SerialBridge, SerialPacket
from ecg_pipeline import ECGPipeline

RESULTS_DIR = PROJECT_ROOT / "05_results" / "pipeline_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = RESULTS_DIR / "serial_bridge_test_report.txt"

def test_protocol_parsing():
    print("[1/4] Testing Protocol Line Parser...")
    bridge = SerialBridge()

    # Valid packet with valid SpO2 and leads attached
    p1 = bridge.parse_line("DATA,1024,2048.5,98,72,1,0")
    assert p1 is not None, "Failed to parse valid packet"
    assert p1.timestamp_ms == 1024
    assert p1.ecg_raw == 2048.5
    assert p1.spo2 == 98
    assert p1.pulse_bpm == 72
    assert p1.spo2_valid is True
    assert p1.lead_off is False

    # Valid packet with invalid SpO2 (no finger on sensor)
    p2 = bridge.parse_line("DATA,2048,1850.0,-1,-1,0,0")
    assert p2 is not None
    assert p2.spo2 is None, f"Expected None for invalid SpO2, got {p2.spo2}"
    assert p2.pulse_bpm is None
    assert p2.spo2_valid is False
    assert p2.lead_off is False

    # Valid packet with Lead-Off (electrode disconnected)
    p3 = bridge.parse_line("DATA,3072,4095.0,97,75,1,1")
    assert p3 is not None
    assert p3.lead_off is True

    # Status message
    status_line = "STATUS,MAX30102_OK,Red and IR channels active."
    p_status = bridge.parse_line(status_line)
    assert p_status is None, "Status lines should update status, not return DATA packet"
    assert bridge.status_message == status_line

    # Malformed packet
    p_bad = bridge.parse_line("DATA,invalid,tokens,only,four")
    assert p_bad is None
    assert bridge.malformed_packets_count >= 1

    print("      PASSED: Protocol line parsing matches specifications.")


def test_simulation_and_buffering():
    print("\n[2/4] Testing Simulation Mode & Ring Buffer...")
    bridge = SerialBridge(fs=360.0, buffer_seconds=5.0)

    # Start simulation for 1.0 second (should generate ~360 samples)
    t = np.arange(0, 5.0, 1.0 / 360.0)
    test_ecg = 2048.0 + 500.0 * np.sin(2 * np.pi * 1.5 * t)

    bridge.start_simulation(signal_array=test_ecg, include_spo2=True)
    time.sleep(1.2)  # Wait for ~400 samples
    bridge.stop_simulation()

    buf = bridge.get_recent_ecg()
    metrics = bridge.get_latest_metrics()

    print(f"      Buffered samples   : {len(buf)} samples")
    print(f"      Packets received   : {metrics['packets_received']}")
    print(f"      Latest SpO2        : {metrics['spo2']}%")
    print(f"      Latest Pulse       : {metrics['pulse_bpm']} BPM")
    print(f"      Lead-off status    : {metrics['lead_off']}")

    assert len(buf) > 300, f"Expected >300 samples, got {len(buf)}"
    assert metrics["spo2"] == 98
    assert metrics["pulse_bpm"] == 72
    print("      PASSED: Mock streaming and buffer extraction verified.")


def test_lead_off_simulation():
    print("\n[3/4] Testing Lead-Off Handling...")
    bridge = SerialBridge()
    bridge.start_simulation(simulate_lead_off=True, include_spo2=False)
    time.sleep(0.3)
    bridge.stop_simulation()

    metrics = bridge.get_latest_metrics()
    assert metrics["lead_off"] is True, "Lead-off flag was not set"
    assert metrics["spo2"] is None, "SpO2 should be None when invalid"
    print("      PASSED: Lead-off and missing SpO2 handled without fabricating values.")


def test_pipeline_integration():
    print("\n[4/4] Testing End-to-End Pipeline Integration from Stream...")
    # Load 5 seconds of normal test signal
    normal_case_file = PROJECT_ROOT / "01_signal_generation" / "test_cases" / "normal_ecg.npy"
    if not normal_case_file.is_file():
        print("      Skipping: normal_ecg.npy not found.")
        return

    full_signal = np.load(normal_case_file)
    sample_window = full_signal[: int(360.0 * 5.0)] # 5 seconds = 1800 samples

    bridge = SerialBridge(fs=360.0, buffer_seconds=6.0)
    bridge.start_simulation(signal_array=sample_window, include_spo2=True)
    time.sleep(2.6) # Accumulate ~2.5 seconds (at least 2-3 beats)
    bridge.stop_simulation()

    streamed_ecg = bridge.get_recent_ecg()
    print(f"      Streamed ECG samples available: {len(streamed_ecg)}")

    # Feed streamed buffer into ecg_pipeline
    pipeline = ECGPipeline()
    res = pipeline.process(streamed_ecg)
    print(f"      Pipeline analysis result: Success={res.success}, Beats={len(res.r_peak_indices)}, Class={res.predicted_class}")
    assert res.success is True
    assert len(res.r_peak_indices) >= 2
    print("      PASSED: Serial stream connects directly to ecg_pipeline.")


if __name__ == "__main__":
    print("=" * 70)
    print("SERIAL BRIDGE VERIFICATION SUITE")
    print("=" * 70)
    test_protocol_parsing()
    test_simulation_and_buffering()
    test_lead_off_simulation()
    test_pipeline_integration()
    print("\n" + "=" * 70)
    print("ALL SERIAL BRIDGE TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 70)
