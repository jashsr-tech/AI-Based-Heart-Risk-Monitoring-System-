"""
AI-Based Heart Risk Monitoring System — Serial Communication Bridge
File: serial_bridge.py

Provides:
  - Robust serial connection to ESP32 over USB (configurable COM port & 115200 baud).
  - Background daemon thread for continuous non-blocking packet ingestion.
  - Line-based protocol parser supporting:
      DATA,timestamp_ms,ecg_raw,spo2,pulse_bpm,spo2_valid,lead_off
      STATUS,message
  - Rolling ring-buffer storing recent ECG samples for live visualization and pipeline analysis.
  - Strict differentiation between:
      * ECG-derived heart rate (calculated from R-peak detection in ecg_pipeline.py)
      * MAX30102 optical pulse rate (from red/IR photoplethysmography)
  - Honest missing-value handling (returns None when finger is absent or sensor fails; no fabricated data).
  - Lead-off detection tracking (AD8232 electrode disconnect).
  - Built-in Mock/Simulation generator for rigorous verification prior to hardware attachment.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Callable
from pathlib import Path

import numpy as np

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    serial = None
    PYSERIAL_AVAILABLE = False


@dataclass
class SerialPacket:
    """Parsed data packet received from ESP32."""
    timestamp_ms: int
    ecg_raw: float
    spo2: Optional[int]
    pulse_bpm: Optional[int]
    spo2_valid: bool
    lead_off: bool
    raw_line: str


class SerialBridge:
    """
    Manages USB serial communications with the ESP32 biosensing module.
    Runs a non-blocking background reader thread to update thread-safe buffers.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        buffer_seconds: float = 15.0,
        fs: float = 360.0
    ):
        self.port = port
        self.baudrate = baudrate
        self.fs = fs
        self.buffer_size = int(round(buffer_seconds * fs))

        # Thread-safe sample buffer & latest metrics
        self._lock = threading.Lock()
        self._ecg_buffer: deque = deque(maxlen=self.buffer_size)
        self._time_buffer: deque = deque(maxlen=self.buffer_size)

        self.latest_packet: Optional[SerialPacket] = None
        self.latest_spo2: Optional[int] = None
        self.latest_pulse_bpm: Optional[int] = None
        self.lead_off: bool = False
        self.spo2_valid: bool = False

        # Status tracking
        self.is_connected: bool = False
        self.status_message: str = "Disconnected"
        self.last_packet_time: float = 0.0
        self.total_packets_received: int = 0
        self.malformed_packets_count: int = 0

        # Worker control
        self._serial_handle = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Simulation mode for testing
        self.is_simulating: bool = False
        self._sim_thread: Optional[threading.Thread] = None

        # Callbacks (optional listeners for UI updates)
        self.on_packet_callback: Optional[Callable[[SerialPacket], None]] = None
        self.on_status_callback: Optional[Callable[[str], None]] = None

    # --------------------------------------------------------
    # PORT ENUMERATION
    # --------------------------------------------------------
    @staticmethod
    def get_available_ports() -> List[Tuple[str, str]]:
        """Return list of (device, description) for detected COM ports."""
        if not PYSERIAL_AVAILABLE:
            return []
        try:
            ports = serial.tools.list_ports.comports()
            return [(p.device, p.description) for p in ports]
        except Exception:
            return []

    # --------------------------------------------------------
    # CONNECTION MANAGEMENT
    # --------------------------------------------------------
    def connect(self, port: Optional[str] = None) -> bool:
        """Open serial connection and launch reader thread."""
        if self.is_connected or self.is_simulating:
            self.disconnect()

        target_port = port or self.port
        if not target_port:
            self._update_status("No COM port specified.")
            return False

        if not PYSERIAL_AVAILABLE:
            self._update_status("pyserial package is not available.")
            return False

        try:
            self._serial_handle = serial.Serial(
                port=target_port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            time.sleep(1.5)  # Allow ESP32 bootloader reset cycle
            self._serial_handle.reset_input_buffer()

            self.port = target_port
            self.is_connected = True
            self.last_packet_time = time.time()
            self._stop_event.clear()

            self._worker_thread = threading.Thread(
                target=self._read_worker,
                daemon=True,
                name="ESP32_Serial_Reader"
            )
            self._worker_thread.start()

            self._update_status(f"Connected to ESP32 on {target_port} @ {self.baudrate} baud.")
            return True

        except Exception as exc:
            self.is_connected = False
            self._serial_handle = None
            self._update_status(f"Connection failed on {target_port}: {exc}")
            return False

    def disconnect(self) -> None:
        """Safely terminate reader thread and close serial port."""
        self._stop_event.set()

        if self.is_simulating:
            self.stop_simulation()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        if self._serial_handle:
            try:
                self._serial_handle.close()
            except Exception:
                pass
            self._serial_handle = None

        self.is_connected = False
        self._update_status("Disconnected")

    def _update_status(self, msg: str) -> None:
        self.status_message = msg
        if self.on_status_callback:
            try:
                self.on_status_callback(msg)
            except Exception:
                pass

    # --------------------------------------------------------
    # PROTOCOL PARSER
    # --------------------------------------------------------
    def parse_line(self, line: str) -> Optional[SerialPacket]:
        """
        Parse raw line from ESP32.
        Expected format:
          DATA,timestamp_ms,ecg_raw,spo2,pulse_bpm,spo2_valid,lead_off
        """
        line_clean = line.strip()
        if not line_clean:
            return None

        # Handle STATUS or LOG messages
        if line_clean.startswith("STATUS,") or line_clean.startswith("LOG,"):
            self._update_status(line_clean)
            return None

        if not line_clean.startswith("DATA,"):
            # Unknown prefix or debug text
            return None

        parts = line_clean.split(",")
        if len(parts) < 7:
            self.malformed_packets_count += 1
            return None

        try:
            t_ms = int(parts[1])
            ecg_raw = float(parts[2])
            raw_spo2 = int(parts[3])
            raw_bpm = int(parts[4])
            spo2_val = (int(parts[5]) == 1)
            lead_off = (int(parts[6]) == 1)

            # Honest missing-value conversion: -1 or invalid flag means unavailable.
            # SpO2 and optical pulse rate are validated independently.
            spo2 = raw_spo2 if (spo2_val and 70 <= raw_spo2 <= 100) else None
            pulse_bpm = raw_bpm if (30 <= raw_bpm <= 220) else None

            packet = SerialPacket(
                timestamp_ms=t_ms,
                ecg_raw=ecg_raw,
                spo2=spo2,
                pulse_bpm=pulse_bpm,
                spo2_valid=spo2_val,
                lead_off=lead_off,
                raw_line=line_clean
            )
            return packet

        except ValueError:
            self.malformed_packets_count += 1
            return None

    # --------------------------------------------------------
    # BACKGROUND READER THREAD
    # --------------------------------------------------------
    def _read_worker(self) -> None:
        """Continuous serial read loop."""
        while not self._stop_event.is_set():
            if not self._serial_handle or not self._serial_handle.is_open:
                break

            try:
                raw_bytes = self._serial_handle.readline()
                if not raw_bytes:
                    # Timeout check: if no data for > 2.5s, report timeout
                    if time.time() - self.last_packet_time > 2.5:
                        self._update_status("Warning: Serial data timeout (no packets received).")
                    continue

                line = raw_bytes.decode("utf-8", errors="ignore")
                packet = self.parse_line(line)

                if packet:
                    self.last_packet_time = time.time()
                    self.total_packets_received += 1

                    with self._lock:
                        self.latest_packet = packet
                        self.latest_spo2 = packet.spo2
                        self.latest_pulse_bpm = packet.pulse_bpm
                        self.lead_off = packet.lead_off
                        self.spo2_valid = packet.spo2_valid

                        self._ecg_buffer.append(packet.ecg_raw)
                        self._time_buffer.append(packet.timestamp_ms / 1000.0)

                    if self.on_packet_callback:
                        self.on_packet_callback(packet)

            except Exception as exc:
                self._update_status(f"Serial read error: {exc}")
                break

        self.is_connected = False
        self._update_status("Disconnected")

    # --------------------------------------------------------
    # BUFFER ACCESS
    # --------------------------------------------------------
    def get_recent_ecg(self, duration_sec: Optional[float] = None) -> np.ndarray:
        """Return the most recent ECG samples as a 1D NumPy array."""
        with self._lock:
            if not self._ecg_buffer:
                return np.array([], dtype=float)

            data = np.asarray(self._ecg_buffer, dtype=float)

            if duration_sec is not None:
                samples_needed = int(round(duration_sec * self.fs))
                if len(data) > samples_needed:
                    data = data[-samples_needed:]

            return data

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Return current status dictionary."""
        with self._lock:
            return {
                "connected": self.is_connected or self.is_simulating,
                "status": self.status_message,
                "buffer_len": len(self._ecg_buffer),
                "spo2": self.latest_spo2,
                "pulse_bpm": self.latest_pulse_bpm,
                "spo2_valid": self.spo2_valid,
                "lead_off": self.lead_off,
                "packets_received": self.total_packets_received,
                "malformed_packets": self.malformed_packets_count,
            }

    # --------------------------------------------------------
    # SIMULATION / MOCK STREAMER FOR TESTING
    # --------------------------------------------------------
    def start_simulation(
        self,
        signal_array: Optional[np.ndarray] = None,
        include_spo2: bool = False,
        simulate_lead_off: bool = False
    ) -> None:
        """
        Launch a mock packet generator delivering packets at 360 Hz.
        Used for testing the dashboard and serial bridge when hardware is absent.
        """
        self.disconnect()

        if signal_array is None:
            # Default to small synthetic pattern
            t = np.arange(0, 5.0, 1.0 / self.fs)
            signal_array = 2048.0 + 800.0 * np.sin(2 * np.pi * 1.2 * t)

        self.is_simulating = True
        self._stop_event.clear()
        self._update_status("Simulation Mode Active: Streaming 360 Hz mock packets.")

        def sim_worker():
            sample_period_s = 1.0 / self.fs
            idx = 0
            n_samples = len(signal_array)
            start_t = time.time()

            while not self._stop_event.is_set():
                now_t = time.time()
                t_ms = int((now_t - start_t) * 1000)
                ecg_val = float(signal_array[idx % n_samples])
                idx += 1

                # Format packet according to protocol
                spo2_val = 98 if include_spo2 else -1
                pulse_val = 72 if include_spo2 else -1
                spo2_valid_flag = 1 if include_spo2 else 0
                lead_off_flag = 1 if simulate_lead_off else 0

                mock_line = f"DATA,{t_ms},{ecg_val:.2f},{spo2_val},{pulse_val},{spo2_valid_flag},{lead_off_flag}"
                packet = self.parse_line(mock_line)

                if packet:
                    self.last_packet_time = now_t
                    self.total_packets_received += 1

                    with self._lock:
                        self.latest_packet = packet
                        self.latest_spo2 = packet.spo2
                        self.latest_pulse_bpm = packet.pulse_bpm
                        self.lead_off = packet.lead_off
                        self.spo2_valid = packet.spo2_valid

                        self._ecg_buffer.append(packet.ecg_raw)
                        self._time_buffer.append(t_ms / 1000.0)

                    if self.on_packet_callback:
                        self.on_packet_callback(packet)

                time.sleep(sample_period_s)

            self.is_simulating = False
            self._update_status("Simulation stopped.")

        self._sim_thread = threading.Thread(target=sim_worker, daemon=True, name="Simulated_Serial_Reader")
        self._sim_thread.start()

    def stop_simulation(self) -> None:
        self._stop_event.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None
        self.is_simulating = False
