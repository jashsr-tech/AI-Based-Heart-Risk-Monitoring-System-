
import time
from collections import Counter

import serial

PORT = "COM5"
BAUD = 115200
DURATION_SECONDS = 10

packets = []
status_messages = []
timestamps = []

print("Keep your finger steady over the MAX30102.")
print("Collecting serial data for 10 seconds...\n")

try:
    with serial.Serial(PORT, BAUD, timeout=0.2) as port:
        start = time.monotonic()

        while time.monotonic() - start < DURATION_SECONDS:
            line = port.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            if not line:
                continue

            if line.startswith("STATUS,"):
                status_messages.append(line)
                print(line)

            elif line.startswith("DATA,"):
                fields = line.split(",")

                if len(fields) != 7:
                    continue

                try:
                    timestamp = int(fields[1])
                    ecg = int(fields[2])
                    spo2 = float(fields[3])
                    pulse = float(fields[4])
                    valid = int(fields[5])
                    lead_off = int(fields[6])
                except ValueError:
                    continue

                packets.append({
                    "timestamp": timestamp,
                    "ecg": ecg,
                    "spo2": spo2,
                    "pulse": pulse,
                    "valid": valid,
                    "lead_off": lead_off,
                })

except serial.SerialException as exc:
    print(f"Serial connection error: {exc}")
    raise SystemExit(1)

print("\n========== SENSOR TEST SUMMARY ==========")
print(f"Packets received: {len(packets)}")

if not packets:
    print("No DATA packets received. Check the COM port and firmware.")
    raise SystemExit(0)

valid_spo2 = [
    p["spo2"] for p in packets
    if p["valid"] == 1 and 70 <= p["spo2"] <= 100
]

valid_pulse = [
    p["pulse"] for p in packets
    if 30 <= p["pulse"] <= 220
]

print(f"SpO2-valid packets: {len(valid_spo2)}")
print(f"Valid pulse-rate packets: {len(valid_pulse)}")
print(f"ECG values observed: {len(set(p['ecg'] for p in packets))}")
print(f"Lead-off values: {dict(Counter(p['lead_off'] for p in packets))}")

if valid_spo2:
    print(
        "SpO2 estimate range: "
        f"{min(valid_spo2):.1f}% to {max(valid_spo2):.1f}%"
    )
    print(
        "Median SpO2 estimate: "
        f"{sorted(valid_spo2)[len(valid_spo2) // 2]:.1f}%"
    )
else:
    print("No valid SpO2 estimates were reported.")

if valid_pulse:
    print(
        "Median optical pulse estimate: "
        f"{sorted(valid_pulse)[len(valid_pulse) // 2]:.1f} BPM"
    )
else:
    print("No valid optical pulse-rate estimate was reported.")

if len(packets) > 1:
    elapsed_ms = packets[-1]["timestamp"] - packets[0]["timestamp"]
    if elapsed_ms > 0:
        estimated_rate = (len(packets) - 1) * 1000 / elapsed_ms
        print(f"Timestamp-based sample rate: {estimated_rate:.1f} Hz")

print("\nThis is an engineering check, not medical validation.")
