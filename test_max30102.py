import time
import serial

PORT = "COM5"
BAUD = 115200
DURATION_SECONDS = 10

packets = []

print("Keep your finger gently on the MAX30102.")
print("Collecting data for 10 seconds...\n")

try:
    with serial.Serial(PORT, BAUD, timeout=0.2) as port:
        start = time.monotonic()

        while time.monotonic() - start < DURATION_SECONDS:
            line = port.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            if line.startswith("STATUS,"):
                print(line)

            elif line.startswith("DATA,"):
                fields = line.split(",")

                if len(fields) != 7:
                    continue

                try:
                    packets.append({
                        "timestamp": int(fields[1]),
                        "ecg": int(fields[2]),
                        "spo2": float(fields[3]),
                        "pulse": float(fields[4]),
                        "valid": int(fields[5]),
                        "lead_off": int(fields[6]),
                    })
                except ValueError:
                    continue

except serial.SerialException as exc:
    print(f"\nSerial connection error: {exc}")
    raise SystemExit(1)

print("\n========== SENSOR TEST SUMMARY ==========")
print(f"Packets received: {len(packets)}")

if not packets:
    print("No DATA packets received.")
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

if valid_spo2:
    print(
        f"SpO2 estimate range: {min(valid_spo2):.1f}% "
        f"to {max(valid_spo2):.1f}%"
    )
    print(
        "Median SpO2 estimate: "
        f"{sorted(valid_spo2)[len(valid_spo2) // 2]:.1f}%"
    )
else:
    print("No valid SpO2 estimates reported.")

if valid_pulse:
    print(
        "Median optical pulse estimate: "
        f"{sorted(valid_pulse)[len(valid_pulse) // 2]:.1f} BPM"
    )
else:
    print("No valid optical pulse-rate estimate reported.")

timestamps = [p["timestamp"] for p in packets]
elapsed_ms = timestamps[-1] - timestamps[0]

if len(timestamps) > 1 and elapsed_ms > 0:
    rate = (len(timestamps) - 1) * 1000 / elapsed_ms
    print(f"Timestamp-based sample rate: {rate:.1f} Hz")

print("\nEngineering test only; readings are not medically validated.")