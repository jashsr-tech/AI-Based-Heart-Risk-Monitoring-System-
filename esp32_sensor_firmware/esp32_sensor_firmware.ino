/*
  AI-Based Heart Risk Monitoring System — MAX30102 SpO2/Pulse Firmware

  Purpose for this project version:
    - ECG is generated synthetically in Python.
    - Physical ESP32 is used ONLY for the MAX30102 optical sensor.
    - AD8232 is intentionally not read in this firmware.

  Board target: esp32:esp32:esp32 (ESP32 Dev Module)
  MAX30102 I2C: SDA=GPIO21, SCL=GPIO22
  Serial: 115200 baud

  Compatibility serial protocol (7 fields):
    DATA,<timestamp_ms>,<ecg_placeholder>,<spo2>,<pulse_bpm>,<spo2_valid>,<lead_off_placeholder>

  IMPORTANT:
    ecg_placeholder is always 0; it is NOT ECG data.
    lead_off_placeholder is always 0; AD8232 lead-off is NOT monitored.
    The Python dashboard must use its synthetic ECG signal, not the zero field.
    SpO2 and pulse numbers are sensor-derived estimates for an academic prototype,
    not medical-grade measurements or a diagnosis.

  Algorithm:
    Uses the installed SparkFun MAX3010x library's Maxim/Maxim Integrated
    reference implementation from spo2_algorithm.h. That implementation uses
    100 samples at 25 Hz. Sensor configuration uses 100 Hz with averaging 4,
    matching the common SparkFun example configuration for this algorithm.
*/

#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

// ----------------------- Configuration -----------------------
static const uint8_t PIN_I2C_SDA = 21;
static const uint8_t PIN_I2C_SCL = 22;
static const uint32_t SERIAL_BAUD = 115200;

// Keep the existing 360 Hz packet cadence for serial bridge compatibility.
// ECG data itself is NOT acquired from hardware in this project mode.
static const uint32_t STREAM_PERIOD_US = 2777;
static const uint32_t FINGER_IR_THRESHOLD = 35000;
static const uint32_t PPG_STALE_TIMEOUT_MS = 1500;
static const int32_t PPG_BUFFER_LENGTH = BUFFER_SIZE; // from spo2_algorithm.h: 100
static const int32_t PPG_UPDATE_INTERVAL = 25;        // update every 25 new PPG samples

MAX30105 particleSensor;
bool max30102Connected = false;

uint32_t irBuffer[BUFFER_SIZE];
uint32_t redBuffer[BUFFER_SIZE];
int32_t ppgBufferCount = 0;
int32_t samplesSinceAlgorithm = 0;
uint32_t lastPpgSampleMs = 0;

int currentSpO2 = -1;
int currentPulseBpm = -1;
bool currentSpO2Valid = false;
bool currentPulseValid = false;

uint32_t nextStreamMicros = 0;

// ----------------------- Helpers -----------------------
void invalidateOpticalReadings()
{
  currentSpO2 = -1;
  currentPulseBpm = -1;
  currentSpO2Valid = false;
  currentPulseValid = false;
}

void clearPpgBuffer()
{
  ppgBufferCount = 0;
  samplesSinceAlgorithm = 0;
  invalidateOpticalReadings();
}

void calculateOpticalMetrics()
{
  if (ppgBufferCount < PPG_BUFFER_LENGTH)
    return;

  int32_t calculatedSpO2 = -1;
  int32_t calculatedPulse = -1;
  int8_t spo2ValidFlag = 0;
  int8_t pulseValidFlag = 0;

  // The library expects the 100-sample IR buffer first, then RED buffer.
  maxim_heart_rate_and_oxygen_saturation(
      irBuffer,
      PPG_BUFFER_LENGTH,
      redBuffer,
      &calculatedSpO2,
      &spo2ValidFlag,
      &calculatedPulse,
      &pulseValidFlag);

  // Treat each output independently. A valid pulse estimate does not imply
  // that SpO2 is valid, and vice versa.
  if (spo2ValidFlag == 1 && calculatedSpO2 >= 70 && calculatedSpO2 <= 100)
  {
    currentSpO2 = (int)calculatedSpO2;
    currentSpO2Valid = true;
  }
  else
  {
    currentSpO2 = -1;
    currentSpO2Valid = false;
  }

  if (pulseValidFlag == 1 && calculatedPulse >= 30 && calculatedPulse <= 220)
  {
    currentPulseBpm = (int)calculatedPulse;
    currentPulseValid = true;
  }
  else
  {
    currentPulseBpm = -1;
    currentPulseValid = false;
  }
}

void appendPpgSample(uint32_t irValue, uint32_t redValue)
{
  // The original reference algorithm assumes the input sequence represents
  // an actual finger PPG waveform. Reject samples below our simple finger
  // presence threshold and restart warm-up instead of reporting stale values.
  if (irValue < FINGER_IR_THRESHOLD)
  {
    clearPpgBuffer();
    return;
  }

  lastPpgSampleMs = millis();

  if (ppgBufferCount < PPG_BUFFER_LENGTH)
  {
    irBuffer[ppgBufferCount] = irValue;
    redBuffer[ppgBufferCount] = redValue;
    ppgBufferCount++;
  }
  else
  {
    // Sliding 4-second window. Shift both channels together to preserve pairing.
    for (int32_t i = 1; i < PPG_BUFFER_LENGTH; ++i)
    {
      irBuffer[i - 1] = irBuffer[i];
      redBuffer[i - 1] = redBuffer[i];
    }
    irBuffer[PPG_BUFFER_LENGTH - 1] = irValue;
    redBuffer[PPG_BUFFER_LENGTH - 1] = redValue;
  }

  if (ppgBufferCount == PPG_BUFFER_LENGTH)
  {
    samplesSinceAlgorithm++;
    if (samplesSinceAlgorithm >= PPG_UPDATE_INTERVAL)
    {
      samplesSinceAlgorithm = 0;
      calculateOpticalMetrics();
    }
  }
}

// ----------------------- Setup -----------------------
void setup()
{
  Serial.begin(SERIAL_BAUD);
  delay(500);

  Serial.println("STATUS,BOOTING,MAX30102 Sensor Monitor");
  Serial.println("STATUS,ECG_SOURCE,SYNTHETIC_IN_PYTHON");
  Serial.println("STATUS,AD8232,NOT_USED");
  Serial.println("STATUS,CONFIG,PPG_CONFIG=100Hz_AVG4,ALGORITHM=25Hz_100_SAMPLES,BAUD=115200,SDA=21,SCL=22");

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST))
  {
    max30102Connected = false;
    Serial.println("STATUS,MAX30102_ERROR,Sensor not detected. Check power, GND, SDA=21, SCL=22.");
  }
  else
  {
    max30102Connected = true;

    // Common SparkFun MAX3010x example configuration:
    // 4-sample averaging with a 100 Hz sensor setting is paired with the
    // installed Maxim reference algorithm, whose buffer rate is 25 Hz.
    const byte ledBrightness = 30;
    const byte sampleAverage = 4;
    const byte ledMode = 2;        // Red + IR
    const int sampleRate = 100;
    const int pulseWidth = 411;
    const int adcRange = 4096;

    particleSensor.setup(
        ledBrightness, sampleAverage, ledMode,
        sampleRate, pulseWidth, adcRange);
    particleSensor.setPulseAmplitudeRed(0x1F);
    particleSensor.setPulseAmplitudeIR(0x1F);
    particleSensor.setPulseAmplitudeGreen(0);

    Serial.println("STATUS,MAX30102_OK,Red and IR channels active; warming up 4-second PPG buffer.");
  }

  Serial.println("STATUS,READY,ECG is synthetic in Python; firmware provides MAX30102 metrics only.");
  nextStreamMicros = micros();
}

// ----------------------- Main Loop -----------------------
void loop()
{
  // Drain the MAX30102 FIFO as promptly as possible; calculations run only
  // after a full buffer and then once per 25 new samples.
  if (max30102Connected)
  {
    particleSensor.check();

    while (particleSensor.available())
    {
      const uint32_t irValue = particleSensor.getFIFOIR();
      const uint32_t redValue = particleSensor.getFIFORed();
      particleSensor.nextSample();
      appendPpgSample(irValue, redValue);
    }

    if (lastPpgSampleMs != 0 &&
        (uint32_t)(millis() - lastPpgSampleMs) > PPG_STALE_TIMEOUT_MS)
    {
      invalidateOpticalReadings();
    }
  }
  else
  {
    invalidateOpticalReadings();
  }

  // Emit packets at the previous bridge-compatible cadence. Field 3 is a
  // deliberate zero placeholder, NOT a physical ECG sample. Keep the Python
  // dashboard on a synthetic ECG mode until bridge/dashboard are configured
  // to merge this sensor telemetry with the Python-generated waveform.
  const uint32_t nowMicros = micros();
  if ((int32_t)(nowMicros - nextStreamMicros) >= 0)
  {
    nextStreamMicros += STREAM_PERIOD_US;

    const uint32_t timestampMs = millis();
    const int ecgPlaceholder = 0;
    const int spo2Out = currentSpO2Valid ? currentSpO2 : -1;
    const int pulseOut = currentPulseValid ? currentPulseBpm : -1;
    const int spo2ValidOut = currentSpO2Valid ? 1 : 0;
    const int leadOffPlaceholder = 0; // AD8232 is not used; this is not a lead-status reading.

    Serial.print("DATA,");
    Serial.print(timestampMs);
    Serial.print(',');
    Serial.print(ecgPlaceholder);
    Serial.print(',');
    Serial.print(spo2Out);
    Serial.print(',');
    Serial.print(pulseOut);
    Serial.print(',');
    Serial.print(spo2ValidOut);
    Serial.print(',');
    Serial.println(leadOffPlaceholder);
  }
}
