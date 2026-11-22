/*
  =============================================================================
  AI-Based Heart Risk Monitoring System — ESP32 Sensor Acquisition Firmware
  File: esp32_sensor_firmware.ino
  Board: ESP32 Dev Module (esp32:esp32:esp32)

  Sensors:
    - AD8232 ECG Sensor:
        Output (Analog) : GPIO 34 (ADC1_CH6)
        LO+ (Lead-off+) : GPIO 26
        LO- (Lead-off-) : GPIO 27
    - MAX30102 Pulse Oximeter & Heart Rate:
        SDA             : GPIO 21
        SCL             : GPIO 22
        I2C Speed       : 400 kHz (Fast Mode)

  Serial Protocol (115200 baud):
    DATA,<timestamp_ms>,<ecg_raw>,<spo2>,<pulse_bpm>,<spo2_valid>,<lead_off>
    STATUS,<message>

  Disclaimers:
    This firmware is an academic demonstration tool for signal acquisition.
    It does not provide medical-grade diagnosis or clinical monitoring.
  =============================================================================
*/

#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

// =============================================================================
// PIN DEFINITIONS & CONSTANTS
// =============================================================================

// AD8232 ECG Pins
const int PIN_ECG_ANALOG   = 34; // ADC1 channel 6 (input only)
const int PIN_LEAD_OFF_P   = 26; // LO+
const int PIN_LEAD_OFF_M   = 27; // LO-

// MAX30102 I2C Pins
const int PIN_I2C_SDA      = 21;
const int PIN_I2C_SCL      = 22;

// Acquisition parameters
const float SAMPLING_FREQ_HZ = 360.0;
const unsigned long SAMPLE_PERIOD_US = (unsigned long)(1000000.0 / SAMPLING_FREQ_HZ); // 2777 us
const long SERIAL_BAUD_RATE  = 115200;

// =============================================================================
// SENSOR OBJECTS & STATE VARIABLES
// =============================================================================

MAX30105 particleSensor;
bool max30102Connected = false;

// Optical Pulse & SpO2 estimation state
const byte RATE_SIZE = 4; // Average across 4 beats for smooth display
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeatTime = 0;
int opticalBpm = -1;

float filterRatioAvg = 0.0;
int currentSpO2 = -1;
bool currentSpO2Valid = false;

// Timing
unsigned long nextSampleMicros = 0;

// Optical DC/AC tracking
float dcRed = 0.0;
float dcIR  = 0.0;
float acRed = 0.0;
float acIR  = 0.0;

// Finger detection threshold on IR channel
const uint32_t FINGER_THRESHOLD = 35000;

// =============================================================================
// SETUP
// =============================================================================

void setup()
{
  Serial.begin(SERIAL_BAUD_RATE);
  delay(1000); // Allow USB serial stabilization

  Serial.println("STATUS,BOOTING,AI-Based Heart Risk Monitoring System");
  Serial.println("STATUS,CONFIG,FS=360Hz,BAUD=115200,ECG_PIN=34,LO_PLUS=26,LO_MINUS=27");

  // 1. Configure AD8232 Pins
  pinMode(PIN_LEAD_OFF_P, INPUT);
  pinMode(PIN_LEAD_OFF_M, INPUT);
  analogReadResolution(12); // 0–4095 on ESP32

  // 2. Configure I2C & MAX30102
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000); // 400 kHz Fast I2C

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST))
  {
    max30102Connected = false;
    Serial.println("STATUS,MAX30102_ERROR,Sensor not detected at I2C address 0x57. Check wiring SDA=21 SCL=22.");
  }
  else
  {
    max30102Connected = true;

    // Configure MAX30102:
    // LED Brightness: 0x1F (approx 6.4mA)
    // Sample Average: 4
    // LED Mode: 2 (Red + IR)
    // Sample Rate: 100 Hz
    // Pulse Width: 411 us (18-bit resolution)
    // ADC Range: 4096
    byte ledBrightness = 30; // Options: 0=Off to 255=50mA
    byte sampleAverage = 4;  // Options: 1, 2, 4, 8, 16, 32
    byte ledMode = 2;        // Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
    int sampleRate = 100;    // Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
    int pulseWidth = 411;    // Options: 69, 118, 215, 411
    int adcRange = 4096;     // Options: 2048, 4096, 8192, 16384

    particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    particleSensor.setPulseAmplitudeRed(0x1F);
    particleSensor.setPulseAmplitudeIR(0x1F);

    Serial.println("STATUS,MAX30102_OK,Red and IR channels active at 100 Hz.");
  }

  Serial.println("STATUS,READY,Acquiring 360 Hz ECG data.");
  nextSampleMicros = micros();
}

// =============================================================================
// MAIN LOOP (360 Hz TIMED LOOP)
// =============================================================================

void loop()
{
  unsigned long currentMicros = micros();

  // Check if target sample period has elapsed
  if ((long)(currentMicros - nextSampleMicros) >= 0)
  {
    nextSampleMicros += SAMPLE_PERIOD_US;

    unsigned long timestamp = millis();

    // -------------------------------------------------------------------------
    // 1. SAMPLE AD8232 ECG
    // -------------------------------------------------------------------------
    int ecgRaw = analogRead(PIN_ECG_ANALOG);

    int loPlus = digitalRead(PIN_LEAD_OFF_P);
    int loMinus = digitalRead(PIN_LEAD_OFF_M);
    int leadOff = (loPlus == HIGH || loMinus == HIGH) ? 1 : 0;

    // -------------------------------------------------------------------------
    // 2. CHECK MAX30102 NON-BLOCKING
    // -------------------------------------------------------------------------
    if (max30102Connected)
    {
      // Check FIFO without blocking loop
      particleSensor.check();

      while (particleSensor.available())
      {
        uint32_t irVal = particleSensor.getFIFOIR();
        uint32_t redVal = particleSensor.getFIFORed();
        particleSensor.nextSample();

        // Finger detection
        if (irVal < FINGER_THRESHOLD)
        {
          currentSpO2Valid = false;
          currentSpO2 = -1;
          opticalBpm = -1;
        }
        else
        {
          // Check optical heart beat
          if (checkForBeat(irVal))
          {
            long delta = millis() - lastBeatTime;
            lastBeatTime = millis();

            float beatsPerMinute = 60.0 / (delta / 1000.0);

            if (beatsPerMinute > 30.0 && beatsPerMinute < 220.0)
            {
              rates[rateSpot++] = (byte)beatsPerMinute;
              rateSpot %= RATE_SIZE;

              int totalBpm = 0;
              for (byte x = 0; x < RATE_SIZE; x++)
              {
                totalBpm += rates[x];
              }
              opticalBpm = totalBpm / RATE_SIZE;
            }
          }

          // DC / AC Tracking for SpO2 calculation
          // High-pass & low-pass filtering on red and IR
          dcRed = (dcRed * 0.95) + ((float)redVal * 0.05);
          dcIR  = (dcIR  * 0.95) + ((float)irVal  * 0.05);

          float curAcRed = abs((float)redVal - dcRed);
          float curAcIR  = abs((float)irVal  - dcIR);

          acRed = (acRed * 0.90) + (curAcRed * 0.10);
          acIR  = (acIR  * 0.90) + (curAcIR  * 0.10);

          if (dcRed > 0 && dcIR > 0 && acIR > 0)
          {
            float ratio = (acRed / dcRed) / (acIR / dcIR);
            // Empirical calibrated SpO2 polynomial approximation
            float calcSpO2 = 110.0 - 25.0 * ratio;

            if (calcSpO2 >= 70.0 && calcSpO2 <= 100.0)
            {
              currentSpO2 = (int)(calcSpO2 + 0.5);
              currentSpO2Valid = true;
            }
            else
            {
              currentSpO2Valid = false;
            }
          }
        }
      }
    }
    else
    {
      currentSpO2Valid = false;
      currentSpO2 = -1;
      opticalBpm = -1;
    }

    // -------------------------------------------------------------------------
    // 3. EMIT STRUCTURED PROTOCOL PACKET OVER SERIAL
    //    Format: DATA,timestamp_ms,ecg_raw,spo2,pulse_bpm,spo2_valid,lead_off
    // -------------------------------------------------------------------------
    int spo2Out = currentSpO2Valid ? currentSpO2 : -1;
    int bpmOut  = (opticalBpm > 0 && currentSpO2Valid) ? opticalBpm : -1;
    int validFlag = currentSpO2Valid ? 1 : 0;

    Serial.print("DATA,");
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(ecgRaw);
    Serial.print(",");
    Serial.print(spo2Out);
    Serial.print(",");
    Serial.print(bpmOut);
    Serial.print(",");
    Serial.print(validFlag);
    Serial.print(",");
    Serial.println(leadOff);
  }
}
