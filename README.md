# ❤️ AI-Based Heart Risk Monitoring System

An AI-assisted portable heart monitoring prototype integrating ECG signal analysis, SpO₂ monitoring, and machine learning to identify abnormal cardiac rhythm patterns.

## 📌 Project Overview

The **AI-Based Heart Risk Monitoring System** is an academic engineering prototype designed to monitor selected cardiac parameters and analyze ECG signals using signal-processing techniques and machine learning.

The system combines hardware-based physiological data acquisition with software-based ECG analysis and an interactive dashboard.

It uses an ESP32 microcontroller with an AD8232 ECG sensor and MAX30102 pulse oximeter to acquire physiological measurements. ECG signals are processed to extract cardiac features, which are analyzed using a trained Random Forest classification model.

The system also supports MIT-BIH ECG dataset replay and synthetic ECG simulations for demonstration and validation.

## 🎯 Objectives

- Develop a portable heart monitoring prototype.
- Acquire ECG signals using the AD8232 sensor.
- Monitor SpO₂ and optical pulse rate using MAX30102.
- Process ECG signals and detect R-peaks.
- Extract important ECG features, including P, Q, R, S and T wave characteristics.
- Analyze heart rhythm patterns using machine learning.
- Display physiological measurements and classification results through an interactive dashboard.
- Demonstrate normal and abnormal ECG patterns.

## 🏗️ System Architecture

```text
       PATIENT / TEST INPUT
               |
       +-------+-------+
       |               |
    AD8232           MAX30102
   ECG Sensor       Pulse Oximeter
       |               |
       +-------+-------+
               |
             ESP32
               |
        USB Serial Data
               |
         Serial Bridge
               |
       +-------+--------+
       |                |
   ECG Processing   SpO2 / Pulse
       |                |
  Bandpass Filter       |
  R-Peak Detection      |
  PQRST Detection      |
  Feature Extraction   |
       |                |
       +-------+--------+
               |
       Machine Learning
        Random Forest
               |
       Classification
        Normal / Abnormal
               |
       Streamlit Dashboard
               |
      Visual Monitoring
```

## ⚙️ Hardware Components

| Component | Purpose |
|---|---|
| ESP32 | Main microcontroller |
| AD8232 ECG Module | ECG signal acquisition |
| MAX30102 | SpO₂ and optical pulse-rate monitoring |
| ECG Electrodes | ECG signal collection |
| USB Connection | Serial communication with computer |

## 💻 Software & Technologies

| Technology | Application |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| Pandas | Dataset processing |
| SciPy | ECG signal filtering and analysis |
| Scikit-learn | Machine learning |
| Streamlit | Interactive monitoring dashboard |
| WFDB | MIT-BIH ECG dataset access |
| PySerial | ESP32 serial communication |
| Joblib / Pickle | Model and preprocessing asset loading |
| Arduino IDE | ESP32 firmware development |

## 🧠 ECG Signal Processing

The ECG processing pipeline performs:

1. ECG signal acquisition.
2. Fourth-order Butterworth bandpass filtering (0.5–40 Hz).
3. R-peak detection.
4. P-Q-R-S-T morphological peak detection.
5. RR interval calculation.
6. Heart-rate and HRV estimation.
7. ECG feature extraction.
8. Machine-learning inference.

### Extracted ECG Features

- Heart Rate
- RR Interval
- Heart Rate Variability (HRV)
- PR Interval
- QRS Duration
- QT Interval
- ST Segment
- P Wave Amplitude
- R Wave Amplitude
- T Wave Amplitude

## 🤖 Machine Learning

The project uses a trained Random Forest classification model to distinguish between normal and abnormal ECG patterns.

**Dataset:** MIT-BIH Arrhythmia Database.

**Classification output:**
- Normal
- Abnormal

The dashboard uses a pretrained model along with its corresponding imputer and scaler.

Synthetic PVC-like and tachycardia-like ECG signals are also supported for demonstration and software validation.

## 📊 Interactive Dashboard

The Streamlit dashboard provides:

- Normal ECG replay using MIT-BIH Record 100.
- Synthetic abnormal ECG simulation.
- PVC and tachycardia demonstration presets.
- ECG waveform visualization.
- Beat navigation and playback controls.
- Random Forest classification results.
- ECG feature telemetry.
- ESP32 sensor telemetry.
- SpO₂ and optical pulse-rate display.
- System connection and operational status.

## 📁 Project Structure

```text
Heart-Risk-Monitoring-System/
│
├── 01_signal_generation/
│
├── 02_signal_processing/
│
├── 03_feature_extraction/
│
├── 04_machine_learning/
│
├── 05_results/
│
├── esp32_sensor_firmware/
│
├── models/
│   ├── stage7_rf_combined_with_ind.pkl
│   ├── ecg_feature_imputer.pkl
│   └── ecg_feature_scaler.pkl
│
├── dashboard.py
├── ecg_pipeline.py
├── serial_bridge.py
├── streamlit_app.py
│
├── test_app_modes.py
├── test_dashboard_backend.py
├── test_max30102.py
├── verify_all_9_points.py
│
└── README.md
```

*The folder structure above highlights the main components; individual folders may contain additional datasets, scripts and results.*

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/jashsr-tech/Heart-Risk-Monitoring-System-.git

cd Heart-Risk-Monitoring-System-
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn streamlit wfdb pyserial joblib
```

### 4. Configure hardware

- Connect the AD8232 ECG module to the ESP32.
- Connect the MAX30102 sensor through I2C.
- Upload the firmware from `esp32_sensor_firmware/` using Arduino IDE.
- Connect the ESP32 to the computer using USB.

### 5. Configure serial communication

Check the ESP32 COM port in Windows Device Manager.

Update the serial port configuration in `streamlit_app.py` if required.

Default baud rate: `115200`.

### 6. Configure dataset location

The current dashboard references a local MIT-BIH database path. Update `MITBIH_RAW_DIR` in `streamlit_app.py` to the location of your downloaded MIT-BIH Arrhythmia Database.

Ensure the required model files and demonstration data are available in their expected directories.

### 7. Run the dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard should open in your default web browser.

## 🧪 Demonstration Modes

**1. Normal ECG Mode**

Uses MIT-BIH Record 100 to demonstrate ECG waveform replay and feature analysis.

**2. Abnormal ECG Mode**

Uses synthetic ECG signals to demonstrate PVC-like and tachycardia-like patterns and machine-learning classification.

**3. Hardware Sensor Mode**

Receives ECG and MAX30102 sensor telemetry from the ESP32 through USB serial communication.

*Synthetic ECG is used as a simulation and validation input, not represented as a live patient ECG recording.*

## 🔮 Future Enhancements

- Improve real-time ECG acquisition and signal quality.
- Enhance machine-learning classification performance.
- Integrate additional ECG datasets.
- Improve motion-artifact handling.
- Develop a standalone portable monitoring interface.
- Add secure patient data storage and historical analysis.
- Validate the system against clinically reviewed ECG recordings.

## ⚠️ Disclaimer

This project is an academic engineering prototype intended for educational, research and demonstration purposes only.

It is not a certified medical device and must not be used for clinical diagnosis, treatment decisions or emergency medical assessment. Machine-learning predictions are experimental and are not clinically validated.

## 👨‍💻 Project Contributors

- D R Vishal
- Harshith Kumar K M
- Jashwanth S R

**Project Guide:** Dr. Harshada J Patil

## 📜 License

No license has currently been specified. Please add an appropriate open-source license if you intend to permit reuse or distribution.

---

**Developed as a Final-Year Engineering Project**  
Electronics and Communication Engineering (ECE)
