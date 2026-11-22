# ❤️ Portable Heart Risk Monitoring System

**Real-Time ECG and SpO₂ Monitoring Using ESP32, Digital Signal Processing and Machine Learning**

## 📌 Project Overview

The **Portable Heart Risk Monitoring System** is an engineering project developed to monitor important cardiac parameters and analyze ECG signals using hardware integration, digital signal processing and machine learning techniques.

The system integrates an **ESP32 microcontroller, AD8232 ECG sensor and MAX30102 pulse oximeter** to acquire real-time physiological measurements.

The AD8232 sensor captures the electrical activity of the heart, while the MAX30102 sensor measures blood oxygen saturation (SpO₂) and pulse rate.

The acquired ECG signal is transmitted through the ESP32 to a Python-based application, where it undergoes digital signal processing, noise filtering, R-peak detection and PQRST feature extraction.

The extracted ECG features are analyzed using a trained Random Forest classification model to identify normal and abnormal ECG rhythm patterns.

An interactive Streamlit dashboard provides real-time visualization of sensor readings, ECG waveforms, extracted cardiac features and classification results.

The project also incorporates the MIT-BIH Arrhythmia Database and synthetic ECG signals for additional analysis, model development, testing and validation.

## 🎯 Project Objectives

- Develop a portable cardiac monitoring prototype.
- Acquire ECG signals in real time using the AD8232 sensor.
- Measure real-time SpO₂ and pulse rate using the MAX30102 sensor.
- Interface multiple physiological sensors with an ESP32 microcontroller.
- Establish serial communication between ESP32 and a Python application.
- Process ECG signals using digital filtering techniques.
- Detect R-peaks and identify PQRST waveform characteristics.
- Extract important ECG parameters.
- Analyze ECG rhythm patterns using machine learning.
- Display real-time physiological measurements through an interactive dashboard.
- Support dataset-based ECG analysis and synthetic signal testing.

## 🏗️ System Architecture

The system consists of four major stages: sensor acquisition, microcontroller communication, signal processing and data visualization.

```text
                  USER / PATIENT
                        |
             +----------+----------+
             |                     |
          AD8232                 MAX30102
        ECG Sensor             Pulse Oximeter
             |                     |
         ECG Signal            SpO2 / Pulse
             |                     |
             +----------+----------+
                        |
                       ESP32
                  Microcontroller
                        |
                 USB Serial Data
                        |
                 Python Application
                        |
             +----------+----------+
             |                     |
        ECG Processing        SpO2 Processing
             |                     |
       Noise Filtering             |
       R-Peak Detection            |
       PQRST Detection             |
       Feature Extraction          |
             |                     |
             +----------+----------+
                        |
                 Random Forest
                   Classifier
                        |
                Normal / Abnormal
                   Classification
                        |
                Streamlit Dashboard
                        |
              Monitoring and Display
```

## ⚙️ Hardware Components

| Component | Function |
|---|---|
| ESP32 Development Board | Main microcontroller |
| AD8232 ECG Sensor | Real-time ECG signal acquisition |
| MAX30102 Sensor | SpO₂ and pulse-rate monitoring |
| ECG Electrodes | Collection of electrical cardiac signals |
| USB Cable | Power and serial communication |

### Hardware Pin Connections

**1. AD8232 ECG Sensor – ESP32**

| AD8232 Pin | ESP32 Pin |
|---|---|
| OUTPUT | GPIO 34 |
| LO+ | GPIO 26 |
| LO- | GPIO 27 |
| 3.3V | 3.3V |
| GND | GND |

**2. MAX30102 – ESP32**

| MAX30102 Pin | ESP32 Pin |
|---|---|
| SDA | GPIO 21 |
| SCL | GPIO 22 |
| VIN | 3.3V |
| GND | GND |

*Note: Connections are based on the project's configured hardware setup. Verify the power requirements of the particular sensor breakout boards before connecting them.*

## 💻 Software and Technologies

| Technology | Application |
|---|---|
| Embedded C / Arduino | ESP32 firmware development |
| Python | Application development and processing |
| NumPy | Numerical computations |
| Pandas | Dataset handling and processing |
| SciPy | ECG signal filtering and analysis |
| Scikit-learn | Machine learning classification |
| Streamlit | Interactive monitoring dashboard |
| WFDB | ECG dataset access and processing |
| PySerial | Serial communication with ESP32 |
| Joblib / Pickle | Model and preprocessing asset loading |
| Arduino IDE | Microcontroller programming |

## 📡 Real-Time Sensor Monitoring

Real-time hardware monitoring is the primary operating functionality of the system.

### 1. ECG Signal Acquisition

The AD8232 ECG sensor captures the electrical activity associated with the heart using ECG electrodes.

The acquired analog signal is supplied to the ESP32 through its analog input pin.

The ESP32 samples the ECG signal and transmits the readings to the Python application through USB serial communication.

The received ECG data is processed to identify waveform characteristics and derive cardiac parameters.

### 2. SpO₂ and Pulse-Rate Monitoring

The MAX30102 is an optical sensor that uses red and infrared light to estimate blood oxygen saturation.

The sensor also detects pulse-related information from optical measurements.

The ESP32 communicates with the MAX30102 through the I2C protocol and retrieves the sensor measurements.

The acquired SpO₂ and pulse-rate values are transmitted to the monitoring application for real-time display.

### 3. Real-Time Dashboard

The monitoring dashboard provides:

- Live ECG waveform visualization.
- Real-time SpO₂ readings.
- Pulse-rate measurements.
- ECG-derived heart-rate estimates.
- Extracted ECG features.
- Classification results.
- Sensor and serial communication status.

## 🧠 ECG Signal Processing

ECG signal processing is performed to convert the acquired waveform into meaningful cardiac features.

### Signal Processing Workflow

1. **Signal Acquisition:** Receives ECG samples from the AD8232 sensor.
2. **Noise Filtering:** Applies digital filtering to reduce unwanted signal noise.
3. **R-Peak Detection:** Identifies R-peaks in the ECG waveform.
4. **PQRST Detection:** Identifies important waveform components.
5. **RR Interval Calculation:** Determines the time between consecutive R-peaks.
6. **Heart-Rate Estimation:** Calculates heart rate using RR intervals.
7. **HRV Analysis:** Extracts heart-rate variability parameters.
8. **Feature Extraction:** Generates relevant features for classification.
9. **Model Inference:** Supplies the extracted features to the trained classification model.

### Digital Filtering

The ECG processing pipeline uses a fourth-order Butterworth bandpass filter with a frequency range of 0.5–40 Hz in the configured processing stage.

Filtering helps reduce unwanted baseline variations and high-frequency noise.

### Extracted ECG Features

The system's ECG feature extraction includes:

| Feature | Description |
|---|---|
| Heart Rate | Estimated beats per minute |
| RR Interval | Time between consecutive R-peaks |
| HRV | Heart-rate variability |
| SDNN | Standard deviation of normal RR intervals |
| RMSSD | Root mean square of successive RR differences |
| PR Interval | Time between P-wave onset and QRS onset |
| QRS Duration | Duration of ventricular depolarization |
| QT Interval | Duration from QRS onset to T-wave end |
| QTc Interval | Corrected QT interval |
| ST Segment | ST-segment characteristics |
| P Wave | P-wave characteristics |
| R Wave | R-wave characteristics |
| T Wave | T-wave characteristics |

*The reliability and availability of individual ECG features depend on signal quality and the processing mode.*

## 🤖 Machine Learning and ECG Classification

The project incorporates a trained **Random Forest Classifier** to analyze extracted ECG features and distinguish between normal and abnormal ECG patterns.

### Dataset

**MIT-BIH Arrhythmia Database**

The MIT-BIH Arrhythmia Database is used for ECG data analysis, model development and classification experiments.

### Model Development Workflow

1. Load ECG recordings and annotations.
2. Preprocess the ECG signals.
3. Extract relevant cardiac features.
4. Prepare and label the feature dataset.
5. Train the Random Forest classification model.
6. Evaluate model performance.
7. Save the trained model and preprocessing assets.
8. Load the trained model into the monitoring application.
9. Perform classification using extracted ECG features.

### Classification Output

The model provides the following classification categories:

- Normal ECG pattern.
- Abnormal ECG pattern.

The classification output is displayed on the Streamlit dashboard.

The project includes saved model and preprocessing assets used by the application.

## 📊 Interactive Monitoring Dashboard

The Streamlit application acts as the main interface for monitoring and analyzing the acquired data.

### Dashboard Features

- Real-time ECG waveform visualization.
- Live SpO₂ monitoring.
- Pulse-rate display.
- ECG-derived heart-rate estimation.
- ECG feature extraction.
- Random Forest classification output.
- MIT-BIH ECG waveform analysis.
- Synthetic ECG simulation.
- PVC-like and tachycardia-like demonstration patterns.
- Sensor communication status.
- Monitoring and testing modes.

## 🧪 Operating Modes

The system supports three different operating modes.

### 1. Real-Time Hardware Monitoring

This mode uses the physical sensors connected to the ESP32.

- ECG acquisition through the AD8232 sensor.
- SpO₂ measurement through the MAX30102 sensor.
- Pulse-rate monitoring.
- ESP32-based sensor interfacing.
- USB serial communication.
- ECG signal processing and feature extraction.
- Machine learning classification.
- Dashboard visualization of sensor measurements and analysis.

### 2. MIT-BIH Dataset Analysis

The MIT-BIH Arrhythmia Database is used for:

- ECG waveform visualization.
- Normal and abnormal rhythm analysis.
- ECG feature extraction.
- Model development.
- Classification testing and evaluation.

The dashboard supports ECG recording replay, including MIT-BIH Record 100.

### 3. Synthetic ECG Simulation

Synthetic ECG signals are used for software testing and demonstration.

Supported demonstrations include:

- Normal ECG waveform generation.
- PVC-like ECG pattern simulation.
- Tachycardia-like ECG pattern simulation.
- ECG feature extraction testing.
- Signal-processing pipeline validation.
- Classification testing.

**Important:** Real-time ECG and SpO₂ measurements are acquired using physical sensors. MIT-BIH recordings and synthetic ECG signals are separate inputs used for dataset analysis, model development, testing and simulation.

## 📁 Project Structure

```text
AI-Based-Heart-Risk-Monitoring-System-/
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

The folders contain the major components of the project:

| Directory / File | Description |
|---|---|
| `01_signal_generation/` | ECG signal generation and input preparation |
| `02_signal_processing/` | ECG signal processing |
| `03_feature_extraction/` | Cardiac feature extraction |
| `04_machine_learning/` | Model development and classification |
| `05_results/` | Experimental results and outputs |
| `esp32_sensor_firmware/` | ESP32 firmware |
| `models/` | Trained model and preprocessing assets |
| `dashboard.py` | Dashboard-related functionality |
| `ecg_pipeline.py` | ECG processing pipeline |
| `serial_bridge.py` | Serial communication handling |
| `streamlit_app.py` | Main Streamlit application |
| `test_app_modes.py` | Application mode testing |
| `test_dashboard_backend.py` | Dashboard backend testing |
| `test_max30102.py` | MAX30102 sensor testing |
| `verify_all_9_points.py` | Project verification script |

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Arduino IDE.
- ESP32 development board.
- AD8232 ECG sensor.
- MAX30102 sensor.
- USB data cable.
- Windows, Linux or macOS computer.

### Step 1: Clone the Repository

Open your terminal or command prompt and execute:

```bash
git clone https://github.com/jashsr-tech/AI-Based-Heart-Risk-Monitoring-System-.git
```

Navigate into the project directory:

```bash
cd AI-Based-Heart-Risk-Monitoring-System-
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python libraries:

```bash
pip install numpy pandas scipy scikit-learn streamlit wfdb pyserial joblib
```

If a `requirements.txt` file is added to the repository, dependencies can also be installed using:

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Hardware

1. Connect the AD8232 ECG sensor to the ESP32.
2. Connect the MAX30102 sensor using I2C.
3. Open the ESP32 firmware from the `esp32_sensor_firmware/` directory.
4. Select the appropriate ESP32 board in Arduino IDE.
5. Select the correct serial port.
6. Upload the firmware.
7. Connect the ESP32 to the computer using a USB data cable.

### Step 5: Configure Serial Communication

Identify the ESP32 serial port through Device Manager on Windows or the appropriate serial-device listing on other operating systems.

Configure the correct port in the Python application.

The configured communication baud rate is:

```text
115200
```

Make sure another application, such as Arduino Serial Monitor, is not occupying the same serial port while the dashboard is running.

### Step 6: Configure the Dataset

Download the MIT-BIH Arrhythmia Database and configure its local directory in the application.

Update the dataset path if required.

Ensure that the trained model and corresponding preprocessing files are available in their expected locations.

### Step 7: Run the Application

Execute:

```bash
streamlit run streamlit_app.py
```

The Streamlit dashboard should open in your default web browser.

## 🔬 Testing and Validation

Different components of the system can be tested independently.

The project includes testing activities for:

- AD8232 ECG sensor acquisition.
- MAX30102 sensor communication.
- ESP32 serial communication.
- ECG signal filtering.
- R-peak detection.
- PQRST feature extraction.
- ECG parameter calculation.
- Random Forest classification.
- MIT-BIH dataset analysis.
- Synthetic ECG simulation.
- Dashboard functionality.

## 🔮 Future Enhancements

- Improve real-time ECG signal quality.
- Enhance ECG classification performance.
- Improve motion-artifact rejection.
- Expand ECG dataset coverage.
- Improve the reliability of extracted cardiac features.
- Integrate additional physiological parameters.
- Develop a standalone portable monitoring interface.
- Add secure patient data storage.
- Implement historical monitoring and trend analysis.
- Perform extensive validation using clinically reviewed ECG recordings.

## ⚠️ Disclaimer

This project is an engineering prototype intended for educational, research and demonstration purposes.

It is not a certified medical device and must not be used for clinical diagnosis, treatment decisions or emergency medical assessment.

The classification results are experimental and have not been clinically validated. Sensor readings and extracted parameters may be affected by electrode placement, movement, signal quality and hardware limitations.
