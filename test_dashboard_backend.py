"""
Test script to verify all backend components of the Streamlit dashboard:
1. MIT-BIH raw record loading via wfdb.
2. Feature dataset loading and beat indexing.
3. Imputer + Scaler transformation with exact feature ordering.
4. Model inference via Stage 7 Random Forest.
5. SerialBridge initialization and mock/simulation stream.
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
import wfdb

print("=======================================================")
print("RUNNING STREAMLIT BACKEND PRE-FLIGHT TEST")
print("=======================================================\n")

# 1. Test MIT-BIH raw reading
print("1. Testing WFDB record loading...")
mitbih_dir = r"C:\Users\srjas\OneDrive\Desktop\ECG_Project - Copy\mit-bih-arrhythmia-database-1.0.0"
rec = wfdb.rdrecord(f"{mitbih_dir}\\100", sampto=3600)
print(f"   -> Record 100 loaded: {rec.p_signal.shape[0]} samples, {rec.fs} Hz, Channels: {rec.sig_name}")

# 2. Test Development Dataset
print("\n2. Testing Development Dataset loading...")
dev_csv = "04_machine_learning/mitbih_development_dataset.csv"
df = pd.read_csv(dev_csv)
print(f"   -> Loaded {len(df):,} total beats from {dev_csv}")
df_100 = df[df["Record"] == 100].reset_index(drop=True)
print(f"   -> Record 100 has {len(df_100):,} beats")

# 3. Test Preprocessing Artifacts
print("\n3. Testing Preprocessing Artifacts...")
with open("models/ecg_feature_imputer.pkl", "rb") as f:
    imputer = pickle.load(f)
with open("models/ecg_feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

num_cols = [
    "RRInterval", "HRV", "QRSDuration", "QTInterval",
    "STSegment", "PRInterval", "PAmplitude", "RAmplitude", "TAmplitude"
]
ind_cols = [
    "P_Valid", "PR_Valid", "QRS_Valid", "ST_Valid", "QT_Valid", "Baseline_Valid"
]

row = df_100.iloc[10]
x_num = np.array([[row[c] for c in num_cols]])
x_imp = imputer.transform(x_num)
z_num = scaler.transform(x_imp)
x_ind = np.array([[int(row[c]) for c in ind_cols]])
x_vec = np.hstack([z_num, x_ind])

print(f"   -> Preprocessed vector shape: {x_vec.shape} (Expected 1, 15)")
assert x_vec.shape == (1, 15), "Feature vector shape mismatch!"

# 4. Test Model Inference
print("\n4. Testing Stage 7 Random Forest Model Inference...")
with open("models/stage7_rf_combined_with_ind.pkl", "rb") as f:
    model = pickle.load(f)

probs = model.predict_proba(x_vec)[0]
p_norm = probs[0]
p_abnorm = probs[1]
pred_class = int(p_abnorm >= 0.50)
pred_label = "Abnormal / Ectopic" if pred_class == 1 else "Normal Sinus"

print(f"   -> Model: stage7_rf_combined_with_ind.pkl ({model.n_estimators} trees)")
print(f"   -> P(Normal): {p_norm*100:.2f}%, P(Abnormal): {p_abnorm*100:.2f}%")
print(f"   -> Prediction: {pred_label} (Class {pred_class})")
print(f"   -> (Ground Truth reference only: Class {row['Label']} - '{row['Annotation_Symbol']}')")

# 5. Test Serial Bridge & Simulation
print("\n5. Testing Serial Bridge & Missing Value Handling...")
from serial_bridge import SerialBridge
bridge = SerialBridge(port="COM5", baudrate=115200)
ports = SerialBridge.get_available_ports()
print(f"   -> Available COM ports detected: {ports}")
print(f"   -> Initial connection state: {bridge.is_connected} ({bridge.status_message})")

# Test metrics dictionary
metrics = bridge.get_latest_metrics()
print(f"   -> Metrics when disconnected: SpO2={metrics['spo2']}, Pulse={metrics['pulse_bpm']}, Connected={metrics['connected']}")

# Test simulation mode
bridge.start_simulation(include_spo2=True)
time.sleep(0.5)
sim_metrics = bridge.get_latest_metrics()
print(f"   -> Metrics during simulation: SpO2={sim_metrics['spo2']}%, Pulse={sim_metrics['pulse_bpm']} BPM, Connected={sim_metrics['connected']}")
bridge.stop_simulation()
print(f"   -> Stopped simulation cleanly.")

print("\n=======================================================")
print("ALL BACKEND CHECKS PASSED PERFECTLY!")
print("=======================================================")
