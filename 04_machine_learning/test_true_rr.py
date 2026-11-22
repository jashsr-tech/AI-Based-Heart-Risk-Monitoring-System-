import wfdb, numpy as np, pandas as pd
from pathlib import Path

raw_db = Path(r"C:\Users\srjas\OneDrive\Desktop\ECG_Project - Copy\mit-bih-arrhythmia-database-1.0.0")
subj_map = pd.read_csv("05_results/stage1_subject_mapping.csv")

ALL_BEATS = {'N', 'L', 'R', 'e', 'j', 'V', 'E', 'A', 'a', 'J', 'S', 'F', '/', 'f', 'Q', '?'}
TARGET_BEATS = {'N', 'L', 'R', 'e', 'j', 'V', 'E', 'A', 'a', 'J', 'S', 'F'}

rr_values = []
for rec_id in subj_map['Record']:
    ann = wfdb.rdann(str(raw_db / str(rec_id)), 'atr')
    samples = ann.sample
    symbols = ann.symbol
    
    last_beat_sample = None
    for s, sym in zip(samples, symbols):
        if sym in ALL_BEATS:
            if sym in TARGET_BEATS:
                if last_beat_sample is not None:
                    rr = (s - last_beat_sample) / 360.0
                    rr_values.append(rr)
            last_beat_sample = s

rr_arr = np.array(rr_values)
print(f"Total evaluated RR intervals: {len(rr_arr):,}")
print(f"Min RR: {rr_arr.min():.4f} s (HR: {60/rr_arr.min():.1f} bpm)")
print(f"Max RR: {rr_arr.max():.4f} s (HR: {60/rr_arr.max():.1f} bpm)")
print(f"Mean RR: {rr_arr.mean():.4f} s (HR: {60/rr_arr.mean():.1f} bpm)")
print(f"Median RR: {np.median(rr_arr):.4f} s")
print(f"Beats with RR > 3.0s: {(rr_arr > 3.0).sum()}")
print(f"Beats with RR < 0.25s: {(rr_arr < 0.25).sum()}")
