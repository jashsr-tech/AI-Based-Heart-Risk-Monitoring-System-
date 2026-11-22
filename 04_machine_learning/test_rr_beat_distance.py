import wfdb

ALL_BEAT_SYMBOLS = {'N', 'L', 'R', 'e', 'j', 'V', 'E', 'A', 'a', 'J', 'S', 'F', '/', 'f', 'Q', '?'}

ann101 = wfdb.rdann(r"C:\Users\srjas\OneDrive\Desktop\ECG_Project - Copy\mit-bih-arrhythmia-database-1.0.0\101", "atr")
prev_any = ann101.sample[132] # '~' at 40603
prev_beat = [s for s, sym in zip(ann101.sample[:133], ann101.symbol[:133]) if sym in ALL_BEAT_SYMBOLS][-1]
print("Rec 101 sample 40668:")
print(f"  Distance to previous annotation (~): {(40668 - prev_any)/360.0:.4f} s")
print(f"  Distance to previous BEAT (N): {(40668 - prev_beat)/360.0:.4f} s")

ann102 = wfdb.rdann(r"C:\Users\srjas\OneDrive\Desktop\ECG_Project - Copy\mit-bih-arrhythmia-database-1.0.0\102", "atr")
prev_any_102 = ann102.sample[705] # '/' at 207541
prev_beat_102 = [s for s, sym in zip(ann102.sample[:706], ann102.symbol[:706]) if sym in ALL_BEAT_SYMBOLS][-1]
print("\nRec 102 sample 207743:")
print(f"  Distance to previous annotation (/): {(207743 - prev_any_102)/360.0:.4f} s")
print(f"  Distance to previous BEAT (/): {(207743 - prev_beat_102)/360.0:.4f} s")
