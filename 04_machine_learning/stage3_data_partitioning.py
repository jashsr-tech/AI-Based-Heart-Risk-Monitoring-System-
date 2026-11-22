"""
ECG FINAL PRO — STAGE 3: DATA PARTITIONING & LEAKAGE PREVENTION
Establishes Development, Validation, and Independent Test subject partitions.
Preserves Records 201 & 202 in Subject_201_202 within the Development partition.
Ensures the 14 exploratory subjects from Stage 2 are NEVER in the Independent Test set.
File: 04_machine_learning/stage3_data_partitioning.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DB_DIR = Path(r"C:\Users\srjas\OneDrive\Desktop\ECG_Project - Copy\mit-bih-arrhythmia-database-1.0.0")
RESULTS_DIR = PROJECT_ROOT / "05_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_0_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
CLASS_1_SYMBOLS = {'V', 'E', 'A', 'a', 'J', 'S', 'F'}
EXCLUDED_SYMBOLS = {'/', 'f', 'Q', '?', 'x', '+', '~', '|', '"', '[', ']', '!', '@'}

EXPLORATORY_RECORDS = {100, 113, 115, 106, 119, 200, 109, 111, 214, 118, 124, 212, 201, 202, 210}

def main():
    subj_map = pd.read_csv(RESULTS_DIR / "stage1_subject_mapping.csv")
    
    # Audit each subject's beats
    subject_stats = {}
    for _, row in subj_map.iterrows():
        rec_id = int(row['Record'])
        subj_id = row['Subject_ID']
        
        ann = wfdb.rdann(str(RAW_DB_DIR / str(rec_id)), 'atr')
        c0 = sum(1 for s in ann.symbol if s in CLASS_0_SYMBOLS)
        c1 = sum(1 for s in ann.symbol if s in CLASS_1_SYMBOLS)
        excl = sum(1 for s in ann.symbol if s in EXCLUDED_SYMBOLS)
        
        if subj_id not in subject_stats:
            subject_stats[subj_id] = {
                'Subject_ID': subj_id,
                'Records': [],
                'Class_0': 0,
                'Class_1': 0,
                'Excluded': 0,
                'Total_Valid': 0,
                'Exploratory_Stage2': False
            }
        subject_stats[subj_id]['Records'].append(rec_id)
        subject_stats[subj_id]['Class_0'] += c0
        subject_stats[subj_id]['Class_1'] += c1
        subject_stats[subj_id]['Excluded'] += excl
        subject_stats[subj_id]['Total_Valid'] += (c0 + c1)
        if rec_id in EXPLORATORY_RECORDS:
            subject_stats[subj_id]['Exploratory_Stage2'] = True

    df = pd.DataFrame(list(subject_stats.values()))
    df['Class_1_Ratio'] = df['Class_1'] / df['Total_Valid']
    
    print(f"Total Subjects: {len(df)}")
    print(f"Stage 2 Exploratory Subjects: {df['Exploratory_Stage2'].sum()}")
    print(f"Genuinely Unseen Subjects: {(~df['Exploratory_Stage2']).sum()}")
    
    # Strategy:
    # 47 subjects total.
    # Exploratory (14 subjects) -> MUST be in Development (Training) partition.
    # Unseen (33 subjects) -> Split between Development, Validation, and Independent Test.
    # Independent Test set target: 10 completely unseen subjects (~21% of subjects, ~20% of beats).
    # Validation set target: 7 completely unseen subjects (~15% of subjects).
    # Development / Training set target: 30 subjects (14 exploratory + 16 unseen) (~64% of subjects).
    
    # Sort unseen subjects by Class 1 ratio for stratified distribution across splits
    unseen_df = df[~df['Exploratory_Stage2']].sort_values(by='Class_1_Ratio', ascending=False).reset_index(drop=True)
    
    # Stratified allocation of the 33 unseen subjects:
    # We want balanced representation of arrhythmias in Dev, Val, and Independent Test
    # Using systematic 3-way distribution:
    # Ratio: approx 16 Dev, 7 Val, 10 Test
    test_subjs = []
    val_subjs = []
    dev_unseen_subjs = []
    
    # Group unseen into high, medium, low arrhythmia burden
    for i, row in unseen_df.iterrows():
        # Cycle through: Dev, Dev, Val, Test, Test
        mod = i % 5
        if mod in (0, 1):
            if len(dev_unseen_subjs) < 16:
                dev_unseen_subjs.append(row['Subject_ID'])
            elif len(val_subjs) < 7:
                val_subjs.append(row['Subject_ID'])
            else:
                test_subjs.append(row['Subject_ID'])
        elif mod == 2:
            if len(val_subjs) < 7:
                val_subjs.append(row['Subject_ID'])
            elif len(test_subjs) < 10:
                test_subjs.append(row['Subject_ID'])
            else:
                dev_unseen_subjs.append(row['Subject_ID'])
        else: # mod in (3, 4)
            if len(test_subjs) < 10:
                test_subjs.append(row['Subject_ID'])
            elif len(val_subjs) < 7:
                val_subjs.append(row['Subject_ID'])
            else:
                dev_unseen_subjs.append(row['Subject_ID'])

    exploratory_subjs = df[df['Exploratory_Stage2']]['Subject_ID'].tolist()
    train_dev_subjs = sorted(exploratory_subjs + dev_unseen_subjs)
    val_subjs = sorted(val_subjs)
    test_subjs = sorted(test_subjs)
    
    # Assign partition column
    def get_partition(s_id):
        if s_id in test_subjs:
            return "Independent_Test"
        elif s_id in val_subjs:
            return "Validation"
        else:
            return "Train_Dev"

    df['Partition'] = df['Subject_ID'].apply(get_partition)
    
    # Verify Subject_201_202 is in Train_Dev
    assert df[df['Subject_ID'] == 'Subject_201_202']['Partition'].iloc[0] == 'Train_Dev', "Subject_201_202 must be in Train_Dev!"
    # Verify no exploratory subject is in Independent_Test
    test_expl = df[(df['Partition'] == 'Independent_Test') & (df['Exploratory_Stage2'])]
    assert len(test_expl) == 0, f"Exploratory subjects found in Independent_Test: {test_expl['Subject_ID'].tolist()}"
    
    # Print Partition Summary
    part_summary = df.groupby('Partition').agg({
        'Subject_ID': 'count',
        'Class_0': 'sum',
        'Class_1': 'sum',
        'Total_Valid': 'sum'
    }).reset_index()
    part_summary['Class_1_Pct'] = part_summary['Class_1'] / part_summary['Total_Valid'] * 100
    
    print("\n" + "=" * 76)
    print("DATASET PARTITION SUMMARY (Zero-Leakage Patient-Level Grouping)")
    print("=" * 76)
    for _, r in part_summary.iterrows():
        print(f"Partition: {r.Partition:<16} | Subjects: {r.Subject_ID:2d} | Valid Beats: {r.Total_Valid:6,d} | Class 0: {r.Class_0:6,d} | Class 1: {r.Class_1:5,d} ({r.Class_1_Pct:4.1f}%)")
    
    # Save partitions
    df.to_csv(RESULTS_DIR / "stage3_subject_partitions.csv", index=False)
    
    partition_manifest = {
        "Train_Dev": {
            "count_subjects": len(train_dev_subjs),
            "subjects": train_dev_subjs,
            "contains_stage2_exploratory": True,
            "exploratory_subjects": exploratory_subjs,
        },
        "Validation": {
            "count_subjects": len(val_subjs),
            "subjects": val_subjs,
            "contains_stage2_exploratory": False,
        },
        "Independent_Test": {
            "count_subjects": len(test_subjs),
            "subjects": test_subjs,
            "contains_stage2_exploratory": False,
            "certification": "100% Genuinely Unseen Subjects reserved for final evaluation (Stage 10). Frozen from tuning."
        }
    }
    
    with open(RESULTS_DIR / "stage3_subject_partitions.json", "w") as f:
        json.dump(partition_manifest, f, indent=2)
        
    print(f"\nSaved partition map to: {RESULTS_DIR / 'stage3_subject_partitions.csv'}")
    print(f"Saved partition manifest to: {RESULTS_DIR / 'stage3_subject_partitions.json'}")

if __name__ == "__main__":
    main()
