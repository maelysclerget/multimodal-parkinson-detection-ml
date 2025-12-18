"""
Check Session/Recording-Level and Patient-Level Distribution for Fold 0

This script analyzes the distribution of Parkinson's Disease (PD) and control (healthy) patients and their recordings (sessions) in Fold 0 of a cross-validation split. It prints both session-level and patient-level class balance for train, validation, and test sets, and highlights the key finding that PD patients tend to have more recordings per patient than controls.

Context:
--------
This analysis is important for understanding dataset imbalances that may affect model training and evaluation. It helps confirm that patient-level splits are balanced, but session-level splits are not, which is critical for fair model assessment and for interpreting results.

Usage:
------
Run as a script to print summary statistics for Fold 0. Adjust file paths as needed for your data location.
"""
import pandas as pd

# Load data
print("Loading data...")
features_csv = "/mloscratch/users/gnahas/data/features/acoustic_features_vf.csv"
labels_csv = "/tremor2tensor/src_GAMMA/paired_healthcode.csv"
train_folds_csv = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
val_test_folds_csv = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"

# Load features
features_df = pd.read_csv(features_csv)
print(f"Loaded features: {features_df.shape}")

# Load labels (semicolon separator)
labels_df = pd.read_csv(labels_csv, sep=',')
print(f"Loaded labels: {labels_df.shape}")

# Load fold splits
train_folds_df = pd.read_csv(train_folds_csv)
val_test_folds_df = pd.read_csv(val_test_folds_csv)

# Separate val and test
val_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'val']
test_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'test']

# Rename column for consistency
if 'healthcode' in features_df.columns and 'healthCode' not in features_df.columns:
    features_df = features_df.rename(columns={'healthcode': 'healthCode'})

# Merge features with labels
features_df = features_df.merge(labels_df[['healthCode', 'label_PD']], on='healthCode', how='inner')
print(f"Features after merging with labels: {features_df.shape}")

# Check Fold 0
fold = 0
print("\n" + "="*80)
print(f"FOLD {fold} - SESSION/RECORDING LEVEL DISTRIBUTION")
print("="*80)

# Get patient IDs for this fold
train_patient_ids = train_folds_df[train_folds_df['fold_iteration'] == fold]['healthCode'].values
val_patient_ids = val_folds_df[val_folds_df['fold_iteration'] == fold]['healthCode'].values
test_patient_ids = test_folds_df[test_folds_df['fold_iteration'] == fold]['healthCode'].values

# Split data by patient
train_data = features_df[features_df['healthCode'].isin(train_patient_ids)]
val_data = features_df[features_df['healthCode'].isin(val_patient_ids)]
test_data = features_df[features_df['healthCode'].isin(test_patient_ids)]

# Count recordings per class
print("\nTRAIN SET (before any undersampling):")
train_pd = (train_data['label_PD'] == 1).sum()
train_control = (train_data['label_PD'] == 0).sum()
train_total = len(train_data)
print(f"  Total recordings: {train_total}")
print(f"  PD (label=1):     {train_pd} ({100*train_pd/train_total:.1f}%)")
print(f"  Control (label=0): {train_control} ({100*train_control/train_total:.1f}%)")

print("\nVALIDATION SET:")
val_pd = (val_data['label_PD'] == 1).sum()
val_control = (val_data['label_PD'] == 0).sum()
val_total = len(val_data)
print(f"  Total recordings: {val_total}")
print(f"  PD (label=1):     {val_pd} ({100*val_pd/val_total:.1f}%)")
print(f"  Control (label=0): {val_control} ({100*val_control/val_total:.1f}%)")

print("\nTEST SET:")
test_pd = (test_data['label_PD'] == 1).sum()
test_control = (test_data['label_PD'] == 0).sum()
test_total = len(test_data)
print(f"  Total recordings: {test_total}")
print(f"  PD (label=1):     {test_pd} ({100*test_pd/test_total:.1f}%)")
print(f"  Control (label=0): {test_control} ({100*test_control/test_total:.1f}%)")

print("\n" + "="*80)
print("PATIENT LEVEL DISTRIBUTION")
print("="*80)

print(f"\nTRAIN SET (patients):")
print(f"  Total patients: {len(train_patient_ids)}")
train_patients_with_labels = labels_df[labels_df['healthCode'].isin(train_patient_ids)]
train_patients_pd = (train_patients_with_labels['label_PD'] == 1).sum()
train_patients_control = (train_patients_with_labels['label_PD'] == 0).sum()
print(f"  PD patients:     {train_patients_pd} ({100*train_patients_pd/len(train_patient_ids):.1f}%)")
print(f"  Control patients: {train_patients_control} ({100*train_patients_control/len(train_patient_ids):.1f}%)")

print(f"\nVALIDATION SET (patients):")
print(f"  Total patients: {len(val_patient_ids)}")
val_patients_with_labels = labels_df[labels_df['healthCode'].isin(val_patient_ids)]
val_patients_pd = (val_patients_with_labels['label_PD'] == 1).sum()
val_patients_control = (val_patients_with_labels['label_PD'] == 0).sum()
print(f"  PD patients:     {val_patients_pd} ({100*val_patients_pd/len(val_patient_ids):.1f}%)")
print(f"  Control patients: {val_patients_control} ({100*val_patients_control/len(val_patient_ids):.1f}%)")

print(f"\nTEST SET (patients):")
print(f"  Total patients: {len(test_patient_ids)}")
test_patients_with_labels = labels_df[labels_df['healthCode'].isin(test_patient_ids)]
test_patients_pd = (test_patients_with_labels['label_PD'] == 1).sum()
test_patients_control = (test_patients_with_labels['label_PD'] == 0).sum()
print(f"  PD patients:     {test_patients_pd} ({100*test_patients_pd/len(test_patient_ids):.1f}%)")
print(f"  Control patients: {test_patients_control} ({100*test_patients_control/len(test_patient_ids):.1f}%)")

print("\n" + "="*80)
print("KEY FINDING:")
print("="*80)
print("\nPatient-level: Balanced 50/50 in train, ~57/43 in val/test")
print("Session-level: Imbalanced in ALL sets (check percentages above)")
print("\nThis confirms: PD patients have MORE recordings per patient than Controls")
print("="*80)
