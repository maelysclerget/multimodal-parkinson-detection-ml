"""
Patient and Session Distribution Analysis for Acoustic Features Dataset

This script analyzes and displays the distribution of Parkinson's Disease (PD) and healthy (control) patients, as well as their session counts, in the acoustic features dataset. It merges feature and label tables, computes per-patient session statistics, and prints a summary for both groups.

Context:
--------
This script is useful for dataset exploration, quality control, and reporting. It helps ensure that patient/session splits are balanced and provides insight into the data structure before model training or cross-validation.

Usage:
------
Run as a script or import the function in a notebook or pipeline. Adjust the CSV paths as needed for your data location.
"""
import pandas as pd
from pathlib import Path

def analyze_patient_session_distribution(
    features_csv="/mloscratch/users/gnahas/data/features/acoustic_features_vf.csv",
    labels_csv="/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
):
    """
    Analyze and display the distribution of PD vs. healthy patients and their session counts.

    Args:
        features_csv (str): Path to features CSV (must have healthCode column).
        labels_csv (str): Path to paired healthcode CSV with label_PD column.

    Returns:
        tuple:
            pd_patients_dict (dict): Mapping of PD healthCode to session count.
            healthy_patients_dict (dict): Mapping of healthy healthCode to session count.
    """
    print("\n" + "="*80)
    print(" "*20 + "PATIENT & SESSION STATISTICS")
    print("="*80)

    # Load data
    features_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_csv, sep=',' if labels_csv.endswith('.csv') else ',')

    # Ensure consistent column names
    if 'healthcode' in features_df.columns and 'healthCode' not in features_df.columns:
        features_df = features_df.rename(columns={'healthcode': 'healthCode'})

    # Merge features with labels
    data_with_labels = features_df.merge(labels_df[['healthCode', 'label_PD']], on='healthCode', how='inner')

    # PD patients
    pd_patients = data_with_labels[data_with_labels['label_PD'] == 1]
    pd_unique_patients = pd_patients['healthCode'].unique()
    pd_num_patients = len(pd_unique_patients)
    pd_num_sessions = len(pd_patients)
    pd_sessions_per_patient = pd_patients['healthCode'].value_counts()

    # Healthy patients
    healthy_patients = data_with_labels[data_with_labels['label_PD'] == 0]
    healthy_unique_patients = healthy_patients['healthCode'].unique()
    healthy_num_patients = len(healthy_unique_patients)
    healthy_num_sessions = len(healthy_patients)
    healthy_sessions_per_patient = healthy_patients['healthCode'].value_counts()

    print(f"\n📊 PD (Positive) Patients:")
    print(f"  Total PD Patients:         {pd_num_patients}")
    print(f"  Total PD Sessions:         {pd_num_sessions}")
    print(f"  Sessions per PD Patient:   {pd_sessions_per_patient.mean():.2f} ± {pd_sessions_per_patient.std():.2f}")
    print(f"                             (min={pd_sessions_per_patient.min()}, max={pd_sessions_per_patient.max()})")
    print(f"  PD Patient IDs (healthCode):")
    for hc in sorted(pd_unique_patients):
        print(f"    - {hc}: {pd_sessions_per_patient[hc]} session(s)")

    print(f"\n📊 Healthy (Control) Patients:")
    print(f"  Total Healthy Patients:    {healthy_num_patients}")
    print(f"  Total Healthy Sessions:    {healthy_num_sessions}")
    print(f"  Sessions per Healthy Patient: {healthy_sessions_per_patient.mean():.2f} ± {healthy_sessions_per_patient.std():.2f}")
    print(f"                             (min={healthy_sessions_per_patient.min()}, max={healthy_sessions_per_patient.max()})")
    print(f"  Healthy Patient IDs (healthCode):")
    for hc in sorted(healthy_unique_patients):
        print(f"    - {hc}: {healthy_sessions_per_patient[hc]} session(s)")

    print(f"\n📊 Summary:")
    print(f"  Total Unique Patients:  {pd_num_patients + healthy_num_patients}")
    print(f"  Total Sessions:         {pd_num_sessions + healthy_num_sessions}")
    print(f"  Ratio (PD:Healthy):     {pd_num_patients}:{healthy_num_patients} (patients)")
    print(f"                          {pd_num_sessions}:{healthy_num_sessions} (sessions)")
    print("="*80)

    pd_dict = dict(zip(sorted(pd_unique_patients), [pd_sessions_per_patient[hc] for hc in sorted(pd_unique_patients)]))
    healthy_dict = dict(zip(sorted(healthy_unique_patients), [healthy_sessions_per_patient[hc] for hc in sorted(healthy_unique_patients)]))

    return pd_dict, healthy_dict

if __name__ == "__main__":
    analyze_patient_session_distribution()