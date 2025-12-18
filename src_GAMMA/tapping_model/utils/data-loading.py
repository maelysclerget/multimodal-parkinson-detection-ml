"""
Tapping Features Extraction and Aggregation

This script extracts tapping test features from raw JSON files for paired healthcodes
and saves them as CSV files at both session-level and patient-level (aggregated).

Workflow:
1. Load list of paired healthcodes (PD patients with matched healthy controls)
2. Process each tapping JSON file to extract:
   - Event sequences: Inter-tap intervals (Δt) and button pressed
   - Statistical features: Mean/std tap intervals, coordinates, miss ratios
3. Save session-level data to CSV
4. Aggregate session-level data to patient-level (mean of all sessions per patient)

Output Files:
- tapping_event_sequences_session.csv: Event-level data (Δt, button per tap)
- tapping_statistical_features_session.csv: Session-level statistical features
- tapping_statistical_features_patient.csv: Patient-level aggregated features

Dependencies:
- Raw JSON files: /mloscratch/users/clerget/data/raw_tapping/
- Paired healthcodes: /mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ===== Load Paired HealthCodes =====
paired_hc_df = pd.read_csv('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv')
paired_healthcodes = set(paired_hc_df["healthCode"].unique())
print(f"Paired healthcodes to extract: {len(paired_healthcodes)}")


# ===== Feature Extraction Functions =====

def extract_event_sequences(json_file_path):
    """
    Extract event sequences from tapping JSON file.
    
    Extracts inter-tap intervals (Δt) and which button was tapped for each event.
    These are event-level features useful for sequence analysis.
    
    Args:
        json_file_path (str or Path): Path to tapping JSON file
    
    Returns:
        list: List of tuples (delta_t, button_pressed) for each event after first tap.
              Returns None if less than 2 taps in file.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    timestamps = []
    buttons = []
    
    for entry in data:
        timestamps.append(float(entry["TapTimeStamp"]))
        button = entry.get("TappedButtonId", "unknown")
        buttons.append(button)
    
    if len(timestamps) < 2:
        return None
    
    timestamps = np.array(timestamps)
    dt = np.diff(timestamps)
    buttons = buttons[1:]
    
    events = list(zip(dt, buttons))
    return events

#%%
def extract_statistical_features(json_file_path):
    """
    Extract statistical features from tapping test.
    
    Computes session-level statistics including:
    - Temporal: mean/std inter-tap intervals
    - Spatial: mean/std X,Y coordinates
    - Accuracy: total taps, missed taps
    
    Args:
        json_file_path (str or Path): Path to tapping JSON file
    
    Returns:
        dict: Dictionary with keys:
            - mean_dt, std_dt: Mean and std of inter-tap intervals
            - total_taps: Total number of taps
            - mean_x, mean_y, std_x, std_y: Coordinate statistics
            - missed_taps: Number of failed/missed taps
          Returns None if less than 2 taps in file.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    timestamps = []
    buttons = []
    xs, ys = [], []
    
    for entry in data:
        timestamps.append(float(entry["TapTimeStamp"]))
        button = entry.get("TappedButtonId", "unknown")
        buttons.append(button)

        coord_str = entry.get("TapCoordinate", None)
        if coord_str:
            try:
                x_str, y_str = coord_str.strip("{} ").split(",")
                xs.append(float(x_str))
                ys.append(float(y_str))
            except:
                xs.append(np.nan)
                ys.append(np.nan)
        else:
            xs.append(np.nan)
            ys.append(np.nan)
    
    if len(timestamps) < 2:
        return None
    
    timestamps = np.array(timestamps)
    dt = np.diff(timestamps)
    buttons = buttons[1:]
    
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    
    left_count = sum(1 for b in buttons if "left" in str(b).lower())
    right_count = sum(1 for b in buttons if "right" in str(b).lower())
    missed_taps = len(buttons) - left_count - right_count
    mean_x = np.nanmean(xs)
    mean_y = np.nanmean(ys)
    std_x = np.nanstd(xs)
    std_y = np.nanstd(ys)

    features = {
        'mean_dt': mean_dt,
        'std_dt': std_dt,
        'total_taps': len(buttons),
        'mean_x': mean_x,
        'mean_y': mean_y,
        'std_x': std_x,
        'std_y': std_y,
        'missed_taps': missed_taps
    }
    
    return features

# ===== Process Tapping Files =====
print("\n===== Extracting Tapping Features =====")
data_dir = Path('/mloscratch/users/clerget/data/raw_tapping')
json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))
print(f"Found {len(json_files)} tapping JSON files")

sequences_data = []
features_data = []

for json_file in json_files:
    try:
        filename = json_file.stem
        parts = filename.split('_')
        health_code = parts[0] if len(parts) > 0 else "unknown"
        trial_id = parts[1] if len(parts) > 1 else "unknown"
        
        # Skip if healthcode not in paired dataset
        if health_code not in paired_healthcodes:
            continue
        
        # Extract event sequences
        events = extract_event_sequences(json_file)
        if events is not None:
            for dt, button in events:
                sequences_data.append({
                    'healthCode': health_code,
                    'trial_id': trial_id,
                    'delta_t': dt,
                    'button': button
                })
        
        # Extract statistical features
        stats = extract_statistical_features(json_file)
        if stats is not None:
            stats['healthCode'] = health_code
            stats['trial_id'] = trial_id
            features_data.append(stats)

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        continue

# ===== Save Session-Level Data =====
print("\n===== Saving Data =====")
csv_dir = Path('/mloscratch/users/clerget/data/csv')
csv_dir.mkdir(parents=True, exist_ok=True)

# Save event sequences
if sequences_data:
    seq_df = pd.DataFrame(sequences_data)
    seq_path = csv_dir / 'tapping_event_sequences_session.csv'
    seq_df.to_csv(seq_path, index=False)
    print(f"✓ Event sequences saved: {len(seq_df)} rows")
    print(f"  File: {seq_path}")

# Save statistical features (session-level)
if features_data:
    feat_df = pd.DataFrame(features_data)
    feat_path = csv_dir / 'tapping_statistical_features_session.csv'
    feat_df.to_csv(feat_path, index=False)
    print(f"✓ Statistical features (session-level) saved: {len(feat_df)} rows")
    print(f"  File: {feat_path}")
    
    # ===== Aggregate to Patient-Level =====
    # Group by healthCode and compute mean of all numeric features
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    aggregated_df = feat_df.groupby('healthCode')[numeric_cols].mean()
    aggregated_df = aggregated_df.reset_index()

    agg_path = csv_dir / 'tapping_statistical_features_patient.csv'
    aggregated_df.to_csv(agg_path, index=False)
    print(f"✓ Statistical features (patient-level) saved: {len(aggregated_df)} rows")
    print(f"  File: {agg_path}")
    print(f"\n  Sample of aggregated data:\n{aggregated_df.head()}")

print("\n" + "="*80)
print("✓ Data extraction and aggregation complete!")
print("="*80)


# ===== ANALYSIS: Patient and Session Statistics =====
def analyze_patient_session_distribution():
    """
    Analyze and display distribution of PD vs Healthy patients and their sessions.
    
    Reads the paired healthcodes CSV with diagnosis labels and provides:
    - Number of PD patients and their total sessions
    - Number of Healthy (control) patients and their total sessions
    - Individual healthCode IDs for each group
    - Sessions per patient statistics (mean, std, min, max)
    
    Returns:
        tuple: (pd_patients_dict, healthy_patients_dict)
            - pd_patients_dict: {healthCode: num_sessions, ...} for PD patients
            - healthy_patients_dict: {healthCode: num_sessions, ...} for Healthy patients
    """
    
    print("\n" + "="*80)
    print(" "*20 + "PATIENT & SESSION STATISTICS")
    print("="*80)
    
    # Load labels and feature data
    labels_df = pd.read_csv('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv')
    csv_dir = Path('/mloscratch/users/clerget/data/csv')
    feat_path = csv_dir / 'tapping_statistical_features_session.csv'
    
    if not feat_path.exists():
        print(f"⚠ Warning: Feature file not found at {feat_path}")
        print(f"  Cannot analyze sessions. Please run feature extraction first.")
        return {}, {}
    
    # Load session data
    sessions_df = pd.read_csv(feat_path)
    
    # Merge with labels
    data_with_labels = sessions_df.merge(labels_df, on='healthCode', how='inner')
    
    # Analyze PD patients
    pd_patients = data_with_labels[data_with_labels['label_PD'] == 1]
    pd_unique_patients = pd_patients['healthCode'].unique()
    pd_num_patients = len(pd_unique_patients)
    pd_num_sessions = len(pd_patients)
    pd_sessions_per_patient = pd_patients['healthCode'].value_counts()
    
    # Analyze Healthy patients
    healthy_patients = data_with_labels[data_with_labels['label_PD'] == 0]
    healthy_unique_patients = healthy_patients['healthCode'].unique()
    healthy_num_patients = len(healthy_unique_patients)
    healthy_num_sessions = len(healthy_patients)
    healthy_sessions_per_patient = healthy_patients['healthCode'].value_counts()
    
    print(f"\n📊 PD (Positive) Patients:")
    print(f"  Total PD Patients:         {pd_num_patients}")
    print(f"  Total PD Sessions/Trials:  {pd_num_sessions}")
    print(f"  Sessions per PD Patient:   {pd_sessions_per_patient.mean():.2f} ± {pd_sessions_per_patient.std():.2f}")
    print(f"                             (min={pd_sessions_per_patient.min()}, max={pd_sessions_per_patient.max()})")
    print(f"  PD Patient IDs (healthCode):")
    for hc in sorted(pd_unique_patients):
        num_sessions = pd_sessions_per_patient[hc]
        print(f"    - {hc}: {num_sessions} session(s)")
    
    print(f"\n📊 Healthy (Control) Patients:")
    print(f"  Total Healthy Patients:    {healthy_num_patients}")
    print(f"  Total Healthy Sessions/Trials: {healthy_num_sessions}")
    print(f"  Sessions per Healthy Patient: {healthy_sessions_per_patient.mean():.2f} ± {healthy_sessions_per_patient.std():.2f}")
    print(f"                             (min={healthy_sessions_per_patient.min()}, max={healthy_sessions_per_patient.max()})")
    print(f"  Healthy Patient IDs (healthCode):")
    for hc in sorted(healthy_unique_patients):
        num_sessions = healthy_sessions_per_patient[hc]
        print(f"    - {hc}: {num_sessions} session(s)")
    
    print(f"\n📊 Summary:")
    print(f"  Total Unique Patients:  {pd_num_patients + healthy_num_patients}")
    print(f"  Total Sessions/Trials:  {pd_num_sessions + healthy_num_sessions}")
    print(f"  Ratio (PD:Healthy):     {pd_num_patients}:{healthy_num_patients} (patients)")
    print(f"                          {pd_num_sessions}:{healthy_num_sessions} (sessions)")
    print("="*80)
    
    # Create return dictionaries
    pd_dict = dict(zip(sorted(pd_unique_patients), 
                       [pd_sessions_per_patient[hc] for hc in sorted(pd_unique_patients)]))
    healthy_dict = dict(zip(sorted(healthy_unique_patients),
                            [healthy_sessions_per_patient[hc] for hc in sorted(healthy_unique_patients)]))
    
    return pd_dict, healthy_dict


# Run analysis if script is executed directly
if __name__ == "__main__":
    pd_patients, healthy_patients = analyze_patient_session_distribution()