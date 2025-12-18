"""
Advanced Tapping Features Extraction and Aggregation

This script computes advanced features from tapping test data that capture
fatigue, consistency, spatial behavior, and temporal fluctuations.

Workflow:
1. Extract advanced features from raw tapping JSON files (session-level)
2. Aggregate to patient-level (mean and std across all sessions)
3. Combine with basic statistical features for comprehensive feature set

Advanced Features:
- Fatigue Slope: Reaction time change from 1st to 4th quarter (% change)
- Variability Index: Coefficient of variation of inter-tap intervals
- Tap Distance: Mean/std distance between consecutive tap coordinates
- Fluctuation: Std dev of changes in reaction times (unpredictability)

Output Files:
- tapping_advanced_features_session.csv: Session-level advanced features
- tapping_advanced_features_patient.csv: Patient-level aggregated features
- tapping_combined_features_session.csv: Basic + Advanced features combined

Dependencies:
- Raw JSON files: /mloscratch/users/clerget/data/raw_tapping/
- Basic features: tapping_statistical_features_sessions.csv
- Paired healthcodes: /mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ===== Load Paired HealthCodes =====
paired_hc_df = pd.read_csv('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv')
paired_healthcodes = set(paired_hc_df["healthCode"].unique())
print(f"Paired healthcodes to extract: {len(paired_healthcodes)}")


# ===== Advanced Feature Extraction Functions =====

def extract_advanced_features(json_file_path):
    """
    Extract advanced features from a single tapping test session.
    
    Computes 5 advanced features that capture different aspects of tapping behavior:
    1. Fatigue Slope: % change in reaction time from 1st to 4th quarter
    2. Variability Index: Coefficient of variation (std/mean) of inter-tap intervals
    3. Tap Distance: Mean and std Euclidean distance between consecutive taps
    4. Fluctuation: Unpredictability in reaction time changes
    
    Args:
        json_file_path (str or Path): Path to tapping JSON file
    
    Returns:
        dict: Dictionary with keys:
            - fatigue_slope: % change in RT (negative = fatigue)
            - variability_index: Coefficient of variation of inter-tap intervals
            - mean_tap_distance, std_tap_distance: Spatial consistency
            - fluctuation: Temporal inconsistency
          Returns None if less than 10 taps in file.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    timestamps = []
    xs, ys = [], []
    
    for entry in data:
        timestamps.append(float(entry["TapTimeStamp"]))
        coord_str = entry.get("TapCoordinate", "")
        if coord_str:
            try:
                numbers = re.findall(r'[-+]?\d*\.?\d+', coord_str)
                if len(numbers) >= 2:
                    xs.append(float(numbers[0]))
                    ys.append(float(numbers[1]))
            except:
                pass
    
    if len(timestamps) < 10:
        return None
    
    timestamps = np.array(timestamps)
    dt = np.diff(timestamps)
    
    # ===== 1. Fatigue Slope =====
    # Divide into 4 quarters and check if reaction time increases (fatigue)
    quarter_len = len(dt) // 4  
    if quarter_len > 0:
        q1_mean = np.mean(dt[:quarter_len])  
        q4_mean = np.mean(dt[-quarter_len:])  
        fatigue_slope = (q4_mean - q1_mean) / q1_mean if q1_mean > 0 else 0
    else:
        fatigue_slope = 0
    
    # ===== 2. Variability Index =====
    # Coefficient of variation (std / mean) of reaction times
    dt_mean = np.mean(dt)
    dt_std = np.std(dt)
    variability_index = dt_std / dt_mean if dt_mean > 0 else 0
    
    # ===== 3. Tap Distance =====
    # Mean distance between consecutive taps (spatial consistency)
    if len(xs) > 1 and len(ys) > 1:
        distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        mean_tap_distance = np.mean(distances)
        std_tap_distance = np.std(distances)
    else:
        mean_tap_distance = 0
        std_tap_distance = 0
    
    # ===== 4. Fluctuation =====
    # Variability in reaction time differences
    if len(dt) > 1:
        dt_diff = np.diff(dt)
        fluctuation = np.std(dt_diff)
    else:
        fluctuation = 0

    features = {
        'fatigue_slope': fatigue_slope,
        'variability_index': variability_index,
        'mean_tap_distance': mean_tap_distance,
        'std_tap_distance': std_tap_distance,
        'fluctuation': fluctuation,
    }
    
    return features


def extract_all_advanced_features(data_dir):
    """
    Extract advanced features for all tapping JSON files in a directory.
    
    Args:
        data_dir (str or Path): Directory containing raw tapping JSON files
    
    Returns:
        pd.DataFrame: Session-level advanced features with columns:
            [healthCode, trial_id, fatigue_slope, variability_index, 
             mean_tap_distance, std_tap_distance, fluctuation]
    """
    data_dir = Path(data_dir)
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))

    all_features = []

    for json_file in json_files:
        try:
            filename = json_file.stem
            parts = filename.split('_')
            health_code = parts[0] if len(parts) > 0 else "unknown"
            trial_id = parts[1] if len(parts) > 1 else "unknown"  
            
            if health_code not in paired_healthcodes:
                continue
            
            features = extract_advanced_features(json_file)
            if features:
                features['healthCode'] = health_code
                features['trial_id'] = trial_id
                all_features.append(features)
        except Exception as e:
            continue
    
    return pd.DataFrame(all_features)

# ===== Patient-Level Aggregation =====

def aggregate_to_patient_level(features_df):
    """
    Aggregate session-level advanced features to patient-level.
    
    Computes mean and std of each feature across all sessions for each patient.
    This creates a single row per patient with aggregated statistics.
    
    Args:
        features_df (pd.DataFrame): Session-level features from extract_all_advanced_features()
    
    Returns:
        pd.DataFrame: Patient-level features with columns:
            [healthCode, fatigue_slope_mean, fatigue_slope_std, 
             variability_index_mean, variability_index_std, ...]
    """
    patient_features = features_df.groupby('healthCode').agg({
        'fatigue_slope': ['mean', 'std'],
        'variability_index': ['mean', 'std'],
        'mean_tap_distance': ['mean', 'std'],
        'std_tap_distance': ['mean', 'std'],
        'fluctuation': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    patient_features.columns = ['_'.join(col).strip('_') for col in patient_features.columns.values]
    patient_features.columns = patient_features.columns.str.replace('healthCode_', 'healthCode')
    
    return patient_features


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVANCED FEATURES EXTRACTION AND AGGREGATION")
    print("="*80)
    
    data_dir = "/mloscratch/users/clerget/data/raw_tapping"
    csv_dir = Path('/mloscratch/users/clerget/data/csv')
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Extract Session-Level Advanced Features =====
    print("\n[1/3] Extracting advanced features from raw tapping files...")
    adv_features_df = extract_all_advanced_features(data_dir)
    print(f"✓ Extracted {len(adv_features_df)} sessions")
    
    # Save session-level features
    session_path = csv_dir / 'tapping_advanced_features_session.csv'
    adv_features_df.to_csv(session_path, index=False)
    print(f"✓ Session-level features saved: {session_path}")
    
    # ===== Aggregate to Patient Level =====
    print("\n[2/3] Aggregating to patient-level...")
    patient_features = aggregate_to_patient_level(adv_features_df)
    print(f"✓ Aggregated to {len(patient_features)} patients")
    
    # Save patient-level features
    patient_path = csv_dir / 'tapping_advanced_features_patient.csv'
    patient_features.to_csv(patient_path, index=False)
    print(f"✓ Patient-level features saved: {patient_path}")
    
    # ===== Combine with Basic Features =====
    print("\n[3/3] Combining with basic statistical features...")
    basic_features_path = csv_dir / 'tapping_statistical_features_sessions.csv'
    
    if basic_features_path.exists():
        basic_features = pd.read_csv(basic_features_path)
        print(f"  Basic features: {basic_features.shape}")
        print(f"  Advanced features: {adv_features_df.shape}")
        
        # Merge on both healthCode and trial_id at session level
        combined_features = basic_features.merge(
            adv_features_df, 
            on=["healthCode", "trial_id"], 
            how="inner"
        )
        print(f"✓ Combined features: {combined_features.shape}")
        
        # Save combined features
        combined_path = csv_dir / 'tapping_combined_features_session.csv'
        combined_features.to_csv(combined_path, index=False)
        print(f"✓ Combined session-level features saved: {combined_path}")
    else:
        print(f"\n⚠ Warning: Basic features not found at {basic_features_path}")
        print("  Run data-loading.py first to generate basic features")
    
    print("\n" + "="*80)
    print("✓ Advanced features extraction complete!")
    print("="*80)