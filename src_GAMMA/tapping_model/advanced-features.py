import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy import stats


paired_hc_df = pd.read_csv('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv')
paired_healthcodes = set(paired_hc_df["healthCode"].unique())
print(f"Paired healthcodes to extract: {len(paired_healthcodes)}")

# ===== Advanced Feature Extraction =====

def extract_advanced_features(json_file_path):
    """Extract advanced features: fatigue slope, variability index, tap distance."""
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
    quarter_len = len(dt) // 4 #number of taps per quarter
    if quarter_len > 0:
        q1_mean = np.mean(dt[:quarter_len]) # average rt in q1 
        q4_mean = np.mean(dt[-quarter_len:]) #  average rt in q4
        fatigue_slope = (q4_mean - q1_mean) / q1_mean if q1_mean > 0 else 0 # shows percentage increase/decrease
    else:
        fatigue_slope = 0
    
    # ===== 2. Variability Index =====
    # Coefficient of variation (std / mean) of reaction times
    dt_mean = np.mean(dt)
    dt_std = np.std(dt)
    variability_index = dt_std / dt_mean if dt_mean > 0 else 0 # coefficient of variation 
    
    # ===== 3. Tap Distance =====
    # Mean distance between consecutive taps
    if len(xs) > 1 and len(ys) > 1:
        distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        mean_tap_distance = np.mean(distances)
        std_tap_distance = np.std(distances)
    else:
        mean_tap_distance = 0
        std_tap_distance = 0
    
    # ===== 4. Fluctuation =====
    # Variability of reaction time differences, high fluctuation = inconsistent tapping
    if len(dt) > 1:
        dt_diff = np.diff(dt) # differences between consecutive reaction times
        fluctuation = np.std(dt_diff) # variability of reaction time differences
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
    """Extract advanced features for all JSON files."""
    data_dir = Path(data_dir)
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))

    all_features = []

    for json_file in json_files:
        try:
            filename = json_file.stem
            parts = filename.split('_')
            health_code = parts[0] if len(parts) > 0 else "unknown"
            trial_id = parts[1] if len(parts) > 1 else "unknown"  # Extract trial/session ID
            
            # ADDED: Filter by paired healthcodes
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
    """Aggregate session-level features to patient-level (mean of all sessions)."""
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
    import re
    
    data_dir = "/mloscratch/users/clerget/data/raw_tapping"
    csv_dir = Path('/mloscratch/users/clerget/data/csv')
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting advanced features...")
    adv_features_df = extract_all_advanced_features(data_dir)
    
    # Save session-level features
    adv_features_df.to_csv(csv_dir / 'tapping_advanced_features_session.csv', index=False)
    print("✓ Session-level features saved")
    
    # Aggregate to patient level
    print("\nAggregating to patient level...")
    patient_features = aggregate_to_patient_level(adv_features_df)
    
    # Save patient-level features
    patient_features.to_csv(csv_dir / 'tapping_advanced_features_patient.csv', index=False)
    print("✓ Patient-level features saved")
    
    
    # ===== Concatenate Basic and Advanced Features =====
    print("\n===== Concatenating Basic and Advanced Features =====")
    
    # Load basic features (session/trial-level)
    basic_features_path = csv_dir / 'tapping_statistical_features.csv'
    if basic_features_path.exists():
        basic_features = pd.read_csv(basic_features_path)
        
        # Load advanced features (session/trial-level)
        advanced_features_session = adv_features_df
        
        print(f"Basic features shape (sessions): {basic_features.shape}")
        print(f"Advanced features shape (sessions): {advanced_features_session.shape}")
        
        # Merge on both healthCode AND trial_id at session level
        combined_features = basic_features.merge(advanced_features_session, on=["healthCode", "trial_id"], how="inner")
        print(f"Combined features shape: {combined_features.shape}")
        
        # Save combined features
        combined_features.to_csv(csv_dir / 'tapping_combined_features_session.csv', index=False)
        print(f"✓ Combined session-level features saved: {combined_features.shape[0]} sessions with {combined_features.shape[1]} features")
    else:
        print(f"⚠ Warning: Basic features not found at {basic_features_path}")
        print("  Run data-loading.py first to generate basic features")