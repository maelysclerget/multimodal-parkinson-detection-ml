import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

paired_hc_df = pd.read_csv('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv')
paired_healthcodes = set(paired_hc_df["healthCode"].unique())
print(f"Paired healthcodes to extract: {len(paired_healthcodes)}")

#%%
# ===== Tapping Features Extraction =====

def extract_event_sequences(json_file_path):
    """Extract event sequences: (Δt, button pressed)."""
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
    """Extract statistical features: mean Δt, std, miss ratio, L/R ratio."""
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
    lr_ratio = left_count / right_count if right_count > 0 else 0
    missed_taps = len(buttons) - left_count - right_count
    mean_x = np.nanmean(xs)
    mean_y = np.nanmean(ys)
    std_x = np.nanstd(xs)
    std_y = np.nanstd(ys)

    features = {
        'mean_dt': mean_dt,
        'std_dt': std_dt,
        'lr_ratio': lr_ratio,
        'left_count': left_count,
        'right_count': right_count,
        'total_taps': len(buttons),
        'mean_x': mean_x,
        'mean_y': mean_y,
        'std_x': std_x,
        'std_y': std_y,
        'missed_taps': missed_taps
    }
    
    return features

# Extract tapping features
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
        trial_id = parts[1] if len(parts) > 1 else "unknown"  # Extract trial/session ID
        
        if health_code not in paired_healthcodes:
            continue
        
        # ---- Event sequences ----
        events = extract_event_sequences(json_file)
        if events is not None:
            for dt, button in events:
                sequences_data.append({
                    'healthCode': health_code,
                    'trial_id': trial_id,
                    'delta_t': dt,
                    'button': button
                })
        
        # ---- Statistical features ----
        stats = extract_statistical_features(json_file)
        if stats is not None:
            stats['healthCode'] = health_code
            stats['trial_id'] = trial_id
            features_data.append(stats)

    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        continue

# Create csv directory if it doesn't exist
csv_dir = Path('/mloscratch/users/clerget/data/csv')
csv_dir.mkdir(parents=True, exist_ok=True)

# Save to CSV
if sequences_data:
    seq_df = pd.DataFrame(sequences_data)
    seq_df.to_csv(csv_dir / 'tapping_event_sequences.csv', index=False)
    print(f"✓ Event sequences saved: {len(seq_df)} rows")

if features_data:
    feat_df = pd.DataFrame(features_data)
    print(f"✓ Statistical features saved: {len(feat_df)} rows")
    feat_df.to_csv(csv_dir / 'tapping_statistical_features.csv', index=False)