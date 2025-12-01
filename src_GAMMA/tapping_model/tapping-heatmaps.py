import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict
import random
import re

def process_tapping_file(json_file_path):
    """Parse JSON tapping file and return DataFrame with coordinates and reaction times."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    timestamps = []
    xs, ys = [], []

    for entry in data:
        timestamps.append(float(entry["TapTimeStamp"]))
        coord_str = entry["TapCoordinate"]
        numbers = re.findall(r'[-+]?\d*\.?\d+', coord_str)
        if len(numbers) >= 2:
            x, y = float(numbers[0]), float(numbers[1])
            xs.append(x)
            ys.append(y)

    if len(timestamps) < 2:
        return None

    timestamps = np.array(timestamps)
    dt = np.diff(timestamps)

    xs = xs[1:]
    ys = ys[1:]

    df = pd.DataFrame({
        'x': xs,
        'y': ys,
        'dt': dt
    })

    return df


def get_global_dt_limits(data_dir, valid_healthcodes):
    """Calculate global dt limits using percentiles to ignore outliers."""
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))
    all_dts = []
    
    for json_file in json_files:
        try:
            # ADDED: Filter by paired healthcodes
            filename = json_file.stem
            parts = filename.split('_')
            health_code = parts[0] if len(parts) > 0 else "unknown"
            
            if health_code not in valid_healthcodes:
                continue
            
            df = process_tapping_file(json_file)
            if df is not None and len(df) > 0:
                all_dts.extend(df['dt'].values)
        except:
            continue
    
    if not all_dts:
        return 0, 1  # fallback
    
    all_dts = np.array(all_dts)
    
    # Use percentiles instead of min/max to ignore outliers
    dt_min = np.percentile(all_dts, 1)    # 1st percentile (ignore bottom 1%)
    dt_max = np.percentile(all_dts, 99)   # 99th percentile (ignore top 1%)
    
    # Also print statistics
    print(f"Global dt statistics:")
    print(f"  Min (absolute): {np.min(all_dts):.6f}")
    print(f"  Max (absolute): {np.max(all_dts):.6f}")
    print(f"  1st percentile: {dt_min:.6f}")
    print(f"  99th percentile: {dt_max:.6f}")
    print(f"  Mean: {np.mean(all_dts):.6f}")
    print(f"  Median: {np.median(all_dts):.6f}")
    
    return dt_min, dt_max

def plot_speed_heatmap(df, patient_id, record_id, output_path, vmin, vmax):
    """Create and save heatmap visualization with fixed color scale."""
    if df is None or len(df) == 0:
        return False

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['x'], df['y'], c=df['dt'], cmap='plasma', 
                         s=70, alpha=0.6, edgecolors='k', linewidth=0.5,
                         vmin=vmin, vmax=vmax)  # Fixed scale
    cbar = plt.colorbar(scatter, label="Reaction time (Δt in seconds)")
    plt.title(f"Tap Duration Heatmap - Patient {patient_id}\nRecord {record_id}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    return True


def main():
    
    # Paths
    data_dir = Path("/mloscratch/users/clerget/data/raw_tapping")
    output_base_dir = Path("/mloscratch/users/clerget/data/tapping_heatmaps")
    paired_file = Path("/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv")
    
    output_base_dir.mkdir(exist_ok=True, parents=True)

    # Load paired healthcodes
    print(f"Loading paired healthcodes from {paired_file}...")
    paired_df = pd.read_csv(paired_file)
    valid_healthcodes = set(paired_df["healthCode"].unique())
    print(f"Found {len(valid_healthcodes)} valid healthCodes")
    
    # Remove all previous heatmaps
    if output_base_dir.exists():
        import shutil
        shutil.rmtree(output_base_dir)
        output_base_dir.mkdir(exist_ok=True, parents=True)

    
    # Calculate global dt limits
    print("Calculating global dt limits...")
    vmin, vmax = get_global_dt_limits(data_dir, valid_healthcodes)

    # Find all JSON tapping files
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))
    print(f"Found {len(json_files)} tapping JSON files\n")

    # Group by patient (healthCode)
    patients = defaultdict(list)
    for json_file in json_files:
        filename = json_file.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            health_code = parts[0]
            
            if health_code not in valid_healthcodes:
                continue
            
            patients[health_code].append(json_file)
    
    total_saved = 0
    total_errors = 0

    # Process only first 3 patients
    for patient_id, files in patients.items(): #for patient_id, files in sorted(patients.items())[:3]:      #for all patients, replace by for patient_id, files in patients.items():
        print(f"\nProcessing patient {patient_id} ({len(files)} total trials)...")

        selected_files = files  # Use all files
        print(f"  Selected {len(selected_files)} trials")

        #create patient folder
        patient_dir = output_base_dir / patient_id
        patient_dir.mkdir(exist_ok=True, parents=True)


        # Process each trial
        for idx, json_file in enumerate(selected_files, 1):
            try:
                df = process_tapping_file(json_file)
                
                if df is None:
                    print(f"  ✗ Trial {idx}: Empty or invalid data")
                    total_errors += 1
                    continue

                filename = json_file.stem
                parts = filename.split('_')
                record_id = parts[1] if len(parts) > 1 else "unknown"

                # Save heatmap
                output_file = patient_dir / f"{idx:02d}_{record_id}_heatmap.png"
                success = plot_speed_heatmap(df, patient_id, record_id, output_file, vmin, vmax)
                
                if success:
                    print(f"  ✓ Trial {idx}: Saved {output_file.name}")
                    total_saved += 1
                else:
                    total_errors += 1

            except Exception as e:
                print(f"  ✗ Trial {idx}: Error - {str(e)}")
                total_errors += 1

    print(f"\n{'='*50}")
    print(f"✓ Completed: {total_saved} heatmaps saved")
    print(f"✗ Errors: {total_errors}")
    print(f"Color scale: {vmin:.6f} to {vmax:.6f} seconds")
    print(f"Output folder: {output_base_dir}")


if __name__ == "__main__":
    main()