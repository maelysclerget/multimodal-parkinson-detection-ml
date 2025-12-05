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

def get_global_coordinate_limits(data_dir, valid_healthcodes):
    """Calculate global X and Y coordinate limits using percentiles to ignore outliers."""
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))
    all_xs = []
    all_ys = []
    
    for json_file in json_files:
        try:
            filename = json_file.stem
            parts = filename.split('_')
            health_code = parts[0] if len(parts) > 0 else "unknown"
            
            if health_code not in valid_healthcodes:
                continue
            
            df = process_tapping_file(json_file)
            if df is not None and len(df) > 0:
                all_xs.extend(df['x'].values)
                all_ys.extend(df['y'].values)
        except:
            continue
    
    if not all_xs or not all_ys:
        return 0, 100, 0, 100  # fallback
    
    all_xs = np.array(all_xs)
    all_ys = np.array(all_ys)
    
    # Use percentiles to crop outliers and maintain spatial resolution
    x_min = np.percentile(all_xs, 1)      # 1st percentile
    x_max = np.percentile(all_xs, 99)     # 99th percentile
    y_min = np.percentile(all_ys, 1)      # 1st percentile
    y_max = np.percentile(all_ys, 99)     # 99th percentile
    
    # Print statistics
    print(f"Global coordinate statistics (1st-99th percentile):")
    print(f"  X range: {x_min:.2f} to {x_max:.2f}")
    print(f"  Y range: {y_min:.2f} to {y_max:.2f}")
    
    return x_min, x_max, y_min, y_max

def plot_speed_heatmap(df, patient_id, record_id, output_path, x_min, x_max, y_min, y_max, dt_min, dt_max):
    """Create and save cropped heatmap visualization with fixed coordinate and color scales."""
    if df is None or len(df) == 0:
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(df['x'], df['y'], c=df['dt'], cmap='plasma', 
                         s=70, alpha=0.6, edgecolors='k', linewidth=0.5,
                         vmin=dt_min, vmax=dt_max)  # Fixed scale
    
    # Set fixed axis limits (crop to percentile range with small margin to avoid clipping)
    x_margin = (x_max - x_min) * 0.085  # 8.5% margin
    y_margin = (y_max - y_min) * 0.085  # 8.5% margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_max + y_margin, y_min - y_margin)  # Inverted Y-axis with margin
    
    # Remove labels, ticks, and title for clean output
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    
    # Remove spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save without colorbar or any extra elements
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
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

    
    # Calculate global coordinate and dt limits
    print("Calculating global coordinate and dt limits...")
    x_min, x_max, y_min, y_max = get_global_coordinate_limits(data_dir, valid_healthcodes)
    print()
    dt_min, dt_max = get_global_dt_limits(data_dir, valid_healthcodes)
    print()

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
    for patient_id, files in patients.items(): #for patient_id, files in sorted(patients.items())[:3]:     
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
                success = plot_speed_heatmap(df, patient_id, record_id, output_file, x_min, x_max, y_min, y_max, dt_min, dt_max)
                
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
    print(f"Coordinate scales (1st-99th percentile):")
    print(f"  X range: {x_min:.2f} to {x_max:.2f}")
    print(f"  Y range: {y_min:.2f} to {y_max:.2f}")
    print(f"  dt (reaction time) scale: {dt_min:.6f} to {dt_max:.6f} seconds")
    print(f"Output folder: {output_base_dir}")


if __name__ == "__main__":
    main()