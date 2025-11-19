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


def get_global_dt_limits(data_dir):
    """Calculate global min/max dt across all JSON files."""
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))
    all_dts = []
    
    for json_file in json_files:
        try:
            df = process_tapping_file(json_file)
            if df is not None and len(df) > 0:
                all_dts.extend(df['dt'].values)
        except:
            continue
    
    if not all_dts:
        return 0, 1  # fallback
    
    dt_min = np.min(all_dts)
    dt_max = np.max(all_dts)
    print(f"Global dt range: {dt_min:.6f} to {dt_max:.6f} seconds")
    
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
    data_dir = Path("src_GAMMA/data")
    output_base_dir = Path("src_GAMMA/tapping-heatmaps")
    output_base_dir.mkdir(exist_ok=True, parents=True)

    # Remove all previous heatmaps
    if output_base_dir.exists():
        import shutil
        shutil.rmtree(output_base_dir)
        output_base_dir.mkdir(exist_ok=True, parents=True)

    # Calculate global dt limits
    print("Calculating global dt limits...")
    vmin, vmax = get_global_dt_limits(data_dir)

    # Find all JSON tapping files
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))
    print(f"Found {len(json_files)} tapping JSON files\n")

    # Group by patient (healthCode)
    patients = defaultdict(list)
    for json_file in json_files:
        filename = json_file.stem # remove .json suffix
        parts = filename.split('_')
        if len(parts) >= 2:
            health_code = parts[0]
            patients[health_code].append(json_file)

    print(f"Found {len(patients)} unique patients")

    total_saved = 0
    total_errors = 0

    # Process only first 3 patients
    for patient_id, files in sorted(patients.items())[:3]:      #for all patients, replace by for patient_id, files in patients.items():
        print(f"\nProcessing patient {patient_id} ({len(files)} total trials)...")

        selected_files = random.sample(files, min(20, len(files)))
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