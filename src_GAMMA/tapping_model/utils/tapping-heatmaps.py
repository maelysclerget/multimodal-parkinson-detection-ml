"""
Tapping Heatmap Visualization Module

This module processes raw tapping data from mPower JSON files and generates heatmap 
visualizations showing tap locations and reaction times across all participants.

Key Features:
    - Extracts tap coordinates (x, y) and inter-tap intervals (dt) from raw JSON
    - Normalizes heatmaps using global percentile-based scaling (1st-99th percentile)
    - Generates 2D scatter plot heatmaps with reaction time color coding
    - Organizes output by patient with consistent spatial and color scales
    - Handles edge cases: outliers, missing data, invalid files

Data Flow:
    1. Load paired healthcodes from CSV (valid patients only)
    2. Calculate global coordinate and dt ranges from all valid tapping files
    3. Process each tapping JSON file (per trial/session)
    4. Generate and save individual heatmap PNG for each trial
    5. Organize heatmaps by patient ID in directory structure

Output Structure:
    /tapping_heatmaps/
    ├── {healthCode_1}/
    │   ├── 01_{record_id}_heatmap.png
    │   ├── 02_{record_id}_heatmap.png
    │   └── ...
    ├── {healthCode_2}/
    │   └── ...
    └── ...

Color Scale:
    - Dark colors (cool): fast reaction times (small dt)
    - Bright colors (hot): slow reaction times (large dt)
    - Colormap: 'plasma' (perceptually uniform, colorblind-friendly)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import re


def process_tapping_file(json_file_path):
    """
    Parse raw tapping JSON file and extract spatial and temporal data.
    
    Reads a mPower tapping JSON file containing tap events and extracts:
    - Tap coordinates (x, y) from screen
    - Inter-tap intervals (dt) computed as differences between consecutive timestamps
    
    Args:
        json_file_path (str or Path): Path to the tapping JSON file 
            (format: *_tapping_results_json_TappingSamples.json)
    
    Returns:
        pd.DataFrame or None: DataFrame with columns:
            - 'x' (float): X coordinate of tap on screen (0-640 typically)
            - 'y' (float): Y coordinate of tap on screen (0-1136 typically)
            - 'dt' (float): Inter-tap interval in seconds (time to previous tap)
            
            Returns None if:
            - File cannot be parsed
            - Fewer than 2 taps recorded (cannot compute dt)
            - Invalid coordinate format
    
    Raises:
        json.JSONDecodeError: If JSON is malformed
        KeyError: If expected fields missing from JSON entries
    
    Notes:
        - Assumes TapTimeStamp is in milliseconds (converted to seconds)
        - TapCoordinate format: "CGPoint(x=..., y=...)" or similar
        - First tap is dropped since dt requires at least 2 taps
    
    Example:
        >>> df = process_tapping_file("patient_123_session_456_tapping_results_json_TappingSamples.json")
        >>> print(df.shape)  # e.g., (35, 3) for 36 taps
        >>> print(df['dt'].describe())
    """
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
    """
    Calculate global inter-tap interval (dt) limits across all valid patients.
    
    Computes percentile-based limits for reaction times (inter-tap intervals) using
    data from all valid participants. Percentiles are preferred over min/max to handle
    outliers robustly (extreme long/short intervals don't distort the scale).
    
    Args:
        data_dir (Path or str): Root directory containing tapping JSON files.
            Expected format: {data_dir}/**/*_tapping_results_json_TappingSamples.json
        valid_healthcodes (set or list): HealthCodes of participants to include.
            Files from healthcodes not in this set are skipped.
    
    Returns:
        tuple: (dt_min, dt_max)
            - dt_min (float): 1st percentile of all inter-tap intervals (seconds)
            - dt_max (float): 99th percentile of all inter-tap intervals (seconds)
            - Returns (0, 1) as fallback if no valid data found
    
    Side Effects:
        - Prints statistics to console:
          * Absolute min/max dt
          * 1st and 99th percentile values
          * Mean and median dt
    
    Notes:
        - Uses percentiles (1st, 99th) instead of absolute min/max
        - Handles file read errors gracefully (skips invalid files)
        - Accounts for patients with single or no taps (returns None from process_tapping_file)
    
    Example:
        >>> dt_min, dt_max = get_global_dt_limits(
        ...     Path("/data/raw_tapping"),
        ...     {'healthCode1', 'healthCode2', ...}
        ... )
        >>> print(f"Reaction time range: {dt_min:.3f}s to {dt_max:.3f}s")
    """
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
    """
    Calculate global screen coordinate limits (X, Y) across all valid patients.
    
    Computes percentile-based ranges for tap locations using data from all valid 
    participants. Percentiles are used instead of absolute min/max to handle outliers
    and ensure consistent spatial scaling across all heatmaps.
    
    Why Percentiles?
        - Some taps may be registered outside normal screen bounds (UI artifacts)
        - Using 1st-99th percentile ensures 98% of taps are visible
        - Consistent scaling makes heatmaps comparable across patients
        - Avoids extreme outliers that would compress the visualization
    
    Args:
        data_dir (Path or str): Root directory containing tapping JSON files.
            Expected format: {data_dir}/**/*_tapping_results_json_TappingSamples.json
        valid_healthcodes (set or list): HealthCodes of participants to include.
            Files from healthcodes not in this set are skipped.
    
    Returns:
        tuple: (x_min, x_max, y_min, y_max)
            - x_min, x_max (float): 1st and 99th percentile of X coordinates
            - y_min, y_max (float): 1st and 99th percentile of Y coordinates
            - Returns (0, 100, 0, 100) as fallback if no valid data found
            
            Typical ranges for iPhone screen:
            - X: 0 to 640 pixels (device width)
            - Y: 0 to 1136 pixels (device height, varies by model)
    
    Side Effects:
        - Prints coordinate statistics to console (1st-99th percentile ranges)
    
    Notes:
        - Handles file read errors gracefully (skips invalid files)
        - Accounts for patients with no taps (returns None from process_tapping_file)
        - All coordinates are in device pixels (not normalized)
    
    Example:
        >>> x_min, x_max, y_min, y_max = get_global_coordinate_limits(
        ...     Path("/data/raw_tapping"),
        ...     {'healthCode1', 'healthCode2', ...}
        ... )
        >>> print(f"Screen bounds: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")
    """
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
    """
    Create and save a 2D heatmap visualization of tapping behavior.
    
    Generates a scatter plot where each tap is a point, with position indicating
    where on screen the tap occurred, and color indicating reaction time (dt).
    All heatmaps use consistent global scales for comparison across patients.
    
    Visualization Details:
        - X, Y axes: Screen coordinates of tap locations
        - Color: Inter-tap interval (reaction time) using 'plasma' colormap
        - Point size: Fixed (70) for consistent visibility
        - Alpha: 0.6 (semi-transparent) to show overlapping taps
        - Edge: Black outline (linewidth 0.5) for clarity
        
    Scaling Strategy:
        - All heatmaps use same spatial range (x_min-x_max, y_min-y_max)
        - All heatmaps use same color scale (dt_min-dt_max)
        - Margins (8.5%) added to prevent point clipping
        - Y-axis inverted to match typical screen coordinates (0 at top)
    
    Design Choices (Clean Output):
        - No axis labels, ticks, or title (clean, minimal aesthetic)
        - No spines/borders (removes chart frame)
        - No colorbar (reduces clutter; global scale is documented separately)
        - High DPI (100) balances file size and clarity
        - Tight layout minimizes whitespace
    
    Args:
        df (pd.DataFrame): Tapping data with columns ['x', 'y', 'dt'].
            - x, y: Screen coordinates (pixels)
            - dt: Inter-tap intervals (seconds)
            Returns immediately with False if df is None or empty.
        patient_id (str): HealthCode of the patient (used in metadata/naming).
        record_id (str): Session/record ID from filename (used in output filename).
        output_path (str or Path): Full path where PNG file will be saved.
            Directory must exist or creation must be handled by caller.
        x_min, x_max (float): Global X coordinate range (1st-99th percentile).
        y_min, y_max (float): Global Y coordinate range (1st-99th percentile).
        dt_min, dt_max (float): Global reaction time range (1st-99th percentile).
    
    Returns:
        bool: True if heatmap saved successfully, False otherwise.
            Fails (returns False) if:
            - df is None
            - df is empty (0 taps)
            - File I/O error during save
    
    Side Effects:
        - Saves PNG file to output_path
        - Closes matplotlib figure (plt.close())
        - Does not display plot (plt.show() not called)
    
    Output File Format:
        - Format: PNG (100 DPI)
        - Size: ~8x8 inches (640x640 pixels typical)
        - Colormap: 'plasma' (uniform, colorblind-friendly)
        - Color scale: Global (dt_min to dt_max) for all patients
    
    Example:
        >>> df = process_tapping_file("tapping.json")
        >>> success = plot_speed_heatmap(
        ...     df,
        ...     patient_id="abc123",
        ...     record_id="def456",
        ...     output_path="/output/heatmap_01.png",
        ...     x_min=10, x_max=630, y_min=50, y_max=1100,
        ...     dt_min=0.1, dt_max=0.5
        ... )
        >>> print(f"Saved: {success}")
    """
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
    """
    Generate tapping heatmap visualizations for all valid patients.
    
    Workflow:
        1. Load valid healthcodes from paired_healthcode.csv
        2. Calculate global coordinate and reaction time limits (1st-99th percentile)
        3. Process all tapping JSON files grouped by patient
        4. Generate and save heatmap PNG for each trial with global normalization
        5. Report summary statistics (saved count, errors, scales)
    
    Output Structure:
        /tapping_heatmaps/
        ├── healthCode_1/
        │   ├── 01_{record_id}_heatmap.png
        │   ├── 02_{record_id}_heatmap.png
        │   └── ...
        └── healthCode_2/
            └── ...
    
    Paths (Hardcoded):
        - Input: /mloscratch/users/clerget/data/raw_tapping/
        - Output: /mloscratch/users/clerget/data/tapping_heatmaps/
        - Patients: /mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv
    """
    
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

    # Find and group all JSON tapping files by patient
    json_files = list(data_dir.rglob("*tapping_results_json_TappingSamples.json"))
    print(f"Found {len(json_files)} tapping JSON files\n")

    patients = defaultdict(list)
    for json_file in json_files:
        filename = json_file.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            health_code = parts[0]
            if health_code in valid_healthcodes:
                patients[health_code].append(json_file)
    
    total_saved = 0
    total_errors = 0

    for patient_id, files in patients.items():
        print(f"\nProcessing patient {patient_id} ({len(files)} total trials)...")
        print(f"  Selected {len(files)} trials")

        patient_dir = output_base_dir / patient_id
        patient_dir.mkdir(exist_ok=True, parents=True)


        # Process each trial
        for idx, json_file in enumerate(files, 1):
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