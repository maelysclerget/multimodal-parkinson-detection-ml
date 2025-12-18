
"""
inspect_all_wav.py
-------------------
Script to inspect all WAV files in a directory.
Generates a CSV report and summary statistics on:
    - Length (duration) uniformity
    - Sampling rate uniformity
    - Channels
    - File sizes
    - Bit depth / format

Functions:
    inspect_wav_file_quick(file_path):
        Quickly inspects a WAV file and returns key properties.
    analyze_wav_directory(directory, pattern="**/*.wav", max_files=None):
        Analyzes all WAV files in a directory, prints summary statistics, and saves reports.
"""

import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import glob
from pathlib import Path

def inspect_wav_file_quick(file_path):
    """
    Quickly inspects a WAV file and returns key properties.

    Args:
        file_path (str): Path to the WAV file.

    Returns:
        dict: Dictionary containing file properties:
            - filename (str): Name of the file
            - filepath (str): Full path to the file
            - duration_sec (float or None): Duration in seconds
            - sampling_rate (int or None): Sampling rate in Hz
            - channels (int or None): Number of channels
            - frames (int or None): Number of frames
            - subtype (str or None): Bit depth/format subtype
            - format (str or None): File format
            - file_size_mb (float or None): File size in MB
            - status (str): 'success' or error message
    """
    try:
        # Using soundfile for wav files
        info = sf.info(file_path)
        
        return {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'duration_sec': info.duration,
            'sampling_rate': info.samplerate,
            'channels': info.channels,
            'frames': info.frames,
            'subtype': info.subtype,
            'format': info.format,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'status': 'success'
        }
    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'duration_sec': None,
            'sampling_rate': None,
            'channels': None,
            'frames': None,
            'subtype': None,
            'format': None,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else None,
            'status': f'error: {str(e)}'
        }


def analyze_wav_directory(directory, pattern="**/*.wav", max_files=None):
    """
    Analyze all WAV files in a directory, print summary statistics, and save detailed and summary CSV reports.

    Args:
        directory (str): Path to directory containing WAV files.
        pattern (str, optional): Glob pattern for finding WAV files (default: "**/*.wav").
        max_files (int, optional): Maximum number of files to process (None = all files).

    Returns:
        pandas.DataFrame: DataFrame containing inspection results for each file.
    """
    print(f"Searching for WAV files in: {directory}")
    print(f"Pattern: {pattern}")
    
    # Find all WAV files
    search_path = os.path.join(directory, pattern)
    wav_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(wav_files)} WAV files")
    
    if max_files:
        wav_files = wav_files[:max_files]
        print(f"Processing first {max_files} files")
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    # Process each file
    results = []
    print("\nProcessing files...")
    for file_path in tqdm(wav_files):
        result = inspect_wav_file_quick(file_path)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal files processed: {len(df)}")
    print(f"Successful: {(df['status'] == 'success').sum()}")
    print(f"Failed: {(df['status'] != 'success').sum()}")
    
    if (df['status'] == 'success').sum() > 0:
        df_success = df[df['status'] == 'success']
        
        print("\n--- Duration Statistics ---")
        print(f"Mean duration: {df_success['duration_sec'].mean():.2f} seconds ({df_success['duration_sec'].mean()/60:.2f} minutes)")
        print(f"Std deviation: {df_success['duration_sec'].std():.2f} seconds")
        print(f"Min duration: {df_success['duration_sec'].min():.2f} seconds")
        print(f"Max duration: {df_success['duration_sec'].max():.2f} seconds")
        print(f"Median duration: {df_success['duration_sec'].median():.2f} seconds")
        
        print("\n--- Sampling Rate Statistics ---")
        unique_rates = df_success['sampling_rate'].unique()
        print(f"Unique sampling rates: {sorted(unique_rates)}")
        for rate in sorted(unique_rates):
            count = (df_success['sampling_rate'] == rate).sum()
            percentage = (count / len(df_success)) * 100
            print(f"  {rate} Hz: {count} files ({percentage:.1f}%)")
        
        print("\n--- Channel Statistics ---")
        unique_channels = df_success['channels'].unique()
        for ch in sorted(unique_channels):
            count = (df_success['channels'] == ch).sum()
            percentage = (count / len(df_success)) * 100
            ch_type = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch}-channel'
            print(f"  {ch_type}: {count} files ({percentage:.1f}%)")
        
        print("\n--- Format Statistics ---")
        unique_formats = df_success['subtype'].unique()
        for fmt in unique_formats:
            count = (df_success['subtype'] == fmt).sum()
            percentage = (count / len(df_success)) * 100
            print(f"  {fmt}: {count} files ({percentage:.1f}%)")
        
        print("\n--- File Size Statistics ---")
        print(f"Mean file size: {df_success['file_size_mb'].mean():.2f} MB")
        print(f"Total size: {df_success['file_size_mb'].sum():.2f} MB ({df_success['file_size_mb'].sum()/1024:.2f} GB)")
        print(f"Min file size: {df_success['file_size_mb'].min():.2f} MB")
        print(f"Max file size: {df_success['file_size_mb'].max():.2f} MB")
    
    # Check for uniformity
    print("\n" + "="*80)
    print("UNIFORMITY CHECKS")
    print("="*80)
    
    if (df['status'] == 'success').sum() > 0:
        df_success = df[df['status'] == 'success']
        
        # Duration uniformity
        duration_std = df_success['duration_sec'].std()
        duration_cv = duration_std / df_success['duration_sec'].mean() if df_success['duration_sec'].mean() > 0 else 0
        print(f"\nDuration uniformity:")
        print(f"  Coefficient of variation: {duration_cv:.2%}")
        if duration_cv < 0.01:
            print(f"  ✓ Highly uniform durations")
        elif duration_cv < 0.1:
            print(f"  ✓ Moderately uniform durations")
        else:
            print(f"  ✗ Variable durations (CV > 10%)")
        
        # Sampling rate uniformity
        unique_rates = len(df_success['sampling_rate'].unique())
        print(f"\nSampling rate uniformity:")
        if unique_rates == 1:
            print(f"  ✓ All files have the same sampling rate: {df_success['sampling_rate'].iloc[0]} Hz")
        else:
            print(f"  ✗ Multiple sampling rates found ({unique_rates} different rates)")
        
        # Channel uniformity
        unique_channels = len(df_success['channels'].unique())
        print(f"\nChannel uniformity:")
        if unique_channels == 1:
            ch = df_success['channels'].iloc[0]
            ch_type = 'Mono' if ch == 1 else 'Stereo' if ch == 2 else f'{ch}-channel'
            print(f"  ✓ All files have the same channel configuration: {ch_type}")
        else:
            print(f"  ✗ Multiple channel configurations found")
    
    # Save detailed report to CSV
    output_dir = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model"
    output_file = os.path.join(output_dir, "wav_inspection_report.csv")
    df.to_csv(output_file, index=False)
    
    # Also save summary statistics to a separate CSV
    summary_file = os.path.join(output_dir, "wav_inspection_summary.csv")
    if (df['status'] == 'success').sum() > 0:
        df_success = df[df['status'] == 'success']
        
        summary_data = {
            'Metric': [
                'Total Files',
                'Successful',
                'Failed',
                'Mean Duration (sec)',
                'Std Duration (sec)',
                'Min Duration (sec)',
                'Max Duration (sec)',
                'Median Duration (sec)',
                'Duration CV (%)',
                'Unique Sampling Rates',
                'Most Common Sampling Rate (Hz)',
                'Unique Channels',
                'Most Common Channels',
                'Mean File Size (MB)',
                'Total Size (GB)'
            ],
            'Value': [
                len(df),
                (df['status'] == 'success').sum(),
                (df['status'] != 'success').sum(),
                f"{df_success['duration_sec'].mean():.2f}",
                f"{df_success['duration_sec'].std():.2f}",
                f"{df_success['duration_sec'].min():.2f}",
                f"{df_success['duration_sec'].max():.2f}",
                f"{df_success['duration_sec'].median():.2f}",
                f"{duration_cv * 100:.2f}",
                len(df_success['sampling_rate'].unique()),
                int(df_success['sampling_rate'].mode()[0]) if len(df_success) > 0 else 'N/A',
                len(df_success['channels'].unique()),
                int(df_success['channels'].mode()[0]) if len(df_success) > 0 else 'N/A',
                f"{df_success['file_size_mb'].mean():.2f}",
                f"{df_success['file_size_mb'].sum()/1024:.2f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
    
    print(f"\n" + "="*80)
    print(f"Detailed report saved to: {output_file}")
    print(f"Summary statistics saved to: {summary_file}")
    print("="*80 + "\n")
    
    # Show files with errors if any
    df_errors = df[df['status'] != 'success']
    if len(df_errors) > 0:
        print("\nFiles with errors:")
        for idx, row in df_errors.iterrows():
            print(f"  - {row['filename']}: {row['status']}")
    
    return df


if __name__ == "__main__":
    """
    Command-line interface for inspecting all WAV files in a directory.
    Usage:
        python inspect_all_wav.py [directory] [max_files]
    Args:
        directory (str, optional): Directory to search for WAV files. Defaults to '/mloscratch/users/gnahas/data/wav'.
        max_files (int, optional): Maximum number of files to process. Defaults to None (all files).
    """
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "/mloscratch/users/gnahas/data/wav"

    max_files = None
    if len(sys.argv) > 2:
        max_files = int(sys.argv[2])

    analyze_wav_directory(directory, max_files=max_files)
