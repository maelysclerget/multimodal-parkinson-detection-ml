"""
Check WAV File Durations and Identify Outliers

This script analyzes the durations of all WAV files in a directory, prints summary statistics, and identifies files that are unusually short (below a specified threshold). It is useful for quality control and for detecting potentially corrupted or invalid audio recordings before feature extraction or model training.

Context:
--------
Duration analysis is important for ensuring that all audio files are suitable for downstream processing. Very short files may indicate failed or incomplete recordings, while a large number of short files may bias the dataset. This script helps flag such issues early in the pipeline.

Usage:
------
Run as a script to print duration statistics and lists of short or problematic files. Adjust the WAV_DIR and DURATION_THRESHOLD as needed for your dataset.
"""

import soundfile as sf
import glob
import os
from tqdm import tqdm

WAV_DIR = "/mloscratch/users/gnahas/data/wav"
DURATION_THRESHOLD = 5.0  # seconds

print("="*80)
print("Checking WAV file durations")
print("="*80)

# Find all WAV files
wav_files = glob.glob(os.path.join(WAV_DIR, "*.wav"))
print(f"\nFound {len(wav_files)} WAV files")
print(f"Threshold: {DURATION_THRESHOLD} seconds\n")

# Check durations
durations = []
short_files = []
errors = []

print("Analyzing durations...")
for wav_path in tqdm(wav_files):
    try:
        info = sf.info(wav_path)
        duration = info.duration
        durations.append(duration)
        
        if duration < DURATION_THRESHOLD:
            short_files.append({
                'filename': os.path.basename(wav_path),
                'duration': duration
            })
    except Exception as e:
        errors.append({
            'filename': os.path.basename(wav_path),
            'error': str(e)
        })

# Statistics
print(f"\n{'='*80}")
print("DURATION STATISTICS")
print(f"{'='*80}")

if durations:
    import numpy as np
    durations = np.array(durations)
    
    print(f"\nTotal files analyzed: {len(durations)}")
    print(f"Mean duration: {durations.mean():.2f} seconds")
    print(f"Median duration: {np.median(durations):.2f} seconds")
    print(f"Min duration: {durations.min():.2f} seconds")
    print(f"Max duration: {durations.max():.2f} seconds")
    print(f"Std deviation: {durations.std():.2f} seconds")
    
    # Percentiles
    print(f"\nPercentiles:")
    print(f"  1st percentile: {np.percentile(durations, 1):.2f} seconds")
    print(f"  5th percentile: {np.percentile(durations, 5):.2f} seconds")
    print(f"  25th percentile: {np.percentile(durations, 25):.2f} seconds")
    print(f"  75th percentile: {np.percentile(durations, 75):.2f} seconds")
    print(f"  95th percentile: {np.percentile(durations, 95):.2f} seconds")
    print(f"  99th percentile: {np.percentile(durations, 99):.2f} seconds")

# Short files
print(f"\n{'='*80}")
print(f"FILES WITH DURATION < {DURATION_THRESHOLD} SECONDS")
print(f"{'='*80}")
print(f"\nCount: {len(short_files)} files ({len(short_files)/len(wav_files)*100:.2f}%)")

if short_files:
    print(f"\nFirst 20 short files:")
    for item in sorted(short_files, key=lambda x: x['duration'])[:20]:
        print(f"  {item['filename']}: {item['duration']:.2f} seconds")
    
    if len(short_files) > 20:
        print(f"\n  ... and {len(short_files) - 20} more")

# Errors
if errors:
    print(f"\n{'='*80}")
    print(f"FILES WITH ERRORS")
    print(f"{'='*80}")
    print(f"\nCount: {len(errors)}")
    for item in errors[:10]:
        print(f"  {item['filename']}: {item['error']}")

# Recommendations
print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")

if len(durations) > 0:
    very_short = sum(d < 1.0 for d in durations)
    short = sum(1.0 <= d < DURATION_THRESHOLD for d in durations)
    
    print(f"\nDuration breakdown:")
    print(f"  < 1 second: {very_short} files ({very_short/len(durations)*100:.2f}%)")
    print(f"  1-{DURATION_THRESHOLD} seconds: {short} files ({short/len(durations)*100:.2f}%)")
    print(f"  >= {DURATION_THRESHOLD} seconds: {len(durations)-very_short-short} files ({(len(durations)-very_short-short)/len(durations)*100:.2f}%)")
    
    if very_short > 0:
        print(f"\n⚠️  Warning: {very_short} files are extremely short (< 1 second)")
        print(f"   Consider excluding these as they may be corrupted or invalid recordings")
    
    if short > 0:
        print(f"\n⚠️  Note: {short} files are short ({DURATION_THRESHOLD} seconds or less)")
        print(f"   These may be valid but consider if they provide enough data for analysis")

print(f"\n{'='*80}\n")
