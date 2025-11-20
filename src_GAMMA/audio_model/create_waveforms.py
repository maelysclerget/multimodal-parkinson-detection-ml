"""
Script to create normalized waveforms from WAV files

This script:
1. Loads WAV files and extracts the raw audio waveform
2. Normalizes the waveform to [-1, 1] range (standard audio normalization)
3. Saves as .npy files with the same filename
4. Limits each healthCode to maximum 20 recordings (randomly sampled if more)

Normalization details:
- The waveform is normalized to the range [-1.0, 1.0]
- This is done by dividing by the maximum absolute value in the audio
- This removes dependence on bit depth and recording volume
- Makes all audio comparable regardless of original recording levels
"""

import soundfile as sf
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import glob
from collections import defaultdict
import random

def load_and_normalize_waveform(wav_path):
    """
    Load a WAV file and return normalized waveform
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        Normalized waveform array (values in [-1, 1])
        Sample rate
        
    What is stored in the .npy file:
        - A 1D NumPy array of float32 values
        - Each value represents the amplitude at a specific time point
        - Values are normalized to the range [-1.0, 1.0]
        - Array length = duration_seconds × sample_rate
        - Example: 10-second audio at 16000 Hz = array of 160,000 values
        
    Normalization:
        - Waveform is loaded and automatically normalized by soundfile to [-1, 1]
        - Further normalized by dividing by max absolute value
        - All files are mono (single channel)
    """
    # Load audio - soundfile automatically normalizes to [-1, 1]
    # All files are mono, so waveform is a 1D array
    waveform, sample_rate = sf.read(wav_path, dtype='float32')
    
    # Ensure normalization to [-1, 1] range
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    
    return waveform, sample_rate


def parse_filename(filename):
    """
    Parse filename to extract healthCode and record_id
    
    Format: {healthCode}_{record_id}_{column_id}.wav
    Example: 0a76e74d-888a-4c9f-bc44-ddb1f73d64fa_440a0466-aa89-4054-ae6b-1e1436ac1238_audio_audio_m4a.wav
    
    Returns:
        (healthCode, record_id, full_basename)
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    
    if len(parts) >= 2:
        healthCode = parts[0]
        record_id = parts[1]
        return healthCode, record_id, basename
    else:
        # Fallback if format doesn't match
        return None, None, basename


def process_single_wav_example(wav_path, output_dir):
    """
    Example: Process a single WAV file and save normalized waveform
    
    Args:
        wav_path: Path to input WAV file
        output_dir: Directory to save output .npy file
    """
    print(f"Processing single example: {wav_path}")
    
    # Load and normalize
    waveform, sample_rate = load_and_normalize_waveform(wav_path)
    
    # Create output filename
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    output_path = os.path.join(output_dir, f"{basename}.npy")
    
    # Save as numpy array
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, waveform)
    
    print(f"  - Waveform shape: {waveform.shape}")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Duration: {len(waveform)/sample_rate:.2f} seconds")
    print(f"  - Value range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    print(f"  - Saved to: {output_path}")
    
    return waveform, sample_rate


def process_all_wavs_with_limit(input_dir, output_dir, max_records_per_patient=20, seed=42):
    """
    Process all WAV files with limit on recordings per healthCode
    
    Args:
        input_dir: Directory containing WAV files
        output_dir: Directory to save normalized waveforms (.npy)
        max_records_per_patient: Maximum number of records per healthCode (default: 20)
        seed: Random seed for reproducible sampling
    """
    print(f"\n{'='*80}")
    print("Processing all WAV files with patient limits")
    print(f"{'='*80}\n")
    
    random.seed(seed)
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(wav_files)} WAV files\n")
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    # Group files by healthCode
    healthcode_to_files = defaultdict(list)
    files_without_healthcode = []
    
    print("Grouping files by healthCode...")
    for wav_path in tqdm(wav_files):
        filename = os.path.basename(wav_path)
        healthCode, record_id, basename = parse_filename(filename)
        
        if healthCode and record_id:
            healthcode_to_files[healthCode].append({
                'path': wav_path,
                'record_id': record_id,
                'basename': basename
            })
        else:
            files_without_healthcode.append({
                'path': wav_path,
                'basename': basename
            })
    
    # Statistics
    print(f"\nGrouping results:")
    print(f"  - Unique healthCodes: {len(healthcode_to_files)}")
    print(f"  - Files without proper format: {len(files_without_healthcode)}")
    
    # Show distribution
    record_counts = [len(files) for files in healthcode_to_files.values()]
    if record_counts:
        print(f"\nRecords per healthCode:")
        print(f"  - Mean: {np.mean(record_counts):.1f}")
        print(f"  - Median: {np.median(record_counts):.1f}")
        print(f"  - Min: {min(record_counts)}")
        print(f"  - Max: {max(record_counts)}")
        print(f"  - HealthCodes with >{max_records_per_patient} records: {sum(1 for c in record_counts if c > max_records_per_patient)}")
    
    # Sample files if needed
    files_to_process = []
    sampled_healthcodes = 0
    
    for healthCode, file_list in healthcode_to_files.items():
        if len(file_list) > max_records_per_patient:
            # Randomly sample max_records_per_patient records
            sampled = random.sample(file_list, max_records_per_patient)
            files_to_process.extend(sampled)
            sampled_healthcodes += 1
            if sampled_healthcodes <= 5:  # Show first 5 examples
                print(f"  HealthCode {healthCode[:8]}... has {len(file_list)} records → sampling {max_records_per_patient}")
        else:
            # Use all records
            files_to_process.extend(file_list)
    
    if sampled_healthcodes > 5:
        print(f"  ... and {sampled_healthcodes - 5} more healthCodes were sampled")
    
    # Add files without healthCode (process all of them)
    files_to_process.extend(files_without_healthcode)
    
    print(f"\nFiles to process after sampling: {len(files_to_process)}")
    print(f"Files excluded due to sampling: {len(wav_files) - len(files_to_process)}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    print("Processing and saving waveforms...")
    successful = 0
    failed = 0
    
    for file_info in tqdm(files_to_process):
        try:
            wav_path = file_info['path']
            basename = file_info['basename']
            
            # Load and normalize
            waveform, sample_rate = load_and_normalize_waveform(wav_path)
            
            # Save as .npy
            output_path = os.path.join(output_dir, f"{basename}.npy")
            np.save(output_path, waveform)
            
            successful += 1
            
        except Exception as e:
            failed += 1
            print(f"\nError processing {os.path.basename(wav_path)}: {str(e)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    return successful, failed


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/mloscratch/users/gnahas/data/wav"
    OUTPUT_DIR = "/mloscratch/users/gnahas/data/waveform_norm"
    MAX_RECORDS_PER_PATIENT = 20
    
    print("="*80)
    print("WAV to Normalized Waveform Converter")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Input directory: {INPUT_DIR}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Max records per patient: {MAX_RECORDS_PER_PATIENT}")
    print(f"\nNormalization: Waveforms normalized to [-1.0, 1.0] range")
    print(f"Output format: .npy files (NumPy arrays)")
    print()
    
    # Example: Process one file first (if files exist)
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    if not wav_files:
        wav_files = glob.glob(os.path.join(INPUT_DIR, "**/*.wav"), recursive=True)
    
    if wav_files:
        print("\n" + "="*80)
        print("EXAMPLE: Processing single WAV file")
        print("="*80 + "\n")
        example_wav = wav_files[0]
        process_single_wav_example(example_wav, OUTPUT_DIR)
        print()
    
    # Uncomment the line below to process all files
    # (Comment it out to only process the single example file)
    process_all_wavs_with_limit(INPUT_DIR, OUTPUT_DIR, MAX_RECORDS_PER_PATIENT)
    