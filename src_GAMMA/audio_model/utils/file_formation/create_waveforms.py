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

def trim_silence(waveform, threshold_db=-40, frame_length=2048, hop_length=512):
    """
    Trim silence from the beginning and end of the waveform
    
    Args:
        waveform: Input audio waveform
        threshold_db: Threshold in dB below which audio is considered silence
        frame_length: Frame size for energy calculation
        hop_length: Hop size between frames
        
    Returns:
        Trimmed waveform
    """
    # Convert threshold from dB to amplitude
    threshold = 10 ** (threshold_db / 20)
    
    # Calculate energy for each frame
    energy = np.array([
        np.sqrt(np.mean(waveform[i:i+frame_length]**2))
        for i in range(0, len(waveform) - frame_length + 1, hop_length)
    ])
    
    # Find non-silent frames
    non_silent = energy > threshold
    
    if not np.any(non_silent):
        # If all frames are silent, return original waveform
        return waveform
    
    # Find first and last non-silent frame
    non_silent_indices = np.where(non_silent)[0]
    start_frame = non_silent_indices[0]
    end_frame = non_silent_indices[-1]
    
    # Convert frame indices to sample indices
    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length + frame_length, len(waveform))
    
    return waveform[start_sample:end_sample]


def load_and_normalize_waveform(wav_path, trim_silence_flag=True, silence_threshold_db=-40):
    """
    Load a WAV file and return normalized waveform with silence trimming
    
    Args:
        wav_path: Path to WAV file
        trim_silence_flag: Whether to trim silence from beginning/end
        silence_threshold_db: Threshold in dB for silence detection (default: -40)
        
    Returns:
        Normalized waveform array (values in [-1, 1])
        Sample rate
        
    What is stored in the .npy file:
        - A 1D NumPy array of float32 values
        - Each value represents the amplitude at a specific time point
        - Values are normalized to the range [-1.0, 1.0]
        - Array length = duration_seconds × sample_rate
        - Example: 10-second audio at 16000 Hz = array of 160,000 values
        
    Normalization steps:
        1. Load waveform (soundfile normalizes to [-1, 1])
        2. Trim silence from beginning and end (optional)
        3. Normalize amplitude to [-1, 1] range
        - All files are mono (single channel)
        
    Note on windowing/stride:
        - NOT done in this file
        - These are hyperparameters for your model's dataloader
        - Apply windowing when loading .npy files during training
    """
    # Load audio - soundfile automatically normalizes to [-1, 1]
    # All files are mono, so waveform is a 1D array
    waveform, sample_rate = sf.read(wav_path, dtype='float32')
    
    # Trim silence if enabled
    if trim_silence_flag:
        waveform = trim_silence(waveform, threshold_db=silence_threshold_db)
    
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


def process_all_wavs_with_limit(input_dir, output_dir, max_records_per_patient=None, seed=42, 
                                trim_silence=True, silence_threshold_db=-40):
    """
    Process all WAV files (no limit on recordings per healthCode)
    
    Args:
        input_dir: Directory containing WAV files
        output_dir: Directory to save normalized waveforms (.npy)
        max_records_per_patient: Not used (kept for compatibility)
        seed: Random seed (not used when no sampling)
        trim_silence: Whether to trim silence from audio
        silence_threshold_db: Threshold in dB for silence detection
    """
    print(f"\n{'='*80}")
    print("Processing ALL WAV files (no patient limits)")
    print(f"Silence trimming: {'ENABLED' if trim_silence else 'DISABLED'}")
    if trim_silence:
        print(f"Silence threshold: {silence_threshold_db} dB")
    print(f"{'='*80}\n")
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(wav_files)} WAV files\n")
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    # Group files by healthCode for statistics only
    healthcode_to_files = defaultdict(list)
    files_without_healthcode = []
    
    print("Grouping files by healthCode (for statistics)...")
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
    
    # Process ALL files (no sampling)
    files_to_process = []
    
    for healthCode, file_list in healthcode_to_files.items():
        # Use all records for every patient
        files_to_process.extend(file_list)
    
    # Add files without healthCode (process all of them)
    files_to_process.extend(files_without_healthcode)
    
    print(f"\nProcessing all {len(files_to_process)} files (no sampling)\n")
    
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
            waveform, sample_rate = load_and_normalize_waveform(
                wav_path, 
                trim_silence_flag=trim_silence,
                silence_threshold_db=silence_threshold_db
            )
            
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
    OUTPUT_DIR = "/mloscratch/users/gnahas/data/waveform_norm_silence_trimmed"
    
    # Silence trimming parameters
    TRIM_SILENCE = True  # Set to False to disable silence trimming
    SILENCE_THRESHOLD_DB = -40  # Adjust threshold: -40 (moderate), -30 (aggressive), -50 (conservative)
    
    print("="*80)
    print("WAV to Normalized Waveform Converter")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Input directory: {INPUT_DIR}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Processing: ALL files (no limit per patient)")
    print(f"  - Silence trimming: {'ENABLED' if TRIM_SILENCE else 'DISABLED'}")
    if TRIM_SILENCE:
        print(f"  - Silence threshold: {SILENCE_THRESHOLD_DB} dB")
    print(f"\nNormalization: Waveforms normalized to [-1.0, 1.0] range")
    print(f"Output format: .npy files (NumPy arrays)")
    print(f"\nNote: Windowing & stride are NOT in this file.")
    print(f"      Add them as hyperparameters in your model's dataloader/dataset class.")
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
    process_all_wavs_with_limit(INPUT_DIR, OUTPUT_DIR, 
                                trim_silence=TRIM_SILENCE,
                                silence_threshold_db=SILENCE_THRESHOLD_DB)
    