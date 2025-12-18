
"""
Script to create normalized waveforms from WAV files for downstream ML tasks.

Overview:
---------
This script processes a directory of WAV audio files, normalizes each waveform to the range [-1.0, 1.0], optionally trims silence from the beginning and end, and saves the result as a .npy file with the same base filename. It is designed for large-scale audio datasets, such as those used in biomedical or speech research, and ensures all audio is comparable regardless of original recording levels, bit depth, or volume.

Features:
- Loads WAV files (mono, float32) and extracts the raw audio waveform.
- Normalizes each waveform to [-1, 1] by dividing by the maximum absolute value.
- Optionally trims silence from the start/end using an energy threshold in dB.
- Saves each waveform as a .npy file in the output directory, preserving the base filename.
- Groups files by patient (healthCode) for statistics.
- Example and batch processing modes included.

Expected file structure:
- Input directory: Contains .wav files named as {healthCode}_{record_id}_{column_id}.wav
- Output directory: Will be created if it does not exist; .npy files will be saved here.

Usage:
------
Set INPUT_DIR and OUTPUT_DIR in the main block. Run the script to process all files. Adjust silence trimming and threshold as needed.
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
    Trim silence from the beginning and end of the waveform using an energy threshold.

    Args:
        waveform (np.ndarray): Input audio waveform (1D float32 array).
        threshold_db (float): Threshold in dB below which audio is considered silence (default: -40).
        frame_length (int): Frame size for energy calculation (default: 2048).
        hop_length (int): Hop size between frames (default: 512).

    Returns:
        np.ndarray: Trimmed waveform (1D float32 array).
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
    Load a WAV file, optionally trim silence, and normalize the waveform to [-1, 1].

    Args:
        wav_path (str or Path): Path to the input WAV file.
        trim_silence_flag (bool): Whether to trim silence from beginning/end (default: True).
        silence_threshold_db (float): Threshold in dB for silence detection (default: -40).

    Returns:
        tuple: (waveform, sample_rate)
            waveform (np.ndarray): 1D float32 array, normalized to [-1, 1].
            sample_rate (int): Sample rate of the audio file.

    What is stored in the .npy file:
        - 1D NumPy array of float32 values (amplitude at each time point)
        - Values normalized to [-1.0, 1.0]
        - Array length = duration_seconds × sample_rate
        - Example: 10-second audio at 16000 Hz = array of 160,000 values

    Note:
        - All files are assumed mono (single channel).
        - Windowing/stride is NOT performed here; apply during model training.
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
    Parse filename to extract healthCode and record_id.

    Args:
        filename (str): Filename of the WAV file (not full path).

    Format:
        {healthCode}_{record_id}_{column_id}.wav
        Example: 0a76e74d-888a-4c9f-bc44-ddb1f73d64fa_440a0466-aa89-4054-ae6b-1e1436ac1238_audio_audio_m4a.wav

    Returns:
        tuple: (healthCode, record_id, full_basename)
            healthCode (str or None): Patient identifier, or None if not found.
            record_id (str or None): Recording identifier, or None if not found.
            full_basename (str): Filename without extension.
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
    Process a single WAV file: load, normalize, and save as .npy.

    Args:
        wav_path (str or Path): Path to input WAV file.
        output_dir (str or Path): Directory to save output .npy file. Will be created if it does not exist.

    Returns:
        tuple: (waveform, sample_rate)
            waveform (np.ndarray): 1D float32 array, normalized to [-1, 1].
            sample_rate (int): Sample rate of the audio file.
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
    Process all WAV files in a directory, normalize, and save as .npy files.

    Args:
        input_dir (str or Path): Directory containing WAV files (searches recursively).
        output_dir (str or Path): Directory to save normalized waveforms (.npy). Will be created if it does not exist.
        max_records_per_patient (int or None): Not used (kept for compatibility; all files processed).
        seed (int): Random seed (not used when no sampling).
        trim_silence (bool): Whether to trim silence from audio (default: True).
        silence_threshold_db (float): Threshold in dB for silence detection (default: -40).

    Returns:
        tuple: (successful, failed)
            successful (int): Number of files processed successfully.
            failed (int): Number of files that failed to process.
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
    