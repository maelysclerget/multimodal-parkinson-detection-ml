
"""
Script to extract acoustic features from normalized waveform .npy files.

Overview:
---------
This script processes a directory of normalized waveform .npy files (one per audio recording), extracts a comprehensive set of acoustic features optimized for sustained phonation tasks (e.g., 10-second "AHHH" sounds), and saves the results as a CSV file (one row per recording).

Features extracted (per recording):
- MFCCs (Mel-frequency cepstral coefficients): 26 features (mean/std of 13 coefficients)
- Spectral characteristics: 8 features (centroid, bandwidth, rolloff, ZCR; mean/std)
- Pitch features: 4 features (F0 mean, std, range, variation)
- Voice quality: 6 features (jitter, shimmer, HNR)
- Energy dynamics: 13 features (variation, entropy, range, roughness, stability, tremor, interruptions, hachuré index, etc.)

Key capabilities for Parkinson's detection in sustained vowels:
- Vocal tremor: frequency and magnitude
- Voice instability: jitter, shimmer, roughness
- Choppy/fragmented voice (hachuré): combined index measuring trembling + choppiness
- Voice interruptions: detects stop/start events and their duration
- Voice quality: breathiness (HNR), hoarseness
- Pitch control: stability and variation

Expected input:
- Input directory: Contains .npy files (normalized waveforms, 1D float32 arrays)
- Each .npy file is named as <healthcode>_<recordid>_...npy

Output:
- CSV file with one row per recording, columns for all features and metadata

Usage:
------
Run as a script with --input and --output arguments, or use the functions in your own pipeline.
"""

import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import os
from pathlib import Path
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings('ignore')


def extract_mfcc_features(waveform, sr):
    """
    Extract MFCC (Mel-frequency cepstral coefficient) features from a waveform.

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: Keys are mfcc_1_mean, mfcc_1_std, ..., mfcc_13_std (26 features).
    """
    features = {}
    
    # Extract 13 MFCCs
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    
    # Compute mean and std for each coefficient
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    return features


def extract_spectral_features(waveform, sr):
    """
    Extract spectral features: centroid, bandwidth, rolloff, and zero-crossing rate.

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: 8 features (mean and std of each spectral property).
    """
    features = {}
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(centroid)
    features['spectral_centroid_std'] = np.std(centroid)
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)
    features['spectral_bandwidth_std'] = np.std(bandwidth)
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_rolloff_std'] = np.std(rolloff)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(waveform)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    return features


def extract_prosody_features(waveform, sr):
    """
    Extract pitch features using Parselmouth (Praat).
    For sustained phonation ("AHHH"), focuses on pitch stability.

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: pitch_mean, pitch_std, pitch_range, pitch_variation_coef (4 features).
    """
    features = {}
    
    try:
        
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(waveform, sampling_frequency=sr)
        
        # Extract pitch (F0)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            features['pitch_variation_coef'] = np.std(pitch_values) / (np.mean(pitch_values) + 1e-8)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
            features['pitch_variation_coef'] = 0
        
    except Exception as e:
        # Fallback values if Parselmouth fails
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_range'] = 0
        features['pitch_variation_coef'] = 0
    
    return features


def extract_voice_quality_features(waveform, sr):
    """
    Extract voice quality features: jitter, shimmer, and HNR (harmonic-to-noise ratio).

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: jitter_local, jitter_rap, shimmer_local, shimmer_apq3, hnr_mean, hnr_std (6 features).
    """
    features = {}
    
    try:
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(waveform, sampling_frequency=sr)
        
        # Extract PointProcess for jitter/shimmer
        pitch = sound.to_pitch()
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        # Jitter (local, relative)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features['jitter_local'] = jitter_local if not np.isnan(jitter_local) else 0
        features['jitter_rap'] = jitter_rap if not np.isnan(jitter_rap) else 0
        
        # Shimmer (local, apq3)
        shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['shimmer_local'] = shimmer_local if not np.isnan(shimmer_local) else 0
        features['shimmer_apq3'] = shimmer_apq3 if not np.isnan(shimmer_apq3) else 0
        
        # Harmonic-to-Noise Ratio
        harmonicity = sound.to_harmonicity()
        hnr_mean = call(harmonicity, "Get mean", 0, 0)
        hnr_std = call(harmonicity, "Get standard deviation", 0, 0)
        features['hnr_mean'] = hnr_mean if not np.isnan(hnr_mean) else 0
        features['hnr_std'] = hnr_std if not np.isnan(hnr_std) else 0
        
    except Exception as e:
        # Fallback values
        features['jitter_local'] = 0
        features['jitter_rap'] = 0
        features['shimmer_local'] = 0
        features['shimmer_apq3'] = 0
        features['hnr_mean'] = 0
        features['hnr_std'] = 0
    
    return features


def extract_energy_features(waveform, sr):
    """
    Extract energy dynamics features from a waveform.

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: 13 features (energy variation, entropy, range, roughness, instability, tremor, interruptions, hachuré index, etc.).
    """
    features = {}
    
    # Frame-wise energy (RMS)
    rms = librosa.feature.rms(y=waveform)[0]
    
    # Energy variation (coefficient of variation)
    mean_energy = np.mean(rms)
    std_energy = np.std(rms)
    features['energy_variation'] = std_energy / (mean_energy + 1e-8)
    
    # Dynamic range (max/percentile ratio to avoid outliers)
    max_energy = np.max(rms)
    min_energy = np.percentile(rms, 10)
    features['dynamic_range'] = max_energy / (min_energy + 1e-8)
    
    # Energy entropy
    # Normalize energy to probability distribution
    energy_prob = rms / (np.sum(rms) + 1e-8)
    energy_prob = energy_prob[energy_prob > 0]  # Remove zeros
    features['energy_entropy'] = -np.sum(energy_prob * np.log2(energy_prob + 1e-8))
    
    # Low-energy frame ratio
    threshold = np.percentile(rms, 10)
    low_energy_frames = np.sum(rms < threshold)
    features['low_energy_frame_ratio'] = low_energy_frames / len(rms)
    
    # Vocal roughness/choppiness features 
    # Frame-to-frame energy changes (how choppy/unstable the sound is)
    rms_diff = np.diff(rms)  # Changes between consecutive frames
    features['energy_instability'] = np.std(rms_diff)  # How much energy jumps around
    features['energy_roughness'] = np.mean(np.abs(rms_diff))  # Average magnitude of changes
    
    # Number of abrupt energy changes (choppy voice detection)
    # Count frames where energy changes by more than 2 standard deviations
    threshold_change = 2 * np.std(rms_diff)
    abrupt_changes = np.sum(np.abs(rms_diff) > threshold_change)
    features['abrupt_changes_ratio'] = abrupt_changes / len(rms_diff)
    
    # Voice breaks: frames where energy drops significantly
    # (indicates voice cutting out)
    energy_drops = np.sum(rms < (mean_energy * 0.5))
    features['voice_breaks_ratio'] = energy_drops / len(rms)
    
    # Voice interruptions - counts stop/start events
    # Detects how many times voice drops then recovers (actual interruptions)
    silence_threshold = mean_energy * 0.3  # Below 30% of mean = silence/very weak
    is_silent = rms < silence_threshold
    
    # Count transitions from voice -> silence -> voice
    # This is more specific than just counting low-energy frames
    voice_to_silence = np.diff(is_silent.astype(int))  # 1 = became silent, -1 = became voiced
    silence_starts = np.sum(voice_to_silence == 1)  # Number of times voice stopped
    silence_ends = np.sum(voice_to_silence == -1)    # Number of times voice restarted
    
    features['num_interruptions'] = min(silence_starts, silence_ends)  # Complete interruption cycles
    
    # Average duration of interruptions (in frames)
    if features['num_interruptions'] > 0:
        
        # Find continuous silent segments
        silent_segments = []
        in_silence = False
        silence_length = 0
        
        for silent in is_silent:
            if silent:
                if not in_silence:
                    in_silence = True
                    silence_length = 1
                else:
                    silence_length += 1
            else:
                if in_silence:
                    silent_segments.append(silence_length)
                    in_silence = False
                    silence_length = 0
        
        if silent_segments:
            avg_interruption_length = np.mean(silent_segments)
            # Convert to seconds (frame_rate = sr/512 for librosa RMS)
            frame_duration = 512 / sr  # seconds per frame
            features['avg_interruption_duration'] = avg_interruption_length * frame_duration
        else:
            features['avg_interruption_duration'] = 0
    else:
        features['avg_interruption_duration'] = 0
    
    # Tremor rate/speed features
    # Autocorrelation of energy to find dominant oscillation frequency
    if len(rms) > 10:
        # Detrend the signal (remove overall trend)
        rms_detrended = rms - np.mean(rms)
        
        # Autocorrelation
        autocorr = np.correlate(rms_detrended, rms_detrended, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first peak after lag 0 (dominant periodicity)
        # Look for peaks in autocorrelation
        if len(autocorr) > 3:
            peaks = []
            for i in range(2, min(len(autocorr), 100)):  # Look up to ~2 seconds
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.1:  # Only significant peaks
                        peaks.append((i, autocorr[i]))
            
            if peaks:
                # Dominant tremor period (in frames)
                dominant_lag = peaks[0][0]
                # Convert to Hz (tremor frequency)
                # Frame hop length in librosa.feature.rms is 512 samples by default
                frame_rate = sr / 512
                features['tremor_frequency'] = frame_rate / dominant_lag
            else:
                features['tremor_frequency'] = 0
        else:
            features['tremor_frequency'] = 0
    else:
        features['tremor_frequency'] = 0
    
    # Zero-crossing rate of energy signal (how often it oscillates around mean)
    # High ZCR of energy = rapid fluctuations
    energy_centered = rms - np.mean(rms)
    energy_zcr = np.sum(np.diff(np.sign(energy_centered)) != 0) / len(rms)
    features['energy_oscillation_rate'] = energy_zcr
    
    # Combines tremor frequency with roughness and abrupt changes
    # High value = voice is both trembling AND choppy
    if features['tremor_frequency'] > 0:
        # Normalize components to 0-1 range for combination
        tremor_component = min(features['tremor_frequency'] / 10.0, 1.0)  # 10 Hz max expected
        roughness_component = min(features['energy_roughness'] * 100, 1.0)  # Scale roughness
        abrupt_component = features['abrupt_changes_ratio']
        
        # Combined index: weighted average
        features['hachure_index'] = (0.4 * tremor_component + 
                                     0.3 * roughness_component + 
                                     0.3 * abrupt_component)
    else:
        # No tremor detected, but still can be choppy
        features['hachure_index'] = (0.5 * min(features['energy_roughness'] * 100, 1.0) + 
                                     0.5 * features['abrupt_changes_ratio'])
    
    return features


def extract_all_features(npy_path, sr=44100):
    """
    Extract all acoustic features from a .npy waveform file.

    Args:
        npy_path (str or Path): Path to .npy file containing a normalized waveform.
        sr (int): Sample rate in Hz (default: 44100).

    Returns:
        dict: All extracted features and metadata (filename, healthcode, record_id).
    """
    # Load waveform
    waveform = np.load(npy_path)
    
    # Initialize feature dict
    all_features = {}
    
    # Add metadata
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    all_features['filename'] = basename
    
    # Parse healthcode from filename (format: healthcode_recordid_...)
    parts = basename.split('_')
    if len(parts) >= 2:
        all_features['healthcode'] = parts[0]
        all_features['record_id'] = parts[1]
    else:
        all_features['healthcode'] = 'unknown'
        all_features['record_id'] = 'unknown'
    
    # Extract feature groups
    try:
        mfcc_feats = extract_mfcc_features(waveform, sr)
        spectral_feats = extract_spectral_features(waveform, sr)
        prosody_feats = extract_prosody_features(waveform, sr)
        voice_quality_feats = extract_voice_quality_features(waveform, sr)
        energy_feats = extract_energy_features(waveform, sr)
        
        # Combine all features
        all_features.update(mfcc_feats)
        all_features.update(spectral_feats)
        all_features.update(prosody_feats)
        all_features.update(voice_quality_feats)
        all_features.update(energy_feats)
        
    except Exception as e:
        print(f"Error extracting features from {basename}: {str(e)}")
        return None
    
    return all_features


def process_all_npy_files(input_dir, output_csv, sr=44100, start_from_file=None):
    """
    Process all .npy files in a directory, extract features, and save to CSV with quarterly checkpoints.

    Args:
        input_dir (str or Path): Directory containing .npy files (normalized waveforms).
        output_csv (str or Path): Path to output CSV file.
        sr (int): Sample rate in Hz (default: 44100).
        start_from_file (str or None): Optional filename (without .npy extension) to resume from. If provided, processing starts from this file.

    Returns:
        pd.DataFrame: DataFrame with all extracted features (one row per file).
    """
    print("="*80)
    print("Acoustic Feature Extraction with Quarterly Checkpoints")
    print("="*80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"Sample rate: {sr} Hz")
    if start_from_file:
        print(f"Resuming from: {start_from_file}")
    print()
    
    # Find all .npy files and sort them alphabetically (chronological order)
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"Found {len(npy_files)} total .npy files")
    
    if not npy_files:
        print("No .npy files found!")
        return None
    
    # If resuming, find the starting position
    if start_from_file:
        start_idx = None
        for idx, file_path in enumerate(npy_files):
            basename = os.path.splitext(os.path.basename(file_path))[0]
            if basename == start_from_file:
                start_idx = idx
                break
        
        if start_idx is None:
            print(f"ERROR: Start file '{start_from_file}' not found in directory!")
            return None
        
        npy_files = npy_files[start_idx:]
        print(f"Resuming from file {start_idx + 1}, processing {len(npy_files)} remaining files\n")
    
    # Calculate quarter checkpoints
    total_files = len(npy_files)
    quarter_size = total_files // 4
    if quarter_size == 0:
        quarter_size = max(1, total_files // 2)  # If less than 4 files, use half
    
    checkpoints = [quarter_size, 2 * quarter_size, 3 * quarter_size, total_files]
    print(f"Will save checkpoints at: {checkpoints[0]}, {checkpoints[1]}, {checkpoints[2]}, and {checkpoints[3]} files")
    print(f"Quarter size: {quarter_size} files\n")
    
    # Process first file as example
    print("="*80)
    print("EXAMPLE: Processing first file")
    print("="*80)
    example_features = extract_all_features(npy_files[0], sr)
    if example_features:
        print(f"\nFile: {example_features['filename']}")
        print(f"Total features extracted: {len(example_features) - 3}")  # Exclude filename, healthcode, record_id
        print(f"\nSample features:")
        feature_items = list(example_features.items())[3:10]  # Show first 7 actual features
        for key, value in feature_items:
            print(f"  {key}: {value:.6f}")
        print("  ...")
    print()
    
    # Process all files with quarterly saves
    print("="*80)
    print("Processing all files...")
    print("="*80)
    
    all_results = []
    failed = 0
    last_processed_file = None
    checkpoint_num = 0
    
    output_dir = os.path.dirname(output_csv) or "."
    base_name = os.path.splitext(os.path.basename(output_csv))[0]
    
    for idx, npy_path in enumerate(tqdm(npy_files, desc="Extracting features"), 1):
        features = extract_all_features(npy_path, sr)
        if features is not None:
            all_results.append(features)
            last_processed_file = features['filename']
        else:
            failed += 1
        
        # Check if we've reached a checkpoint
        if idx in checkpoints:
            checkpoint_num += 1
            
            # Create DataFrame from current results
            df_checkpoint = pd.DataFrame(all_results)
            
            # Save checkpoint CSV
            checkpoint_csv = os.path.join(output_dir, f"{base_name}_checkpoint_{checkpoint_num}.csv")
            os.makedirs(output_dir, exist_ok=True)
            df_checkpoint.to_csv(checkpoint_csv, index=False)
            
            # Save progress info
            progress_file = os.path.join(output_dir, f"{base_name}_progress.txt")
            with open(progress_file, 'w') as f:
                f.write(f"Last processed file: {last_processed_file}\n")
                f.write(f"Files processed: {len(all_results)}\n")
                f.write(f"Files failed: {failed}\n")
                f.write(f"Checkpoint: {checkpoint_num}/4\n")
                f.write(f"Progress: {idx}/{total_files} ({100*idx/total_files:.1f}%)\n")
            
            print(f"\n{'='*80}")
            print(f"✓ CHECKPOINT {checkpoint_num}/4 SAVED")
            print(f"{'='*80}")
            print(f"Progress: {idx}/{total_files} files ({100*idx/total_files:.1f}%)")
            print(f"Processed: {len(all_results)} | Failed: {failed}")
            print(f"Last file: {last_processed_file}")
            print(f"Saved to: {checkpoint_csv}")
            print(f"Progress file: {progress_file}")
            print(f"{'='*80}\n")
    
    # Create final DataFrame
    df = pd.DataFrame(all_results)
    
    # Save final CSV
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # Final summary
    print(f"\n{'='*80}")
    print("Feature Extraction Complete!")
    print(f"{'='*80}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {failed}")
    print(f"Total features per file: {len(df.columns) - 3}")
    print(f"Output saved to: {output_csv}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}...")
    if last_processed_file:
        print(f"\nLast processed file: {last_processed_file}")
        print(f"\n*** TO RESUME: Use start_from_file parameter with the next file after '{last_processed_file}' ***")
    print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract acoustic features with quarterly checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Process all files from beginning:
  python extract_acoustic_features.py
  
  # Resume from a specific file:
  python extract_acoustic_features.py --start-from "008b878d-8b12-428a-99bb-d39e1db26512_0646b2a0-16ee-4091-a0bb-001ac144c1c0_audio_audio_m4a"
  
  # Use custom directories:
  python extract_acoustic_features.py --input /path/to/input --output /path/to/output
        """
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default="/mloscratch/users/gnahas/data/waveform_norm",
        help='Input directory containing .npy files'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default="/mloscratch/users/gnahas/data/features",
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--sample-rate', 
        type=int, 
        default=44100,
        help='Sample rate in Hz (default: 44100)'
    )
    parser.add_argument(
        '--start-from', 
        type=str, 
        default=None,
        help='Filename (without .npy extension) to resume processing from'
    )
    
    args = parser.parse_args()
    
    # Configuration
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "acoustic_features_v1.csv")
    SAMPLE_RATE = args.sample_rate
    START_FROM = args.start_from
    
    print("\n" + "="*80)
    print("Configuration:")
    print(f"  Input: {INPUT_DIR}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    if START_FROM:
        print(f"  Resume from: {START_FROM}")
    print("="*80 + "\n")
    
    # Run feature extraction
    df = process_all_npy_files(INPUT_DIR, OUTPUT_CSV, SAMPLE_RATE, start_from_file=START_FROM)
    
    if df is not None:
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        print("\nFeature statistics:")
        print(df.describe())
