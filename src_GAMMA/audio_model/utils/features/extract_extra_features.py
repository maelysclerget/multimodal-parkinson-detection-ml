
"""
Script to extract additional (v2) acoustic features from normalized waveform .npy files and merge with existing V1 features.

Overview:
---------
This script processes a directory of normalized waveform .npy files, extracts a set of advanced acoustic features (v2), and merges them with previously extracted V1 features (acoustic_features_v1.csv) to create a comprehensive feature set for each recording.

v2 Features extracted (per recording):
- Delta MFCCs (13 coefficients × mean/std): 26 features
- Delta-Delta MFCCs (13 coefficients × mean/std): 26 features
- Spectral Contrast (7 bands × mean/std): 14 features
- Spectral Flatness (mean/std): 2 features
- Spectral Flux (mean/std): 2 features
- Chroma features (12 × mean/std): 24 features
- Pitch percentiles (10th, 90th): 2 features
- Energy variance: 1 feature

Total v2 additions: ~97 new features
Total vf features: 57 (V1) + 97 (v2) = 154 features

Expected input:
- Input directory: Contains .npy files (normalized waveforms, 1D float32 arrays)
- Each .npy file is named as <healthcode>_<recordid>_...npy
- V1 CSV: acoustic_features.csv (from extract_acoustic_features.py)

Output:
- v2 CSV: acoustic_features_vf_only.csv (new features only)
- vf CSV: acoustic_features_vf.csv (merged V1+v2 features)

Usage:
------
Run as a script to extract v2 features and merge with V1 features, or use the functions in your own pipeline.
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


def extract_delta_features(waveform, sr):
    """
    Extract delta and delta-delta MFCC features (temporal dynamics).

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: 52 features (13 delta + 13 delta-delta, each with mean/std).
    """
    features = {}
    
    # Extract 13 MFCCs
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    
    # Compute deltas (first derivative - velocity of change)
    delta_mfccs = librosa.feature.delta(mfccs)
    
    # Compute delta-deltas (second derivative - acceleration of change)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Extract mean and std for delta MFCCs
    for i in range(13):
        features[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfccs[i])
        features[f'delta_mfcc_{i+1}_std'] = np.std(delta_mfccs[i])
    
    # Extract mean and std for delta-delta MFCCs
    for i in range(13):
        features[f'delta2_mfcc_{i+1}_mean'] = np.mean(delta2_mfccs[i])
        features[f'delta2_mfcc_{i+1}_std'] = np.std(delta2_mfccs[i])
    
    return features


def extract_advanced_spectral_features(waveform, sr):
    """
    Extract advanced spectral features: spectral contrast, flatness, and flux.

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: 18 features (spectral contrast, flatness, flux; mean/std for each).
    """
    features = {}
    
    # Spectral contrast (7 frequency bands)
    # Measures difference between peaks and valleys in spectrum
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_bands=7)
    for i in range(7):
        features[f'spectral_contrast_band{i+1}_mean'] = np.mean(contrast[i])
        features[f'spectral_contrast_band{i+1}_std'] = np.std(contrast[i])
    
    # Spectral flatness (how noise-like vs tone-like)
    flatness = librosa.feature.spectral_flatness(y=waveform)[0]
    features['spectral_flatness_mean'] = np.mean(flatness)
    features['spectral_flatness_std'] = np.std(flatness)
    
    # Compute STFT for spectral flux
    stft = librosa.stft(waveform)
    magnitude = np.abs(stft)
    
    # Spectral flux (rate of change in spectrum)
    flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))
    features['spectral_flux_mean'] = np.mean(flux)
    features['spectral_flux_std'] = np.std(flux)
    
    return features


def extract_chroma_features(waveform, sr):
    """
    Extract chroma features (pitch class profiles).

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: 24 features (12 pitch classes × mean/std).
    """
    features = {}
    
    # Extract chromagram
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
    
    # Mean and std for each of 12 pitch classes
    for i in range(12):
        features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
        features[f'chroma_{i+1}_std'] = np.std(chroma[i])
    
    return features


def extract_pitch_percentiles(waveform, sr):
    """
    Extract pitch percentiles (10th and 90th) for better tremor characterization.

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: 2 features (pitch_p10, pitch_p90).
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
            features['pitch_p10'] = np.percentile(pitch_values, 10)
            features['pitch_p90'] = np.percentile(pitch_values, 90)
        else:
            features['pitch_p10'] = 0
            features['pitch_p90'] = 0
            
    except Exception as e:
        features['pitch_p10'] = 0
        features['pitch_p90'] = 0
    
    return features


def extract_energy_variance(waveform, sr):
    """
    Extract frame-level energy variance (additional tremor indicator).

    Args:
        waveform (np.ndarray): 1D float32 array of audio samples.
        sr (int): Sample rate in Hz.

    Returns:
        dict: 1 feature (energy_frame_variance).
    """
    features = {}
    
    # Frame-wise energy (RMS)
    rms = librosa.feature.rms(y=waveform)[0]
    
    # Variance of energy across frames
    features['energy_frame_variance'] = np.var(rms)
    
    return features


def extract_v2_features(npy_path, sr=44100):
    """
    Extract ONLY the new v2 features from a .npy waveform file.

    Args:
        npy_path (str or Path): Path to .npy file containing a normalized waveform.
        sr (int): Sample rate in Hz (default: 44100).

    Returns:
        dict: v2 features and metadata (filename, healthcode, record_id).
    """
    # Load waveform
    waveform = np.load(npy_path)
    
    # Initialize feature dict
    all_features = {}
    
    # Add metadata (for merging with V1 features)
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
    
    # Extract NEW v2 feature groups
    try:
        delta_feats = extract_delta_features(waveform, sr)
        spectral_adv_feats = extract_advanced_spectral_features(waveform, sr)
        chroma_feats = extract_chroma_features(waveform, sr)
        pitch_percentile_feats = extract_pitch_percentiles(waveform, sr)
        energy_var_feats = extract_energy_variance(waveform, sr)
        
        # Combine all NEW features
        all_features.update(delta_feats)
        all_features.update(spectral_adv_feats)
        all_features.update(chroma_feats)
        all_features.update(pitch_percentile_feats)
        all_features.update(energy_var_feats)
        
    except Exception as e:
        print(f"Error extracting v2 features from {basename}: {str(e)}")
        return None
    
    return all_features


def process_all_npy_files(input_dir, output_csv, sr=44100):
    """
    Process all .npy files in a directory, extract v2 features, and save to CSV.

    Args:
        input_dir (str or Path): Directory containing .npy files (normalized waveforms).
        output_csv (str or Path): Path to output CSV file for v2 features only.
        sr (int): Sample rate in Hz (default: 44100).

    Returns:
        pd.DataFrame: DataFrame with all extracted v2 features (one row per file).
    """
    print("="*80)
    print("v2 Feature Extraction (Additional Features Only)")
    print("="*80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"Sample rate: {sr} Hz")
    print()
    
    # Find all .npy files
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"Found {len(npy_files)} .npy files\n")
    
    if not npy_files:
        print("No .npy files found!")
        return None
    
    # Process first file as example
    print("="*80)
    print("EXAMPLE: Processing first file")
    print("="*80)
    example_features = extract_v2_features(npy_files[0], sr)
    if example_features:
        print(f"\nFile: {example_features['filename']}")
        print(f"Total NEW v2 features extracted: {len(example_features) - 3}")
        print(f"\nSample NEW features:")
        feature_items = list(example_features.items())[3:10]
        for key, value in feature_items:
            print(f"  {key}: {value:.6f}")
        print("  ...")
    print()
    
    # Process all files
    print("="*80)
    print("Processing all files...")
    print("="*80)
    
    all_results = []
    failed = 0
    
    for npy_path in tqdm(npy_files, desc="Extracting v2 features"):
        features = extract_v2_features(npy_path, sr)
        if features is not None:
            all_results.append(features)
        else:
            failed += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save v2 features CSV
    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("v2 Feature Extraction Complete!")
    print(f"{'='*80}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {failed}")
    print(f"Total NEW features per file: {len(df.columns) - 3}")
    print(f"v2 features saved to: {output_csv}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"{'='*80}\n")
    
    return df


def merge_v1_and_v2_features(v1_csv, v2_csv, output_csv):
    """
    Merge V1 (existing) and v2 (new) features into a single CSV file.

    Args:
        v1_csv (str or Path): Path to acoustic_features_v1.csv (57 features).
        v2_csv (str or Path): Path to v2 features CSV (~97 features).
        output_csv (str or Path): Path to merged output (acoustic_features_vf.csv).

    Returns:
        pd.DataFrame: DataFrame with merged features (one row per file).
    """
    print("="*80)
    print("Merging V1 and v2 Features")
    print("="*80)
    
    # Load both CSVs
    print(f"\nLoading V1 features from: {v1_csv}")
    df_v1 = pd.read_csv(v1_csv)
    print(f"  Shape: {df_v1.shape}")
    print(f"  Features: {len(df_v1.columns) - 3}")
    
    print(f"\nLoading v2 features from: {v2_csv}")
    df_v2 = pd.read_csv(v2_csv)
    print(f"  Shape: {df_v2.shape}")
    print(f"  Features: {len(df_v2.columns) - 3}")
    
    # Merge on filename (should match exactly)
    print(f"\nMerging on 'filename' column...")
    df_v2_features_only = df_v2.drop(columns=['healthcode', 'record_id'])  # Drop duplicate metadata
    df_merged_temp = pd.merge(df_v1, df_v2_features_only, on='filename', how='inner')

    # Dimension and filename checks before merge
    print("\n--- Pre-merge checks ---")
    v1_filenames = set(df_v1['filename'])
    v2_filenames = set(df_v2['filename'])
    print(f"  V1 unique filenames: {len(v1_filenames)}")
    print(f"  v2 unique filenames: {len(v2_filenames)}")
    common_filenames = v1_filenames & v2_filenames
    print(f"  Filenames in both: {len(common_filenames)}")
    missing_in_v1 = v2_filenames - v1_filenames
    missing_in_v2 = v1_filenames - v2_filenames
    if missing_in_v1:
        print(f"  WARNING: {len(missing_in_v1)} filenames in v2 not in V1")
    if missing_in_v2:
        print(f"  WARNING: {len(missing_in_v2)} filenames in V1 not in v2")
    print(f"  V1 columns: {df_v1.shape[1]}")
    print(f"  v2 columns: {df_v2.shape[1]}")

    
    # Reorder columns to ensure correct order
    # Order: filename, healthcode, record_id, then V1 features, then v2 features
    metadata_cols = ['filename', 'healthcode', 'record_id']

    # Post-merge checks
    print("\n--- Post-merge checks ---")
    print(f"  Merged shape: {df_merged_temp.shape}")
    print(f"  Expected rows: {len(common_filenames)}")
    expected_cols = df_v1.shape[1] + df_v2_features_only.shape[1] - 1  # minus 1 for duplicate filename
    print(f"  Expected columns: {expected_cols}")
    if df_merged_temp.shape[0] != len(common_filenames):
        print(f"  WARNING: Merged rows ({df_merged_temp.shape[0]}) != common filenames ({len(common_filenames)})")
    if df_merged_temp.shape[1] != expected_cols:
        print(f"  WARNING: Merged columns ({df_merged_temp.shape[1]}) != expected ({expected_cols})")
            
    v1_feature_cols = [col for col in df_v1.columns if col not in metadata_cols]
    v2_feature_cols = [col for col in df_v2_features_only.columns if col not in metadata_cols]
    
    # Construct final column order
    final_column_order = metadata_cols + v1_feature_cols + v2_feature_cols
    df_merged = df_merged_temp[final_column_order]
    
    print(f"  Merged shape: {df_merged.shape}")
    print(f"  Total features: {len(df_merged.columns) - 3}")
    print(f"  V1 features: {len(v1_feature_cols)}")
    print(f"  v2 features: {len(v2_feature_cols)}")
    print(f"  Total: {len(v1_feature_cols)} + {len(v2_feature_cols)} = {len(df_merged.columns) - 3}")
    
    # Verify column order
    print(f"\n✓ Column order verification:")
    print(f"  Columns 1-3 (metadata): {list(df_merged.columns[:3])}")
    print(f"  Columns 4-{3+len(v1_feature_cols)} (V1 features): {df_merged.columns[3]} ... {df_merged.columns[3+len(v1_feature_cols)-1]}")
    print(f"  Columns {4+len(v1_feature_cols)}-{3+len(v1_feature_cols)+len(v2_feature_cols)} (v2 features): {df_merged.columns[3+len(v1_feature_cols)]} ... {df_merged.columns[-1]}")
    
    # Check for any mismatches
    if len(df_merged) != len(df_v1):
        print(f"\n⚠️  WARNING: Merged rows ({len(df_merged)}) != V1 rows ({len(df_v1)})")
        print(f"  Some files may not have matched!")
    else:
        print(f"\n✓ All {len(df_merged)} files matched successfully")
    
    # Save merged CSV
    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print("Merge Complete!")
    print(f"{'='*80}")
    print(f"Output saved to: {output_csv}")
    print(f"\nFINAL Column Structure (VERIFIED):")
    print(f"  Columns 1-3: Metadata")
    print(f"    - {list(df_merged.columns[:3])}")
    print(f"  Columns 4-{3+len(v1_feature_cols)}: V1 Features (MFCCs, spectral, pitch, voice quality, energy)")
    print(f"    - First: {df_merged.columns[3]}")
    print(f"    - Last: {df_merged.columns[3+len(v1_feature_cols)-1]}")
    print(f"  Columns {4+len(v1_feature_cols)}-{len(df_merged.columns)}: v2 Features (Delta MFCCs, spectral advanced, chroma)")
    print(f"    - First: {df_merged.columns[3+len(v1_feature_cols)]}")
    print(f"    - Last: {df_merged.columns[-1]}")
    print(f"\nFirst 10 columns: {list(df_merged.columns[:10])}")
    print(f"Last 10 columns: {list(df_merged.columns[-10:])}")
    print(f"{'='*80}\n")
    
    return df_merged


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/mloscratch/users/gnahas/data/waveform_norm"
    FEATURES_DIR = "/mloscratch/users/gnahas/data/features"
    
    V1_CSV = os.path.join(FEATURES_DIR, "acoustic_features_v1.csv")
    v2_CSV = os.path.join(FEATURES_DIR, "acoustic_features_vf_only.csv")
    vf_CSV = os.path.join(FEATURES_DIR, "acoustic_features_vf.csv")
    
    SAMPLE_RATE = 44100
    
    print("\n" + "="*80)
    print("v2 Feature Extraction and Merging Pipeline")
    print("="*80)
    print(f"\nStep 1: Extract NEW v2 features from .npy files")
    print(f"  Input: {INPUT_DIR}")
    print(f"  Output: {v2_CSV}")
    print(f"\nStep 2: Merge V1 + v2 features")
    print(f"  V1 features: {V1_CSV}")
    print(f"  v2 features: {v2_CSV}")
    print(f"  Merged output: {vf_CSV}")
    print("="*80 + "\n")
    
    # Step 1: Extract v2 features
    print("\n" + "="*80)
    print("STEP 1: Extracting v2 Features")
    print("="*80 + "\n")
    df_v2 = process_all_npy_files(INPUT_DIR, v2_CSV, SAMPLE_RATE)
    
    if df_v2 is None:
        print("ERROR: v2 feature extraction failed!")
        exit(1)
    
    # Step 2: Merge V1 and v2
    print("\n" + "="*80)
    print("STEP 2: Merging V1 and v2 Features")
    print("="*80 + "\n")
    
    if not os.path.exists(V1_CSV):
        print(f"ERROR: V1 features not found at {V1_CSV}")
        print("Please run extract_acoustic_features.py first!")
        exit(1)
    
    df_vf = merge_v1_and_v2_features(V1_CSV, v2_CSV, vf_CSV)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nFinal output: {vf_CSV}")
    print(f"Total features: {len(df_vf.columns) - 3}")
    print(f"Total recordings: {len(df_vf)}")
    print(f"\nFirst few rows:")
    print(df_vf.head())
    print(f"\nFeature summary:")
    print(df_vf.describe())
    print("="*80 + "\n")
