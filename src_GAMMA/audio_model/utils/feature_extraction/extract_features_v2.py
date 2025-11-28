"""
Script to extract ADDITIONAL V4 features and merge with existing V1 features
Only extracts NEW features, then combines with acoustic_features.csv

NEW V4 Features (~40 additional features):
- Delta MFCCs (13 coefficients × mean/std) - 26 features
- Delta-Delta MFCCs (13 coefficients × mean/std) - 26 features  
- Spectral Contrast (7 bands × mean/std) - 14 features
- Spectral Flatness (mean/std) - 2 features
- Spectral Flux (mean/std) - 2 features
- Chroma features (12 × mean/std) - 24 features
- Pitch percentiles (10th, 90th) - 2 features
- Energy variance - 1 feature

Total V4 additions: ~97 new features
Total V2 features: 57 (V1) + 97 (V4) = 154 features
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
    Extract delta and delta-delta MFCC features (temporal dynamics)
    
    Returns:
        dict with 52 features (13 delta + 13 delta-delta, each with mean/std)
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
    Extract advanced spectral features: contrast, flatness, flux
    
    Returns:
        dict with 18 features
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
    Extract chroma features (pitch class profiles)
    
    Returns:
        dict with 24 features (12 pitch classes × mean/std)
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
    Extract pitch percentiles (10th and 90th) for better tremor characterization
    
    Returns:
        dict with 2 features
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
    Extract frame-level energy variance (additional tremor indicator)
    
    Returns:
        dict with 1 feature
    """
    features = {}
    
    # Frame-wise energy (RMS)
    rms = librosa.feature.rms(y=waveform)[0]
    
    # Variance of energy across frames
    features['energy_frame_variance'] = np.var(rms)
    
    return features


def extract_v4_features(npy_path, sr=44100):
    """
    Extract ONLY the new V4 features from a .npy waveform file
    
    Args:
        npy_path: Path to .npy file
        sr: Sample rate (default: 44100)
    
    Returns:
        dict with V4 features and metadata
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
    
    # Extract NEW V4 feature groups
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
        print(f"Error extracting V4 features from {basename}: {str(e)}")
        return None
    
    return all_features


def process_all_npy_files(input_dir, output_csv, sr=44100):
    """
    Process all .npy files and extract V4 features
    
    Args:
        input_dir: Directory containing .npy files
        output_csv: Path to output CSV file for V4 features only
        sr: Sample rate
    """
    print("="*80)
    print("V4 Feature Extraction (Additional Features Only)")
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
    example_features = extract_v4_features(npy_files[0], sr)
    if example_features:
        print(f"\nFile: {example_features['filename']}")
        print(f"Total NEW V4 features extracted: {len(example_features) - 3}")
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
    
    for npy_path in tqdm(npy_files, desc="Extracting V4 features"):
        features = extract_v4_features(npy_path, sr)
        if features is not None:
            all_results.append(features)
        else:
            failed += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save V4 features CSV
    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("V4 Feature Extraction Complete!")
    print(f"{'='*80}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {failed}")
    print(f"Total NEW features per file: {len(df.columns) - 3}")
    print(f"V4 features saved to: {output_csv}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"{'='*80}\n")
    
    return df


def merge_v1_and_v4_features(v1_csv, v4_csv, output_csv):
    """
    Merge V1 (existing) and V4 (new) features into single CSV
    
    Args:
        v1_csv: Path to acoustic_features.csv (57 features)
        v4_csv: Path to V4 features CSV (~97 features)
        output_csv: Path to merged output (acoustic_features_v2.csv)
    """
    print("="*80)
    print("Merging V1 and V4 Features")
    print("="*80)
    
    # Load both CSVs
    print(f"\nLoading V1 features from: {v1_csv}")
    df_v1 = pd.read_csv(v1_csv)
    print(f"  Shape: {df_v1.shape}")
    print(f"  Features: {len(df_v1.columns) - 3}")
    
    print(f"\nLoading V4 features from: {v4_csv}")
    df_v4 = pd.read_csv(v4_csv)
    print(f"  Shape: {df_v4.shape}")
    print(f"  Features: {len(df_v4.columns) - 3}")
    
    # Merge on filename (should match exactly)
    print(f"\nMerging on 'filename' column...")
    df_v4_features_only = df_v4.drop(columns=['healthcode', 'record_id'])  # Drop duplicate metadata
    df_merged_temp = pd.merge(df_v1, df_v4_features_only, on='filename', how='inner')

    # Dimension and filename checks before merge
    print("\n--- Pre-merge checks ---")
    v1_filenames = set(df_v1['filename'])
    v4_filenames = set(df_v4['filename'])
    print(f"  V1 unique filenames: {len(v1_filenames)}")
    print(f"  V4 unique filenames: {len(v4_filenames)}")
    common_filenames = v1_filenames & v4_filenames
    print(f"  Filenames in both: {len(common_filenames)}")
    missing_in_v1 = v4_filenames - v1_filenames
    missing_in_v4 = v1_filenames - v4_filenames
    if missing_in_v1:
        print(f"  WARNING: {len(missing_in_v1)} filenames in V4 not in V1")
    if missing_in_v4:
        print(f"  WARNING: {len(missing_in_v4)} filenames in V1 not in V4")
    print(f"  V1 columns: {df_v1.shape[1]}")
    print(f"  V4 columns: {df_v4.shape[1]}")

    
    # CRITICAL: Reorder columns to ensure correct order
    # Order: filename, healthcode, record_id, then V1 features, then V4 features
    metadata_cols = ['filename', 'healthcode', 'record_id']

    # Post-merge checks
    print("\n--- Post-merge checks ---")
    print(f"  Merged shape: {df_merged_temp.shape}")
    print(f"  Expected rows: {len(common_filenames)}")
    expected_cols = df_v1.shape[1] + df_v4_features_only.shape[1] - 1  # minus 1 for duplicate filename
    print(f"  Expected columns: {expected_cols}")
    if df_merged_temp.shape[0] != len(common_filenames):
        print(f"  WARNING: Merged rows ({df_merged_temp.shape[0]}) != common filenames ({len(common_filenames)})")
    if df_merged_temp.shape[1] != expected_cols:
        print(f"  WARNING: Merged columns ({df_merged_temp.shape[1]}) != expected ({expected_cols})")
            
    v1_feature_cols = [col for col in df_v1.columns if col not in metadata_cols]
    v4_feature_cols = [col for col in df_v4_features_only.columns if col not in metadata_cols]
    
    # Construct final column order
    final_column_order = metadata_cols + v1_feature_cols + v4_feature_cols
    df_merged = df_merged_temp[final_column_order]
    
    print(f"  Merged shape: {df_merged.shape}")
    print(f"  Total features: {len(df_merged.columns) - 3}")
    print(f"  V1 features: {len(v1_feature_cols)}")
    print(f"  V4 features: {len(v4_feature_cols)}")
    print(f"  Total: {len(v1_feature_cols)} + {len(v4_feature_cols)} = {len(df_merged.columns) - 3}")
    
    # Verify column order
    print(f"\n✓ Column order verification:")
    print(f"  Columns 1-3 (metadata): {list(df_merged.columns[:3])}")
    print(f"  Columns 4-{3+len(v1_feature_cols)} (V1 features): {df_merged.columns[3]} ... {df_merged.columns[3+len(v1_feature_cols)-1]}")
    print(f"  Columns {4+len(v1_feature_cols)}-{3+len(v1_feature_cols)+len(v4_feature_cols)} (V4 features): {df_merged.columns[3+len(v1_feature_cols)]} ... {df_merged.columns[-1]}")
    
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
    print(f"  Columns {4+len(v1_feature_cols)}-{len(df_merged.columns)}: V4 Features (Delta MFCCs, spectral advanced, chroma)")
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
    
    V1_CSV = os.path.join(FEATURES_DIR, "acoustic_features.csv")
    V4_CSV = os.path.join(FEATURES_DIR, "acoustic_features_v4_only.csv")
    V2_CSV = os.path.join(FEATURES_DIR, "acoustic_features_vf.csv")
    
    SAMPLE_RATE = 44100
    
    print("\n" + "="*80)
    print("V4 Feature Extraction and Merging Pipeline")
    print("="*80)
    print(f"\nStep 1: Extract NEW V4 features from .npy files")
    print(f"  Input: {INPUT_DIR}")
    print(f"  Output: {V4_CSV}")
    print(f"\nStep 2: Merge V1 + V4 features")
    print(f"  V1 features: {V1_CSV}")
    print(f"  V4 features: {V4_CSV}")
    print(f"  Merged output: {V2_CSV}")
    print("="*80 + "\n")
    
    # Step 1: Extract V4 features
    print("\n" + "="*80)
    print("STEP 1: Extracting V4 Features")
    print("="*80 + "\n")
    df_v4 = process_all_npy_files(INPUT_DIR, V4_CSV, SAMPLE_RATE)
    
    if df_v4 is None:
        print("ERROR: V4 feature extraction failed!")
        exit(1)
    
    # Step 2: Merge V1 and V4
    print("\n" + "="*80)
    print("STEP 2: Merging V1 and V4 Features")
    print("="*80 + "\n")
    
    if not os.path.exists(V1_CSV):
        print(f"ERROR: V1 features not found at {V1_CSV}")
        print("Please run extract_acoustic_features.py first!")
        exit(1)
    
    df_v2 = merge_v1_and_v4_features(V1_CSV, V4_CSV, V2_CSV)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nFinal output: {V2_CSV}")
    print(f"Total features: {len(df_v2.columns) - 3}")
    print(f"Total recordings: {len(df_v2)}")
    print(f"\nFirst few rows:")
    print(df_v2.head())
    print(f"\nFeature summary:")
    print(df_v2.describe())
    print("="*80 + "\n")
