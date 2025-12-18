
"""
Feature preprocessing pipeline for acoustic features.

Overview:
---------
This script implements a robust feature selection pipeline for acoustic features, ensuring no data leakage by computing all statistics on the training set only. The pipeline includes:
1. Variance filtering: Remove low-variance (near-constant) features.
2. Correlation filtering: Remove highly correlated features (keep one from each pair).
3. Statistical testing: Remove features with no class discrimination (t-test between classes).

Note:
-----
This feature selection pipeline was not ultimately used in the final deep learning model, 
which was designed to learn discriminative features directly from the data. However, 
this script is kept in the repository as a reference and for exploration of potential 
feature selection techniques, which may be useful for classical machine learning models or for interpretability studies.

Expected input:
- Features CSV: Table of extracted features (one row per recording, columns for features and metadata).
- Labels CSV: Mapping from healthCode to class label (e.g., PD/control).
- Training folds CSV: Specifies which healthCodes are in the training set (to avoid data leakage).

Output:
- Preprocessed features CSV: Only selected features, ready for model training.
- Report: Text file summarizing feature selection steps and results.

Usage:
------
Run as a script with --input, --output, --labels, and --train-folds arguments, or use the functions in your own pipeline.
"""

import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')


def load_data(features_csv, labels_csv, train_folds_csv):
    """
    Load features and labels, and extract the training set only.

    Args:
        features_csv (str or Path): Path to features CSV file.
        labels_csv (str or Path): Path to labels CSV (healthcode -> label mapping).
        train_folds_csv (str or Path): Path to training folds CSV.

    Returns:
        tuple:
            df_features (pd.DataFrame): Full feature DataFrame.
            train_healthcodes (set): Set of training healthCodes.
            labels_dict (dict): healthCode -> label mapping.
    """
    print("="*80)
    print("Loading Data")
    print("="*80)
    
    # Load features
    print(f"\nLoading features from: {features_csv}")
    df_features = pd.read_csv(features_csv)
    print(f"  Shape: {df_features.shape}")
    print(f"  Features: {len(df_features.columns) - 3}")  # Exclude filename, healthcode, record_id
    
    # Rename 'healthcode' to 'healthCode' for consistency with labels CSV
    if 'healthcode' in df_features.columns and 'healthCode' not in df_features.columns:
        df_features = df_features.rename(columns={'healthcode': 'healthCode'})
        print(f"  Renamed 'healthcode' → 'healthCode' for consistency")
    
    # Load labels
    print(f"\nLoading labels from: {labels_csv}")
    df_labels = pd.read_csv(labels_csv, sep=',')  
    print(f"  Shape: {df_labels.shape}")
    
    # Create healthcode -> label mapping
    labels_dict = dict(zip(df_labels['healthCode'], df_labels['label_PD']))  
    print(f"  Unique labels: {df_labels['label_PD'].value_counts().to_dict()}")
    
    # Load training folds (to avoid data leakage)
    print(f"\nLoading training folds from: {train_folds_csv}")
    df_train = pd.read_csv(train_folds_csv)
    
    # Extract healthcodes from 'healthCode' column where subset == 'train'
    train_healthcodes = set(df_train[df_train['subset'] == 'train']['healthCode'].unique())
    
    print(f"  Training healthcodes (across all folds): {len(train_healthcodes)}")
    print(f"  Training recordings: {df_features[df_features['healthCode'].isin(train_healthcodes)].shape[0]}")
    
    print("="*80 + "\n")
    
    return df_features, train_healthcodes, labels_dict


def filter_low_variance(X_train, feature_names, variance_threshold=0.01):
    """
    Remove features with low variance (near-constant features).

    Args:
        X_train (np.ndarray): Training data (n_samples, n_features).
        feature_names (list): List of feature names.
        variance_threshold (float): Minimum variance threshold.

    Returns:
        list: Names of features to keep.
    """
    print("="*80)
    print(f"Step 1: Variance Filtering (threshold={variance_threshold})")
    print("="*80)
    
    # Compute variance for each feature
    variances = np.var(X_train, axis=0)
    
    # Find low-variance features
    low_var_mask = variances < variance_threshold
    kept_mask = ~low_var_mask
    
    removed_features = [f for f, remove in zip(feature_names, low_var_mask) if remove]
    kept_features = [f for f, keep in zip(feature_names, kept_mask) if keep]
    
    print(f"\nVariance statistics:")
    print(f"  Min variance: {variances.min():.6f}")
    print(f"  Max variance: {variances.max():.6f}")
    print(f"  Mean variance: {variances.mean():.6f}")
    
    print(f"\nFiltering results:")
    print(f"  Features removed: {len(removed_features)}")
    print(f"  Features kept: {len(kept_features)}")
    
    if removed_features:
        print(f"\nRemoved features (variance < {variance_threshold}):")
        for feat in removed_features[:10]:  # Show first 10
            var = variances[feature_names.index(feat)]
            print(f"  - {feat}: variance={var:.6f}")
        if len(removed_features) > 10:
            print(f"  ... and {len(removed_features) - 10} more")
    
    print("="*80 + "\n")
    
    return kept_features


def filter_correlated_features(X_train, feature_names, correlation_threshold=0.95):
    """
    Remove highly correlated features (keep one from each pair).

    Args:
        X_train (np.ndarray): Training data (n_samples, n_features).
        feature_names (list): List of feature names.
        correlation_threshold (float): Correlation threshold for removal.

    Returns:
        list: Names of features to keep.
    """
    print("="*80)
    print(f"Step 2: Correlation Filtering (threshold={correlation_threshold})")
    print("="*80)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    
    # Find highly correlated pairs
    removed_features = set()
    correlated_pairs = []
    
    n_features = len(feature_names)
    for i in range(n_features):
        if feature_names[i] in removed_features:
            continue
        
        for j in range(i + 1, n_features):
            if feature_names[j] in removed_features:
                continue
            
            if abs(corr_matrix[i, j]) > correlation_threshold:
                # Remove feature j (arbitrary choice, could use other criteria)
                removed_features.add(feature_names[j])
                correlated_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    kept_features = [f for f in feature_names if f not in removed_features]
    
    print(f"\nCorrelation statistics:")
    # Get upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    print(f"  Max correlation: {np.max(np.abs(upper_tri)):.4f}")
    print(f"  Mean correlation: {np.mean(np.abs(upper_tri)):.4f}")
    print(f"  Pairs with |corr| > {correlation_threshold}: {len(correlated_pairs)}")
    
    print(f"\nFiltering results:")
    print(f"  Features removed: {len(removed_features)}")
    print(f"  Features kept: {len(kept_features)}")
    
    if correlated_pairs:
        print(f"\nHighly correlated pairs (showing first 10):")
        for feat1, feat2, corr in correlated_pairs[:10]:
            print(f"  - {feat1} <-> {feat2}: corr={corr:.4f} (removed {feat2})")
        if len(correlated_pairs) > 10:
            print(f"  ... and {len(correlated_pairs) - 10} more pairs")
    
    print("="*80 + "\n")
    
    return kept_features


def filter_non_discriminative_features(X_train, y_train, feature_names, p_value_threshold=0.05):
    """
    Remove features with no statistical difference between classes (t-test).

    Args:
        X_train (np.ndarray): Training data (n_samples, n_features).
        y_train (np.ndarray): Training labels (n_samples,).
        feature_names (list): List of feature names.
        p_value_threshold (float): P-value threshold for significance.

    Returns:
        list: Names of features to keep.
    """
    print("="*80)
    print(f"Step 3: Statistical Testing (p-value threshold={p_value_threshold})")
    print("="*80)
    
    # Perform t-test for each feature
    p_values = []
    t_stats = []
    
    class_0_mask = y_train == 0
    class_1_mask = y_train == 1
    
    for i, feat in enumerate(feature_names):
        class_0_values = X_train[class_0_mask, i]
        class_1_values = X_train[class_1_mask, i]
        
        # Independent t-test
        t_stat, p_val = stats.ttest_ind(class_0_values, class_1_values)
        t_stats.append(abs(t_stat))
        p_values.append(p_val)
    
    # Find non-discriminative features
    p_values = np.array(p_values)
    t_stats = np.array(t_stats)
    
    non_sig_mask = p_values > p_value_threshold
    kept_mask = ~non_sig_mask
    
    removed_features = [f for f, remove in zip(feature_names, non_sig_mask) if remove]
    kept_features = [f for f, keep in zip(feature_names, kept_mask) if keep]
    
    print(f"\nStatistical testing results:")
    print(f"  Min p-value: {p_values.min():.6f}")
    print(f"  Max p-value: {p_values.max():.6f}")
    print(f"  Median p-value: {np.median(p_values):.6f}")
    print(f"  Features with p < {p_value_threshold}: {np.sum(p_values < p_value_threshold)}")
    
    print(f"\nFiltering results:")
    print(f"  Features removed: {len(removed_features)}")
    print(f"  Features kept: {len(kept_features)}")
    
    if removed_features:
        print(f"\nNon-discriminative features (p > {p_value_threshold}, showing first 10):")
        for feat in removed_features[:10]:
            idx = feature_names.index(feat)
            print(f"  - {feat}: p={p_values[idx]:.4f}, |t|={t_stats[idx]:.2f}")
        if len(removed_features) > 10:
            print(f"  ... and {len(removed_features) - 10} more")
    
    # Show most discriminative features
    print(f"\nMost discriminative features (top 10):")
    top_indices = np.argsort(p_values)[:10]
    for idx in top_indices:
        print(f"  - {feature_names[idx]}: p={p_values[idx]:.6f}, |t|={t_stats[idx]:.2f}")
    
    print("="*80 + "\n")
    
    return kept_features


def preprocess_features(features_csv, labels_csv, train_folds_csv, output_csv,
                       variance_threshold=0.01, 
                       correlation_threshold=0.95,
                       p_value_threshold=0.05):
    """
    Complete preprocessing pipeline with no data leakage.

    Args:
        features_csv (str or Path): Path to input features CSV.
        labels_csv (str or Path): Path to labels CSV.
        train_folds_csv (str or Path): Path to training folds CSV.
        output_csv (str or Path): Path to output preprocessed CSV.
        variance_threshold (float): Minimum variance for feature selection.
        correlation_threshold (float): Maximum allowed correlation for feature selection.
        p_value_threshold (float): Maximum p-value for t-test (feature selection).

    Returns:
        pd.DataFrame: DataFrame with preprocessed features (one row per file).
    """
    print("\n" + "="*80)
    print("FEATURE PREPROCESSING PIPELINE")
    print("="*80)
    print(f"\nInput: {features_csv}")
    print(f"Output: {output_csv}")
    print(f"\nParameters:")
    print(f"  Variance threshold: {variance_threshold}")
    print(f"  Correlation threshold: {correlation_threshold}")
    print(f"  P-value threshold: {p_value_threshold}")
    print("="*80 + "\n")
    
    # Step 0: Load data
    df_features, train_healthcodes, labels_dict = load_data(
        features_csv, labels_csv, train_folds_csv
    )
    
    # Extract feature columns (exclude metadata)
    metadata_cols = ['filename', 'healthCode', 'record_id']  
    feature_cols = [col for col in df_features.columns if col not in metadata_cols]
    
    print(f"Initial feature count: {len(feature_cols)}\n")
    
    # Get training data only (NO DATA LEAKAGE)
    train_mask = df_features['healthCode'].isin(train_healthcodes)  
    df_train = df_features[train_mask].copy()
    
    # Add labels to training data
    df_train['label'] = df_train['healthCode'].map(labels_dict)  
    
    # Remove any rows without labels
    df_train = df_train.dropna(subset=['label'])
    
    print(f"Training data:")
    print(f"  Recordings: {len(df_train)}")
    print(f"  Class distribution: {df_train['label'].value_counts().to_dict()}")
    print()
    
    # Extract training features and labels
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    
    # Step 1: Variance filtering
    kept_features = filter_low_variance(X_train, feature_cols, variance_threshold)
    
    # Update X_train with kept features
    kept_indices = [i for i, f in enumerate(feature_cols) if f in kept_features]
    X_train = X_train[:, kept_indices]
    feature_cols = kept_features
    
    # Step 2: Correlation filtering
    kept_features = filter_correlated_features(X_train, feature_cols, correlation_threshold)
    
    # Update X_train with kept features
    kept_indices = [i for i, f in enumerate(feature_cols) if f in kept_features]
    X_train = X_train[:, kept_indices]
    feature_cols = kept_features
    
    # Step 3: Statistical testing
    kept_features = filter_non_discriminative_features(
        X_train, y_train, feature_cols, p_value_threshold
    )
    
    # Final feature selection
    final_cols = metadata_cols + kept_features
    df_output = df_features[final_cols].copy()
    
    # Save preprocessed features
    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    
    # Summary
    print("="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nFeature reduction:")
    print(f"  Original features: {len(df_features.columns) - 3}")
    print(f"  Final features: {len(kept_features)}")
    print(f"  Removed: {len(df_features.columns) - 3 - len(kept_features)}")
    print(f"  Reduction: {100 * (1 - len(kept_features) / (len(df_features.columns) - 3)):.1f}%")
    
    print(f"\nOutput saved to: {output_csv}")
    print(f"  Shape: {df_output.shape}")
    print(f"  Columns: {list(df_output.columns[:10])}... (+{len(df_output.columns) - 10} more)")
    
    # Save feature selection report
    report_path = output_csv.replace('.csv', '_report.txt')
    with open(report_path, 'w') as f:
        f.write("Feature Preprocessing Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input: {features_csv}\n")
        f.write(f"Output: {output_csv}\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Variance threshold: {variance_threshold}\n")
        f.write(f"  Correlation threshold: {correlation_threshold}\n")
        f.write(f"  P-value threshold: {p_value_threshold}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Original features: {len(df_features.columns) - 3}\n")
        f.write(f"  Final features: {len(kept_features)}\n")
        f.write(f"  Removed: {len(df_features.columns) - 3 - len(kept_features)}\n\n")
        f.write(f"Kept features:\n")
        for feat in kept_features:
            f.write(f"  - {feat}\n")
    
    print(f"\nReport saved to: {report_path}")
    print("="*80 + "\n")
    
    return df_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Feature preprocessing with variance, correlation, and statistical filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Default parameters:
  python preprocess_features.py
  
  # Custom thresholds:
  python preprocess_features.py --variance 0.005 --correlation 0.90 --pvalue 0.01
  
  # Process V1 features:
  python preprocess_features.py --input acoustic_features.csv --output acoustic_features_clean.csv
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='/mloscratch/users/gnahas/data/features/acoustic_features_vf.csv',
        help='Input features CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/mloscratch/users/gnahas/data/features/Preprocessed/acoustic_features_vf_clean.csv',
        help='Output preprocessed CSV'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='/tremor2tensor/src_GAMMA/paired_healthcode.csv',
        help='Labels CSV (healthcode -> label mapping)'
    )
    parser.add_argument(
        '--train-folds',
        type=str,
        default='/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv',
        help='Training folds CSV (to avoid data leakage)'
    )
    parser.add_argument(
        '--variance',
        type=float,
        default=0.01,
        help='Variance threshold (default: 0.01)'
    )
    parser.add_argument(
        '--correlation',
        type=float,
        default=0.95,
        help='Correlation threshold (default: 0.95)'
    )
    parser.add_argument(
        '--pvalue',
        type=float,
        default=0.05,
        help='P-value threshold for t-test (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    df = preprocess_features(
        features_csv=args.input,
        labels_csv=args.labels,
        train_folds_csv=args.train_folds,
        output_csv=args.output,
        variance_threshold=args.variance,
        correlation_threshold=args.correlation,
        p_value_threshold=args.pvalue
    )
