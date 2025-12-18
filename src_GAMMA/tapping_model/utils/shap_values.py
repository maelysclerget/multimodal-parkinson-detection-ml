"""
SHAP Feature Importance Analysis for MLP Classifier.

This module computes SHAP (SHapley Additive exPlanations) values for each
cross-validation fold's test set to understand feature importance in the
Parkinson's Disease classification task.

For each fold, the module generates:
    - Beeswarm plot showing top 10 most important features
    - CSV file with feature importance scores (mean absolute SHAP values)

SHAP values explain how each feature contributes to individual predictions,
helping identify which tapping features are most predictive of PD.

Input:
    - Trained MLP models (one per fold)
    - Feature CSV with tapping metrics
    - Labels CSV with healthCode and PD diagnosis
    - Cross-validation split CSVs

Output:
    - Per-fold SHAP plots: shap_top10_fold{i}.png
    - Per-fold importance CSVs: shap_importance_fold{i}.csv
"""

from pathlib import Path
from typing import Tuple, List, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt


# Configuration
TRAIN_SPLIT = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_train.csv"
VALTEST_SPLIT = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_val_test.csv"
LABELS_PATH = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
FEATURES_PATH = "/mloscratch/users/clerget/data/csv/tapping_combined_features_session.csv"
MODEL_DIR = Path("/mloscratch/users/clerget/data/saved_models")
OUTPUT_DIR = Path("/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results/SHAP")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FOLDS = [0, 1, 2, 3, 4]
BEST_HYPERPARAMS = {"hidden_dim_1": 32, "hidden_dim_2": 16, "dropout_rate": 0.2}
NON_FEATURE_COLS = {"healthCode", "trial_id", "label_PD", "filename", "record_id", "modality"}


class MLP(nn.Module):
    """
    Multi-layer perceptron for binary classification (HC vs PD).
    
    Architecture:
        - Input layer → hidden_dim_1 (ReLU + Dropout)
        - hidden_dim_1 → hidden_dim_2 (ReLU + Dropout)
        - hidden_dim_2 → 16 (ReLU + Dropout 0.2)
        - 16 → 2 (output logits)
    
    Args:
        input_dim: Number of input features
        hidden_dim_1: Size of first hidden layer (default: 32)
        hidden_dim_2: Size of second hidden layer (default: 16)
        dropout_rate: Dropout probability for first two layers (default: 0.5)
    
    Input shape:
        (batch_size, input_dim)
    
    Output shape:
        (batch_size, 2) - logits for binary classification
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 32,
        hidden_dim_2: int = 16,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output logits of shape (batch_size, 2)
        """
        return self.net(x)


def get_test_data_for_fold(
    fold: int,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract and preprocess test data for a specific cross-validation fold.
    
    Workflow:
        1. Load fold split and identify test healthCodes
        2. Merge features with labels
        3. Handle missing values (mean imputation for numeric columns)
        4. Filter to test set only
        5. Extract feature columns (exclude metadata)
        6. Standardize features using statistics from full dataset
    
    Args:
        fold: Fold number (0-4)
        features_df: DataFrame with tapping features and healthCode
        labels_df: DataFrame with healthCode and label_PD columns
    
    Returns:
        Tuple containing:
            - X_test: Standardized test features (n_samples, n_features)
            - y_test: Test labels (n_samples,)
            - feature_cols: List of feature column names
    
    Example:
        >>> X_test, y_test, features = get_test_data_for_fold(0, feat_df, label_df)
        >>> print(f"Test set: {len(X_test)} samples, {len(features)} features")
    """
    # Load test healthCodes for this fold
    valtest_split = pd.read_csv(VALTEST_SPLIT)
    test_hc = valtest_split[
        (valtest_split["fold_iteration"] == fold) & 
        (valtest_split["subset"] == "test")
    ]["healthCode"]
    
    # Merge features with labels
    data_df = features_df.merge(labels_df, on="healthCode", how="inner")
    
    # Remove modality column if present
    if "modality" in data_df.columns:
        data_df = data_df.drop(columns=["modality"])
    
    # Handle missing values with mean imputation
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    data_df[numeric_cols] = data_df[numeric_cols].fillna(data_df[numeric_cols].mean())
    
    # Filter to test set
    test_df = data_df[data_df["healthCode"].isin(test_hc)].copy()
    
    # Extract feature columns (exclude metadata)
    feature_cols = [
        c for c in data_df.select_dtypes(include=[np.number]).columns 
        if c not in NON_FEATURE_COLS
    ]
    
    # Standardize features using full dataset statistics
    scaler = StandardScaler()
    scaler.fit(data_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    y_test = test_df["label_PD"].values.astype(np.int32)
    
    return X_test, y_test, feature_cols


def compute_shap_for_fold(
    fold: int,
    model: nn.Module,
    X_test: np.ndarray,
    feature_cols: List[str],
    device: torch.device
) -> None:
    """
    Compute SHAP values and generate visualizations for a single fold.
    
    Uses KernelExplainer with a background sample to compute SHAP values
    for all test samples. Generates a beeswarm plot showing how feature
    values (color) relate to their impact on predictions (x-axis).
    
    Args:
        fold: Fold number for output file naming
        model: Trained MLP model in eval mode
        X_test: Test features array (n_samples, n_features)
        feature_cols: List of feature names
        device: Device where model is loaded (cuda/cpu)
    
    Output files:
        - shap_top10_fold{fold}.png: Beeswarm plot of top 10 features
        - shap_importance_fold{fold}.csv: All features ranked by importance
    
    Note:
        Uses 100 background samples (or half of test set if smaller) for
        KernelExplainer to balance computation time and accuracy.
    """
    
    def model_predict(X: np.ndarray) -> np.ndarray:
        """
        Prediction wrapper for SHAP explainer.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Probability of PD class (n_samples,)
        """
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            proba = torch.softmax(logits, dim=1)
        return proba[:, 1].cpu().numpy()
    
    # Select background samples for KernelExplainer
    n_background = min(100, len(X_test) // 2)
    background_indices = np.random.choice(len(X_test), size=n_background, replace=False)
    X_background = X_test[background_indices]
    
    print(f"  Computing SHAP values (background size: {n_background})...")
    explainer = shap.KernelExplainer(model_predict, X_background)
    shap_values = explainer.shap_values(X_test)
    
    # Calculate feature importance as mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Select top 10 features
    top_features = feature_importance.head(10)
    top_indices = [feature_cols.index(f) for f in top_features['feature']]
    
    # Generate beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[:, top_indices],
        X_test[:, top_indices],
        feature_names=top_features['feature'].tolist(),
        show=False,
        plot_type="dot",
        color_bar=True
    )
    plt.title(
        f"Top 10 SHAP Features (Fold {fold})\n"
        "Red=High feature value | Blue=Low feature value",
        fontweight='bold'
    )
    plt.tight_layout()
    
    # Save plot
    plot_path = OUTPUT_DIR / f"shap_top10_fold{fold}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved: {plot_path}")
    plt.close()
    
    # Save full feature importance CSV
    csv_path = OUTPUT_DIR / f"shap_importance_fold{fold}.csv"
    feature_importance.to_csv(csv_path, index=False)
    print(f"  ✓ CSV saved: {csv_path}")


def main():
    """
    Run SHAP analysis for all cross-validation folds.
    
    Workflow for each fold:
        1. Check if trained model exists
        2. Load and preprocess test data
        3. Load trained model
        4. Compute SHAP values
        5. Generate and save visualizations
    
    Skips folds where model file is not found.
    """
    print("="*80)
    print("SHAP Feature Importance Analysis")
    print("="*80 + "\n")
    
    print("Loading data...")
    features_df = pd.read_csv(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    labels_df["label_PD"] = labels_df["label_PD"].astype(int)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    for fold in FOLDS:
        print(f"=== Fold {fold} ===")
        
        # Check if model exists
        model_path = MODEL_DIR / f"best_model_hp135_fold{fold}.pth"
        if not model_path.exists():
            print(f"  ✗ Model not found: {model_path}")
            print(f"  Skipping fold {fold}\n")
            continue
        
        # Load test data
        X_test, y_test, feature_cols = get_test_data_for_fold(
            fold, features_df, labels_df
        )
        print(f"  Test samples: {len(X_test)}, Features: {len(feature_cols)}")
        
        # Load model
        model = MLP(len(feature_cols), **BEST_HYPERPARAMS).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"  ✓ Model loaded: {model_path}")
        
        # Compute SHAP values
        compute_shap_for_fold(fold, model, X_test, feature_cols, device)
        print()
    
    print("="*80)
    print("✓ SHAP analysis complete!")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()