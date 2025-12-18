"""
SHAP Feature Importance Analysis for MLP (Single Fold, Top 10 Features)

This script computes and visualizes SHAP feature importances for a classical MLP model trained on acoustic features, for each fold in a cross-validation setup. It loads the best model weights for each fold, computes SHAP values for the test set, and plots the 10 most important features (by mean |SHAP|) in a beeswarm plot. The results are saved as both PNG plots and CSV files for each fold.

Context:
--------
This script is intended for classical machine learning models (e.g., MLPs) where feature importance can be directly interpreted. In the final deep learning pipeline, feature selection and importance were handled differently (e.g., via end-to-end learning or Grad-CAM), but this script is kept for reference and for interpretability studies on classical models.

Expected file structure:
- Model weights: Results/MLP_v2_HP_Search/undersampling_mean/PTH/best_model_hp92_fold{fold}.pth
- Features CSV: data/features/acoustic_features_vf.csv
- Labels CSV: src_GAMMA/paired_healthcode.csv
- Per-fold SHAP outputs: Results/MLP_v2_SHAP/shap_feature_importance_fold{fold}_top10.csv/png

Usage:
------
Run this script after training MLP models for each fold. It will aggregate, plot, and save summary files for reporting and interpretation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import os

# ===== Paths and Configurations =====
train_split_path = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
valtest_split_path = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
labels_path = "/tremor2tensor/src_GAMMA/paired_healthcode.csv"
combined_features_path = "/mloscratch/users/gnahas/data/features/acoustic_features_vf.csv"
model_dir = Path("/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/MLP_v2_HP_Search")
output_dir = Path("/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/MLP_v2_SHAP")
output_dir.mkdir(exist_ok=True, parents=True)

FOLDS = [0, 1, 2, 3, 4]  # Run for all folds
BATCH_SIZE = 64

# ===== MLP Model (same as mlp_simplified.py) =====
class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) for tabular feature classification.

    Args:
        input_dim (int): Number of input features.
        hidden_dim_1 (int): Size of first hidden layer.
        hidden_dim_2 (int): Size of second hidden layer.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, input_dim, hidden_dim_1=32, hidden_dim_2=64, dropout_rate=0.5):
        super(MLP, self).__init__()
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
    def forward(self, x):
        return self.net(x)

# ===== Utility: Get test set for a fold =====
def get_test_data_for_fold(fold, features_df, labels_df):
    """
    Prepare the test set for a given fold, including feature scaling and label extraction.

    Args:
        fold (int): Fold index (0-4).
        features_df (pd.DataFrame): DataFrame of all features.
        labels_df (pd.DataFrame): DataFrame of healthCode/label mapping.

    Returns:
        tuple:
            X_test (np.ndarray): Scaled test features.
            y_test (np.ndarray): Test labels.
            test_df (pd.DataFrame): Test set DataFrame (with metadata).
            feature_cols (list): List of feature column names.
    """
    
    valtest_split = pd.read_csv(valtest_split_path)
    test_hc = valtest_split[(valtest_split["fold_iteration"] == fold) & (valtest_split["subset"] == "test")]["healthCode"]
    data_df = features_df.merge(labels_df, on="healthCode", how="inner")
    
    if "modality" in data_df.columns:
        data_df = data_df.drop(columns=["modality"])
    
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    
    data_df[numeric_cols] = data_df[numeric_cols].fillna(data_df[numeric_cols].mean())
    test_df = data_df[data_df["healthCode"].isin(test_hc)].copy()
    
    feature_cols = [c for c in data_df.columns if c not in ["healthCode", "record_id", "label_PD", "filename"]]
    
    scaler = StandardScaler()
    scaler.fit(data_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    y_test = test_df["label_PD"].values.astype(np.int32)
    
    return X_test, y_test, test_df, feature_cols

# ===== Main SHAP Analysis =====
def main():
    features_df = pd.read_csv(combined_features_path)
    labels_df = pd.read_csv(labels_path, delimiter=",")
    labels_df["label_PD"] = labels_df["label_PD"].astype(int)
    
    if 'healthcode' in features_df.columns:
        features_df = features_df.rename(columns={'healthcode': 'healthCode'})

    # Hyperparameters: Use the best found in your grid search (update as needed)
    best_hp = dict(hidden_dim_1=32, hidden_dim_2=64, dropout_rate=0.5)

    for fold in FOLDS:
        print(f"\n=== SHAP Analysis: Fold {fold} ===")
        
        # Model path (update if your pathing is different)
        model_path = model_dir / f"undersampling_mean/PTH/best_model_hp92_fold{fold}.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        # Prepare test data
        X_test, y_test, test_df, feature_cols = get_test_data_for_fold(fold, features_df, labels_df)
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(len(feature_cols), **best_hp).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # SHAP wrapper
        def model_predict(X):
            X_tensor = torch.from_numpy(X).float().to(device)
            with torch.no_grad():
                logits = model(X_tensor)
                proba = torch.softmax(logits, dim=1)
            return proba[:, 1].cpu().numpy()
        
        # Use a subset for background for efficiency
        background_indices = np.random.choice(len(X_test), size=min(100, len(X_test)//2), replace=False)
        X_background = X_test[background_indices]
        explainer = shap.KernelExplainer(model_predict, X_background)
        
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(X_test)
        
        # Feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Select top 10 features
        selected_features = feature_importance.head(10)
        selected_indices = [feature_cols.index(f) for f in selected_features['feature']]
        
        # Beeswarm plot for selected features
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, selected_indices],
            X_test[:, selected_indices],
            feature_names=selected_features['feature'].tolist(),
            show=False,
            plot_type="dot",
            color_bar=True
        )
        
        plt.title(f"SHAP Feature Importance Summary (Fold {fold})\n(Red = High feature value, Blue = Low feature value)")
        plt.tight_layout()
        plot_path = output_dir / f"shap_feature_importance_fold{fold}_top10.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP plot saved: {plot_path}")
        plt.close()
        
        # Save CSV
        csv_path = output_dir / f"shap_feature_importance_fold{fold}_top10.csv"
        feature_importance.to_csv(csv_path, index=False)
        print(f"✓ SHAP CSV saved: {csv_path}")

if __name__ == "__main__":
    main()
