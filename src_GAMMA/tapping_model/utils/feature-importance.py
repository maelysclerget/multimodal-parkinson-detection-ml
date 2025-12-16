"""
Feature Importance Analysis for MLP - Tapping Modality

Analyzes which tapping features are most important for PD prediction using
permutation importance on a trained MLP model.

Uses the best hyperparameters found from V2 grid search.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import shap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# ===== Paths =====
train_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_train.csv"
valtest_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_val_test.csv"
labels_path = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
combined_features_path = "/mloscratch/users/clerget/data/csv/tapping_combined_features_session.csv"

NUM_FOLDS = 5
batch_size = 64

# Best hyperparameters from MLP V2 grid search
BEST_HYPERPARAMS = {
    'learning_rate': 0.0005,
    'weight_decay': 0.005,
    'dropout_rate': 0.2,
    'hidden_dim_1': 32,
    'hidden_dim_2': 16,
}


# ===== MLP Model =====
class MLP(nn.Module):
    """Multi-layer Perceptron for binary classification."""
    def __init__(self, input_dim, hidden_dim_1=64, hidden_dim_2=32, dropout_rate=0.5):
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


def create_balanced_sampler(y_train):
    """Create weighted sampler for 50-50 balance"""
    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1)
    pd_weight_factor = num_class_0 / num_class_1
    weights = np.where(y_train == 0, 1.0, pd_weight_factor)
    
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler


def aggregate_predictions(test_df, test_preds_proba, test_preds_binary, test_labels, aggregation_method='mean'):
    """Aggregate trial-level predictions to patient-level predictions."""
    results_df = pd.DataFrame({
        'healthCode': test_df['healthCode'].values,
        'pred_proba': test_preds_proba,
        'pred_binary': test_preds_binary,
        'label': test_labels
    })
    
    if aggregation_method == 'mean':
        patient_preds = results_df.groupby('healthCode').agg({
            'pred_proba': 'mean',
            'label': 'first'
        }).reset_index()
        patient_preds['pred_binary'] = (patient_preds['pred_proba'] > 0.5).astype(int)
    
    return patient_preds


def mlp_predict_proba(model, X, device, batch_size=64):
    """Get probability predictions from MLP"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            logits = model(X_batch)
            proba = torch.softmax(logits, dim=1)
            predictions.extend(proba[:, 1].cpu().numpy().tolist())
    
    return np.array(predictions)


# ===== Main =====
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS - MLP TAPPING MODEL")
    print("="*80)
    print("Permutation importance: drop in AUC when each feature is shuffled")
    
    # Load data
    combined_features = pd.read_csv(combined_features_path)
    labels_df = pd.read_csv(labels_path)
    labels_df["label_PD"] = labels_df["label_PD"].astype(int)
    
    # Merge
    data_df = combined_features.merge(labels_df, on="healthCode", how="inner")
    if "modality" in data_df.columns:
        data_df = data_df.drop(columns=["modality"])
    
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    data_df[numeric_cols] = data_df[numeric_cols].fillna(data_df[numeric_cols].mean())
    
    print(f"\nTotal samples: {len(data_df)}")
    
    # Load splits for fold 0 (using first fold for analysis)
    train_split = pd.read_csv(train_split_path)
    valtest_split = pd.read_csv(valtest_split_path)
    
    fold = 0
    train_hc = train_split[(train_split["fold_iteration"] == fold) & (train_split["subset"] == "train")]["healthCode"]
    test_hc = valtest_split[(valtest_split["fold_iteration"] == fold) & (valtest_split["subset"] == "test")]["healthCode"]
    
    train_df = data_df[data_df["healthCode"].isin(train_hc)].copy()
    test_df = data_df[data_df["healthCode"].isin(test_hc)].copy()
    
    print(f"Fold 0 - Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Extract all features (no filtering)
    feature_cols = [c for c in data_df.columns if c not in ["healthCode", "trial_id", "label_PD"]]
    print(f"Total features: {len(feature_cols)}")

    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    
    y_train = train_df["label_PD"].values.astype(np.int32)
    y_test = test_df["label_PD"].values.astype(np.int32)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    X_test_t = torch.from_numpy(X_test)
    y_train_t = torch.from_numpy(y_train).long()
    y_test_t = torch.from_numpy(y_test).long()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # Create balanced sampler
    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1)
    pd_weight_factor = num_class_0 / num_class_1
    weights = np.where(y_train == 0, 1.0, pd_weight_factor)
    train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ===== TRAIN MLP =====
    print("[TRAINING MLP]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    model = MLP(len(feature_cols), 
                hidden_dim_1=BEST_HYPERPARAMS['hidden_dim_1'],
                hidden_dim_2=BEST_HYPERPARAMS['hidden_dim_2'],
                dropout_rate=BEST_HYPERPARAMS['dropout_rate']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=BEST_HYPERPARAMS['learning_rate'],
                                weight_decay=BEST_HYPERPARAMS['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training loop
    num_epochs = 50
    best_val_auc = 0.0
    patience = 15
    patience_counter = 0
    
    print("Training for 50 epochs (early stopping at patience=15)...")
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_logits = model(X_batch)
            loss = criterion(y_logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation on test set
        model.eval()
        test_preds_proba = []
        test_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_logits = model(X_batch)
                y_proba = torch.softmax(y_logits, dim=1)
                test_preds_proba.extend(y_proba[:, 1].cpu().numpy().tolist())
                test_labels.extend(y_batch.cpu().numpy().tolist())
        
        test_auc = roc_auc_score(test_labels, test_preds_proba)
        
        if test_auc > best_val_auc:
            best_val_auc = test_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(test_auc)
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"✓ Training complete. Best Test AUC: {best_val_auc:.4f}\n")
    
    # ...existing code up to SHAP calculation...

    # ===== SHAP VALUES =====
    print("[CALCULATING SHAP VALUES]")
    print("This measures each feature's contribution to predictions...\n")
    
    # Create a wrapper function for SHAP
    def model_predict(X):
        """Wrapper for SHAP to get probability predictions"""
        X_tensor = torch.from_numpy(X).float().to(device)
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            proba = torch.softmax(logits, dim=1)
        return proba[:, 1].cpu().numpy()
    
    # Use KernelExplainer (model-agnostic, works with any model)
    print("Initializing SHAP KernelExplainer...")
    print("  (Using 100 background samples for efficiency)\n")
    
    # Sample background data for faster computation
    background_indices = np.random.choice(len(X_test), size=min(100, len(X_test)//2), replace=False)
    X_background = X_test[background_indices]
    
    explainer = shap.KernelExplainer(model_predict, X_background)
    
    # Calculate SHAP values for test set
    print("Computing SHAP values for test set...")
    shap_values = explainer.shap_values(X_test)
    
    print("✓ SHAP values computed\n")
    
    # Calculate mean absolute SHAP values for feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"✓ Feature Importance (by Mean |SHAP|):")
    print(feature_importance.to_string(index=False))
    
    # ===== VISUALIZATION - SINGLE SHAP SUMMARY PLOT =====
    print("\n[CREATING SHAP SUMMARY PLOT]")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create beeswarm-style plot
    y_pos = 0
    y_labels = []
    y_ticks = []
    
    for feat_idx, feat_name in enumerate(feature_importance['feature'].values):
        # Find index in original feature_cols
        original_idx = feature_cols.index(feat_name)
        
        shap_vals = shap_values[:, original_idx]
        X_vals = X_test[:, original_idx]
        
        # Normalize X values for color coding (0-1)
        X_norm = (X_vals - X_vals.min()) / (X_vals.max() - X_vals.min() + 1e-8)
        
        # Add jitter for visualization
        y_jitter = np.random.normal(y_pos, 0.04, len(shap_vals))
        
        # Create scatter plot with color gradient
        scatter = ax.scatter(shap_vals, y_jitter, c=X_norm, cmap='coolwarm', 
                            alpha=0.7, s=40, edgecolor='gray', linewidth=0.3, vmin=0, vmax=1)
        
        y_labels.append(feat_name)
        y_ticks.append(y_pos)
        y_pos -= 1
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontweight='bold', fontsize=12)
    ax.set_title('SHAP Feature Importance Summary\n(Red = High feature value, Blue = Low feature value)', 
                fontweight='bold', fontsize=13)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Feature Value\n(Low ← → High)', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_path = output_dir / 'feature_importance_mlp_shap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # ===== SAVE CSV =====
    print("\n[SAVING RESULTS]")
    
    csv_path = output_dir / 'feature_importance_mlp_shap.csv'
    feature_importance.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    print("\n" + "="*80)
    print("✓ SHAP Feature Importance Analysis Complete!")
    print("="*80)
    print(f"\nTop 5 Most Important Features:")
    for idx, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"  {idx}. {row['feature']:<30} (Mean |SHAP|: {row['mean_abs_shap']:.4f})")
    print("="*80)