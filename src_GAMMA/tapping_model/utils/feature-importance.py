"""
Feature Importance Analysis for MLP - Tapping Modality

Analyzes which tapping features are most important for PD prediction using
permutation importance on a trained MLP model.

Uses the best hyperparameters found from V2 grid search.
"""

import numpy as np
import pandas as pd
from pathlib import Path

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
    
    # ===== PERMUTATION IMPORTANCE =====
    print("[CALCULATING PERMUTATION IMPORTANCE]")
    print("This measures how much AUC drops when each feature is shuffled...")
    print("Higher value = feature is more important\n")
    
    # Get model predictions as baseline
    model.eval()
    baseline_pred_proba = mlp_predict_proba(model, X_test, device, batch_size)
    baseline_auc = roc_auc_score(y_test, baseline_pred_proba)
    print(f"Baseline AUC: {baseline_auc:.4f}\n")
    
    # Calculate permutation importance
    feature_importance = []
    
    for feat_idx, feat_name in enumerate(feature_cols):
        print(f"  Testing feature {feat_idx + 1}/{len(feature_cols)}: {feat_name}...", end='', flush=True)
        
        # Shuffle this feature
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled[:, feat_idx])
        
        # Get predictions on shuffled data
        shuffled_pred_proba = mlp_predict_proba(model, X_test_shuffled, device, batch_size)
        shuffled_auc = roc_auc_score(y_test, shuffled_pred_proba)
        
        # Importance = drop in AUC
        importance = baseline_auc - shuffled_auc
        feature_importance.append({
            'feature': feat_name,
            'importance': importance,
            'shuffled_auc': shuffled_auc
        })
        
        print(f" ✓ (importance: {importance:.4f})")
    
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)
    
    print(f"\n✓ Permutation Importance Results:")
    print(importance_df.to_string(index=False))
    
    # ===== VISUALIZATION =====
    print("\n[CREATING VISUALIZATION]")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importance Analysis - MLP Tapping Model\n(Permutation Importance on Test Set)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Feature importance bar chart
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_features)))
    bars = axes[0].barh(range(len(top_features)), top_features['importance'].values, color=colors, alpha=0.8)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'].values, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance (ΔAUC when shuffled)', fontweight='bold', fontsize=12)
    axes[0].set_title(f'Top {top_n} Most Important Features', fontweight='bold', fontsize=12)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        axes[0].text(row['importance'] + 0.0005, i, f"{row['importance']:.4f}", 
                    va='center', fontsize=10)
    
    # Plot 2: Summary statistics
    axes[1].axis('off')
    summary_text = f"""
FEATURE IMPORTANCE ANALYSIS SUMMARY
{'='*55}

Model Configuration:
  • Learning Rate: {BEST_HYPERPARAMS['learning_rate']}
  • Weight Decay: {BEST_HYPERPARAMS['weight_decay']}
  • Dropout Rate: {BEST_HYPERPARAMS['dropout_rate']}
  • Hidden Dims: {BEST_HYPERPARAMS['hidden_dim_1']} → {BEST_HYPERPARAMS['hidden_dim_2']}

Dataset:
  • Total features: {len(feature_cols)}
  • Test samples: {len(test_df)}
  • Baseline AUC: {baseline_auc:.4f}

Top 5 Most Important Features:
"""
    for idx, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        summary_text += f"\n  {idx}. {row['feature']:<25} (importance: {row['importance']:.4f})"
    
    summary_text += f"""

Interpretation:
  • Importance = Drop in AUC when feature is shuffled
  • Higher values = More critical for predictions
  • If shuffling a feature barely changes AUC,
    it's not important for this model's decisions

Bottom 5 Least Important Features:
"""
    for idx, (_, row) in enumerate(importance_df.tail(5).iloc[::-1].iterrows(), 1):
        summary_text += f"\n  {idx}. {row['feature']:<25} (importance: {row['importance']:.4f})"
    
    axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_path = output_dir / 'feature_importance_mlp_permutation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # ===== SAVE CSV =====
    print("\n[SAVING RESULTS]")
    
    csv_path = output_dir / 'feature_importance_mlp_permutation.csv'
    importance_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    print("\n" + "="*80)
    print("✓ Feature Importance Analysis Complete!")
    print("="*80)
    print(f"\nKey Insight:")
    print(f"  Most important feature: {importance_df.iloc[0]['feature']}")
    print(f"  Importance score: {importance_df.iloc[0]['importance']:.4f}")
    print(f"\n  This means shuffling '{importance_df.iloc[0]['feature']}' reduces")
    print(f"  the model's AUC by {importance_df.iloc[0]['importance']:.4f}")
    print("="*80)
