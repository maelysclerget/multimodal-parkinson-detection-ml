"""
MLP-V3: Hyperparameter Grid Search with Correlation Threshold
Based on baseline-MLP-upgraded.py with added feature correlation filtering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

train_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_train.csv"
valtest_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_val_test.csv"
labels_path = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv"

combined_features_path = "/mloscratch/users/clerget/data/csv/tapping_combined_features_session.csv"

NUM_FOLDS = 5
batch_size = 64

# ===== Hyperparameter Grid for Tuning =====
HYPERPARAMETER_GRID = {
    'learning_rate': [0.001, 0.0005],
    'weight_decay': [1e-4, 1e-3, 5e-3],
    'dropout_rate': [0.2, 0.5, 0.7],
    'hidden_dim_1': [32, 64, 128],
    'hidden_dim_2': [16, 32, 64],
}

# Correlation threshold for feature removal
CORRELATION_THRESHOLD = 0.85

# ===== Define MLP Model =====
class MLP(nn.Module):
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
            nn.Linear(16, 2)  # Output 2 classes for CrossEntropyLoss
        )
    
    def forward(self, x):
        # Return raw logits for use with CrossEntropyLoss
        return self.net(x)


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
    
    elif aggregation_method == 'majority':
        patient_preds = results_df.groupby('healthCode').agg({
            'pred_binary': lambda x: 1 if (x > 0.5).sum() > len(x) / 2 else 0,
            'pred_proba': 'mean',
            'label': 'first'
        }).reset_index()
    
    elif aggregation_method == 'max':
        patient_preds = results_df.groupby('healthCode').agg({
            'pred_proba': 'max',
            'label': 'first'
        }).reset_index()
        patient_preds['pred_binary'] = (patient_preds['pred_proba'] > 0.5).astype(int)
    
    return patient_preds


def remove_highly_correlated_features(features_df, threshold):
    """Remove features that are highly correlated above threshold (greedy approach)"""
    
    # Get feature columns only
    feature_cols = [c for c in features_df.columns if c not in ["healthCode", "trial_id", "label_PD"]]
    X = features_df[feature_cols].values
    
    # Compute correlation matrix
    correlation_matrix = pd.DataFrame(X, columns=feature_cols).corr()
    
    # Track features to remove
    features_to_remove = set()
    
    # Get correlation pairs above threshold
    upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    for i in range(len(correlation_matrix.columns)):
        if correlation_matrix.columns[i] in features_to_remove:
            continue
            
        for j in range(i+1, len(correlation_matrix.columns)):
            if upper_triangle[i, j]:
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    features_to_remove.add(correlation_matrix.columns[j])
    
    # Get features to keep
    features_to_keep = [f for f in feature_cols if f not in features_to_remove]
    
    return features_to_keep, len(features_to_remove)


def create_balanced_sampler(y_train):
    """
    Create a WeightedRandomSampler for 50-50 balanced batch sampling at SESSION level.
    
    Uses the ratio num_class_0 / num_class_1 as PD weight to achieve 50-50 split:
    - Healthy sessions weight: 1.0
    - PD sessions weight: num_healthy / num_pd
    - Result: ~50% Healthy, ~50% PD in each batch
    
    Math: Expected batch ratio = n_healthy × 1.0 / (n_healthy × 1.0 + n_pd × (n_healthy/n_pd))
                               = n_healthy / (n_healthy + n_healthy) 
                               = 50-50 split
    """
    
    num_class_0 = np.sum(y_train == 0)  # Healthy sessions
    num_class_1 = np.sum(y_train == 1)  # PD sessions
    
    # Calculate weight factor for 50-50 balance
    pd_weight_factor = num_class_0 / num_class_1
    weights = np.where(y_train == 0, 1.0, pd_weight_factor)
    
    print(f"  Using 50-50 balanced sampler:")
    print(f"    - Healthy sessions: {num_class_0} (weight=1.0)")
    print(f"    - PD sessions: {num_class_1} (weight={pd_weight_factor:.4f})")
    print(f"    - Expected batch ratio: ~50% Healthy, ~50% PD")
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def train_and_evaluate(features_df, model_name, model_prefix, fold=0, hyperparams=None, use_scheduler=True, use_balanced_sampler=True, correlation_threshold=None):
    print(f"\n Model: {model_name} | Fold: {fold}")
    print(f" ├─ Learning Rate Scheduler: {'✓ Enabled' if use_scheduler else '✗ Disabled'}")
    print(f" └─ Balanced Sampler (50-50 Healthy/PD): {'✓ Enabled' if use_balanced_sampler else '✗ Disabled'}")
    
    # Use default hyperparameters if not provided
    if hyperparams is None:
        hyperparams = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'dropout_rate': 0.5,
            'hidden_dim_1': 64,
            'hidden_dim_2': 32,
        }
    
    # Extract hyperparameters
    learning_rate = hyperparams.get('learning_rate', 1e-3)
    weight_decay = hyperparams.get('weight_decay', 1e-3)
    dropout_rate = hyperparams.get('dropout_rate', 0.5)
    hidden_dim_1 = hyperparams.get('hidden_dim_1', 64)
    hidden_dim_2 = hyperparams.get('hidden_dim_2', 32)
    
    if fold == 0:
        print(f"Hyperparameters: lr={learning_rate}, wd={weight_decay}, dropout={dropout_rate}, "
              f"h1={hidden_dim_1}, h2={hidden_dim_2}")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    labels_df["label_PD"] = labels_df["label_PD"].astype(int)
    
    # Merge features with labels
    data_df = features_df.merge(labels_df, on="healthCode", how="inner")
    
    # Drop modality if exists
    if "modality" in data_df.columns:
        data_df = data_df.drop(columns=["modality"])
        
    # if 1 trial, std is NaN --> fill NaN with mean
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    data_df[numeric_cols] = data_df[numeric_cols].fillna(data_df[numeric_cols].mean())
    
    print(f"Total patients: {len(data_df)}")
    
    # Load splits
    train_split = pd.read_csv(train_split_path)
    valtest_split = pd.read_csv(valtest_split_path)
    
    # Get healthCodes for each split
    train_hc = train_split[(train_split["fold_iteration"] == fold) & (train_split["subset"] == "train")]["healthCode"]
    val_hc = valtest_split[(valtest_split["fold_iteration"] == fold) & (valtest_split["subset"] == "val")]["healthCode"]
    test_hc = valtest_split[(valtest_split["fold_iteration"] == fold) & (valtest_split["subset"] == "test")]["healthCode"]
    
    # Filter dataframes
    train_df = data_df[data_df["healthCode"].isin(train_hc)].copy()
    val_df = data_df[data_df["healthCode"].isin(val_hc)].copy()
    test_df = data_df[data_df["healthCode"].isin(test_hc)].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Extract features (exclude healthCode, trial_id, and label)
    feature_cols = [c for c in data_df.columns if c not in ["healthCode", "trial_id", "label_PD"]]
    
    # Remove highly correlated features if threshold specified
    if correlation_threshold is not None and correlation_threshold < 1.0:
        feature_cols, n_removed = remove_highly_correlated_features(data_df, correlation_threshold)
        print(f"  Features after filtering (threshold={correlation_threshold}): {len(feature_cols)} (removed {n_removed})")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    X_val = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    
    y_train = train_df["label_PD"].values.astype(np.float32)
    y_val = val_df["label_PD"].values.astype(np.float32)
    y_test = test_df["label_PD"].values.astype(np.float32)
    
    # Count class distribution for info
    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1)
    
    print(f"Class distribution - Healthy (0): {num_class_0}, PD (1): {num_class_1}")
    print(f"Note: Using 50-50 balanced sampler, so no additional class weights needed")
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)
    
    y_train_t = torch.from_numpy(y_train).long()
    y_val_t = torch.from_numpy(y_val).long()
    y_test_t = torch.from_numpy(y_test).long()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # ===== UPGRADE 1: Create balanced sampler for training =====
    if use_balanced_sampler:
        print("\n[UPGRADE 1] Per-batch 50-50 balancing of Healthy/PD sessions:")
        train_sampler = create_balanced_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP(len(feature_cols), hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, 
                dropout_rate=dropout_rate).to(device)
    
    # Use standard CrossEntropyLoss (no class weights since using 50-50 balanced sampler)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # ===== UPGRADE 2: Dynamic Learning Rate Scheduler =====
    if use_scheduler:
        print("\n[UPGRADE 2] Dynamic Learning Rate Scheduler (ReduceLROnPlateau):")
        print("  ├─ Mode: max (increase LR when metric improves)")
        print("  ├─ Factor: 0.5 (reduce LR by 50% when plateau)")
        print("  ├─ Patience: 5 epochs")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximize validation AUC
            factor=0.5,  # Reduce LR by 50%
            patience=5,  # Wait 5 epochs before reducing
            min_lr=1e-6  # Don't go below this LR
        )
    else:
        scheduler = None
    
    # Training loop
    num_epochs = 50
    best_val_auc = 0.0
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_aucs = []
    learning_rates = []
    
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
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_preds_proba = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_logits = model(X_batch)
                y_proba = torch.softmax(y_logits, dim=1)
                val_preds_proba.extend(y_proba[:, 1].cpu().numpy().tolist())
                val_labels.extend(y_batch.cpu().numpy().tolist())
        
        val_auc = roc_auc_score(val_labels, val_preds_proba)
        val_aucs.append(val_auc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # ===== Step scheduler =====
        if scheduler is not None:
            scheduler.step(val_auc)
        
        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # Save best model based on VAL
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Load best model before test evaluation
    model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_preds_proba = []
    test_preds_binary = []
    test_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_logits = model(X_batch)
            y_proba = torch.softmax(y_logits, dim=1)
            y_pred_proba_class1 = y_proba[:, 1]
            test_preds_proba.extend(y_pred_proba_class1.cpu().numpy().tolist())
            test_preds_binary.extend((y_pred_proba_class1 > 0.5).cpu().numpy().tolist())
            test_labels.extend(y_batch.numpy().tolist())
    
    # Aggregate to patient-level
    patient_preds = aggregate_predictions(test_df, test_preds_proba, test_preds_binary, test_labels, aggregation_method='mean')
    
    patient_accuracy = accuracy_score(patient_preds['label'], patient_preds['pred_binary'])
    patient_f1 = f1_score(patient_preds['label'], patient_preds['pred_binary'])
    patient_auc = roc_auc_score(patient_preds['label'], patient_preds['pred_proba'])
    
    print(f"\nPatient-level Accuracy: {patient_accuracy:.4f}")
    print(f"Patient-level F1-Score: {patient_f1:.4f}")
    print(f"Patient-level AUC-ROC:  {patient_auc:.4f}")
    
    return {
        'model': model_name,
        'fold': fold,
        'accuracy': patient_accuracy,
        'f1_score': patient_f1,
        'auc': patient_auc,
        'num_patients': len(patient_preds),
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout_rate': dropout_rate,
        'hidden_dim_1': hidden_dim_1,
        'hidden_dim_2': hidden_dim_2,
    }


# ===== Main =====
if __name__ == "__main__":
    
    combined_features = pd.read_csv(combined_features_path)
    
    print("\n===== Feature Sets Loaded =====")
    print(f"Combined features shape: {combined_features.shape}\n")
    
    print("\n" + "="*80)
    print("MLP-V3: HYPERPARAMETER GRID SEARCH WITH FEATURE CORRELATION FILTERING")
    print("="*80)
    print("\n[CONFIGURATION]")
    print(f"  Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"  Hyperparameter combinations: {len(list(product(*HYPERPARAMETER_GRID.values())))}")
    print(f"  Cross-validation folds: {NUM_FOLDS}")
    print("="*80)
    
    # Generate all hyperparameter combinations
    hp_combinations = list(product(*HYPERPARAMETER_GRID.values()))
    hp_keys = list(HYPERPARAMETER_GRID.keys())
    
    # Store results for all combinations and folds
    all_results = []
    
    # Loop through all hyperparameter combinations
    for hp_idx, hp_values in enumerate(hp_combinations):
        hyperparams = dict(zip(hp_keys, hp_values))
        
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER COMBINATION {hp_idx + 1}/{len(hp_combinations)}")
        print(f"  lr={hyperparams['learning_rate']}, wd={hyperparams['weight_decay']}, "
              f"dr={hyperparams['dropout_rate']}, h1={hyperparams['hidden_dim_1']}, "
              f"h2={hyperparams['hidden_dim_2']}")
        print(f"{'='*80}")
        
        # Loop through all 5 folds
        for fold in range(NUM_FOLDS):
            print(f"\n  FOLD {fold}/{NUM_FOLDS - 1}")
            
            result = train_and_evaluate(
                combined_features, 
                f"HP{hp_idx}", 
                f"hp{hp_idx}", 
                fold=fold, 
                hyperparams=hyperparams, 
                use_scheduler=True, 
                use_balanced_sampler=True,
                correlation_threshold=CORRELATION_THRESHOLD
            )
            all_results.append(result)
    
    # Convert all results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate mean metrics for each hyperparameter combination
    print("\n" + "="*80)
    print("HYPERPARAMETER GRID SEARCH RESULTS")
    print("="*80)
    
    mean_results = []
    for hp_idx in range(len(hp_combinations)):
        hp_results = results_df[results_df['model'] == f'HP{hp_idx}']
        
        mean_accuracy = hp_results['accuracy'].mean()
        std_accuracy = hp_results['accuracy'].std()
        
        mean_f1 = hp_results['f1_score'].mean()
        std_f1 = hp_results['f1_score'].std()
        
        mean_auc = hp_results['auc'].mean()
        std_auc = hp_results['auc'].std()
        
        hyperparams = dict(zip(hp_keys, hp_combinations[hp_idx]))
        
        mean_results.append({
            'hp_idx': hp_idx,
            'learning_rate': hyperparams['learning_rate'],
            'weight_decay': hyperparams['weight_decay'],
            'dropout_rate': hyperparams['dropout_rate'],
            'hidden_dim_1': hyperparams['hidden_dim_1'],
            'hidden_dim_2': hyperparams['hidden_dim_2'],
            'accuracy_mean': mean_accuracy,
            'accuracy_std': std_accuracy,
            'f1_mean': mean_f1,
            'f1_std': std_f1,
            'auc_mean': mean_auc,
            'auc_std': std_auc,
        })
    
    mean_results_df = pd.DataFrame(mean_results)
    
    # Find best hyperparameters
    best_idx = mean_results_df['auc_mean'].idxmax()
    best_row = mean_results_df.iloc[best_idx]
    
    print(f"\n{mean_results_df.to_string(index=False)}\n")
    
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"Learning Rate: {best_row['learning_rate']}")
    print(f"Weight Decay: {best_row['weight_decay']}")
    print(f"Dropout Rate: {best_row['dropout_rate']}")
    print(f"Hidden Dim 1: {best_row['hidden_dim_1']}")
    print(f"Hidden Dim 2: {best_row['hidden_dim_2']}")
    print(f"\nAccuracy: {best_row['accuracy_mean']:.4f} ± {best_row['accuracy_std']:.4f}")
    print(f"F1-Score: {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}")
    print(f"AUC-ROC:  {best_row['auc_mean']:.4f} ± {best_row['auc_std']:.4f}")
    print(f"{'='*80}")
    
    # Save results
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/cv_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save all fold results
    all_csv = output_dir / f'mlp_v3_hp_search_threshold{CORRELATION_THRESHOLD}_all_folds.csv'
    results_df.to_csv(all_csv, index=False)
    print(f"\n✓ All fold results saved to: {all_csv}")
    
    # Save mean results
    mean_csv = output_dir / f'mlp_v3_hp_search_threshold{CORRELATION_THRESHOLD}_mean.csv'
    mean_results_df.to_csv(mean_csv, index=False)
    print(f"✓ Mean results saved to: {mean_csv}")
    
    print("\n" + "="*80)
    print("✓ MLP-V3 Hyperparameter Grid Search Complete!")
    print("="*80)
