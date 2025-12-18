"""
MLP-v2: Hyperparameter Grid Search with 5-Fold Cross-Validation & Class Imbalance Handling

This script performs a comprehensive hyperparameter grid search for an MLP classifier
using 5-fold cross-validation on tapping features to classify PD vs Healthy subjects.

Key Features:
- Hyperparameter grid search over learning rate, weight decay, dropout, and hidden dimensions
- 5-fold cross-validation with selectable class imbalance handling
- Dynamic learning rate scheduling (ReduceLROnPlateau)
- Early stopping based on validation AUC
- Saves best performing model (all 5 folds) to data/saved_models/
- Generates detailed results CSV with per-fold and mean metrics
- Creates comprehensive visualizations of results

Class Imbalance Handling (SELECT ONE):
- 'weighted_sampler': WeightedRandomSampler for 50-50 per-batch balancing (DEFAULT)
- 'class_weights': Loss-level weighting via CrossEntropyLoss weights
- 'undersampling': Randomly remove majority class samples to match minority size
- 'baseline': No balancing, train on raw imbalanced data

Usage:
BALANCING_METHOD = 'weighted_sampler'  # Change to: 'class_weights', 'undersampling', 'baseline'

Output Files (with dynamic method identifier):
- CV results CSV:
  * mlp_v2_hp_search_mean_{method}.csv - Mean metrics per hyperparameter combination
  
- Visualization figures (in results/):
  * mlp_v2_best_model_summary_{method}.png - Detailed summary of best model across all folds
  
- Best model weights (in data/saved_models/):
  * best_model_hp{X}_fold{Y}.pth - PyTorch model weights for all 5 folds
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
import matplotlib.gridspec as gridspec
import seaborn as sns

# ===== Paths and Configurations =====
train_split_path = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
valtest_split_path = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
labels_path = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
combined_features_path = "/mloscratch/users/clerget/data/csv/tapping_combined_features_session.csv"

NUM_FOLDS = 5
batch_size = 64

# ===== SELECT CLASS IMBALANCE HANDLING METHOD =====
# Change to one of: 'weighted_sampler', 'class_weights', 'undersampling', 'baseline'
BALANCING_METHOD = 'undersampling'

# ===== Hyperparameter Grid for Tuning =====
HYPERPARAMETER_GRID = {
    'learning_rate': [0.001, 0.0005],
    'weight_decay': [1e-4, 1e-3, 5e-3],
    'dropout_rate': [0.2, 0.5, 0.7],
    'hidden_dim_1': [32, 64, 128],
    'hidden_dim_2': [16, 32, 64],
}

# ===== Define MLP Model =====
class MLP(nn.Module):
    """
    Multi-layer Perceptron for binary classification (PD vs Healthy).
    
    Architecture:
    - Input Layer: variable input_dim
    - Hidden Layer 1: hidden_dim_1 neurons with ReLU + Dropout
    - Hidden Layer 2: hidden_dim_2 neurons with ReLU + Dropout
    - Hidden Layer 3: 16 neurons with ReLU + Dropout(0.2)
    - Output Layer: 2 neurons (logits for CrossEntropyLoss)
    
    Args:
        input_dim (int): Number of input features
        hidden_dim_1 (int): Number of neurons in first hidden layer (default: 64)
        hidden_dim_2 (int): Number of neurons in second hidden layer (default: 32)
        dropout_rate (float): Dropout rate for first two hidden layers (default: 0.5)
    """
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
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        return self.net(x)


def aggregate_predictions(test_df, test_preds_proba, test_preds_binary, test_labels, aggregation_method='majority'):
    """
    Aggregate trial-level predictions to patient-level predictions.
    
    Since a patient can have multiple tapping sessions/trials, this function aggregates
    the session-level predictions to get a single prediction per patient.
    
    Args:
        test_df (pd.DataFrame): Test dataframe containing healthCode column
        test_preds_proba (list): Predicted probabilities for class 1 (PD)
        test_preds_binary (list): Binary predictions (0 or 1)
        test_labels (list): True labels
        aggregation_method (str): Method to aggregate predictions. Options:
            - 'mean': Average probability across all sessions for each patient (default)
            - 'majority': Majority vote of binary predictions
            - 'max': Maximum probability across all sessions
    
    Returns:
        pd.DataFrame: Patient-level predictions with columns [healthCode, pred_proba, pred_binary, label]
    """
    
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


def create_best_model_summary(best_row, best_hp_results, output_dir, balancing_method='weighted_sampler'):
    """
    Create a summary figure for the best model across all folds.
    
    Args:
        best_row (pd.Series): Row with best hyperparameters from mean_results_df
        best_hp_results (pd.DataFrame): Results for all folds of best hyperparameters
        output_dir (Path): Directory to save figure
    
    Returns:
        None (saves figure to output_dir)
    """
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Best Model Summary (HP{int(best_row["hp_idx"])})', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Metrics across folds
    ax1 = fig.add_subplot(gs[0, :])
    folds = best_hp_results['fold'].values
    ax1.plot(folds, best_hp_results['accuracy'].values, 'o-', linewidth=2, 
            markersize=8, label='Accuracy', color='seagreen')
    ax1.plot(folds, best_hp_results['f1_score'].values, 's-', linewidth=2, 
            markersize=8, label='F1-Score', color='coral')
    ax1.plot(folds, best_hp_results['auc'].values, '^-', linewidth=2, 
            markersize=8, label='AUC-ROC', color='steelblue')
    ax1.axhline(y=best_row['accuracy_mean'], color='seagreen', linestyle='--', alpha=0.5)
    ax1.axhline(y=best_row['f1_mean'], color='coral', linestyle='--', alpha=0.5)
    ax1.axhline(y=best_row['auc_mean'], color='steelblue', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Across Folds')
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Mean metrics with std
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
    means = [best_row['accuracy_mean'], best_row['f1_mean'], best_row['auc_mean']]
    stds = [best_row['accuracy_std'], best_row['f1_std'], best_row['auc_std']]
    colors = ['seagreen', 'coral', 'steelblue']
    
    ax2.bar(metrics, means, yerr=stds, capsize=10, alpha=0.7, color=colors)
    ax2.set_ylabel('Score')
    ax2.set_title('Mean Performance ± Std Dev')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax2.text(i, m + s + 0.05, f'{m:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Hyperparameter values
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    hyperparams_text = f"""
    BEST HYPERPARAMETERS:
    
    Learning Rate:    {best_row['learning_rate']}
    Weight Decay:     {best_row['weight_decay']}
    Dropout Rate:     {best_row['dropout_rate']}
    Hidden Dim 1:     {int(best_row['hidden_dim_1'])}
    Hidden Dim 2:     {int(best_row['hidden_dim_2'])}
    
    ─────────────────────────
    
    MEAN PERFORMANCE:
    
    Accuracy:         {best_row['accuracy_mean']:.4f} ± {best_row['accuracy_std']:.4f}
    F1-Score:         {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}
    AUC-ROC:          {best_row['auc_mean']:.4f} ± {best_row['auc_std']:.4f}
    """
    
    ax3.text(0.1, 0.5, hyperparams_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Fold-by-fold comparison
    ax4 = fig.add_subplot(gs[2, :])
    x = np.arange(len(best_hp_results))
    width = 0.25
    
    ax4.bar(x - width, best_hp_results['accuracy'].values, width, 
           label='Accuracy', alpha=0.8, color='seagreen')
    ax4.bar(x, best_hp_results['f1_score'].values, width, 
           label='F1-Score', alpha=0.8, color='coral')
    ax4.bar(x + width, best_hp_results['auc'].values, width, 
           label='AUC-ROC', alpha=0.8, color='steelblue')
    
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Score')
    ax4.set_title('Per-Fold Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Fold {int(f)}' for f in best_hp_results['fold'].values])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.savefig(output_dir / f'mlp_v2_best_model_summary_{balancing_method}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Best model summary figure saved to: {output_dir / f'mlp_v2_best_model_summary_{balancing_method}.png'}")
    plt.close()


def apply_balancing_method(X_train, y_train, method='weighted_sampler'):
    """
    Apply class imbalance handling using one of four methods:
    - 'weighted_sampler': WeightedRandomSampler for 50-50 balanced batch sampling (default)
    - 'class_weights': Loss-level weighting via CrossEntropyLoss weights
    - 'undersampling': Randomly remove majority class samples to match minority size
    - 'baseline': No balancing, train on raw imbalanced data
    
    Args:
        X_train (np.ndarray): Training features [n_samples, n_features]
        y_train (np.ndarray): Training labels [n_samples]
        method (str): Balancing method - 'weighted_sampler' (default), 'class_weights', 'undersampling', 'baseline'
    
    Returns:
        tuple: (X_train_balanced, y_train_balanced, sampler, loss_weights, info_dict)
            - X_train_balanced: Modified features (for undersampling/baseline, original for others)
            - y_train_balanced: Modified labels (for undersampling/baseline, original for others)
            - sampler: WeightedRandomSampler (for weighted_sampler method) or None
            - loss_weights: Class weights tensor (for class_weights method) or None
            - info_dict: Dictionary with method statistics
    """
    
    num_class_0 = np.sum(y_train == 0)  # Healthy sessions
    num_class_1 = np.sum(y_train == 1)  # PD sessions
    total_samples = len(y_train)
    
    print(f"\nClass distribution - Healthy (0): {num_class_0}, PD (1): {num_class_1}")
    print(f"Total sessions: {total_samples}")
    
    if method == 'weighted_sampler':
        """
        Create a WeightedRandomSampler for 50-50 balanced batch sampling at SESSION level.
        Uses the ratio num_class_0 / num_class_1 as PD weight to achieve 50-50 split:
        - Healthy sessions weight: 1.0
        - PD sessions weight: num_healthy / num_pd
        - Result: ~50% Healthy, ~50% PD in each batch
        """
        pd_weight_factor = num_class_0 / num_class_1
        weights = np.where(y_train == 0, 1.0, pd_weight_factor)
        
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
        print(f"\n[BALANCING METHOD] Weighted Random Sampler (50-50 per-batch):")
        print(f"  ├─ Healthy sessions: {num_class_0} (weight=1.0)")
        print(f"  ├─ PD sessions: {num_class_1} (weight={pd_weight_factor:.4f})")
        print(f"  └─ Expected batch ratio: ~50% Healthy, ~50% PD")
        
        info_dict = {
            'method': 'weighted_sampler',
            'healthy_count': num_class_0,
            'pd_count': num_class_1,
            'pd_weight_factor': pd_weight_factor
        }
        
        return X_train, y_train, sampler, None, info_dict
    
    elif method == 'class_weights':
        """
        Use CrossEntropyLoss with class weights to handle imbalance.
        Weight for each class = total_samples / (2 × num_samples_in_class)
        """
        weight_class_0 = total_samples / (2 * num_class_0)
        weight_class_1 = total_samples / (2 * num_class_1)
        loss_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
        
        print(f"\n[BALANCING METHOD] Class Weights (via CrossEntropyLoss):")
        print(f"  ├─ Healthy (0) weight: {weight_class_0:.4f}")
        print(f"  ├─ PD (1) weight: {weight_class_1:.4f}")
        print(f"  └─ Training on raw {num_class_0 + num_class_1} sessions (no data modification)")
        
        info_dict = {
            'method': 'class_weights',
            'healthy_count': num_class_0,
            'pd_count': num_class_1,
            'weight_class_0': weight_class_0,
            'weight_class_1': weight_class_1
        }
        
        return X_train, y_train, None, loss_weights, info_dict
    
    elif method == 'undersampling':
        """
        Randomly remove majority class samples to match minority size.
        Then shuffle to mix both classes throughout the dataset.
        """
        min_count = min(num_class_0, num_class_1)
        
        # Get indices for each class
        idx_class_0 = np.where(y_train == 0)[0]
        idx_class_1 = np.where(y_train == 1)[0]
        
        # Randomly sample min_count from majority class
        rng = np.random.RandomState(42)
        if num_class_0 > num_class_1:
            idx_class_0 = rng.choice(idx_class_0, size=min_count, replace=False)
        else:
            idx_class_1 = rng.choice(idx_class_1, size=min_count, replace=False)
        
        # Combine and shuffle
        selected_idx = np.concatenate([idx_class_0, idx_class_1])
        rng.shuffle(selected_idx)
        
        X_train_balanced = X_train[selected_idx]
        y_train_balanced = y_train[selected_idx]
        
        print(f"\n[BALANCING METHOD] Undersampling (random removal of majority):")
        print(f"  ├─ Removed {num_class_0 - min_count if num_class_0 > num_class_1 else 0} Healthy samples")
        print(f"  ├─ Removed {num_class_1 - min_count if num_class_1 > num_class_0 else 0} PD samples")
        print(f"  ├─ Training on balanced {2 * min_count} sessions ({min_count} per class)")
        print(f"  └─ RandomState(42) for reproducibility")
        
        info_dict = {
            'method': 'undersampling',
            'healthy_count_before': num_class_0,
            'pd_count_before': num_class_1,
            'healthy_count_after': np.sum(y_train_balanced == 0),
            'pd_count_after': np.sum(y_train_balanced == 1),
            'total_removed': total_samples - len(y_train_balanced)
        }
        
        return X_train_balanced, y_train_balanced, None, None, info_dict
    
    elif method == 'baseline':
        """
        No balancing. Train on raw imbalanced data with standard shuffling.
        """
        print(f"\n[BALANCING METHOD] Baseline (no balancing):")
        print(f"  ├─ Healthy sessions: {num_class_0}")
        print(f"  ├─ PD sessions: {num_class_1}")
        print(f"  ├─ Imbalance ratio: {num_class_0 / num_class_1:.2f}:1")
        print(f"  └─ Training on raw imbalanced {total_samples} sessions")
        
        info_dict = {
            'method': 'baseline',
            'healthy_count': num_class_0,
            'pd_count': num_class_1,
            'imbalance_ratio': num_class_0 / num_class_1
        }
        
        return X_train, y_train, None, None, info_dict
    
    else:
        raise ValueError(f"Unknown balancing method: {method}. Choose from: 'weighted_sampler', 'class_weights', 'undersampling', 'baseline'")


def train_and_evaluate(features_df, model_name, model_prefix, fold=0, hyperparams=None, use_scheduler=True, balancing_method='weighted_sampler', save_model=False):
    """
    Train and evaluate MLP model on a single fold of cross-validation.
    
    Args:
        features_df (pd.DataFrame): Combined features with columns [healthCode, trial_id, label_PD, feature_1, ...]
        model_name (str): Name of the model configuration (e.g., 'HP0') for logging
        model_prefix (str): Prefix for saving model files (e.g., 'hp0')
        fold (int): Fold number in cross-validation (0-4) (default: 0)
        hyperparams (dict): Hyperparameters with keys: learning_rate, weight_decay, dropout_rate,
                           hidden_dim_1, hidden_dim_2. If None, uses defaults.
        use_scheduler (bool): Whether to use ReduceLROnPlateau scheduler (default: True)
        balancing_method (str): Class imbalance handling method (default: 'weighted_sampler')
            - 'weighted_sampler': WeightedRandomSampler for 50-50 per-batch balancing
            - 'class_weights': Loss-level weighting via CrossEntropyLoss
            - 'undersampling': Randomly remove majority class samples
            - 'baseline': No balancing, train on raw imbalanced data
        save_model (bool): Whether to save model (not used in grid search) (default: False)
    
    Returns:
        dict: Results dictionary containing:
            - 'model', 'fold': Configuration info
            - 'accuracy', 'f1_score', 'auc': Patient-level metrics
            - 'num_patients': Number of unique patients in test set
            - 'learning_rate', 'weight_decay', 'dropout_rate', 'hidden_dim_1', 'hidden_dim_2': Hyperparameters
            - 'model_state': Model state dictionary for saving
    """
    
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
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    X_val = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    
    y_train = train_df["label_PD"].values.astype(np.float32)
    y_val = val_df["label_PD"].values.astype(np.float32)
    y_test = test_df["label_PD"].values.astype(np.float32)
    
    # ===== Apply Class Imbalance Handling Method =====
    X_train, y_train, train_sampler, loss_weights, balance_info = apply_balancing_method(
        X_train, y_train, method=balancing_method
    )
    
    # Convert to tensors (use updated X_train, y_train from balancing method)
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
    
    # Create DataLoader based on sampling method
    if train_sampler is not None:
        # Use sampler (weighted_sampler method)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    else:
        # Use standard shuffling (class_weights, undersampling, baseline methods)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP(len(feature_cols), hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, 
                dropout_rate=dropout_rate).to(device)
    
    # Create loss function with optional class weights
    if loss_weights is not None:
        loss_weights = loss_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        print(f"Using CrossEntropyLoss with class weights: {loss_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Dynamic Learning Rate Scheduler
    if use_scheduler:
        print("\n[SCHEDULER] Dynamic Learning Rate (ReduceLROnPlateau):")
        print("  ├─ Mode: max (increase LR when metric improves)")
        print("  ├─ Factor: 0.5 (reduce LR by 50% when plateau)")
        print("  ├─ Patience: 5 epochs")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  
            factor=0.5,  
            patience=5,  
            min_lr=1e-6  
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
            best_model_state = model.state_dict().copy()  
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
        'model_state': best_model_state,  
    }


# ===== Main =====
if __name__ == "__main__":
    
    combined_features = pd.read_csv(combined_features_path)
    
    print("\n===== Feature Sets Loaded =====")
    print(f"Combined features shape: {combined_features.shape}\n")
    
    print("\n" + "="*80)
    print("MLP-v2: HYPERPARAMETER GRID SEARCH")
    print("="*80)
    print("\n[CONFIGURATION]")
    print(f"  Balancing Method: {BALANCING_METHOD}")
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
        
        for fold in range(NUM_FOLDS):
            print(f"\n  FOLD {fold}/{NUM_FOLDS - 1}")
            
            result = train_and_evaluate(
                combined_features, 
                f"HP{hp_idx}", 
                f"hp{hp_idx}", 
                fold=fold, 
                hyperparams=hyperparams, 
                use_scheduler=True, 
                balancing_method=BALANCING_METHOD
            )
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    model_states = {}
    for idx, result in enumerate(all_results):
        key = f"{result['model']}_fold{result['fold']}"
        model_states[key] = result.pop('model_state')
    
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
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save mean results
    mean_csv = output_dir / f'mlp_v2_hp_search_mean_{BALANCING_METHOD}.csv'
    mean_results_df.to_csv(mean_csv, index=False)
    print(f"\n✓ Mean results saved to: {mean_csv}")
    
    # ===== Save Only the Best Model =====
    print("\n" + "="*80)
    print("SAVING BEST MODELS (ALL 5 FOLDS)")
    print("="*80)
    
    best_hp_idx = int(best_row['hp_idx'])
    best_hp_results = results_df[results_df['model'] == f'HP{best_hp_idx}']
    
    model_save_dir = Path('/mloscratch/users/clerget/data/saved_models')
    model_save_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nBest Hyperparameters (HP{best_hp_idx}):")
    print(f"  Learning Rate: {best_row['learning_rate']}")
    print(f"  Weight Decay: {best_row['weight_decay']}")
    print(f"  Dropout Rate: {best_row['dropout_rate']}")
    print(f"  Hidden Dim 1: {best_row['hidden_dim_1']}")
    print(f"  Hidden Dim 2: {best_row['hidden_dim_2']}")
    print(f"\nMean Performance:")
    print(f"  AUC-ROC:  {best_row['auc_mean']:.4f} ± {best_row['auc_std']:.4f}")
    print(f"  Accuracy: {best_row['accuracy_mean']:.4f} ± {best_row['accuracy_std']:.4f}")
    print(f"  F1-Score: {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}")
    
    print(f"\nSaving all 5 folds:")
    for _, row in best_hp_results.iterrows():
        fold_number = int(row['fold'])
        model_key = f"HP{best_hp_idx}_fold{fold_number}"
        
        if model_key in model_states:
            model_save_path = model_save_dir / f'best_model_hp{best_hp_idx}_fold{fold_number}.pth'
            torch.save(model_states[model_key], model_save_path)
            print(f"  ✓ Fold {fold_number}: AUC={row['auc']:.4f}, saved to {model_save_path.name}")
    
    create_best_model_summary(best_row, best_hp_results, output_dir, balancing_method=BALANCING_METHOD)
    
    print("="*80)
    
    print("\n" + "="*80)
    print("✓ MLP-v2 Hyperparameter Grid Search Complete!")
    print("="*80)