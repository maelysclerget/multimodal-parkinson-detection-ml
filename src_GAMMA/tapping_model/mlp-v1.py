"""
MLP-V1: Multi-Feature MLP Classification with 5-Fold Cross-Validation

This script trains MLP classifiers on multiple feature sets (basic, advanced, combined)
using 5-fold cross-validation to classify PD vs Healthy subjects. Includes optional
hyperparameter tuning via random search on Fold 0.

Architecture:
- Input Layer: variable input_dim
- Hidden Layer 1: 64 neurons with ReLU + Dropout
- Hidden Layer 2: 32 neurons with ReLU + Dropout
- Hidden Layer 3: 16 neurons with ReLU + Dropout(0.2)
- Output Layer: 2 neurons (logits for CrossEntropyLoss)

Key Features:
- Trains on 3 feature sets: basic, advanced, and combined features
- Optional random hyperparameter search (disabled by default)
- Per-batch class balancing via WeightedRandomSampler alternative: class weights
- Patient-level aggregation of trial predictions
- Early stopping based on validation AUC (patience=10)
- Generates per-fold and mean CV visualizations
- Comprehensive results CSV files

Input:
- Feature CSVs: basic, advanced, and combined features
- Labels: paired_healthcode.csv with diagnosis labels
- Splits: 5-fold CV split files

Output Files:
- CV results CSVs:
  * 01_basic_Basic_Features_results.csv
  * 02_advanced_Advanced_Features_results.csv
  * 03_combined_Combined_Features_results.csv
  * results_all_models.csv - All models combined
  
- Visualization figures (in results/):
  * CV summary figures: 0X_model_name_results.png (mean ± std)

Dependencies:
- PyTorch, scikit-learn, pandas, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Paths and Configurations =====
train_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_train.csv"
valtest_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_val_test.csv"
labels_path = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv"

basic_features_path = "/mloscratch/users/clerget/data/csv/tapping_statistical_features_session.csv"
advanced_features_path = "/mloscratch/users/clerget/data/csv/tapping_advanced_features_session.csv"
combined_features_path = "/mloscratch/users/clerget/data/csv/tapping_combined_features_session.csv"

NUM_FOLDS = 5
batch_size = 64

# ===== Hyperparameter Grid for Tuning =====
HYPERPARAMETER_GRID = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
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
    
def aggregate_predictions(test_df, test_preds_proba, test_preds_binary, test_labels, aggregation_method='mean'):
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


def train_and_evaluate(features_df, model_name, model_prefix, fold=0, hyperparams=None):
    """
    Train and evaluate MLP model on a single fold of cross-validation.
    
    Args:
        features_df (pd.DataFrame): Combined features with columns [healthCode, trial_id, label_PD, feature_1, ...]
        model_name (str): Name of the model configuration (e.g., 'Basic Features') for logging
        model_prefix (str): Prefix for saving visualization files (e.g., '01_basic')
        fold (int): Fold number in cross-validation (0-4) (default: 0)
        hyperparams (dict): Hyperparameters with keys: learning_rate, weight_decay, dropout_rate,
                           hidden_dim_1, hidden_dim_2. If None, uses defaults.
    
    Returns:
        dict: Results dictionary containing:
            - 'model', 'fold': Configuration info
            - 'accuracy', 'f1_score', 'auc': Patient-level metrics
            - 'num_patients': Number of unique patients in test set
    """
    
    print(f"\nModel: {model_name} | Fold: {fold}")
    
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
    
    # Calculate class weights to handle imbalance and penalize incorrect predictions
    # More weight on the minority class (PD=1) to avoid overpredicting 0s
    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1)
    total_samples = len(y_train)
    
    # Weight inversely proportional to class frequency
    # This penalizes misclassifying the minority class more heavily
    class_weight = total_samples / (2 * num_class_1) if num_class_1 > 0 else 1.0
    
    print(f"Class distribution - Healthy (0): {num_class_0}, PD (1): {num_class_1}")
    print(f"Class weights - Healthy (0): 1.0, PD (1): {class_weight:.4f}")
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)
    
    y_train_t = torch.from_numpy(y_train).long()  # Convert to long for CrossEntropyLoss
    y_val_t = torch.from_numpy(y_val).long()
    y_test_t = torch.from_numpy(y_test).long()
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP(len(feature_cols), hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, 
                dropout_rate=dropout_rate).to(device)
    
    # Create weighted CrossEntropyLoss with class weights
    class_weights = torch.FloatTensor([1.0, class_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    num_epochs = 50
    best_val_auc = 0.0
    patience = 10
    patience_counter = 0
    
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
        
        # Validation
        model.eval()
        val_preds_proba = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_logits = model(X_batch)
                y_proba = torch.softmax(y_logits, dim=1)  # Get probabilities
                val_preds_proba.extend(y_proba[:, 1].cpu().numpy().tolist())  # Probability of class 1
                val_labels.extend(y_batch.cpu().numpy().tolist())
        
        val_auc = roc_auc_score(val_labels, val_preds_proba)
        
        # Early stopping based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # Save best model based on VAL
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model before test evaluation
    model.load_state_dict(best_model_state)
    
    # Test evaluation - Get predictions only
    model.eval()
    test_preds_proba = []
    test_preds_binary = []
    test_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_logits = model(X_batch)
            y_proba = torch.softmax(y_logits, dim=1)  # Get probabilities
            y_pred_proba_class1 = y_proba[:, 1]  # Probability of class 1
            test_preds_proba.extend(y_pred_proba_class1.cpu().numpy().tolist())
            test_preds_binary.extend((y_pred_proba_class1 > 0.5).cpu().numpy().tolist())
            test_labels.extend(y_batch.numpy().tolist())
    
    # Aggregate to patient-level ONLY
    patient_preds = aggregate_predictions(test_df, test_preds_proba, test_preds_binary, test_labels, aggregation_method='mean')
    
    patient_accuracy = accuracy_score(patient_preds['label'], patient_preds['pred_binary'])
    patient_f1 = f1_score(patient_preds['label'], patient_preds['pred_binary'])
    patient_auc = roc_auc_score(patient_preds['label'], patient_preds['pred_proba'])
    
    print(f"Patient-level Accuracy: {patient_accuracy:.4f}")
    print(f"Patient-level F1-Score: {patient_f1:.4f}")
    print(f"Patient-level AUC-ROC:  {patient_auc:.4f}")
    
    # Return data (visualization will be done later for mean CV results)
    return {
        'model': model_name,
        'fold': fold,
        'accuracy': patient_accuracy,
        'f1_score': patient_f1,
        'auc': patient_auc,
        'num_patients': len(patient_preds)
    }

def create_cv_visualization(model_name, accuracy, f1_score, auc, accuracy_std, f1_score_std, auc_std, model_prefix):
    """
    Create visualization for CV mean results across all folds.
    
    Generates a figure with mean metrics and standard deviations across 5 folds.
    
    Args:
        model_name (str): Name of the model for the title
        accuracy (float): Mean accuracy across folds
        f1_score (float): Mean F1-score across folds
        auc (float): Mean AUC-ROC across folds
        accuracy_std (float): Standard deviation of accuracy
        f1_score_std (float): Standard deviation of F1-score
        auc_std (float): Standard deviation of AUC-ROC
        model_prefix (str): Prefix for saving the figure file
    
    Returns:
        None (saves figure to results/)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Cross-Validation Results (Mean ± Std across 5 Folds)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Metrics Bar Plot with Error Bars
    metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
    values = [accuracy, f1_score, auc]
    stds = [accuracy_std, f1_score_std, auc_std]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    axes[0].bar(metrics, values, yerr=stds, color=colors, alpha=0.7, edgecolor='black', 
                capsize=10, error_kw={'linewidth': 2})
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Performance Metrics (Mean ± Std)', fontsize=12)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (v, s) in enumerate(zip(values, stds)):
        axes[0].text(i, v + s + 0.03, f'{v:.4f}\n±{s:.4f}', ha='center', fontweight='bold', fontsize=10)
    
    # 2. Comparison Table as Text
    axes[1].axis('off')
    table_data = [
        ['Metric', 'Mean', 'Std Dev'],
        ['Accuracy', f'{accuracy:.4f}', f'{accuracy_std:.4f}'],
        ['F1-Score', f'{f1_score:.4f}', f'{f1_score_std:.4f}'],
        ['AUC-ROC', f'{auc:.4f}', f'{auc_std:.4f}']
    ]
    
    table = axes[1].table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1].set_title('Detailed Results', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results')
    fig_path = output_dir / f'{model_prefix}_{model_name.replace(" ", "_")}_results.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ CV Visualization saved: {fig_path}")
    
    plt.close()

# ===== Hyperparameter Tuning =====
def hyperparameter_search(features_df, model_name, model_prefix, num_folds=5, num_trials=10):
    """
    Random search for optimal hyperparameters using Fold 0 validation.
    
    Performs random search over the hyperparameter grid, evaluating each combination
    on Fold 0 only (for speed). Returns the best hyperparameters based on AUC.
    
    Args:
        features_df (pd.DataFrame): Feature dataframe
        model_name (str): Name of the model (e.g., 'Basic Features')
        model_prefix (str): Prefix for saving files
        num_folds (int): Number of folds (not used, kept for compatibility)
        num_trials (int): Number of random trials to run (default: 10)
    
    Returns:
        tuple: (best_hyperparams dict, search_results list)
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH: {model_name}")
    print(f"{'='*70}")
    
    import random
    
    best_auc = 0.0
    best_hyperparams = None
    search_results = []
    
    for trial in range(min(num_trials, 50)):  # Limit to 50 trials
        # Random hyperparameter combination
        hyperparams = {}
        for param_name, param_list in HYPERPARAMETER_GRID.items():
            hyperparams[param_name] = random.choice(param_list)
        
        # Evaluate on fold 0 only (for speed)
        result = train_and_evaluate(features_df, model_name, model_prefix, fold=0, hyperparams=hyperparams)
        fold_auc = result['auc']
        
        search_results.append({
            'trial': trial,
            'auc': fold_auc,
            'hyperparams': hyperparams.copy()
        })
        
        print(f"Trial {trial+1}/{num_trials} - AUC: {fold_auc:.4f} - "
              f"lr={hyperparams['learning_rate']:.0e}, wd={hyperparams['weight_decay']:.0e}, "
              f"dropout={hyperparams['dropout_rate']:.1f}")
        
        if fold_auc > best_auc:
            best_auc = fold_auc
            best_hyperparams = hyperparams.copy()
            print(f"  ✓ New best AUC: {best_auc:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Best Hyperparameters for {model_name}:")
    print(f"  Learning Rate: {best_hyperparams['learning_rate']}")
    print(f"  Weight Decay: {best_hyperparams['weight_decay']}")
    print(f"  Dropout Rate: {best_hyperparams['dropout_rate']}")
    print(f"  Hidden Dim 1: {best_hyperparams['hidden_dim_1']}")
    print(f"  Hidden Dim 2: {best_hyperparams['hidden_dim_2']}")
    #print(f"  Label Smoothing: {best_hyperparams['label_smoothing']}")
    #print(f"  Batch Size: {best_hyperparams['batch_size']}")
    print(f"Best AUC (Fold 0): {best_auc:.4f}")
    print(f"{'='*70}\n")
    
    return best_hyperparams, search_results

# ===== Main Execution =====
if __name__ == "__main__":
    """
    Main execution block for MLP classification on multiple feature sets with 5-fold CV.
    
    Pipeline:
        [1/4] Load feature sets (basic, advanced, combined)
        [2/4] Optional hyperparameter search (disabled by default)
        [3/4] Execute 5-fold cross-validation training
        [4/4] Calculate metrics, save results, and generate visualizations
    """
    
    print("\n" + "="*80)
    print(" "*15 + "MLP-V1: MULTI-FEATURE CLASSIFICATION (5-FOLD CV)")
    print("="*80)
    
    # [1/4] Load feature sets
    print("\n[1/4] Loading feature sets...")
    print("-" * 80)
    
    basic_features = pd.read_csv(basic_features_path)
    advanced_features = pd.read_csv(advanced_features_path)
    combined_features = pd.read_csv(combined_features_path)
    
    print(f"✓ Basic features:      {basic_features.shape[0]} samples × {basic_features.shape[1]} features")
    print(f"✓ Advanced features:   {advanced_features.shape[0]} samples × {advanced_features.shape[1]} features")
    print(f"✓ Combined features:   {combined_features.shape[0]} samples × {combined_features.shape[1]} features")
    
    # [2/4] Hyperparameter search (optional - disabled by default)
    print("\n[2/4] Hyperparameter tuning...")
    print("-" * 80)
    
    PERFORM_HYPERPARAMETER_SEARCH = False  # Set to True to tune hyperparameters
    
    if PERFORM_HYPERPARAMETER_SEARCH:
        print("Performing random search over hyperparameter grid...")
        best_hp_basic, _ = hyperparameter_search(basic_features, "Basic Features", "01_basic", num_trials=10)
        best_hp_advanced, _ = hyperparameter_search(advanced_features, "Advanced Features", "02_advanced", num_trials=10)
        best_hp_combined, _ = hyperparameter_search(combined_features, "Combined Features", "03_combined", num_trials=10)
    else:
        print("Using default hyperparameters (search disabled)")
        best_hp_basic = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'dropout_rate': 0.5,
            'hidden_dim_1': 64,
            'hidden_dim_2': 32,
        }
        best_hp_advanced = best_hp_basic.copy()
        best_hp_combined = best_hp_basic.copy()
    
    # [3/4] Cross-validation with best hyperparameters
    print("\n[3/4] Training across 5 folds...")
    print("-" * 80)
    
    all_results = []
    
    for fold in range(NUM_FOLDS):
        print(f"\nFold {fold + 1}/{NUM_FOLDS}:")
        fold_results = []
        fold_results.append(train_and_evaluate(basic_features, "Basic Features", "01_basic", fold=fold, hyperparams=best_hp_basic))
        fold_results.append(train_and_evaluate(advanced_features, "Advanced Features", "02_advanced", fold=fold, hyperparams=best_hp_advanced))
        fold_results.append(train_and_evaluate(combined_features, "Combined Features", "03_combined", fold=fold, hyperparams=best_hp_combined))
        all_results.extend(fold_results)
    
    # [4/4] Calculate metrics and save results
    print("\n[4/4] Calculating results and generating visualizations...")
    print("-" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    # Calculate mean metrics for each model
    print("\n" + "="*80)
    print(" "*20 + "5-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    
    mean_data = []
    for model_name in ["Basic Features", "Advanced Features", "Combined Features"]:
        model_results = results_df[results_df['model'] == model_name]
        
        mean_accuracy = model_results['accuracy'].mean()
        std_accuracy = model_results['accuracy'].std()
        
        mean_f1 = model_results['f1_score'].mean()
        std_f1 = model_results['f1_score'].std()
        
        mean_auc = model_results['auc'].mean()
        std_auc = model_results['auc'].std()
        
        mean_data.append({
            'model': model_name,
            'fold': 'MEAN',
            'accuracy': mean_accuracy,
            'f1_score': mean_f1,
            'auc': mean_auc,
            'accuracy_std': std_accuracy,
            'f1_score_std': std_f1,
            'auc_std': std_auc,
            'num_patients': ''
        })
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  AUC-ROC:   {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Combine individual fold results with mean results
    results_df['accuracy_std'] = ''
    results_df['f1_score_std'] = ''
    results_df['auc_std'] = ''
    
    final_results_df = pd.concat([results_df, pd.DataFrame(mean_data)], ignore_index=True)
    
    # Save results CSV files
    print(f"\nSaving results CSVs...")
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_prefixes = {
        'Basic Features': '01_basic',
        'Advanced Features': '02_advanced',
        'Combined Features': '03_combined'
    }
    
    for model_name, prefix in model_prefixes.items():
        model_df = final_results_df[final_results_df['model'] == model_name]
        csv_path = output_dir / f'{prefix}_{model_name.replace(" ", "_")}_results.csv'
        model_df.to_csv(csv_path, index=False)
        print(f"  ✓ {csv_path.name}")
    
    # Save all models combined
    all_csv = output_dir / 'results_all_models.csv'
    final_results_df.to_csv(all_csv, index=False)
    print(f"  ✓ {all_csv.name}")
    
    # Find best model based on mean AUC
    mean_results_df = final_results_df[final_results_df['fold'] == 'MEAN']
    best_idx = mean_results_df['auc'].idxmax()
    best_model = mean_results_df.loc[best_idx, 'model']
    best_auc = mean_results_df.loc[best_idx, 'auc']
    print(f"\n✓ Best model: {best_model} (AUC: {best_auc:.4f})")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    for model_name, prefix in model_prefixes.items():
        mean_row = mean_results_df[mean_results_df['model'] == model_name].iloc[0]
        create_cv_visualization(
            model_name=model_name,
            accuracy=mean_row['accuracy'],
            f1_score=mean_row['f1_score'],
            auc=mean_row['auc'],
            accuracy_std=mean_row['accuracy_std'],
            f1_score_std=mean_row['f1_score_std'],
            auc_std=mean_row['auc_std'],
            model_prefix=prefix
        )
    
    # Final summary
    print("\n" + "="*80)
    print(" "*20 + "✓ MLP-V1 TRAINING COMPLETE")
    print("="*80)
    print(f"\nOutput Files:")
    print(f"- CV results CSVs:")
    print(f"  * 01_basic_Basic_Features_results.csv - Per-fold results")
    print(f"  * 02_advanced_Advanced_Features_results.csv - Per-fold results")
    print(f"  * 03_combined_Combined_Features_results.csv - Per-fold results")
    print(f"  * results_all_models.csv - All models combined")
    print(f"\n- Visualization figures (in results/):")
    print(f"  * CV summary figures: 0X_model_name_results.png (mean ± std)")
    print("\n")