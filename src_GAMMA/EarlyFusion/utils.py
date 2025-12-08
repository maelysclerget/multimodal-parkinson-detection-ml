import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import os
from pathlib import Path
from EarlyFusion.EarlyFusionCNN import EarlyFusionCNN
from EarlyFusion.EarlyFusionMLP import EarlyFusionMLP


def cross_validation_5fold_early_fusion(
    features_csv,
    labels_csv,
    train_folds_csv,
    val_test_folds_csv,
    output_dir,
    hidden_dims=[256, 128, 64, 32, 16, 8, 4],
    batch_size=64,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=0.0,
    dropout=0.5,
    class_weight=3.0,
    verbose=True
):
    """
    Perform 5-fold cross-validation training for early fusion models.
    
    Args:
        features_csv: Path to CSV file containing features (first 3 columns are patient IDs, rest are features)
        labels_csv: Path to CSV file containing labels in 'label_PD' column and 'healthcode' for patient IDs
        train_folds_csv: Path to CSV file with 'healthcode', 'fold_iteration', and 'subset' (='train') columns
        val_test_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' (='val'/'test') columns
        output_dir: Directory to save model checkpoints and results
        hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
        batch_size: Batch size for training (default: 64)
        num_epochs: Number of training epochs (default: 100)
        learning_rate: Learning rate for optimizer (default: 0.001)
        weight_decay: L2 regularization weight decay (default: 0.0)
        dropout: Dropout rate (default: 0.5)
        class_weight: Weight for positive class in loss function (default: 1.0)
        verbose: Whether to print training progress (default: True)
        
    Returns:
        tuple: (best_model_path, metrics_history) where:
            - best_model_path (str): Path to the best model saved as .pth file
            - metrics_history (dict): Dictionary with metrics from all folds, containing:
                - 'folds': List of fold numbers (0-4)
                - 'test_loss': List of test losses per fold
                - 'test_acc': List of test accuracies per fold
                - 'test_f1': List of test F1 scores per fold
                - 'test_roc_auc': List of test ROC AUC scores per fold
                - 'test_conf_mat': List of confusion matrices per fold
                - 'test_roc_curve': List of ROC curves per fold
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    features_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_csv)
    train_folds_df = pd.read_csv(train_folds_csv)
    val_test_folds_df = pd.read_csv(val_test_folds_csv)
    
    # Extract features (skip first 3 columns which contain patient ID information)
    feature_columns = features_df.columns[3:]
    
    # Create label mapping
    labels_healthcodes = labels_df['healthCode'].values
    labels_values = labels_df['label_PD'].values
    label_map = {hc: label for hc, label in zip(labels_healthcodes, labels_values)}
    
    input_dim = len(feature_columns)
    
    if verbose:
        print(f"{'='*50}")
        print(f"5-Fold Cross-Validation with")
        print(f"{'='*50}")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimensions: {hidden_dims}")
        print(f"Total samples in features: {len(features_df)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Device: {device}")
    
    fold_results = []
    best_model_path = None
    best_fold_score = -np.inf
    
    # Initialize metrics history
    metrics_history = {
        'folds': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1': [],
        'test_roc_auc': [],
        'test_conf_mat': [],
        'test_roc_curve': []
    }
    
    # Perform 5-fold CV
    for fold in range(5):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/5")
            print(f"{'='*50}")
        
        # Get train healthcodes for this fold
        train_fold_mask = (train_folds_df['fold_iteration'] == fold) & (train_folds_df['subset'] == 'train')
        train_healthcodes = train_folds_df[train_fold_mask]['healthCode'].values
        
        # Get test healthcodes for this fold (using 'healthCode' column name)
        test_fold_mask = (val_test_folds_df['fold_iteration'] == fold) & (val_test_folds_df['subset'] == 'test')
        test_healthcodes = val_test_folds_df[test_fold_mask]['healthCode'].values
        
        # Filter features_df to include ALL trials for the healthcodes in train/test sets
        train_mask = features_df['healthcode'].isin(train_healthcodes) & features_df['healthcode'].isin(label_map.keys())
        test_mask = features_df['healthcode'].isin(test_healthcodes) & features_df['healthcode'].isin(label_map.keys())
        
        train_df = features_df[train_mask]
        test_df = features_df[test_mask]
        
        # Extract features for all trials
        X_train_raw = train_df[feature_columns].values
        X_test_raw = test_df[feature_columns].values
        
        # Standardize features using training data statistics
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Get labels for all trials
        y_train = np.array([label_map[hc] for hc in train_df['healthcode'].values], dtype=np.float32)
        y_test = np.array([label_map[hc] for hc in test_df['healthcode'].values], dtype=np.float32)
        
        if verbose:
            print(f"Train: {len(np.unique(train_healthcodes))} unique healthcodes, {len(X_train)} total trials")
            print(f"Test: {len(np.unique(test_healthcodes))} unique healthcodes, {len(X_test)} total trials")
            print(f"Train label distribution: {np.bincount(y_train.astype(int))}")
            print(f"Test label distribution: {np.bincount(y_test.astype(int))}")
        
        # Initialize model based on NN_type
        
        model = EarlyFusionMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # Train model
        if verbose:
            print(f"Training MLP model...")
        
        model.fit(
            X_train, y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            lr=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose
        )
        
        # Test model
        if verbose:
            print(f"Evaluating MLP model...")
        
        test_metrics = model.test(X_test, y_test)
        
        # Store metrics in history
        metrics_history['folds'].append(fold)
        metrics_history['test_loss'].append(test_metrics['test_loss'])
        metrics_history['test_acc'].append(test_metrics['test_acc'])
        metrics_history['test_f1'].append(test_metrics['test_f1'])
        metrics_history['test_roc_auc'].append(test_metrics['test_roc_auc'])
        metrics_history['test_conf_mat'].append(test_metrics['test_conf_mat'])
        metrics_history['test_roc_curve'].append(test_metrics['test_roc_curve'])
        
        if verbose:
            print(f"Fold {fold + 1} Test Results:")
            for metric_name, metric_value in test_metrics.items():
                if metric_name not in ['test_conf_mat', 'test_roc_curve']:
                    print(f"  {metric_name}: {metric_value:.4f}")
        
        fold_results.append({
            'fold': fold,
            'model': model,
            'metrics': test_metrics
        })
        
        # Track best fold
        primary_metric = test_metrics.get('test_f1', test_metrics.get('test_acc', 0))
        if primary_metric > best_fold_score:
            best_fold_score = primary_metric
            best_model_path = os.path.join(output_dir, f"best_model_fold_{fold}.pth")
            torch.save(model.state_dict(), best_model_path)
    
    # Save best model
    if best_model_path is None:
        best_model_path = os.path.join(output_dir, "best_model.pth")
        torch.save(fold_results[0]['model'].state_dict(), best_model_path)
    
    # Compute and print summary
    if verbose:
        _print_cv_summary(fold_results)
    
    return best_model_path, metrics_history


def _print_cv_summary(fold_results):
    """
    Print summary of cross-validation results.
    
    Args:
        fold_results: List of fold results dictionaries
    """
    print(f"\n{'='*50}")
    print("5-Fold CV Results Summary")
    print(f"{'='*50}")
    
    # Collect all metrics
    metric_keys = fold_results[0]['metrics'].keys()
    metric_dict = {key: [] for key in metric_keys}
    
    for fold_result in fold_results:
        for key, value in fold_result['metrics'].items():
            metric_dict[key].append(value)
    
    # Print mean and std for each metric (skip non-numeric metrics)
    for metric_name, values in metric_dict.items():
        if metric_name not in ['test_conf_mat', 'test_roc_curve']:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
