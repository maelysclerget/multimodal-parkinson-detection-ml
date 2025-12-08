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
    NN_type="MLP",
    hidden_dims=[256, 128, 64],
    batch_size=64,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=0.0,
    dropout=0.5,
    pooling_type="max",
    class_weight=1.0,
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
        NN_type: Type of neural network - "MLP" or "CNN" (default: "MLP")
        hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
        batch_size: Batch size for training (default: 64)
        num_epochs: Number of training epochs (default: 100)
        learning_rate: Learning rate for optimizer (default: 0.001)
        weight_decay: L2 regularization weight decay (default: 0.0)
        dropout: Dropout rate (default: 0.5)
        pooling_type: Pooling type for CNN - 'max', 'avg', or 'adaptive_avg' (default: 'max')
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
    features = features_df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Get patient IDs from features (assuming 'healthcode' is in the first 3 columns)
    # We'll match on healthcode column from features_df
    if 'healthcode' in features_df.columns:
        features_healthcodes = features_df['healthcode'].values
    
    # Create a mapping from healthcode to index in features array
    healthcode_to_idx = {hc: idx for idx, hc in enumerate(features_healthcodes)}
    
    # Extract labels and match with features
    labels_healthcodes = labels_df['healthCode'].values
    labels_values = labels_df['label_PD'].values
    
    # Create label mapping
    label_map = {hc: label for hc, label in zip(labels_healthcodes, labels_values)}
    
    input_dim = features.shape[1]
    
    if verbose:
        print(f"{'='*50}")
        print(f"5-Fold Cross-Validation with {NN_type}")
        print(f"{'='*50}")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimensions: {hidden_dims}")
        print(f"Number of samples: {len(features)}")
    
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
        
        # Map healthcodes to feature indices and get labels
        train_valid_hc = [hc for hc in train_healthcodes if hc in healthcode_to_idx and hc in label_map]
        test_valid_hc = [hc for hc in test_healthcodes if hc in healthcode_to_idx and hc in label_map]
        
        train_idx = np.array([healthcode_to_idx[hc] for hc in train_valid_hc], dtype=np.int64)
        test_idx = np.array([healthcode_to_idx[hc] for hc in test_valid_hc], dtype=np.int64)
        
        # Get training data
        X_train = np.ascontiguousarray(features_scaled[train_idx])
        y_train = np.array([label_map[hc] for hc in train_valid_hc], dtype=np.float32)
        
        # Get test data
        X_test = np.ascontiguousarray(features_scaled[test_idx])
        y_test = np.array([label_map[hc] for hc in test_valid_hc], dtype=np.float32)
        
        if verbose:
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Initialize model based on NN_type
        if NN_type.upper() == "MLP":
            model = EarlyFusionMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif NN_type.upper() == "CNN":
            model = EarlyFusionCNN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                pooling_type=pooling_type
            )
        else:
            raise ValueError(f"Unknown NN_type: {NN_type}. Must be 'MLP' or 'CNN'")
        
        # Train model
        if verbose:
            print(f"Training {NN_type} model...")
        
        model.fit(
            X_train, y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose
        )
        
        # Test model
        if verbose:
            print(f"Evaluating {NN_type} model...")
        
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
