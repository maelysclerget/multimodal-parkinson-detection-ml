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
from IntermediateFusion.IntermediateFusionCNN import IntermediateFusionCNN
from IntermediateFusion.IntermediateFusionMLP import IntermediateFusionMLP


def cross_validation_5fold_intermediate_fusion(
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
    Perform 5-fold cross-validation training.
    
    Args:
        features_csv: Path to CSV file containing features
        labels_csv: Path to CSV file containing labels
        train_folds_csv: Path to CSV file containing training fold indices
        val_test_folds_csv: Path to CSV file containing validation/test fold indices
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
    features = pd.read_csv(features_csv).values
    labels = pd.read_csv(labels_csv).values.flatten()
    train_folds = pd.read_csv(train_folds_csv).values.flatten()
    val_test_folds = pd.read_csv(val_test_folds_csv).values.flatten()
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    input_size = features.shape[1]
    
    if verbose:
        print(f"{'='*50}")
        print(f"5-Fold Cross-Validation with {NN_type}")
        print(f"{'='*50}")
        print(f"Input size: {input_size}")
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
        
        # Get fold indices
        train_idx = np.where(train_folds == fold)[0]
        test_idx = np.where(val_test_folds == fold)[0]
        
        # Get training data
        X_train = features_scaled[train_idx]
        y_train = labels[train_idx]
        
        # Get test data
        X_test = features_scaled[test_idx]
        y_test = labels[test_idx]
        
        if verbose:
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Initialize model based on NN_type
        if NN_type.upper() == "MLP":
            model = IntermediateFusionMLP(
                input_size=input_size,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif NN_type.upper() == "CNN":
            model = IntermediateFusionCNN(
                input_size=input_size,
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
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weight=class_weight,
            verbose=verbose
        )
        
        # Test model
        if verbose:
            print(f"Evaluating {NN_type} model...")
        
        test_metrics = model.test(X_test, y_test, batch_size=batch_size)
        
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
                print(f"  {metric_name}: {metric_value:.4f}")
        
        fold_results.append({
            'fold': fold,
            'model': model,
            'metrics': test_metrics
        })
        
        # Track best fold
        primary_metric = test_metrics.get('f1_score', test_metrics.get('accuracy', 0))
        if primary_metric > best_fold_score:
            best_fold_score = primary_metric
            best_model_path = os.path.join(output_dir, f"best_model_fold_{fold}.pth")
            model.save_model(best_model_path)
    
    # Save best model
    if best_model_path is None:
        best_model_path = os.path.join(output_dir, "best_model.pth")
        fold_results[0]['model'].save_model(best_model_path)
    
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
    
    # Print mean and std for each metric
    for metric_name, values in metric_dict.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

