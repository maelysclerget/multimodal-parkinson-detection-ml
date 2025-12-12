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
from EarlyFusion.IntermediateFusionMLP import IntermediateFusionMLP
from itertools import product
import gc

def cross_validation_5fold_early_fusion(
    features_csv,
    labels_csv,
    train_folds_csv,
    val_test_folds_csv,
    # output_dir,
    hidden_dims=[256, 128, 64],
    batch_size=64,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=0.01,
    dropout=0.3,
    class_weight=3.0,
    verbose=True
):
    """
    Perform 5-fold cross-validation training for early fusion MLP models.
    
    Args:
        features_csv: Path to CSV file containing features with 'healthCode' column
        labels_csv: Path to CSV file containing labels in 'label_PD' column and 'healthCode' for patient IDs
        train_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' columns
        val_test_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' (='val'/'test') columns
        output_dir: Directory to save model checkpoints (.pth files)
        hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
        batch_size: Batch size for training (default: 64)
        num_epochs: Maximum number of training epochs (default: 100)
        learning_rate: Initial learning rate for optimizer (default: 0.001)
        weight_decay: L2 regularization weight decay (default: 0.01)
        dropout: Dropout rate (default: 0.3)
        class_weight: Weight for class 0 (controls) to handle imbalance (default: 3.0)
        verbose: Whether to print training progress (default: True)
        
    Returns:
        dict: Dictionary with summary statistics and per-fold results:
            - 'mean_test_loss': Mean test loss across folds
            - 'std_test_loss': Standard deviation of test loss
            - 'mean_test_acc': Mean test accuracy across folds
            - 'std_test_acc': Standard deviation of test accuracy
            - 'mean_test_f1': Mean test F1 score across folds
            - 'std_test_f1': Standard deviation of test F1 score
            - 'mean_test_roc_auc': Mean test ROC AUC across folds
            - 'std_test_roc_auc': Standard deviation of test ROC AUC
            - 'confusion_matrices': List of 5 confusion matrices (one per fold)
            - 'model_paths': List of paths to saved .pth models (one per fold)
    """
    # Create output directory
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data files
    features_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_csv)
    train_folds_df = pd.read_csv(train_folds_csv)
    val_test_folds_df = pd.read_csv(val_test_folds_csv)
    
    if verbose:
        print(f"Loaded features: {features_df.shape}")
        print(f"  Columns: {list(features_df.columns[:5])}...")
        print(f"Loaded labels: {labels_df.shape}")
        print(f"  Columns: {list(labels_df.columns)}")
        print(f"Train folds: {train_folds_df.shape}")
        print(f"  Columns: {list(train_folds_df.columns)}")
        print(f"Val+Test folds: {val_test_folds_df.shape}")
        print(f"  Columns: {list(val_test_folds_df.columns)}")
    
    # Separate val and test based on 'subset' column
    val_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'val']
    test_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'test']
    
    if verbose:
        print(f"  - Validation folds: {val_folds_df.shape}")
        print(f"  - Test folds: {test_folds_df.shape}")
    
    # Rename 'healthcode' to 'healthCode' for consistency if needed
    if 'healthcode' in features_df.columns and 'healthCode' not in features_df.columns:
        features_df = features_df.rename(columns={'healthcode': 'healthCode'})
        if verbose:
            print(f"Renamed 'healthcode' → 'healthCode' for consistency")
    
    # Merge features with labels on healthCode
    features_df = features_df.merge(labels_df[['healthCode', 'label_PD']], on='healthCode', how='inner')
    if verbose:
        print(f"Features after merging with labels: {features_df.shape}")
    
    # Filter to only include healthcodes in 5-fold splits
    all_fold_healthcodes = set(train_folds_df['healthCode'].unique()) | set(val_test_folds_df['healthCode'].unique())
    features_df = features_df[features_df['healthCode'].isin(all_fold_healthcodes)]
    if verbose:
        print(f"Features after filtering to 5-fold healthcodes: {features_df.shape}")
    
    # Prepare feature columns (exclude metadata)
    # Define all possible metadata columns
    metadata_cols = ['filename', 'healthCode', 'record_id', 'label_PD', 'trial_id', 'row_id', 
                     'trial_id_file1', 'trial_id_file2', 'trial_id_spec', 'trial_id_heatmap',
                     'filename_file1', 'filename_file2', 'record_id_file1', 'record_id_file2']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    if verbose:
        print(f"Number of features: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols[:5]}... (showing first 5)")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}\n")
    
    # Initialize storage for results
    test_losses = []
    test_accs = []
    test_f1s = []
    test_roc_aucs = []
    confusion_matrices = []
    models_list = []
    
    # Perform 5-fold CV
    for fold in range(5):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/5 (Fold iteration {fold + 1})")
            print(f"{'='*50}")
        
        # Get unique patient healthCodes for this fold
        train_patient_ids = train_folds_df[train_folds_df['fold_iteration'] == fold]['healthCode'].values
        val_patient_ids = val_folds_df[val_folds_df['fold_iteration'] == fold]['healthCode'].values
        test_patient_ids = test_folds_df[test_folds_df['fold_iteration'] == fold]['healthCode'].values
        
        if verbose:
            print(f"Train patients: {len(train_patient_ids)}")
            print(f"Val patients: {len(val_patient_ids)}")
            print(f"Test patients: {len(test_patient_ids)}")
        
        # Split data by patient healthCode
        train_data = features_df[features_df['healthCode'].isin(train_patient_ids)]
        val_data = features_df[features_df['healthCode'].isin(val_patient_ids)]
        test_data = features_df[features_df['healthCode'].isin(test_patient_ids)]
        
        if verbose:
            print(f"Train samples: {len(train_data)}")
            print(f"Val samples: {len(val_data)}")
            print(f"Test samples: {len(test_data)}")
        
        # Extract features and labels
        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data['label_PD'].values
        X_val = val_data[feature_cols].values.astype(np.float32)
        y_val = val_data['label_PD'].values
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data['label_PD'].values
        
        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features using StandardScaler (fit on train, transform on val/test)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        # Initialize model
        model = EarlyFusionMLP(
            input_dim=X_train_tensor.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            verbose=False
        )
        
        # Train model
        if verbose:
            print(f"\n{'='*50}")
            print("Training EarlyFusionMLP...")
            print(f"{'='*50}")
        
        model.fit(
            X_train_tensor,
            y_train_tensor,
            val_features=X_val_tensor,
            val_labels=y_val_tensor,
            epochs=num_epochs,
            class_weight=class_weight,
            batch_size=batch_size,
            lr=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose
        )
        
        # Test model with majority voting per patient
        if verbose:
            print(f"\n{'='*50}")
            print("Testing EarlyFusionMLP with Majority Voting...")
            print(f"{'='*50}")
        
        # Prepare test dataframe with standardized features
        test_data_standardized = test_data.copy()
        test_data_standardized[feature_cols] = X_test
        
        # Use the test_with_majority_counting method
        test_results = model.test_with_majority_counting(test_data_standardized)
        
        # Display results
        if verbose:
            print(f"\nPatient-level Results (Majority Voting):")
            print(f"  Number of patients: {test_results['num_patients']}")
            print(f"  Number of trials: {test_results['num_trials']}")
            print(f"  Test Accuracy: {test_results['test_acc']:.4f}")
            print(f"  Test F1 Score: {test_results['test_f1']:.4f}")
            print(f"  Test ROC AUC: {test_results['test_roc_auc']:.4f}")
            print(f"\nConfusion Matrix:\n{test_results['test_conf_mat']}")
        
        # Store results
        # Note: test_with_majority_counting doesn't return test_loss, so we'll use a placeholder
        test_losses.append(0.0)  # Placeholder - majority voting doesn't have a direct loss value
        test_accs.append(test_results['test_acc'])
        test_f1s.append(test_results['test_f1'])
        test_roc_aucs.append(test_results['test_roc_auc'])
        confusion_matrices.append(test_results['test_conf_mat'])
        
        # Save model
        # model_path = os.path.join(output_dir, f"model_fold_{fold}.pth")
        # torch.save(model.state_dict(), model_path)
        models_list.append(model.state_dict())
        # model_paths.append(model_path)
        # if verbose:
        #     print(f"\nModel saved to: {model_path}")
    
    # Calculate summary statistics
    results = {
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'mean_test_acc': np.mean(test_accs),
        'std_test_acc': np.std(test_accs),
        'mean_test_f1': np.mean(test_f1s),
        'std_test_f1': np.std(test_f1s),
        'mean_test_roc_auc': np.mean(test_roc_aucs),
        'std_test_roc_auc': np.std(test_roc_aucs),
        'confusion_matrices': confusion_matrices,
        'models_list': models_list
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("5-Fold Cross-Validation Summary")
        print(f"{'='*50}")
        print(f"Test Accuracy: {results['mean_test_acc']:.4f} ± {results['std_test_acc']:.4f}")
        print(f"Test F1 Score: {results['mean_test_f1']:.4f} ± {results['std_test_f1']:.4f}")
        print(f"Test ROC AUC:  {results['mean_test_roc_auc']:.4f} ± {results['std_test_roc_auc']:.4f}")
    
    return results


def hyperparameter_tuning(
    features_csv,
    labels_csv,
    train_folds_csv,
    val_test_folds_csv,
    output_dir,
    hidden_dims: list[list],
    batch_size,
    weight_decay,
    dropout,
    class_weight,
    num_epochs,
    learning_rate,
    verbose=True        
):
    """
    Perform hyperparameter tuning using grid search over specified hyperparameter ranges.
    
    Evaluates all combinations of hyperparameters using 5-fold cross-validation and selects
    the best model based on mean test ROC AUC score. The best model's state dictionaries
    for all 5 folds are saved to the output directory.
    
    Args:
        features_csv: Path to CSV file containing features with 'healthCode' column
        labels_csv: Path to CSV file containing labels in 'label_PD' column and 'healthCode' for patient IDs
        train_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' columns
        val_test_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' (='val'/'test') columns
        output_dir: Directory to save the best model checkpoints (.pth files for each fold)
        hidden_dims: List of hidden layer dimension configurations to test (e.g., [[256, 128, 64], [512, 256]])
        batch_size: List of batch sizes to test (e.g., [32, 64])
        weight_decay: List of L2 regularization coefficients to test (e.g., [0.0, 0.01])
        dropout: List of dropout rates to test (e.g., [0.3, 0.5, 0.8])
        class_weight: List of class weights for handling class imbalance (e.g., [1, 2, 3])
        num_epochs: List of maximum training epoch values to test (e.g., [100, 150])
        learning_rate: List of initial learning rates to test (e.g., [0.001, 0.0001])
        verbose: Whether to print training progress and results (default: True)
        
    Returns:
        dict: Dictionary containing the best model's results with the following keys:
            - 'mean_test_acc': Mean test accuracy across 5 folds
            - 'std_test_acc': Standard deviation of test accuracy
            - 'mean_test_f1': Mean test F1 score across 5 folds
            - 'std_test_f1': Standard deviation of test F1 score
            - 'mean_test_roc_auc': Mean test ROC AUC across 5 folds
            - 'std_test_roc_auc': Standard deviation of test ROC AUC
            - 'confusion_matrices': List of 5 confusion matrices (one per fold)
            - 'models_list': List of 5 model state dictionaries (one per fold)
            
    Note:
        - Grid search evaluates all possible combinations of the provided hyperparameters
        - Total combinations = len(hidden_dims) × len(batch_size) × len(weight_decay) × 
                               len(dropout) × len(class_weight) × len(num_epochs) × len(learning_rate)
        - Models not selected as best are deleted from memory using garbage collection
        - Selection criterion: highest mean test ROC AUC across 5 folds
        - Early stopping and learning rate scheduling are applied based on validation ROC AUC
    """
    # Convert product iterator to list to allow multiple iterations
    hyperparameter_sets = list(product(hidden_dims, batch_size, weight_decay, dropout, class_weight, num_epochs, learning_rate))

    if verbose:
        print(f"Testing {len(hyperparameter_sets)} Hyperparameter Combinations\n")

    # Counting the number of sets we have tried so far to track progress in verbose version
    hyperparameter_set_counter = 0

    # Creating an empty variable which will hold the results dictionary of the best model
    best_results = None

    # Tracking best Test ROC AUC throughout the tuning
    best_ROC_AUC = 0

    # Tracking best hyperparameters
    BEST_HIDDEN_LAYERS = None
    BEST_BATCH_SIZE = None
    BEST_WEIGHT_DECAY = None
    BEST_DROPOUT = None
    BEST_CLASS_WEIGHT = None
    BEST_NUM_EPOCHS = None
    BEST_LEARNING_RATE = None

    for hyperparam_set in hyperparameter_sets:
        hyperparameter_set_counter += 1

        HIDDEN_LAYERS, BATCH_SIZE, WEIGHT_DECAY, DROPOUT, CLASS_WEIGHT, NUM_EPOCHS, LEARNING_RATE = hyperparam_set

        if verbose: 
            print("------------------------------")
            print(f"Hyperparameters; set {hyperparameter_set_counter}/{len(hyperparameter_sets)}")
            print("------------------------------")
            print(f"Hidden Layers: {HIDDEN_LAYERS}")
            print(f"Batch Size: {BATCH_SIZE}")
            print(f"Weight Decay: {WEIGHT_DECAY}")
            print(f"Dropout: {DROPOUT}")
            print(f"Class Weight: {CLASS_WEIGHT}")
            print(f"Num Epochs: {NUM_EPOCHS}")
            print(f"Learning Rate: {LEARNING_RATE}")
            print("------------------------------", "\n")
            

        hyperparam_set_res = cross_validation_5fold_early_fusion(
            features_csv=features_csv, 
            labels_csv=labels_csv,
            train_folds_csv=train_folds_csv,
            val_test_folds_csv=val_test_folds_csv,
            # output_dir=output_dir,
            hidden_dims=HIDDEN_LAYERS,
            batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            dropout=DROPOUT,
            class_weight=CLASS_WEIGHT,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            verbose=False
        )

        if verbose:
            print("Results Summary")
            print("------------------------------")
            print(f"Test Accuracy: {hyperparam_set_res['mean_test_acc']:.4f} ± {hyperparam_set_res['std_test_acc']:.4f}")
            print(f"Test F1 Score: {hyperparam_set_res['mean_test_f1']:.4f} ± {hyperparam_set_res['std_test_f1']:.4f}")
            print(f"Test ROC AUC:  {hyperparam_set_res['mean_test_roc_auc']:.4f} ± {hyperparam_set_res['std_test_roc_auc']:.4f}")
            print("------------------------------", "\n")

        if hyperparam_set_res['mean_test_roc_auc'] >= best_ROC_AUC:
            if verbose:
                print("Found New Best Model!")
                print("------------------------------", "\n")

            best_results = hyperparam_set_res
            best_ROC_AUC = hyperparam_set_res['mean_test_roc_auc']

            BEST_HIDDEN_LAYERS = HIDDEN_LAYERS
            BEST_BATCH_SIZE = BATCH_SIZE
            BEST_WEIGHT_DECAY = WEIGHT_DECAY
            BEST_DROPOUT = DROPOUT  
            BEST_CLASS_WEIGHT = CLASS_WEIGHT
            BEST_NUM_EPOCHS = NUM_EPOCHS
            BEST_LEARNING_RATE = LEARNING_RATE

        
        else:
            # Force garbage collection of bad models to free up space
            del hyperparam_set_res
            gc.collect()

    if verbose: 
        print("------------------------------")
        print("End Summary")
        print("------------------------------")
        print("Hyperparameters")
        print("------------------------------")
        print(f"Hidden Layers: {BEST_HIDDEN_LAYERS}")
        print(f"Batch Size: {BEST_BATCH_SIZE}")
        print(f"Weight Decay: {BEST_WEIGHT_DECAY}")
        print(f"Dropout: {BEST_DROPOUT}")
        print(f"Class Weight: {BEST_CLASS_WEIGHT}")
        print(f"Num Epochs: {BEST_NUM_EPOCHS}")
        print(f"Learning Rate: {BEST_LEARNING_RATE}")
        print("------------------------------")
        print("Best Model Results")
        print("------------------------------")
        print(f"Best Test Accuracy: {best_results['mean_test_acc']:.4f} ± {best_results['std_test_acc']:.4f}")
        print(f"Best Test F1 Score: {best_results['mean_test_f1']:.4f} ± {best_results['std_test_f1']:.4f}")
        print(f"Best Test ROC AUC:  {best_results['mean_test_roc_auc']:.4f} ± {best_results['std_test_roc_auc']:.4f}")
        print("------------------------------", "\n")

    # Save best model as .pth to output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    for fold in range(5):
        model_path = os.path.join(output_dir, f"model_fold_{fold}.pth")
        torch.save(best_results["models_list"][fold], model_path)

    if verbose:
        print(f"Saved Best Model at the following Path: {model_path}")

    return best_results


def hyperparameter_tuning_intermediate_fusion(
    audio_features_csv,
    tapping_features_csv,
    labels_csv,
    train_folds_csv,
    val_test_folds_csv,
    output_dir,
    hidden_dims: list[list],
    audio_branch: list[dict],
    tapping_branch: list[dict],
    batch_size,
    weight_decay,
    dropout,
    class_weight,
    num_epochs,
    learning_rate,
    verbose=True        
):
    """
    Perform hyperparameter tuning using grid search for intermediate fusion models.
    
    Evaluates all combinations of hyperparameters using 5-fold cross-validation and selects
    the best model based on mean test ROC AUC score. The best model's state dictionaries
    for all 5 folds are saved to the output directory.
    
    Args:
        audio_features_csv: Path to CSV file containing audio features with 'healthCode' column
        tapping_features_csv: Path to CSV file containing tapping features with 'healthCode' column
        labels_csv: Path to CSV file containing labels in 'label_PD' column and 'healthCode' for patient IDs
        train_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' columns
        val_test_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' (='val'/'test') columns
        output_dir: Directory to save the best model checkpoints (.pth files for each fold)
        hidden_dims: List of hidden layer dimension configurations for fusion MLP (e.g., [[256, 128, 64], [512, 256]])
        audio_branch: List of dicts with 'output_dim' and 'hidden_dims' for audio branch (e.g., [{"output_dim": 64, "hidden_dims": [256, 128]}])
        tapping_branch: List of dicts with 'output_dim' and 'hidden_dims' for tapping branch (e.g., [{"output_dim": 64, "hidden_dims": [128, 64]}])
        batch_size: List of batch sizes to test (e.g., [32, 64])
        weight_decay: List of L2 regularization coefficients to test (e.g., [0.0, 0.01])
        dropout: List of dropout rates to test (e.g., [0.3, 0.5, 0.8])
        class_weight: List of class weights for handling class imbalance (e.g., [1, 2, 3])
        num_epochs: List of maximum training epoch values to test (e.g., [100, 150])
        learning_rate: List of initial learning rates to test (e.g., [0.001, 0.0001])
        verbose: Whether to print training progress and results (default: True)
        
    Returns:
        dict: Dictionary containing the best model's results with the following keys:
            - 'mean_test_acc': Mean test accuracy across 5 folds
            - 'std_test_acc': Standard deviation of test accuracy
            - 'mean_test_f1': Mean test F1 score across 5 folds
            - 'std_test_f1': Standard deviation of test F1 score
            - 'mean_test_roc_auc': Mean test ROC AUC across 5 folds
            - 'std_test_roc_auc': Standard deviation of test ROC AUC
            - 'confusion_matrices': List of 5 confusion matrices (one per fold)
            - 'models_list': List of 5 model state dictionaries (one per fold)
            
    Note:
        - Grid search evaluates all possible combinations of the provided hyperparameters
        - Total combinations = len(hidden_dims) × len(audio_branch) × len(tapping_branch) × 
                               len(batch_size) × len(weight_decay) × len(dropout) × 
                               len(class_weight) × len(num_epochs) × len(learning_rate)
        - Models not selected as best are deleted from memory using garbage collection
        - Selection criterion: highest mean test ROC AUC across 5 folds
        - Early stopping and learning rate scheduling are applied based on validation ROC AUC
    """
    # Convert product iterator to list to allow multiple iterations
    hyperparameter_sets = list(product(hidden_dims, audio_branch, tapping_branch, batch_size, 
                                      weight_decay, dropout, class_weight, num_epochs, learning_rate))

    if verbose:
        print(f"Testing {len(hyperparameter_sets)} Hyperparameter Combinations\n")

    # Counting the number of sets we have tried so far to track progress in verbose version
    hyperparameter_set_counter = 0

    # Creating an empty variable which will hold the results dictionary of the best model
    best_results = None

    # Tracking best Test ROC AUC throughout the tuning
    best_ROC_AUC = 0

    # Tracking best hyperparameters
    BEST_HIDDEN_LAYERS = None
    BEST_AUDIO_BRANCH = None
    BEST_TAPPING_BRANCH = None
    BEST_BATCH_SIZE = None
    BEST_WEIGHT_DECAY = None
    BEST_DROPOUT = None
    BEST_CLASS_WEIGHT = None
    BEST_NUM_EPOCHS = None
    BEST_LEARNING_RATE = None

    for hyperparam_set in hyperparameter_sets:
        hyperparameter_set_counter += 1

        HIDDEN_LAYERS, AUDIO_BRANCH, TAPPING_BRANCH, BATCH_SIZE, WEIGHT_DECAY, DROPOUT, CLASS_WEIGHT, NUM_EPOCHS, LEARNING_RATE = hyperparam_set

        if verbose: 
            print("------------------------------")
            print(f"Hyperparameters; set {hyperparameter_set_counter}/{len(hyperparameter_sets)}")
            print("------------------------------")
            print(f"Fusion Hidden Layers: {HIDDEN_LAYERS}")
            print(f"Audio Branch: {AUDIO_BRANCH}")
            print(f"Tapping Branch: {TAPPING_BRANCH}")
            print(f"Batch Size: {BATCH_SIZE}")
            print(f"Weight Decay: {WEIGHT_DECAY}")
            print(f"Dropout: {DROPOUT}")
            print(f"Class Weight: {CLASS_WEIGHT}")
            print(f"Num Epochs: {NUM_EPOCHS}")
            print(f"Learning Rate: {LEARNING_RATE}")
            print("------------------------------", "\n")
            

        hyperparam_set_res = cross_validation_5fold_intermediate_fusion(
            audio_features_csv=audio_features_csv,
            tapping_features_csv=tapping_features_csv,
            labels_csv=labels_csv,
            train_folds_csv=train_folds_csv,
            val_test_folds_csv=val_test_folds_csv,
            hidden_dims=HIDDEN_LAYERS,
            audio_branch=AUDIO_BRANCH,
            tapping_branch=TAPPING_BRANCH,
            batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            dropout=DROPOUT,
            class_weight=CLASS_WEIGHT,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            verbose=False
        )

        if verbose:
            print("Results Summary")
            print("------------------------------")
            print(f"Test Accuracy: {hyperparam_set_res['mean_test_acc']:.4f} ± {hyperparam_set_res['std_test_acc']:.4f}")
            print(f"Test F1 Score: {hyperparam_set_res['mean_test_f1']:.4f} ± {hyperparam_set_res['std_test_f1']:.4f}")
            print(f"Test ROC AUC:  {hyperparam_set_res['mean_test_roc_auc']:.4f} ± {hyperparam_set_res['std_test_roc_auc']:.4f}")
            print("------------------------------", "\n")

        if hyperparam_set_res['mean_test_roc_auc'] >= best_ROC_AUC:
            if verbose:
                print("Found New Best Model!")
                print("------------------------------", "\n")

            best_results = hyperparam_set_res
            best_ROC_AUC = hyperparam_set_res['mean_test_roc_auc']

            BEST_HIDDEN_LAYERS = HIDDEN_LAYERS
            BEST_AUDIO_BRANCH = AUDIO_BRANCH
            BEST_TAPPING_BRANCH = TAPPING_BRANCH
            BEST_BATCH_SIZE = BATCH_SIZE
            BEST_WEIGHT_DECAY = WEIGHT_DECAY
            BEST_DROPOUT = DROPOUT  
            BEST_CLASS_WEIGHT = CLASS_WEIGHT
            BEST_NUM_EPOCHS = NUM_EPOCHS
            BEST_LEARNING_RATE = LEARNING_RATE

        
        else:
            # Force garbage collection of bad models to free up space
            del hyperparam_set_res
            gc.collect()

    if verbose: 
        print("------------------------------")
        print("End Summary")
        print("------------------------------")
        print("Hyperparameters")
        print("------------------------------")
        print(f"Fusion Hidden Layers: {BEST_HIDDEN_LAYERS}")
        print(f"Audio Branch: {BEST_AUDIO_BRANCH}")
        print(f"Tapping Branch: {BEST_TAPPING_BRANCH}")
        print(f"Batch Size: {BEST_BATCH_SIZE}")
        print(f"Weight Decay: {BEST_WEIGHT_DECAY}")
        print(f"Dropout: {BEST_DROPOUT}")
        print(f"Class Weight: {BEST_CLASS_WEIGHT}")
        print(f"Num Epochs: {BEST_NUM_EPOCHS}")
        print(f"Learning Rate: {BEST_LEARNING_RATE}")
        print("------------------------------")
        print("Best Model Results")
        print("------------------------------")
        print(f"Best Test Accuracy: {best_results['mean_test_acc']:.4f} ± {best_results['std_test_acc']:.4f}")
        print(f"Best Test F1 Score: {best_results['mean_test_f1']:.4f} ± {best_results['std_test_f1']:.4f}")
        print(f"Best Test ROC AUC:  {best_results['mean_test_roc_auc']:.4f} ± {best_results['std_test_roc_auc']:.4f}")
        print("------------------------------", "\n")

    # Save best model as .pth to output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    for fold in range(5):
        model_path = os.path.join(output_dir, f"model_fold_{fold}.pth")
        torch.save(best_results["models_list"][fold], model_path)

    if verbose:
        print(f"Saved Best Model at the following Path: {model_path}")

    return best_results


def cross_validation_5fold_intermediate_fusion(
    audio_features_csv,
    tapping_features_csv,
    labels_csv,
    train_folds_csv,
    val_test_folds_csv,
    hidden_dims=[256, 128, 64],
    audio_branch=None,
    tapping_branch=None,
    batch_size=64,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=0.01,
    dropout=0.3,
    class_weight=3.0,
    verbose=True
):
    """
    Perform 5-fold cross-validation training for intermediate fusion MLP models.
    
    Args:
        audio_features_csv: Path to CSV file containing audio features with 'healthCode' column
        tapping_features_csv: Path to CSV file containing tapping features with 'healthCode' column
        labels_csv: Path to CSV file containing labels in 'label_PD' column and 'healthCode' for patient IDs
        train_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' columns
        val_test_folds_csv: Path to CSV file with 'healthCode', 'fold_iteration', and 'subset' (='val'/'test') columns
        hidden_dims: List of hidden layer dimensions for fusion MLP (default: [256, 128, 64])
        audio_branch: Dict with 'input_dim', 'output_dim', 'hidden_dims' for audio branch (default: None - auto-inferred)
        tapping_branch: Dict with 'input_dim', 'output_dim', 'hidden_dims' for tapping branch (default: None - auto-inferred)
        batch_size: Batch size for training (default: 64)
        num_epochs: Maximum number of training epochs (default: 100)
        learning_rate: Initial learning rate for optimizer (default: 0.001)
        weight_decay: L2 regularization weight decay (default: 0.01)
        dropout: Dropout rate (default: 0.3)
        class_weight: Weight for class 0 (controls) to handle imbalance (default: 3.0)
        verbose: Whether to print training progress (default: True)
        
    Returns:
        dict: Dictionary with summary statistics and per-fold results:
            - 'mean_test_loss': Mean test loss across folds
            - 'std_test_loss': Standard deviation of test loss
            - 'mean_test_acc': Mean test accuracy across folds
            - 'std_test_acc': Standard deviation of test accuracy
            - 'mean_test_f1': Mean test F1 score across folds
            - 'std_test_f1': Standard deviation of test F1 score
            - 'mean_test_roc_auc': Mean test ROC AUC across folds
            - 'std_test_roc_auc': Standard deviation of test ROC AUC
            - 'confusion_matrices': List of 5 confusion matrices (one per fold)
            - 'models_list': List of model state dictionaries (one per fold)
    """
    # Load data files
    audio_features_df = pd.read_csv(audio_features_csv)
    tapping_features_df = pd.read_csv(tapping_features_csv)
    labels_df = pd.read_csv(labels_csv)
    train_folds_df = pd.read_csv(train_folds_csv)
    val_test_folds_df = pd.read_csv(val_test_folds_csv)
    
    if verbose:
        print(f"Loaded audio features: {audio_features_df.shape}")
        print(f"  Columns: {list(audio_features_df.columns[:5])}...")
        print(f"Loaded tapping features: {tapping_features_df.shape}")
        print(f"  Columns: {list(tapping_features_df.columns[:5])}...")
        print(f"Loaded labels: {labels_df.shape}")
        print(f"  Columns: {list(labels_df.columns)}")
        print(f"Train folds: {train_folds_df.shape}")
        print(f"  Columns: {list(train_folds_df.columns)}")
        print(f"Val+Test folds: {val_test_folds_df.shape}")
        print(f"  Columns: {list(val_test_folds_df.columns)}")
    
    # Separate val and test based on 'subset' column
    val_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'val']
    test_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'test']
    
    if verbose:
        print(f"  - Validation folds: {val_folds_df.shape}")
        print(f"  - Test folds: {test_folds_df.shape}")
    
    # Rename 'healthcode' to 'healthCode' for consistency if needed
    for df in [audio_features_df, tapping_features_df]:
        if 'healthcode' in df.columns and 'healthCode' not in df.columns:
            df.rename(columns={'healthcode': 'healthCode'}, inplace=True)
            if verbose:
                print(f"Renamed 'healthcode' → 'healthCode' for consistency")
    
    # Merge features with labels on healthCode
    audio_features_df = audio_features_df.merge(labels_df[['healthCode', 'label_PD']], on='healthCode', how='inner')
    tapping_features_df = tapping_features_df.merge(labels_df[['healthCode', 'label_PD']], on='healthCode', how='inner')
    
    if verbose:
        print(f"Audio features after merging with labels: {audio_features_df.shape}")
        print(f"Tapping features after merging with labels: {tapping_features_df.shape}")
    
    # Filter to only include healthcodes in 5-fold splits
    all_fold_healthcodes = set(train_folds_df['healthCode'].unique()) | set(val_test_folds_df['healthCode'].unique())
    audio_features_df = audio_features_df[audio_features_df['healthCode'].isin(all_fold_healthcodes)]
    tapping_features_df = tapping_features_df[tapping_features_df['healthCode'].isin(all_fold_healthcodes)]
    
    if verbose:
        print(f"Audio features after filtering to 5-fold healthcodes: {audio_features_df.shape}")
        print(f"Tapping features after filtering to 5-fold healthcodes: {tapping_features_df.shape}")
    
    # Prepare feature columns (exclude metadata)
    metadata_cols = ['filename', 'healthCode', 'record_id', 'label_PD', 'trial_id', 'row_id', 
                     'filename_file1', 'filename_file2', 'record_id_file1', 'record_id_file2']
    audio_feature_cols = [col for col in audio_features_df.columns if col not in metadata_cols]
    tapping_feature_cols = [col for col in tapping_features_df.columns if col not in metadata_cols]
    
    if verbose:
        print(f"Number of audio features: {len(audio_feature_cols)}")
        print(f"Audio feature columns: {audio_feature_cols[:5]}... (showing first 5)")
        print(f"Number of tapping features: {len(tapping_feature_cols)}")
        print(f"Tapping feature columns: {tapping_feature_cols[:5]}... (showing first 5)")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}\n")
    
    # Initialize storage for results
    test_losses = []
    test_accs = []
    test_f1s = []
    test_roc_aucs = []
    confusion_matrices = []
    models_list = []
    
    # Perform 5-fold CV
    for fold in range(5):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/5 (Fold iteration {fold + 1})")
            print(f"{'='*50}")
        
        # Get unique patient healthCodes for this fold
        train_patient_ids = train_folds_df[train_folds_df['fold_iteration'] == fold]['healthCode'].values
        val_patient_ids = val_folds_df[val_folds_df['fold_iteration'] == fold]['healthCode'].values
        test_patient_ids = test_folds_df[test_folds_df['fold_iteration'] == fold]['healthCode'].values
        
        if verbose:
            print(f"Train patients: {len(train_patient_ids)}")
            print(f"Val patients: {len(val_patient_ids)}")
            print(f"Test patients: {len(test_patient_ids)}")
        
        # Split data by patient healthCode
        train_audio = audio_features_df[audio_features_df['healthCode'].isin(train_patient_ids)]
        val_audio = audio_features_df[audio_features_df['healthCode'].isin(val_patient_ids)]
        test_audio = audio_features_df[audio_features_df['healthCode'].isin(test_patient_ids)]
        
        train_tapping = tapping_features_df[tapping_features_df['healthCode'].isin(train_patient_ids)]
        val_tapping = tapping_features_df[tapping_features_df['healthCode'].isin(val_patient_ids)]
        test_tapping = tapping_features_df[tapping_features_df['healthCode'].isin(test_patient_ids)]
        
        if verbose:
            print(f"Train audio samples: {len(train_audio)}")
            print(f"Train tapping samples: {len(train_tapping)}")
            print(f"Val audio samples: {len(val_audio)}")
            print(f"Val tapping samples: {len(val_tapping)}")
            print(f"Test audio samples: {len(test_audio)}")
            print(f"Test tapping samples: {len(test_tapping)}")
        
        # Extract features and labels
        X_train_audio = train_audio[audio_feature_cols].values.astype(np.float32)
        X_train_tapping = train_tapping[tapping_feature_cols].values.astype(np.float32)
        y_train = train_audio['label_PD'].values
        
        X_val_audio = val_audio[audio_feature_cols].values.astype(np.float32)
        X_val_tapping = val_tapping[tapping_feature_cols].values.astype(np.float32)
        y_val = val_audio['label_PD'].values
        
        X_test_audio = test_audio[audio_feature_cols].values.astype(np.float32)
        X_test_tapping = test_tapping[tapping_feature_cols].values.astype(np.float32)
        y_test = test_audio['label_PD'].values
        
        # Handle NaN values
        X_train_audio = np.nan_to_num(X_train_audio, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_tapping = np.nan_to_num(X_train_tapping, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_audio = np.nan_to_num(X_val_audio, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_tapping = np.nan_to_num(X_val_tapping, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_audio = np.nan_to_num(X_test_audio, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_tapping = np.nan_to_num(X_test_tapping, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features using StandardScaler (fit on train, transform on val/test)
        scaler_audio = StandardScaler()
        X_train_audio = scaler_audio.fit_transform(X_train_audio)
        X_val_audio = scaler_audio.transform(X_val_audio)
        X_test_audio = scaler_audio.transform(X_test_audio)
        
        scaler_tapping = StandardScaler()
        X_train_tapping = scaler_tapping.fit_transform(X_train_tapping)
        X_val_tapping = scaler_tapping.transform(X_val_tapping)
        X_test_tapping = scaler_tapping.transform(X_test_tapping)
        
        # Convert to tensors
        X_train_audio_tensor = torch.tensor(X_train_audio, dtype=torch.float32)
        X_train_tapping_tensor = torch.tensor(X_train_tapping, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        X_val_audio_tensor = torch.tensor(X_val_audio, dtype=torch.float32)
        X_val_tapping_tensor = torch.tensor(X_val_tapping, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        # Auto-configure branch architectures if not provided
        if audio_branch is None:
            audio_branch_config = {
                "input_dim": X_train_audio_tensor.shape[1],
                "output_dim": 64,
                "hidden_dims": [256, 128]
            }
        else:
            # Create a copy to avoid modifying the original dict
            audio_branch_config = audio_branch.copy()
            audio_branch_config["input_dim"] = X_train_audio_tensor.shape[1]
        
        if tapping_branch is None:
            tapping_branch_config = {
                "input_dim": X_train_tapping_tensor.shape[1],
                "output_dim": 64,
                "hidden_dims": [128, 64]
            }
        else:
            # Create a copy to avoid modifying the original dict
            tapping_branch_config = tapping_branch.copy()
            tapping_branch_config["input_dim"] = X_train_tapping_tensor.shape[1]
        
        # Initialize model
        model = IntermediateFusionMLP(
            hidden_dims=hidden_dims,
            dropout=dropout,
            audio_branch=audio_branch_config,
            tapping_branch=tapping_branch_config
        )
        
        # Train model
        if verbose:
            print(f"\n{'='*50}")
            print("Training IntermediateFusionMLP...")
            print(f"{'='*50}")
        
        model.fit(
            X_train_audio_tensor,
            X_train_tapping_tensor,
            y_train_tensor,
            val_audio_features=X_val_audio_tensor,
            val_tapping_features=X_val_tapping_tensor,
            val_labels=y_val_tensor,
            epochs=num_epochs,
            class_weight=class_weight,
            batch_size=batch_size,
            lr=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose
        )
        
        # Test model with majority voting at patient level
        if verbose:
            print(f"\n{'='*50}")
            print("Testing IntermediateFusionMLP with Majority Voting...")
            print(f"{'='*50}")

        # Build separate test dataframes with standardized features
        test_audio_standardized = test_audio.copy()
        test_audio_standardized[audio_feature_cols] = X_test_audio
        
        test_tapping_standardized = test_tapping.copy()
        test_tapping_standardized[tapping_feature_cols] = X_test_tapping
        
        # Construct labels_df for this fold (one row per patient)
        test_labels_df = test_audio[['healthCode', 'label_PD']].drop_duplicates().reset_index(drop=True)

        # Use majority voting per healthCode with separate audio and tapping dataframes
        test_results = model.test_with_majority_counting(test_audio_standardized, test_tapping_standardized, test_labels_df)

        # Display patient-level results
        if verbose:
            print(f"\nPatient-level Results (Majority Voting):")
            print(f"  Number of patients: {test_results['num_patients']}")
            print(f"  Number of trials: {test_results['num_trials']}")
            print(f"  Test Accuracy: {test_results['test_acc']:.4f}")
            print(f"  Test F1 Score: {test_results['test_f1']:.4f}")
            print(f"  Test ROC AUC: {test_results['test_roc_auc']:.4f}")
            print(f"\nConfusion Matrix:\n{test_results['test_conf_mat']}")

        # Store results (test loss not defined for majority voting)
        test_losses.append(0.0)
        test_accs.append(test_results['test_acc'])
        test_f1s.append(test_results['test_f1'])
        test_roc_aucs.append(test_results['test_roc_auc'])
        confusion_matrices.append(test_results['test_conf_mat'])
        
        # Save model state dict
        models_list.append(model.state_dict())
    
    # Calculate summary statistics
    results = {
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'mean_test_acc': np.mean(test_accs),
        'std_test_acc': np.std(test_accs),
        'mean_test_f1': np.mean(test_f1s),
        'std_test_f1': np.std(test_f1s),
        'mean_test_roc_auc': np.mean(test_roc_aucs),
        'std_test_roc_auc': np.std(test_roc_aucs),
        'confusion_matrices': confusion_matrices,
        'models_list': models_list
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("5-Fold Cross-Validation Summary")
        print(f"{'='*50}")
        print(f"Test Loss:     {results['mean_test_loss']:.4f} ± {results['std_test_loss']:.4f}")
        print(f"Test Accuracy: {results['mean_test_acc']:.4f} ± {results['std_test_acc']:.4f}")
        print(f"Test F1 Score: {results['mean_test_f1']:.4f} ± {results['std_test_f1']:.4f}")
        print(f"Test ROC AUC:  {results['mean_test_roc_auc']:.4f} ± {results['std_test_roc_auc']:.4f}")
    
    return results
