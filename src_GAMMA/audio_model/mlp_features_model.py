"""
MLP (Multi-Layer Perceptron) model for acoustic feature classification
Simple architecture for Parkinson's detection from extracted acoustic features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class AcousticFeaturesDataset(Dataset):
    """
    Dataset for loading pre-extracted acoustic features from CSV
    """
    def __init__(self, csv_path=None, features=None, labels=None, feature_columns=None):
        """
        Args:
            csv_path: Path to CSV file with features (optional)
            features: Pre-loaded feature array (optional, if csv_path not provided)
            labels: Pre-loaded label array (optional, if csv_path not provided)
            feature_columns: List of feature column names to use (optional)
        """
        if csv_path is not None:
            # Load from CSV
            df = pd.read_csv(csv_path)
            
            # Extract label column (assume 'label' or 'target' column exists)
            if 'label' in df.columns:
                self.labels = df['label'].values
                df = df.drop(['label'], axis=1)
            elif 'target' in df.columns:
                self.labels = df['target'].values
                df = df.drop(['target'], axis=1)
            else:
                raise ValueError("CSV must contain 'label' or 'target' column")
            
            # Drop non-feature columns (filename, healthcode, record_id, etc.)
            drop_cols = ['filename', 'healthcode', 'record_id']
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
            
            # Select specific feature columns if provided
            if feature_columns is not None:
                df = df[feature_columns]
            
            self.features = df.values.astype(np.float32)
            
        else:
            # Use pre-loaded arrays
            if features is None or labels is None:
                raise ValueError("Either csv_path or (features, labels) must be provided")
            
            self.features = features.astype(np.float32)
            self.labels = labels
        
        # Handle NaN values
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Dataset loaded: {len(self.labels)} samples, {self.features.shape[1]} features")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label


class MLPAcousticClassifier(nn.Module):
    """
    Multi-Layer Perceptron for acoustic feature classification
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_classes=2, dropout=0.5):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout probability
        """
        super(MLPAcousticClassifier, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.network(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Returns:
        Average loss and accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.squeeze().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set
    
    Returns:
        Average loss, accuracy, and AUC
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.squeeze().to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, accuracy, auc


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=0.001,
    class_weight=3.0,
    device='cuda'
):
    """
    Full training loop
    
    Args:
        model: MLP model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        class_weight: Weight for class 0 (controls) to handle imbalance
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model
    """
    model = model.to(device)
    
    # Loss and optimizer with class weights to handle imbalance
    # Label smoothing (0.1) prevents overconfident predictions and improves calibration
    class_weights = torch.FloatTensor([class_weight, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # Increased regularization
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_auc = 0.0
    patience_counter = 0
    early_stop_patience = 20
    
    print("="*80)
    print("Starting Training")
    print("="*80)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
        
        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mlp_features_model.pth')
            print(f"  ✓ Best model saved (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    print("="*80)
    print(f"Training Complete! Best Val AUC: {best_val_auc:.4f}")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_mlp_features_model.pth'))
    
    return model


def normalize_features(train_features, val_features=None, test_features=None):
    """
    Normalize features using training set statistics
    
    Args:
        train_features: Training features array
        val_features: Validation features array (optional)
        test_features: Test features array (optional)
    
    Returns:
        Normalized features (train, val, test) and statistics (mean, std)
    """
    # Calculate statistics from training set
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    # Normalize
    train_norm = (train_features - mean) / std
    
    results = [train_norm]
    
    if val_features is not None:
        val_norm = (val_features - mean) / std
        results.append(val_norm)
    
    if test_features is not None:
        test_norm = (test_features - mean) / std
        results.append(test_norm)
    
    results.extend([mean, std])
    
    return tuple(results)


def aggregate_predictions_by_patient(predictions, probabilities, labels, healthcodes, method='majority'):
    """
    Aggregate recording-level predictions to patient-level
    
    Args:
        predictions: Array of predicted labels for each recording
        probabilities: Array of predicted probabilities (for class 1) for each recording
        labels: Array of true labels for each recording
        healthcodes: Array of healthCodes for each recording
        method: 'majority' or 'average'
    
    Returns:
        patient_preds, patient_labels, patient_healthcodes
    """
    from collections import defaultdict
    
    # Group by healthCode
    patient_data = defaultdict(lambda: {'preds': [], 'probs': [], 'label': None})
    
    for i, hc in enumerate(healthcodes):
        patient_data[hc]['preds'].append(predictions[i])
        patient_data[hc]['probs'].append(probabilities[i])
        patient_data[hc]['label'] = labels[i]  # Same for all recordings from same patient
    
    # Aggregate
    patient_preds = []
    patient_labels = []
    patient_healthcodes = []
    
    for hc, data in patient_data.items():
        if method == 'majority':
            # Majority vote of predictions
            pred = 1 if np.sum(data['preds']) > len(data['preds']) / 2 else 0
        elif method == 'average':
            # Average probability, threshold at 0.5
            avg_prob = np.mean(data['probs'])
            pred = 1 if avg_prob > 0.5 else 0
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        patient_preds.append(pred)
        patient_labels.append(data['label'])
        patient_healthcodes.append(hc)
    
    return np.array(patient_preds), np.array(patient_labels), patient_healthcodes


def train_5fold_cv(
    features_csv,
    labels_csv,
    train_folds_csv,
    val_test_folds_csv,
    output_dir,
    batch_size=64,
    num_epochs=100,
    learning_rate=0.001,
    hidden_sizes=[256, 128, 64],
    dropout=0.5,
    class_weight=3.0
):
    """
    Train MLP with 5-fold cross-validation
    
    Args:
        features_csv: Path to acoustic_features.csv
        labels_csv: Path to paired_healthcode.csv
        train_folds_csv: Path to healthcode_5fold_train.csv
        val_test_folds_csv: Path to healthcode_5fold_val_test.csv (contains 'subset' column)
        output_dir: Directory to save results
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        hidden_sizes: List of hidden layer sizes
        dropout: Dropout probability
        class_weight: Weight for class 0 (controls) to handle imbalance
    """
    import os
    import json
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("="*80)
    print("Loading data...")
    print("="*80)
    
    # Load features
    features_df = pd.read_csv(features_csv)
    print(f"Loaded features: {features_df.shape}")
    print(f"  Columns: {list(features_df.columns[:5])}...")
    
    # Load labels (semicolon separator)
    labels_df = pd.read_csv(labels_csv, sep=';')
    print(f"Loaded labels: {labels_df.shape}")
    print(f"  Columns: {list(labels_df.columns)}")
    
    # Load fold splits
    train_folds_df = pd.read_csv(train_folds_csv)
    val_test_folds_df = pd.read_csv(val_test_folds_csv)
    print(f"Train folds: {train_folds_df.shape}")
    print(f"  Columns: {list(train_folds_df.columns)}")
    print(f"Val+Test folds: {val_test_folds_df.shape}")
    print(f"  Columns: {list(val_test_folds_df.columns)}")
    
    # Separate val and test based on 'subset' column
    val_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'val']
    test_folds_df = val_test_folds_df[val_test_folds_df['subset'] == 'test']
    print(f"  - Validation folds: {val_folds_df.shape}")
    print(f"  - Test folds: {test_folds_df.shape}")
    
    # Rename 'healthcode' to 'healthCode' in features_df to match labels
    # acoustic_features.csv has 'healthcode' (lowercase)
    # paired_healthcode.csv has 'healthCode' (camelCase)
    if 'healthcode' in features_df.columns and 'healthCode' not in features_df.columns:
        features_df = features_df.rename(columns={'healthcode': 'healthCode'})
        print(f"Renamed 'healthcode' → 'healthCode' for consistency")
    
    # Merge features with labels on healthCode
    features_df = features_df.merge(labels_df[['healthCode', 'label_PD']], on='healthCode', how='inner')
    print(f"Features after merging with labels: {features_df.shape}")
    
    # Filter to only include healthcodes in 5-fold splits
    all_fold_healthcodes = set(train_folds_df['healthCode'].unique()) | set(val_test_folds_df['healthCode'].unique())
    features_df = features_df[features_df['healthCode'].isin(all_fold_healthcodes)]
    print(f"Features after filtering to 5-fold healthcodes: {features_df.shape}")
    
    # Prepare feature columns (exclude metadata)
    # After renaming: 'healthcode' → 'healthCode', so only 'healthCode' exists
    metadata_cols = ['filename', 'healthCode', 'record_id', 'label_PD']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:5]}... (showing first 5)")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Store results for all folds
    all_val_results = []
    all_test_results = []
    
    # 5-fold cross-validation
    for fold in range(5):
        print("="*80)
        print(f"FOLD {fold + 1}/5")
        print("="*80)
        
        # Get unique patient healthCodes for this fold (for filtering)
        train_patient_ids = train_folds_df[train_folds_df['fold_iteration'] == fold]['healthCode'].values
        val_patient_ids = val_folds_df[val_folds_df['fold_iteration'] == fold]['healthCode'].values
        test_patient_ids = test_folds_df[test_folds_df['fold_iteration'] == fold]['healthCode'].values
        
        print(f"Train patients: {len(train_patient_ids)}")
        print(f"Val patients: {len(val_patient_ids)}")
        print(f"Test patients: {len(test_patient_ids)}")
        
        # Split data by patient healthCode
        train_data = features_df[features_df['healthCode'].isin(train_patient_ids)]
        val_data = features_df[features_df['healthCode'].isin(val_patient_ids)]
        test_data = features_df[features_df['healthCode'].isin(test_patient_ids)]
        
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
        
        # Normalize features using training set statistics
        X_train, X_val, X_test, mean, std = normalize_features(X_train, X_val, X_test)
        
        # Create datasets
        train_dataset = AcousticFeaturesDataset(features=X_train, labels=y_train)
        val_dataset = AcousticFeaturesDataset(features=X_val, labels=y_val)
        test_dataset = AcousticFeaturesDataset(features=X_test, labels=y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = MLPAcousticClassifier(
            input_size=len(feature_cols),
            hidden_sizes=hidden_sizes,
            num_classes=2,
            dropout=dropout
        )
        
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        
        # Train model (using validation set for early stopping)
        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            class_weight=class_weight,
            device=device
        )
        
        # ====================================================================
        # VALIDATION SET EVALUATION (for model selection)
        # ====================================================================
        print("\n" + "="*80)
        print("VALIDATION SET EVALUATION")
        print("="*80)
        
        trained_model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        
        # Get predictions for all validation recordings
        with torch.no_grad():
            for idx, (features, labels) in enumerate(val_loader):
                features = features.to(device)
                outputs = trained_model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.squeeze().cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())
        
        # Get healthCodes for each recording (for patient-level aggregation)
        val_recording_healthcodes = val_data['healthCode'].values
        
        # === Recording-level metrics ===
        val_rec_accuracy = 100 * np.mean(np.array(val_preds) == np.array(val_labels))
        val_rec_auc = roc_auc_score(val_labels, val_probs)
        val_rec_f1 = f1_score(val_labels, val_preds)
        val_rec_cm = confusion_matrix(val_labels, val_preds)
        
        # Calculate recording-level sensitivity/specificity
        tn_rec, fp_rec, fn_rec, tp_rec = val_rec_cm.ravel()
        val_rec_sensitivity = tp_rec / (tp_rec + fn_rec) if (tp_rec + fn_rec) > 0 else 0
        val_rec_specificity = tn_rec / (tn_rec + fp_rec) if (tn_rec + fp_rec) > 0 else 0
        
        # === Patient-level metrics (majority voting) ===
        val_pat_preds_maj, val_pat_labels_maj, _ = aggregate_predictions_by_patient(
            val_preds, val_probs, val_labels, val_recording_healthcodes, method='majority'
        )
        val_maj_accuracy = 100 * np.mean(val_pat_preds_maj == val_pat_labels_maj)
        val_maj_f1 = f1_score(val_pat_labels_maj, val_pat_preds_maj)
        val_maj_cm = confusion_matrix(val_pat_labels_maj, val_pat_preds_maj)
        
        # === Patient-level metrics (average probability) ===
        val_pat_preds_avg, val_pat_labels_avg, _ = aggregate_predictions_by_patient(
            val_preds, val_probs, val_labels, val_recording_healthcodes, method='average'
        )
        val_avg_accuracy = 100 * np.mean(val_pat_preds_avg == val_pat_labels_avg)
        val_avg_f1 = f1_score(val_pat_labels_avg, val_pat_preds_avg)
        val_avg_cm = confusion_matrix(val_pat_labels_avg, val_pat_preds_avg)
        
        # Calculate sensitivity/specificity for majority voting
        tn, fp, fn, tp = val_maj_cm.ravel()
        val_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate sensitivity/specificity for average probability
        tn_avg, fp_avg, fn_avg, tp_avg = val_avg_cm.ravel()
        val_avg_sensitivity = tp_avg / (tp_avg + fn_avg) if (tp_avg + fn_avg) > 0 else 0
        val_avg_specificity = tn_avg / (tn_avg + fp_avg) if (tn_avg + fp_avg) > 0 else 0
        
        # Store validation results
        val_fold_results = {
            'fold': fold + 1,
            'recording_level': {
                'accuracy': val_rec_accuracy,
                'auc': val_rec_auc,
                'f1_score': val_rec_f1,
                'sensitivity': val_rec_sensitivity,
                'specificity': val_rec_specificity,
                'confusion_matrix': val_rec_cm.tolist()
            },
            'patient_level_majority': {
                'accuracy': val_maj_accuracy,
                'f1_score': val_maj_f1,
                'sensitivity': val_sensitivity,
                'specificity': val_specificity,
                'confusion_matrix': val_maj_cm.tolist()
            },
            'patient_level_average': {
                'accuracy': val_avg_accuracy,
                'f1_score': val_avg_f1,
                'sensitivity': val_avg_sensitivity,
                'specificity': val_avg_specificity,
                'confusion_matrix': val_avg_cm.tolist()
            }
        }
        all_val_results.append(val_fold_results)
        
        # Print validation results
        print(f"\nValidation Results (Fold {fold + 1}):")
        print(f"  Recording-level - Accuracy: {val_rec_accuracy:.2f}%, AUC: {val_rec_auc:.4f}, F1: {val_rec_f1:.4f}")
        print(f"                    Sensitivity: {val_rec_sensitivity:.4f}, Specificity: {val_rec_specificity:.4f}")
        print(f"  Patient-level (Majority) - Accuracy: {val_maj_accuracy:.2f}%, F1: {val_maj_f1:.4f}")
        print(f"  Patient-level (Average) - Accuracy: {val_avg_accuracy:.2f}%, F1: {val_avg_f1:.4f}")
        
        # ====================================================================
        # TEST SET EVALUATION (final performance)
        # ====================================================================
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        
        test_preds = []
        test_labels = []
        test_probs = []
        
        # Get predictions for all test recordings
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = trained_model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.squeeze().cpu().numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())
        
        # Get healthCodes for each recording (for patient-level aggregation)
        test_recording_healthcodes = test_data['healthCode'].values
        
        # === Recording-level metrics ===
        test_rec_accuracy = 100 * np.mean(np.array(test_preds) == np.array(test_labels))
        test_rec_auc = roc_auc_score(test_labels, test_probs)
        test_rec_f1 = f1_score(test_labels, test_preds)
        test_rec_cm = confusion_matrix(test_labels, test_preds)
        
        # Calculate recording-level sensitivity/specificity
        tn_rec, fp_rec, fn_rec, tp_rec = test_rec_cm.ravel()
        test_rec_sensitivity = tp_rec / (tp_rec + fn_rec) if (tp_rec + fn_rec) > 0 else 0
        test_rec_specificity = tn_rec / (tn_rec + fp_rec) if (tn_rec + fp_rec) > 0 else 0
        
        # === Patient-level metrics (majority voting) ===
        test_pat_preds_maj, test_pat_labels_maj, test_pat_hc_maj = aggregate_predictions_by_patient(
            test_preds, test_probs, test_labels, test_recording_healthcodes, method='majority'
        )
        test_maj_accuracy = 100 * np.mean(test_pat_preds_maj == test_pat_labels_maj)
        test_maj_f1 = f1_score(test_pat_labels_maj, test_pat_preds_maj)
        test_maj_cm = confusion_matrix(test_pat_labels_maj, test_pat_preds_maj)
        
        # AUC at patient level (use average probability)
        patient_avg_probs = {}
        for i, hc in enumerate(test_recording_healthcodes):
            if hc not in patient_avg_probs:
                patient_avg_probs[hc] = []
            patient_avg_probs[hc].append(test_probs[i])
        
        pat_probs_for_auc = [np.mean(patient_avg_probs[hc]) for hc in test_pat_hc_maj]
        test_maj_auc = roc_auc_score(test_pat_labels_maj, pat_probs_for_auc)
        
        # === Patient-level metrics (average probability) ===
        test_pat_preds_avg, test_pat_labels_avg, _ = aggregate_predictions_by_patient(
            test_preds, test_probs, test_labels, test_recording_healthcodes, method='average'
        )
        test_avg_accuracy = 100 * np.mean(test_pat_preds_avg == test_pat_labels_avg)
        test_avg_f1 = f1_score(test_pat_labels_avg, test_pat_preds_avg)
        test_avg_cm = confusion_matrix(test_pat_labels_avg, test_pat_preds_avg)
        test_avg_auc = roc_auc_score(test_pat_labels_avg, pat_probs_for_auc)
        
        # Calculate sensitivity/specificity for majority voting
        tn, fp, fn, tp = test_maj_cm.ravel()
        test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate sensitivity/specificity for average probability
        tn_avg, fp_avg, fn_avg, tp_avg = test_avg_cm.ravel()
        test_avg_sensitivity = tp_avg / (tp_avg + fn_avg) if (tp_avg + fn_avg) > 0 else 0
        test_avg_specificity = tn_avg / (tn_avg + fp_avg) if (tn_avg + fp_avg) > 0 else 0
        
        # Store test results
        test_fold_results = {
            'fold': fold + 1,
            'num_patients': len(test_pat_hc_maj),
            'num_recordings': len(test_preds),
            'recording_level': {
                'accuracy': test_rec_accuracy,
                'auc': test_rec_auc,
                'f1_score': test_rec_f1,
                'sensitivity': test_rec_sensitivity,
                'specificity': test_rec_specificity,
                'confusion_matrix': test_rec_cm.tolist()
            },
            'patient_level_majority': {
                'accuracy': test_maj_accuracy,
                'auc': test_maj_auc,
                'f1_score': test_maj_f1,
                'sensitivity': test_sensitivity,
                'specificity': test_specificity,
                'confusion_matrix': test_maj_cm.tolist()
            },
            'patient_level_average': {
                'accuracy': test_avg_accuracy,
                'auc': test_avg_auc,
                'f1_score': test_avg_f1,
                'sensitivity': test_avg_sensitivity,
                'specificity': test_avg_specificity,
                'confusion_matrix': test_avg_cm.tolist()
            }
        }
        all_test_results.append(test_fold_results)
        
        # Print test results
        print(f"\nTest Results (Fold {fold + 1}):")
        print(f"  Patients: {len(test_pat_hc_maj)}, Recordings: {len(test_preds)}")
        print(f"\n  Recording-level:")
        print(f"    Accuracy: {test_rec_accuracy:.2f}%, AUC: {test_rec_auc:.4f}, F1: {test_rec_f1:.4f}")
        print(f"    Sensitivity: {test_rec_sensitivity:.4f}, Specificity: {test_rec_specificity:.4f}")
        print(f"    Confusion Matrix:\n{test_rec_cm}")
        print(f"\n  Patient-level (Majority Voting):")
        print(f"    Accuracy: {test_maj_accuracy:.2f}%, AUC: {test_maj_auc:.4f}, F1: {test_maj_f1:.4f}")
        print(f"    Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}")
        print(f"    Confusion Matrix:\n{test_maj_cm}")
        print(f"\n  Patient-level (Average Probability):")
        print(f"    Accuracy: {test_avg_accuracy:.2f}%, AUC: {test_avg_auc:.4f}, F1: {test_avg_f1:.4f}")
        print(f"    Sensitivity: {test_avg_sensitivity:.4f}, Specificity: {test_avg_specificity:.4f}")
        print(f"    Confusion Matrix:\n{test_avg_cm}")
        
        # Save model
        model_path = os.path.join(output_dir, f'model_fold_{fold+1}.pth')
        torch.save(trained_model.state_dict(), model_path)
        print(f"\n  Model saved to: {model_path}\n")
    
    # ========================================================================
    # AGGREGATE RESULTS ACROSS ALL FOLDS
    # ========================================================================
    
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS 5 FOLDS")
    print("="*80)
    
    # === TEST SET SUMMARY (Patient-level - Majority Voting) ===
    avg_test_maj_accuracy = np.mean([r['patient_level_majority']['accuracy'] for r in all_test_results])
    avg_test_maj_auc = np.mean([r['patient_level_majority']['auc'] for r in all_test_results])
    avg_test_maj_f1 = np.mean([r['patient_level_majority']['f1_score'] for r in all_test_results])
    avg_test_maj_sensitivity = np.mean([r['patient_level_majority']['sensitivity'] for r in all_test_results])
    avg_test_maj_specificity = np.mean([r['patient_level_majority']['specificity'] for r in all_test_results])
    
    std_test_maj_accuracy = np.std([r['patient_level_majority']['accuracy'] for r in all_test_results])
    std_test_maj_auc = np.std([r['patient_level_majority']['auc'] for r in all_test_results])
    std_test_maj_f1 = np.std([r['patient_level_majority']['f1_score'] for r in all_test_results])
    std_test_maj_sensitivity = np.std([r['patient_level_majority']['sensitivity'] for r in all_test_results])
    std_test_maj_specificity = np.std([r['patient_level_majority']['specificity'] for r in all_test_results])
    
    # === TEST SET SUMMARY (Patient-level - Average Probability) ===
    avg_test_avg_accuracy = np.mean([r['patient_level_average']['accuracy'] for r in all_test_results])
    avg_test_avg_auc = np.mean([r['patient_level_average']['auc'] for r in all_test_results])
    avg_test_avg_f1 = np.mean([r['patient_level_average']['f1_score'] for r in all_test_results])
    avg_test_avg_sensitivity = np.mean([r['patient_level_average']['sensitivity'] for r in all_test_results])
    avg_test_avg_specificity = np.mean([r['patient_level_average']['specificity'] for r in all_test_results])
    
    std_test_avg_accuracy = np.std([r['patient_level_average']['accuracy'] for r in all_test_results])
    std_test_avg_auc = np.std([r['patient_level_average']['auc'] for r in all_test_results])
    std_test_avg_f1 = np.std([r['patient_level_average']['f1_score'] for r in all_test_results])
    std_test_avg_sensitivity = np.std([r['patient_level_average']['sensitivity'] for r in all_test_results])
    std_test_avg_specificity = np.std([r['patient_level_average']['specificity'] for r in all_test_results])
    
    # === TEST SET SUMMARY (Recording-level) ===
    avg_test_rec_accuracy = np.mean([r['recording_level']['accuracy'] for r in all_test_results])
    avg_test_rec_auc = np.mean([r['recording_level']['auc'] for r in all_test_results])
    avg_test_rec_f1 = np.mean([r['recording_level']['f1_score'] for r in all_test_results])
    avg_test_rec_sensitivity = np.mean([r['recording_level']['sensitivity'] for r in all_test_results])
    avg_test_rec_specificity = np.mean([r['recording_level']['specificity'] for r in all_test_results])
    
    std_test_rec_accuracy = np.std([r['recording_level']['accuracy'] for r in all_test_results])
    std_test_rec_auc = np.std([r['recording_level']['auc'] for r in all_test_results])
    std_test_rec_f1 = np.std([r['recording_level']['f1_score'] for r in all_test_results])
    std_test_rec_sensitivity = np.std([r['recording_level']['sensitivity'] for r in all_test_results])
    std_test_rec_specificity = np.std([r['recording_level']['specificity'] for r in all_test_results])
    
    # Summary results
    summary = {
        'test_patient_majority': {
            'avg_accuracy': avg_test_maj_accuracy,
            'std_accuracy': std_test_maj_accuracy,
            'avg_auc': avg_test_maj_auc,
            'std_auc': std_test_maj_auc,
            'avg_f1': avg_test_maj_f1,
            'std_f1': std_test_maj_f1,
            'avg_sensitivity': avg_test_maj_sensitivity,
            'std_sensitivity': std_test_maj_sensitivity,
            'avg_specificity': avg_test_maj_specificity,
            'std_specificity': std_test_maj_specificity
        },
        'test_patient_average': {
            'avg_accuracy': avg_test_avg_accuracy,
            'std_accuracy': std_test_avg_accuracy,
            'avg_auc': avg_test_avg_auc,
            'std_auc': std_test_avg_auc,
            'avg_f1': avg_test_avg_f1,
            'std_f1': std_test_avg_f1,
            'avg_sensitivity': avg_test_avg_sensitivity,
            'std_sensitivity': std_test_avg_sensitivity,
            'avg_specificity': avg_test_avg_specificity,
            'std_specificity': std_test_avg_specificity
        },
        'test_recording_level': {
            'avg_accuracy': avg_test_rec_accuracy,
            'std_accuracy': std_test_rec_accuracy,
            'avg_auc': avg_test_rec_auc,
            'std_auc': std_test_rec_auc,
            'avg_f1': avg_test_rec_f1,
            'std_f1': std_test_rec_f1,
            'avg_sensitivity': avg_test_rec_sensitivity,
            'std_sensitivity': std_test_rec_sensitivity,
            'avg_specificity': avg_test_rec_specificity,
            'std_specificity': std_test_rec_specificity
        },
        'validation_results': all_val_results,
        'test_fold_results': all_test_results,
        'config': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'hidden_sizes': hidden_sizes,
            'dropout': dropout,
            'num_features': len(feature_cols)
        }
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, 'V1_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    print("\n--- TEST SET: Patient-level (Majority Voting) ---")
    print(f"Average Accuracy: {avg_test_maj_accuracy:.2f}% ± {std_test_maj_accuracy:.2f}%")
    print(f"Average AUC: {avg_test_maj_auc:.4f} ± {std_test_maj_auc:.4f}")
    print(f"Average F1 Score: {avg_test_maj_f1:.4f} ± {std_test_maj_f1:.4f}")
    print(f"Average Sensitivity: {avg_test_maj_sensitivity:.4f} ± {std_test_maj_sensitivity:.4f}")
    print(f"Average Specificity: {avg_test_maj_specificity:.4f} ± {std_test_maj_specificity:.4f}")
    
    print("\n--- TEST SET: Patient-level (Average Probability) ---")
    print(f"Average Accuracy: {avg_test_avg_accuracy:.2f}% ± {std_test_avg_accuracy:.2f}%")
    print(f"Average AUC: {avg_test_avg_auc:.4f} ± {std_test_avg_auc:.4f}")
    print(f"Average F1 Score: {avg_test_avg_f1:.4f} ± {std_test_avg_f1:.4f}")
    print(f"Average Sensitivity: {avg_test_avg_sensitivity:.4f} ± {std_test_avg_sensitivity:.4f}")
    print(f"Average Specificity: {avg_test_avg_specificity:.4f} ± {std_test_avg_specificity:.4f}")
    
    print("\n--- TEST SET: Recording-level (for reference) ---")
    print(f"Average Accuracy: {avg_test_rec_accuracy:.2f}% ± {std_test_rec_accuracy:.2f}%")
    print(f"Average AUC: {avg_test_rec_auc:.4f} ± {std_test_rec_auc:.4f}")
    print(f"Average F1 Score: {avg_test_rec_f1:.4f} ± {std_test_rec_f1:.4f}")
    print(f"Average Sensitivity: {avg_test_rec_sensitivity:.4f} ± {std_test_rec_sensitivity:.4f}")
    print(f"Average Specificity: {avg_test_rec_specificity:.4f} ± {std_test_rec_specificity:.4f}")
    print(f"Average AUC: {avg_test_rec_auc:.4f} ± {std_test_rec_auc:.4f}")
    print(f"Average F1 Score: {avg_test_rec_f1:.4f} ± {std_test_rec_f1:.4f}")
    
    print(f"\nResults saved to: {summary_path}")
    print("="*80)
    
    return summary


if __name__ == "__main__":
    """
    Run MLP with 5-fold cross-validation - Version 1
    Uses validation set for model selection and test set for final evaluation
    """
    
    # Configuration
    FEATURES_CSV = "/mloscratch/users/gnahas/data/features/acoustic_features.csv"
    LABELS_CSV = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
    TRAIN_FOLDS_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
    VAL_TEST_FOLDS_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
    OUTPUT_DIR = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/MLP"
    
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_SIZES = [256, 128, 64]
    DROPOUT = 0.5
    
    print("="*80)
    print("MLP Training - Version 1 (Baseline)")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Features: {FEATURES_CSV}")
    print(f"  Labels: {LABELS_CSV}")
    print(f"  Train folds: {TRAIN_FOLDS_CSV}")
    print(f"  Val+Test folds: {VAL_TEST_FOLDS_CSV}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Hidden sizes: {HIDDEN_SIZES}")
    print(f"  Dropout: {DROPOUT}")
    print("="*80 + "\n")
    
    # Run 5-fold cross-validation
    summary = train_5fold_cv(
        features_csv=FEATURES_CSV,
        labels_csv=LABELS_CSV,
        train_folds_csv=TRAIN_FOLDS_CSV,
        val_test_folds_csv=VAL_TEST_FOLDS_CSV,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT
    )
