
"""
ResNet1D Model for Acoustic Feature Classification

This script implements a simple 1D ResNet architecture for classifying patients using precomputed acoustic features.
It supports 5-fold cross-validation, patient-level aggregation, and feature importance analysis.

Main components:
- AcousticFeaturesDataset: Loads feature vectors and labels for each patient
- SingleResNet1D: The neural network model (with optional attention)
- Training and evaluation loops
- Patient-level aggregation and metrics
- Feature importance computation and plotting
- 5-fold cross-validation pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class AcousticFeaturesDataset(Dataset):
    """
    Dataset for loading acoustic features and labels, grouped by patient.
    Handles NaN/infinite values and prints class distribution.
    """
    def __init__(self, features_df, feature_columns, label_column='label_PD'):
        """
        Args:
            features_df (DataFrame): DataFrame with features and labels.
            feature_columns (list): List of feature column names.
            label_column (str): Name of label column.
        """
        self.features = features_df[feature_columns].values.astype(np.float32)
        self.labels = features_df[label_column].values.astype(np.int64)
        self.health_codes = features_df['healthCode'].values    
            
        # Handle NaN/inf
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"  Dataset: {len(self.labels)} samples, {self.features.shape[1]} features")
        print(f"  Class distribution: Control={np.sum(self.labels==0)}, PD={np.sum(self.labels==1)}")
    
    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (features, label, health_code)
        """
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label, self.health_codes[idx]


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class ResidualBlock1D(nn.Module):
    """
    1D Residual Block for feature vectors (fully connected, not convolutional).
    Adds a skip connection for better gradient flow.
    """
    def __init__(self, in_features, out_features, dropout=0.3):
        """
        Args:
            in_features (int): Input feature size.
            out_features (int): Output feature size.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, in_features).
        Returns:
            Tensor: Output tensor of shape (batch, out_features).
        """
        residual = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual  
        out = self.relu(out)
        
        return out


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature attention.
    """
    def __init__(self, features, reduction=4):
        """
        Args:
            features (int): Number of input features.
            reduction (int): Reduction ratio for bottleneck.
        """
        super().__init__()
        self.fc1 = nn.Linear(features, features // reduction)
        self.fc2 = nn.Linear(features // reduction, features)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, features).
        Returns:
            Tensor: Output tensor with channel-wise attention applied.
        """
        scale = self.fc1(x)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        
        return x * scale 


# ============================================================================
# ARCHITECTURE 1: SINGLE RESNET1D (SIMPLE & EFFECTIVE)
# ============================================================================

class SingleResNet1D(nn.Module):
    """
    Simple ResNet1D for all features.
    Best for quick baselines and interpretable results.
    Optionally includes channel attention.
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], 
                 num_classes=2, dropout=0.4, use_attention=True):
        """
        Args:
            input_size (int): Number of input features.
            hidden_sizes (list): List of hidden layer sizes.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
            use_attention (bool): Whether to use SE attention block.
        """
        super().__init__()
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock1D(hidden_sizes[i], hidden_sizes[i+1], dropout)
            )
        
        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SEBlock1D(hidden_sizes[-1])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1], num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, input_size).
        Returns:
            Tensor: Output logits of shape (batch, num_classes).
        """
        x = self.input_layer(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        out = self.classifier(x)
        return out


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for features, labels, _ in dataloader:
        features, labels = features.to(device), labels.squeeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), 100 * correct / total


def evaluate(model, dataloader, device):
    """
    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Data loader.
        device: Device to use.
    Returns:
        dict: Metrics for both majority vote and probability threshold aggregation.
    """
    model.eval()
    all_probs, all_preds, all_labels, all_health_codes = [], [], [], []
    
    with torch.no_grad():
        for features, labels, health_codes in dataloader:
            features, labels = features.to(device), labels.squeeze(1).to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_health_codes.extend(health_codes)
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_health_codes = np.array(all_health_codes)

    # Patient-level aggregation
    unique_hcs = np.unique(all_health_codes)
    patient_labels, patient_preds_majority, patient_probs = [], [], []

    for hc in unique_hcs:
        mask = all_health_codes == hc
        patient_labels.append(all_labels[mask][0])
        # Majority vote
        vote_counts = Counter(all_preds[mask])
        patient_preds_majority.append(vote_counts.most_common(1)[0][0])
        # Average probability
        avg_prob = np.mean(all_probs[mask])
        patient_probs.append(avg_prob)

    patient_labels = np.array(patient_labels)
    patient_preds_majority = np.array(patient_preds_majority)
    patient_probs = np.array(patient_probs)

    # Use fixed threshold (0.5) for probability-based predictions
    patient_preds_prob = (patient_probs >= 0.5).astype(int)

    # Metrics (majority vote)
    acc_majority = 100 * np.mean(patient_preds_majority == patient_labels)
    cm_majority = confusion_matrix(patient_labels, patient_preds_majority)

    if cm_majority.ravel().shape[0] == 4:
        tn, fp, fn, tp = cm_majority.ravel()
        sens_majority = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_majority = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens_majority = spec_majority = 0

    # Metrics (probability, with tuned threshold)
    acc_prob = 100 * np.mean(patient_preds_prob == patient_labels)
    cm_prob = confusion_matrix(patient_labels, patient_preds_prob)

    if cm_prob.ravel().shape[0] == 4:
        tn, fp, fn, tp = cm_prob.ravel()
        sens_prob = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_prob = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens_prob = spec_prob = 0

    try:
        auc = roc_auc_score(patient_labels, patient_probs)
    except:
        auc = 0.0

    return {
        'patient_accuracy': acc_majority,
        'patient_sensitivity': sens_majority,
        'patient_specificity': spec_majority,
        'confusion_matrix': cm_majority,
        'patient_preds': patient_preds_majority,
        'patient_accuracy_prob': acc_prob,
        'patient_sensitivity_prob': sens_prob,
        'patient_specificity_prob': spec_prob,
        'confusion_matrix_prob': cm_prob,
        'patient_preds_prob': patient_preds_prob,
        'patient_auc': auc,
        'patient_labels': patient_labels,
        'patient_probs': patient_probs
        }


def train_model(model, train_loader, val_loader, num_epochs, learning_rate,
                device, patience, fold_num, output_dir):
    """
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Maximum number of epochs.
        learning_rate (float): Initial learning rate.
        device: Device to use.
        patience (int): Early stopping patience.
        fold_num (int): Fold number for saving checkpoints.
        output_dir (str): Directory to save checkpoints.
    Returns:
        tuple: (trained model, best validation metrics, training history)
    """
    pth_dir = os.path.join(output_dir, 'PTH')
    os.makedirs(pth_dir, exist_ok=True)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6
    )
    
    best_val_auc = 0.0
    best_metrics = None
    best_model_state = None
    epochs_without_improvement = 0
    current_checkpoint_path = None

    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_patient_auc': [], 'val_patient_f1': [], 'learning_rate': []
    }

    print("\nStarting training...")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        val_auc = val_metrics['patient_auc']
        val_f1 = f1_score(val_metrics['patient_labels'], val_metrics['patient_preds'], zero_division=0)

        scheduler.step(val_auc)  
        current_lr = optimizer.param_groups[0]['lr']

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_patient_auc'].append(val_auc)
        history['val_patient_f1'].append(val_f1)
        history['learning_rate'].append(current_lr)

        if epoch % 10 == 0 or val_auc > best_val_auc:
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
            print(f"    Probability - Acc: {val_metrics['patient_accuracy_prob']:.2f}%, Sens: {val_metrics['patient_sensitivity_prob']:.2f}%, Spec: {val_metrics['patient_specificity_prob']:.2f}%")
            print(f"    Majority - Acc: {val_metrics['patient_accuracy']:.2f}%, Sens: {val_metrics['patient_sensitivity']:.2f}%, Spec: {val_metrics['patient_specificity']:.2f}%")

        # Prevent collapse: require minimum sensitivity AND specificity
        min_sens = val_metrics['patient_sensitivity']
        min_spec = val_metrics['patient_specificity']
        is_balanced = min_sens > 20.0 and min_spec > 20.0  # Stricter thresholds

        if val_auc > best_val_auc and is_balanced:
            best_val_auc = val_auc
            best_metrics = val_metrics.copy()
            best_metrics['val_auc'] = val_auc
            best_metrics['val_f1'] = val_f1
            if isinstance(model, nn.DataParallel):
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()
            epochs_without_improvement = 0

            if current_checkpoint_path and os.path.exists(current_checkpoint_path):
                os.remove(current_checkpoint_path)

            checkpoint_path = os.path.join(pth_dir, f'fold_{fold_num}_best_auc_{best_val_auc:.4f}_epoch_{epoch+1}.pth')
            torch.save({
                'fold': fold_num,
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'val_auc': best_val_auc,
                'val_f1': val_f1,
                'val_metrics': val_metrics
            }, checkpoint_path)
            current_checkpoint_path = checkpoint_path

            print(f"  ✓ Best model saved (AUC: {best_val_auc:.4f}, Sens: {min_sens:.1f}%, Spec: {min_spec:.1f}%) -> {os.path.basename(checkpoint_path)}")
        else:
            epochs_without_improvement += 1
            if not is_balanced:
                print(f"  ⚠ Model imbalanced (Sens: {min_sens:.1f}%, Spec: {min_spec:.1f}%)")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no AUC improvement for {patience} epochs)")
            break

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)

    return model, best_metrics, history


def get_detailed_metrics(labels, preds, probs):
    """
    Args:
        labels (array): True labels.
        preds (array): Predicted labels.
        probs (array): Predicted probabilities for positive class.
    Returns:
        dict: F1, precision, recall, and AUC.
    """
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    return {'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc}


def plot_confusion_matrix(cm, filepath, title):
    """
    Args:
        cm (array): Confusion matrix.
        filepath (str): Path to save the plot.
        title (str): Plot title.
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def compute_feature_importance(model, dataloader, device, feature_columns):
    """
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Data loader.
        device: Device to use.
        feature_columns (list): List of feature names.
    Returns:
        DataFrame: Feature importance scores (sorted).
    """
    model.eval()
    total_importance = np.zeros(len(feature_columns))
    n_samples = 0
    
    for features, labels, _ in dataloader:
        features = features.to(device)
        features.requires_grad = True
        
        outputs = model(features)
        
        # Get gradients for predicted class
        pred_class = outputs.argmax(dim=1)
        outputs[range(len(pred_class)), pred_class].sum().backward()
        
        # Accumulate gradient magnitudes
        total_importance += features.grad.abs().sum(dim=0).cpu().numpy()
        n_samples += features.size(0)
        
        model.zero_grad()
    
    # Average and normalize
    importance = total_importance / n_samples
    importance = importance / importance.max()  # Scale to 0-1
    
    # Create sorted DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df, filepath, title, top_n=30):
    """
    Args:
        importance_df (DataFrame): Feature importance scores.
        filepath (str): Path to save the plot.
        title (str): Plot title.
        top_n (int): Number of top features to plot.
    Returns:
        None
    """
    plt.figure(figsize=(10, max(8, top_n * 0.3)))
    top_features = importance_df.head(top_n)
    
    # Simple color scheme
    colors = plt.cm.YlOrRd(top_features['importance'].values)
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['importance'].values, color=colors, edgecolor='black', linewidth=0.5)
    
    plt.yticks(y_pos, top_features['feature'].values, fontsize=9)
    plt.xlabel('Importance', fontsize=11, fontweight='bold')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================================

def run_5fold_cv(features_csv, label_csv, train_split_csv, val_test_split_csv,
                 batch_size=32, num_epochs=100, learning_rate=0.001,
                 hidden_sizes=[256, 128, 64], dropout=0.4, patience=15,
                 device='cuda', output_dir='resnet1d_results'):
    """
    Args:
        features_csv (str): Path to acoustic features CSV.
        label_csv (str): Path to paired_healthcode.csv.
        train_split_csv (str): Path to train split CSV.
        val_test_split_csv (str): Path to val/test split CSV.
        batch_size (int): Batch size for training.
        num_epochs (int): Maximum number of epochs.
        learning_rate (float): Initial learning rate.
        hidden_sizes (list): List of hidden layer sizes.
        dropout (float): Dropout probability.
        patience (int): Early stopping patience.
        device: Device to train on ('cuda' or 'cpu').
        output_dir (str): Directory to save results and plots.
    Returns:
        DataFrame: Results for all folds.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Plots'), exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ResNet1D Configuration - SINGLE MODEL")
    print(f"{'='*80}")
    print(f"  Dropout: {dropout}, Patience: {patience}")
    print(f"  Learning Rate: {learning_rate}, Batch Size: {batch_size}")
    print(f"  Hidden Sizes: {hidden_sizes}")
    
    # Load labels
    label_df = pd.read_csv(label_csv, sep=',')
    healthcode_to_label = dict(zip(label_df['healthCode'], label_df['label_PD']))
    
    # Load features
    features_df = pd.read_csv(features_csv, sep=';' if ';' in open(features_csv).read(100) else ',')
    if 'healthCode' not in features_df.columns and 'healthcode' in features_df.columns:
        features_df = features_df.rename(columns={'healthcode': 'healthCode'})
    elif 'healthCode' not in features_df.columns:
        raise ValueError("Feature CSV must contain a 'healthCode' or 'healthcode' column.")
    
    # Load splits
    train_splits = pd.read_csv(train_split_csv)
    val_test_splits = pd.read_csv(val_test_split_csv)
    
    # Get feature columns (exclude metadata)
    exclude_cols = ['filename', 'healthCode', 'record_id', 'label', 'label_PD', 'target']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"  Total features: {len(feature_columns)}")
    print(f"  Labels loaded: {len(healthcode_to_label)} healthCodes")
    
    results_summary = []
    
    for fold in range(5):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")
        
        # Get healthCodes for this fold
        train_hcs = set(train_splits[train_splits['fold_iteration'] == fold]['healthCode'])
        val_hcs = set(val_test_splits[
            (val_test_splits['fold_iteration'] == fold) & 
            (val_test_splits['subset'] == 'val')
        ]['healthCode'])
        test_hcs = set(val_test_splits[
            (val_test_splits['fold_iteration'] == fold) & 
            (val_test_splits['subset'] == 'test')
        ]['healthCode'])
        
        # Filter features for this fold and add labels
        train_data = features_df[features_df['healthCode'].isin(train_hcs)].copy()
        val_data = features_df[features_df['healthCode'].isin(val_hcs)].copy()
        test_data = features_df[features_df['healthCode'].isin(test_hcs)].copy()
        
        # Map labels (just like TCN)
        train_data['label'] = train_data['healthCode'].map(healthcode_to_label)
        val_data['label'] = val_data['healthCode'].map(healthcode_to_label)
        test_data['label'] = test_data['healthCode'].map(healthcode_to_label)
        
        # Remove samples without labels
        train_data = train_data.dropna(subset=['label'])
        val_data = val_data.dropna(subset=['label'])
        test_data = test_data.dropna(subset=['label'])

        # Undersample PD (label 1) to match Healthy (label 0) in train_data
        n_healthy = (train_data['label'] == 0).sum()
        n_pd = (train_data['label'] == 1).sum()
        if n_pd > n_healthy:
            pd_indices = train_data[train_data['label'] == 1].index
            undersampled_pd_indices = np.random.choice(pd_indices, n_healthy, replace=False)
            healthy_indices = train_data[train_data['label'] == 0].index
            balanced_indices = np.concatenate([healthy_indices, undersampled_pd_indices])
            train_data = train_data.loc[balanced_indices]
            train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            print(f"  Undersampled PD: {n_pd} -> {n_healthy}")
        else:
            print(f"  No undersampling needed (PD: {n_pd}, Healthy: {n_healthy})")

        print(f"Patients: Train={len(train_hcs)}, Val={len(val_hcs)}, Test={len(test_hcs)}")

        # Create datasets
        train_dataset = AcousticFeaturesDataset(train_data, feature_columns, label_column='label')
        val_dataset = AcousticFeaturesDataset(val_data, feature_columns, label_column='label')
        test_dataset = AcousticFeaturesDataset(test_data, feature_columns, label_column='label')

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
        
        # Create model
        model = SingleResNet1D(
            input_size=len(feature_columns),
            hidden_sizes=hidden_sizes,
            num_classes=2,
            dropout=dropout,
            use_attention=True
        )
        
        # Train
        trained_model, best_val_metrics, history = train_model(
            model, train_loader, val_loader, num_epochs, learning_rate,
            device, patience, fold, output_dir
        )
        
        # Save history
        pd.DataFrame(history).to_csv(f'{output_dir}/fold_{fold}_history.csv', index=False)
        
        # Test evaluation
        test_metrics = evaluate(trained_model, test_loader, device)
        test_detailed = get_detailed_metrics(
            test_metrics['patient_labels'],
            test_metrics['patient_preds'],
            test_metrics['patient_probs']
        )
        
        # Plot confusion matrix
        plots_dir = os.path.join(output_dir, 'Plots')
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            f'{plots_dir}/fold_{fold}_test_cm.png',
            f'Fold {fold} Test Set - ResNet1D'
        )
        
        # Compute feature importance
        print(f"\nComputing feature importance for Fold {fold}...")
        importance_df = compute_feature_importance(
            trained_model, test_loader, device, feature_columns
        )
        
        # Save results
        importance_df.to_csv(f'{output_dir}/fold_{fold}_feature_importance.csv', index=False)
        
        # Plot
        plot_feature_importance(
            importance_df,
            f'{plots_dir}/fold_{fold}_feature_importance.png',
            f'Feature Importance - ResNet1D (Fold {fold})',
            top_n=30
        )
        
        print(f"  Top 5: {', '.join(importance_df.head(5)['feature'].tolist())}")
        
        print(f"\nFold {fold} Test Results:")
        print(f"  AUC: {test_metrics['patient_auc']:.4f}")
        print(f"  Acc: {test_metrics['patient_accuracy']:.2f}%, F1: {test_detailed['f1']:.4f}")
        print(f"  Sens: {test_metrics['patient_sensitivity']:.2f}%, Spec: {test_metrics['patient_specificity']:.2f}%")
        
        results_summary.append({
            'fold': fold,
            'test_auc': test_metrics['patient_auc'],
            'test_acc': test_metrics['patient_accuracy'],
            'test_sens': test_metrics['patient_sensitivity'],
            'test_spec': test_metrics['patient_specificity'],
            'test_f1': test_detailed['f1'],
            'test_precision': test_detailed['precision'],
            'test_recall': test_detailed['recall']
        })
    
    # Final summary
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(f'{output_dir}/results_summary.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"5-FOLD CROSS-VALIDATION RESULTS - ResNet1D")
    print(f"{'='*80}")
    print(f"AUC:       {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
    print(f"Accuracy:  {results_df['test_acc'].mean():.2f}% ± {results_df['test_acc'].std():.2f}%")
    print(f"F1-Score:  {results_df['test_f1'].mean():.4f} ± {results_df['test_f1'].std():.4f}")
    print(f"Sens/Spec: {results_df['test_sens'].mean():.2f}% / {results_df['test_spec'].mean():.2f}%")
    print(f"{'='*80}\n")
    
    return results_df


if __name__ == "__main__":
    
    # Set up file paths and hyperparameters
    FEATURES_CSV = "/mloscratch/users/gnahas/data/features/acoustic_features_vf.csv"
    LABEL_CSV = "/tremor2tensor/src_GAMMA/paired_healthcode.csv"
    TRAIN_SPLIT_CSV = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
    VAL_TEST_SPLIT_CSV = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run 5-fold cross-validation with single ResNet1D
    results = run_5fold_cv(
        features_csv=FEATURES_CSV,
        label_csv=LABEL_CSV,
        train_split_csv=TRAIN_SPLIT_CSV,
        val_test_split_csv=VAL_TEST_SPLIT_CSV,
        batch_size=32,
        num_epochs=100,
        learning_rate=0.001,
        hidden_sizes=[256, 128, 64],
        dropout=0.4,
        patience=20,
        device=device,
        output_dir='/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/ResNet1D/V5'
    )
