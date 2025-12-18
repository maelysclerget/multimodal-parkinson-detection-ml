
"""
TCN Waveform Model for Audio Classification

This script implements a Temporal Convolutional Network (TCN) for classifying raw audio waveforms.
It uses windowing and stride to split long audio into overlapping segments, and supports 5-fold cross-validation.
The model is trained and evaluated at the patient level, with metrics like accuracy, sensitivity, specificity, F1, and AUC.
Dynamic learning rate scheduling and early stopping are included for robust training.

Main components:
- WaveformDataset: Loads audio windows for training/testing
- TCNWaveformClassifier: The neural network model
- Training and evaluation loops
- Patient-level aggregation and metrics
- 5-fold cross-validation pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

class WaveformDataset(Dataset):
    """
    Dataset for loading audio waveforms with windowing and stride.
    Each audio file is split into overlapping windows for more training samples.
    """
    def __init__(self, file_paths, labels, health_codes, window_size=220500, stride=110250):
        """
        Args:
            file_paths (list): List of paths to .npy waveform files.
            labels (list): List of labels (0 or 1).
            health_codes (list): List of health codes for patient-level grouping.
            window_size (int): Number of samples per window (default 220500 = 5s @ 44.1kHz).
            stride (int): Stride between windows (default 110250 = 2.5s).
        """
        self.window_size = window_size
        self.stride = stride
        
        self.windows = []
        for fp, label, hc in zip(file_paths, labels, health_codes):
            waveform = np.load(fp)
            # If waveform is shorter than window, use one window with padding
            if len(waveform) < window_size:
                self.windows.append((fp, 0, label, hc))
            else:
                # Otherwise, split into overlapping windows
                num_windows = (len(waveform) - window_size) // stride + 1
                for w_idx in range(num_windows):
                    self.windows.append((fp, w_idx, label, hc))
    
    def __len__(self):
        """
        Returns:
            int: Number of windows in the dataset.
        """
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the window/sample.
        Returns:
            tuple: (window tensor, label tensor, health code)
        """
        
        filepath, window_idx, label, healthcode = self.windows[idx]
        
        # Load waveform and extract window
        waveform = np.load(filepath)
        start = window_idx * self.stride
        end = start + self.window_size
        if end <= len(waveform):
            window = waveform[start:end]
        else:
            # Pad with zeros if window exceeds waveform length
            window = np.zeros(self.window_size, dtype=np.float32)
            available = len(waveform) - start
            if available > 0:
                window[:available] = waveform[start:]
        
        # Pad if still too short
        if len(window) < self.window_size:
            window = np.pad(window, (0, self.window_size - len(window)))
        
        window = torch.FloatTensor(window).unsqueeze(0)  # Add channel dim
        label_tensor = torch.LongTensor([label])
        
        return window, label_tensor, healthcode


class TemporalBlock(nn.Module):
    """
    A single TCN block with dilated convolutions and residual connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        """
        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Convolution kernel size.
            stride (int): Convolution stride.
            dilation (int): Dilation factor.
            padding (int): Padding size.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, channels, length).
        Returns:
            Tensor: Output tensor of same shape.
        """
        out = self.net(x)
        
        if out.size(2) != x.size(2):
            out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNWaveformClassifier(nn.Module):
    """
    Temporal Convolutional Network for classifying raw audio waveforms.
    """
    def __init__(self, num_inputs=1, num_channels=[32, 64, 128, 256], 
                 kernel_size=7, dropout=0.3, num_classes=2):
        """
        Args:
            num_inputs (int): Number of input channels.
            num_channels (list): List of channel sizes for each TCN block.
            kernel_size (int): Convolution kernel size.
            dropout (float): Dropout probability.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 1, 
                                       dilation_size, padding, dropout))
        
        self.network = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, 1, length).
        Returns:
            Tensor: Output logits of shape (batch, num_classes).
        """
        
        y = self.network(x)
        y = self.global_avg_pool(y).view(y.size(0), -1)
        
        return self.fc(y)


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
    
    for waveforms, labels, _ in dataloader:
        waveforms, labels = waveforms.to(device), labels.squeeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(waveforms)
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
        for waveforms, labels, health_codes in dataloader:
            waveforms, labels = waveforms.to(device), labels.squeeze(1).to(device)
            outputs = model(waveforms)
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
    
    # Patient-level aggregation: compute both methods
    unique_hcs = np.unique(all_health_codes)
    patient_labels = []
    patient_preds_majority = []  
    patient_preds_prob = []      
    patient_probs = []
    
    for hc in unique_hcs:
        mask = all_health_codes == hc
        patient_labels.append(all_labels[mask][0])
        
        # Method 1: Majority vote
        vote_counts = Counter(all_preds[mask])
        patient_preds_majority.append(vote_counts.most_common(1)[0][0])
        
        # Method 2: Average probability with 0.5 threshold
        avg_prob = np.mean(all_probs[mask])
        patient_probs.append(avg_prob)
        patient_preds_prob.append(1 if avg_prob >= 0.5 else 0)
    
    patient_labels = np.array(patient_labels)
    patient_preds_majority = np.array(patient_preds_majority)
    patient_preds_prob = np.array(patient_preds_prob)
    patient_probs = np.array(patient_probs)
    
    # Metrics using majority vote
    acc_majority = 100 * np.mean(patient_preds_majority == patient_labels)
    cm_majority = confusion_matrix(patient_labels, patient_preds_majority)
    tn, fp, fn, tp = cm_majority.ravel()
    sens_majority = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_majority = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Metrics using probability threshold
    acc_prob = 100 * np.mean(patient_preds_prob == patient_labels)
    cm_prob = confusion_matrix(patient_labels, patient_preds_prob)
    tn, fp, fn, tp = cm_prob.ravel()
    sens_prob = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_prob = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        auc = roc_auc_score(patient_labels, patient_probs)
    except:
        auc = 0.0
    
    return {
        # Majority vote metrics
        'patient_accuracy': acc_majority,
        'patient_sensitivity': sens_majority,
        'patient_specificity': spec_majority,
        'confusion_matrix': cm_majority,
        'patient_preds': patient_preds_majority,
        
        # Probability threshold metrics
        'patient_accuracy_prob': acc_prob,
        'patient_sensitivity_prob': sens_prob,
        'patient_specificity_prob': spec_prob,
        'confusion_matrix_prob': cm_prob,
        'patient_preds_prob': patient_preds_prob,
        
        # Common
        'patient_auc': auc,
        'patient_labels': patient_labels,
        'patient_probs': patient_probs
    }


def plot_fold_summary(train_cm, val_cm_maj, val_cm_prob, test_cm_maj, test_cm_prob,
                      test_metrics_maj, test_metrics_prob, filepath, fold_num):
    """
    Args:
        train_cm (array): Training set confusion matrix.
        val_cm_maj (array): Validation set confusion matrix (majority vote).
        val_cm_prob (array): Validation set confusion matrix (probability threshold).
        test_cm_maj (array): Test set confusion matrix (majority vote).
        test_cm_prob (array): Test set confusion matrix (probability threshold).
        test_metrics_maj (dict): Test metrics (majority vote).
        test_metrics_prob (dict): Test metrics (probability threshold).
        filepath (str): Path to save the plot.
        fold_num (int): Fold number.
    Returns:
        None
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Training
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    ax1.set_title(f'Training Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Row 2: Validation - Majority Vote
    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(val_cm_maj, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    ax2.set_title(f'Validation (Majority Vote)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # Row 2: Validation - Prob Threshold
    ax3 = fig.add_subplot(gs[1, 1])
    sns.heatmap(val_cm_prob, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax3,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    ax3.set_title(f'Validation (Prob Threshold)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # Row 3: Test - Majority Vote
    ax4 = fig.add_subplot(gs[2, 0])
    sns.heatmap(test_cm_maj, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax4,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    ax4.set_title(f'Test (Majority Vote)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    
    # Row 3: Test - Prob Threshold
    ax5 = fig.add_subplot(gs[2, 1])
    sns.heatmap(test_cm_prob, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax5,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    ax5.set_title(f'Test (Prob Threshold)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('True Label')
    ax5.set_xlabel('Predicted Label')
    
    # Metrics table for Majority Vote
    ax6 = fig.add_subplot(gs[1:, 2])
    ax6.axis('off')
    metrics_text = f"""
    FOLD {fold_num} - TEST SET METRICS
    {'='*40}
    
    MAJORITY VOTE AGGREGATION:
      AUC:         {test_metrics_maj['auc']:.4f}
      Accuracy:    {test_metrics_maj['accuracy']:.2f}%
      F1-Score:    {test_metrics_maj['f1']:.4f}
      Precision:   {test_metrics_maj['precision']:.4f}
      Recall:      {test_metrics_maj['recall']:.4f}
      Sensitivity: {test_metrics_maj['sensitivity']:.2f}%
      Specificity: {test_metrics_maj['specificity']:.2f}%
      
      TN: {test_cm_maj[0,0]:>4}  FP: {test_cm_maj[0,1]:>4}
      FN: {test_cm_maj[1,0]:>4}  TP: {test_cm_maj[1,1]:>4}
    
    PROBABILITY THRESHOLD (0.5):
      AUC:         {test_metrics_prob['auc']:.4f}
      Accuracy:    {test_metrics_prob['accuracy']:.2f}%
      F1-Score:    {test_metrics_prob['f1']:.4f}
      Precision:   {test_metrics_prob['precision']:.4f}
      Recall:      {test_metrics_prob['recall']:.4f}
      Sensitivity: {test_metrics_prob['sensitivity']:.2f}%
      Specificity: {test_metrics_prob['specificity']:.2f}%
      
      TN: {test_cm_prob[0,0]:>4}  FP: {test_cm_prob[0,1]:>4}
      FN: {test_cm_prob[1,0]:>4}  TP: {test_cm_prob[1,1]:>4}
    """
    ax6.text(0.05, 0.95, metrics_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)
    
    fig.suptitle(f'Fold {fold_num} - Complete Summary', fontsize=16, fontweight='bold')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


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


def get_detailed_metrics(labels, preds, probs):
    """
    Args:
        labels (array): True labels.
        preds (array): Predicted labels.
        probs (array): Predicted probabilities for positive class.
    Returns:
        dict: F1, precision, recall, and AUC.
    """
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'auc': auc}


def plot_test_confusion_matrix(cm, metrics, filepath, title):
    """
    Args:
        cm (array): Confusion matrix.
        metrics (dict): Metrics to display.
        filepath (str): Path to save the plot.
        title (str): Plot title.
    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Metrics table
    ax2.axis('off')
    metrics_text = f"""
    Test Set Metrics (Patient-Level)
    {'='*35}
    
    AUC:         {metrics['auc']:.4f}
    Accuracy:    {metrics['accuracy']:.2f}%
    F1-Score:    {metrics['f1']:.4f}
    Precision:   {metrics['precision']:.4f}
    Recall:      {metrics['recall']:.4f}
    Sensitivity: {metrics['sensitivity']:.2f}%
    Specificity: {metrics['specificity']:.2f}%
    
    Confusion Matrix:
      TN: {cm[0,0]:>4}  FP: {cm[0,1]:>4}
      FN: {cm[1,0]:>4}  TP: {cm[1,1]:>4}
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', 
             verticalalignment='center')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


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
    
    # Create PTH directory for saving model checkpoints
    pth_dir = os.path.join(output_dir, 'PTH')
    os.makedirs(pth_dir, exist_ok=True)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Class weighting (77% PD, 23% Control)
    class_weights = torch.FloatTensor([3.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Dynamic LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_val_auc = 0.0
    best_metrics = None
    best_model_state = None
    epochs_without_improvement = 0
    current_checkpoint_path = None  # Track current checkpoint to delete old ones
    
    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_patient_auc': [], 'learning_rate': []
    }
    
    print("\\nStarting training...")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step(val_metrics['patient_auc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_patient_auc'].append(val_metrics['patient_auc'])
        history['learning_rate'].append(current_lr)
        
        if epoch % 10 == 0 or val_metrics['patient_auc'] > best_val_auc:
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val AUC: {val_metrics['patient_auc']:.4f}")
            print(f"    Majority Vote - Acc: {val_metrics['patient_accuracy']:.2f}%, Sens: {val_metrics['patient_sensitivity']:.2f}%, Spec: {val_metrics['patient_specificity']:.2f}%")
            print(f"    Prob Thresh   - Acc: {val_metrics['patient_accuracy_prob']:.2f}%, Sens: {val_metrics['patient_sensitivity_prob']:.2f}%, Spec: {val_metrics['patient_specificity_prob']:.2f}%")
        
        if val_metrics['patient_auc'] > best_val_auc:
            best_val_auc = val_metrics['patient_auc']
            best_metrics = val_metrics.copy()
            if isinstance(model, nn.DataParallel):
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()
            epochs_without_improvement = 0
            
            # Delete old checkpoint if exists
            if current_checkpoint_path and os.path.exists(current_checkpoint_path):
                os.remove(current_checkpoint_path)
            
            # Save new checkpoint to PTH folder
            checkpoint_path = os.path.join(pth_dir, f'fold_{fold_num}_best_auc_{best_val_auc:.4f}_epoch_{epoch+1}.pth')
            torch.save({
                'fold': fold_num,
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'val_auc': best_val_auc,
                'val_metrics': val_metrics
            }, checkpoint_path)
            current_checkpoint_path = checkpoint_path  # Update tracker
            
            print(f"  ✓ Best model saved (AUC: {best_val_auc:.4f}) -> {os.path.basename(checkpoint_path)}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"\\nEarly stopping after {epoch+1} epochs")
            break
    
    # Load best model
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)
    
    return model, best_metrics, history


def run_5fold_cv(waveform_dir, label_csv, train_split_csv, val_test_split_csv,
                 batch_size=32, num_epochs=100, learning_rate=0.001,
                 num_channels=[32, 64, 128, 256], kernel_size=7, dropout=0.3,
                 patience=10, device='cuda', output_dir='tcn_results',
                 window_size=220500, stride=110250):
    """
    Args:
        waveform_dir (str): Directory with waveform .npy files.
        label_csv (str): Path to CSV file with healthCode and labels.
        train_split_csv (str): Path to train split CSV.
        val_test_split_csv (str): Path to val/test split CSV.
        batch_size (int): Batch size for training.
        num_epochs (int): Maximum number of epochs.
        learning_rate (float): Initial learning rate.
        num_channels (list): List of channel sizes for TCN blocks.
        kernel_size (int): Convolution kernel size.
        dropout (float): Dropout probability.
        patience (int): Early stopping patience.
        device: Device to train on ('cuda' or 'cpu').
        output_dir (str): Directory to save results and plots.
        window_size (int): Number of samples per window.
        stride (int): Stride between windows.
    Returns:
        DataFrame: Results for all folds.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Plots'), exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Window: {window_size/44100:.1f}s, Stride: {stride/44100:.1f}s")
    print(f"  Dropout: {dropout}, Kernel: {kernel_size}")
    
    # Load data
    label_df = pd.read_csv(label_csv, sep=',')
    healthcode_to_label = dict(zip(label_df['healthCode'], label_df['label_PD']))
    train_splits = pd.read_csv(train_split_csv)
    val_test_splits = pd.read_csv(val_test_split_csv)
    
    all_files = [f for f in os.listdir(waveform_dir) if f.endswith('.npy')]
    file_info = []
    for fname in all_files:
        parts = fname.replace('.npy', '').split('_')
        if len(parts) >= 2:
            hc = parts[0]
            if hc in healthcode_to_label:
                file_info.append({
                    'filepath': os.path.join(waveform_dir, fname),
                    'healthCode': hc,
                    'label': healthcode_to_label[hc]
                })
    
    file_df = pd.DataFrame(file_info)
    print(f"\\nLoaded {len(file_df)} recordings")
    
    results_summary = []
    
    for fold in range(5):
        print(f"\\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")
        
        train_hcs = set(train_splits[train_splits['fold_iteration'] == fold]['healthCode'])
        val_hcs = set(val_test_splits[
            (val_test_splits['fold_iteration'] == fold) & 
            (val_test_splits['subset'] == 'val')
        ]['healthCode'])
        test_hcs = set(val_test_splits[
            (val_test_splits['fold_iteration'] == fold) & 
            (val_test_splits['subset'] == 'test')
        ]['healthCode'])
        
        train_data = file_df[file_df['healthCode'].isin(train_hcs)]
        val_data = file_df[file_df['healthCode'].isin(val_hcs)]
        test_data = file_df[file_df['healthCode'].isin(test_hcs)]
        
        print(f"Patients: Train={len(train_hcs)}, Val={len(val_hcs)}, Test={len(test_hcs)}")
        
        # Create datasets with windowing
        train_dataset = WaveformDataset(train_data['filepath'].tolist(),
                                        train_data['label'].tolist(),
                                        train_data['healthCode'].tolist(),
                                        window_size, stride)
        val_dataset = WaveformDataset(val_data['filepath'].tolist(),
                                      val_data['label'].tolist(),
                                      val_data['healthCode'].tolist(),
                                      window_size, stride)
        test_dataset = WaveformDataset(test_data['filepath'].tolist(),
                                       test_data['label'].tolist(),
                                       test_data['healthCode'].tolist(),
                                       window_size, stride)
        
        print(f"Windows: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
        
        model = TCNWaveformClassifier(1, num_channels, kernel_size, dropout, 2)
        
        trained_model, best_val_metrics, history = train_model(
            model, train_loader, val_loader, num_epochs, learning_rate,
            device, patience, fold, output_dir
        )
        
        # Save history
        pd.DataFrame(history).to_csv(f'{output_dir}/fold_{fold}_history.csv', index=False)
        
        # Test evaluation
        test_metrics = evaluate(trained_model, test_loader, device)
        
        # Compute detailed metrics for BOTH aggregation methods
        test_detailed_majority = get_detailed_metrics(
            test_metrics['patient_labels'],
            test_metrics['patient_preds'],
            test_metrics['patient_probs']
        )
        test_detailed_prob = get_detailed_metrics(
            test_metrics['patient_labels'],
            test_metrics['patient_preds_prob'],
            test_metrics['patient_probs']
        )
        
        # Evaluate on train and val for comprehensive plot
        train_metrics = evaluate(trained_model, train_loader, device)
        val_metrics = evaluate(trained_model, val_loader, device)
        
        # Create single comprehensive plot for this fold
        plots_dir = os.path.join(output_dir, 'Plots')
        all_test_metrics_majority = {
            'auc': test_metrics['patient_auc'],
            'accuracy': test_metrics['patient_accuracy'],
            'f1': test_detailed_majority['f1'],
            'precision': test_detailed_majority['precision'],
            'recall': test_detailed_majority['recall'],
            'sensitivity': test_metrics['patient_sensitivity'],
            'specificity': test_metrics['patient_specificity']
        }
        all_test_metrics_prob = {
            'auc': test_metrics['patient_auc'],
            'accuracy': test_metrics['patient_accuracy_prob'],
            'f1': test_detailed_prob['f1'],
            'precision': test_detailed_prob['precision'],
            'recall': test_detailed_prob['recall'],
            'sensitivity': test_metrics['patient_sensitivity_prob'],
            'specificity': test_metrics['patient_specificity_prob']
        }
        
        plot_fold_summary(
            train_metrics['confusion_matrix'],
            val_metrics['confusion_matrix'],
            val_metrics['confusion_matrix_prob'],
            test_metrics['confusion_matrix'],
            test_metrics['confusion_matrix_prob'],
            all_test_metrics_majority,
            all_test_metrics_prob,
            f'{plots_dir}/fold_{fold}_complete_summary.png',
            fold
        )
        
        print(f"\nFold {fold} Test Results:")
        print(f"  AUC: {test_metrics['patient_auc']:.4f}")
        print(f"  MAJORITY VOTE:")
        print(f"    Acc: {test_metrics['patient_accuracy']:.2f}%, F1: {test_detailed_majority['f1']:.4f}, Precision: {test_detailed_majority['precision']:.4f}, Recall: {test_detailed_majority['recall']:.4f}")
        print(f"  PROB THRESHOLD (0.5):")
        print(f"    Acc: {test_metrics['patient_accuracy_prob']:.2f}%, F1: {test_detailed_prob['f1']:.4f}, Precision: {test_detailed_prob['precision']:.4f}, Recall: {test_detailed_prob['recall']:.4f}")
        
        results_summary.append({
            'fold': fold,
            'test_auc': test_metrics['patient_auc'],
            # Majority vote metrics
            'test_acc_majority': test_metrics['patient_accuracy'],
            'test_sens_majority': test_metrics['patient_sensitivity'],
            'test_spec_majority': test_metrics['patient_specificity'],
            'test_f1_majority': test_detailed_majority['f1'],
            'test_precision_majority': test_detailed_majority['precision'],
            'test_recall_majority': test_detailed_majority['recall'],
            # Probability threshold metrics
            'test_acc_prob': test_metrics['patient_accuracy_prob'],
            'test_sens_prob': test_metrics['patient_sensitivity_prob'],
            'test_spec_prob': test_metrics['patient_specificity_prob'],
            'test_f1_prob': test_detailed_prob['f1'],
            'test_precision_prob': test_detailed_prob['precision'],
            'test_recall_prob': test_detailed_prob['recall']
        })
    
    # Final summary
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(f'{output_dir}/results_summary.csv', index=False)
    
    print(f"\n{'='*80}")
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
    print(f"\nMAJORITY VOTE AGGREGATION:")
    print(f"  Accuracy:  {results_df['test_acc_majority'].mean():.2f}% ± {results_df['test_acc_majority'].std():.2f}%")
    print(f"  F1-Score:  {results_df['test_f1_majority'].mean():.4f} ± {results_df['test_f1_majority'].std():.4f}")
    print(f"  Precision: {results_df['test_precision_majority'].mean():.4f} ± {results_df['test_precision_majority'].std():.4f}")
    print(f"  Recall:    {results_df['test_recall_majority'].mean():.4f} ± {results_df['test_recall_majority'].std():.4f}")
    print(f"  Sens/Spec: {results_df['test_sens_majority'].mean():.2f}% / {results_df['test_spec_majority'].mean():.2f}%")
    print(f"\nPROBABILITY THRESHOLD (0.5) AGGREGATION:")
    print(f"  Accuracy:  {results_df['test_acc_prob'].mean():.2f}% ± {results_df['test_acc_prob'].std():.2f}%")
    print(f"  F1-Score:  {results_df['test_f1_prob'].mean():.4f} ± {results_df['test_f1_prob'].std():.4f}")
    print(f"  Precision: {results_df['test_precision_prob'].mean():.4f} ± {results_df['test_precision_prob'].std():.4f}")
    print(f"  Recall:    {results_df['test_recall_prob'].mean():.4f} ± {results_df['test_recall_prob'].std():.4f}")
    print(f"  Sens/Spec: {results_df['test_sens_prob'].mean():.2f}% / {results_df['test_spec_prob'].mean():.2f}%")
    print(f"{'='*80}")
    
    return results_df


if __name__ == "__main__":
    # Set up file paths and hyperparameters
    WAVEFORM_DIR = "/mloscratch/users/gnahas/data/waveform_norm_silence_trimmed"
    LABEL_CSV = "/tremor2tensor/src_GAMMA/paired_healthcode.csv"
    TRAIN_SPLIT_CSV = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
    VAL_TEST_SPLIT_CSV = "/tremor2tensor/src_GAMMA/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"

    # Model and training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    NUM_CHANNELS = [32, 64, 128, 256]
    KERNEL_SIZE = 7
    DROPOUT = 0.3
    PATIENCE = 10
    WINDOW_SIZE = 220500
    STRIDE = 110250

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")

    # Run 5-fold cross-validation
    results = run_5fold_cv(
        waveform_dir=WAVEFORM_DIR,
        label_csv=LABEL_CSV,
        train_split_csv=TRAIN_SPLIT_CSV,
        val_test_split_csv=VAL_TEST_SPLIT_CSV,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        patience=PATIENCE,
        device=device,
        output_dir='/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/TCN/V1_Windowed',
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )
