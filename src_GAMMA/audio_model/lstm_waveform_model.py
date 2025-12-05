"""
LSTM model for audio waveform classification
5-fold CV with recording and patient-level aggregation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class WaveformDataset(Dataset):
    """
    Dataset for loading audio waveforms from .npy files
    """
    def __init__(self, file_paths, labels, health_codes, max_length=441000):
        """
        Args:
            file_paths: List of paths to .npy waveform files
            labels: List of labels (0 or 1 for binary classification)
            health_codes: List of health codes for patient-level grouping
            max_length: Maximum waveform length (default: 10s at 44100 Hz)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.health_codes = health_codes
        self.max_length = max_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load waveform
        waveform = np.load(self.file_paths[idx])
        
        # Pad or truncate to max_length
        if len(waveform) < self.max_length:
            # Pad with zeros
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            # Truncate
            waveform = waveform[:self.max_length]
        
        # Convert to tensor and add channel dimension
        waveform = torch.FloatTensor(waveform).unsqueeze(0)  # Shape: (1, max_length)
        label = torch.LongTensor([self.labels[idx]])
        
        return waveform, label, self.health_codes[idx]


class LSTMWaveformClassifier(nn.Module):
    """
    LSTM-based classifier for raw audio waveforms
    Uses CNN downsampling before LSTM to handle long sequences
    """
    def __init__(
        self,
        input_size=1,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        bidirectional=True
    ):
        """
        Args:
            input_size: Number of input features per timestep (1 for raw audio)
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMWaveformClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # CNN downsampling layers to reduce sequence length
        # Input: (batch, 1, 445000) -> Output: (batch, 64, ~1390)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=80, stride=40, padding=40)  # Downsample by 40
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Downsample by 4
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)  # Downsample by 2
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample by 2
        # Total downsampling: 40 * 4 * 2 * 2 = 640x
        # 445000 / 640 ≈ 695 timesteps (manageable for LSTM)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,  # Now taking CNN features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, sequence_length)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # CNN downsampling: (batch, 1, seq_len) -> (batch, 64, reduced_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        x = x.transpose(1, 2).contiguous()
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        # For bidirectional: concatenate forward and backward hidden states
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch_size, hidden_size)
            # Take last layer forward and backward
            forward_hidden = hidden[-2, :, :]
            backward_hidden = hidden[-1, :, :]
            hidden_concat = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # hidden shape: (num_layers, batch_size, hidden_size)
            hidden_concat = hidden[-1, :, :]
        
        # Apply dropout
        out = self.dropout(hidden_concat)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for waveforms, labels, _ in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.squeeze(1).to(device)  # Keep as 1D tensor
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(waveforms)
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


def evaluate(model, dataloader, device, aggregation_method='average'):
    """
    Evaluate model at recording and patient level
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        aggregation_method: Method for patient-level aggregation
            - 'majority_vote': Most common prediction (hard voting)
            - 'average': Average probabilities (soft voting, uniform weights)
            - 'weighted_average': Confidence-weighted average (high confidence predictions have more influence)
    """
    model.eval()
    
    # Collect recording-level predictions
    all_probs = []
    all_preds = []
    all_labels = []
    all_health_codes = []
    
    with torch.no_grad():
        for waveforms, labels, health_codes in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.squeeze(1).to(device)  # Keep as 1D tensor
            
            # Forward pass
            outputs = model(waveforms)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            probs_pd = probs[:, 1]  # Probability of PD class
            
            all_probs.extend(probs_pd.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_health_codes.extend(health_codes)
    
    # Convert to numpy
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_health_codes = np.array(all_health_codes)
    
    # Recording-level metrics
    rec_accuracy = 100 * np.mean(all_preds == all_labels)
    rec_tn, rec_fp, rec_fn, rec_tp = confusion_matrix(all_labels, all_preds).ravel()
    rec_sens = 100 * rec_tp / (rec_tp + rec_fn) if (rec_tp + rec_fn) > 0 else 0
    rec_spec = 100 * rec_tn / (rec_tn + rec_fp) if (rec_tn + rec_fp) > 0 else 0
    
    try:
        rec_auc = roc_auc_score(all_labels, all_probs)
    except:
        rec_auc = 0.0
    
    # Patient-level aggregation
    unique_health_codes = np.unique(all_health_codes)
    patient_labels = []
    patient_preds = []
    patient_probs = []
    
    for hc in unique_health_codes:
        mask = all_health_codes == hc
        patient_label = all_labels[mask][0]  # All recordings from same patient have same label
        rec_probs = all_probs[mask]
        
        if aggregation_method == 'majority_vote':
            # Majority vote: most common prediction
            rec_preds = all_preds[mask]
            vote_counts = Counter(rec_preds)
            patient_pred = vote_counts.most_common(1)[0][0]
            patient_prob = np.mean(rec_probs)  # For AUC
            
        elif aggregation_method == 'average':
            # Average probability (uniform weights)
            patient_prob = np.mean(rec_probs)
            patient_pred = 1 if patient_prob >= 0.5 else 0
            
        elif aggregation_method == 'weighted_average':
            # Weighted average: higher confidence predictions get more weight
            # Weight = |prob - 0.5| (distance from uncertain threshold)
            # Predictions closer to 0 or 1 have higher influence
            confidence_weights = np.abs(rec_probs - 0.5)
            
            # Avoid division by zero
            if confidence_weights.sum() > 0:
                patient_prob = np.average(rec_probs, weights=confidence_weights)
            else:
                patient_prob = np.mean(rec_probs)
            
            patient_pred = 1 if patient_prob >= 0.5 else 0
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        patient_labels.append(patient_label)
        patient_preds.append(patient_pred)
        patient_probs.append(patient_prob)
    
    patient_labels = np.array(patient_labels)
    patient_preds = np.array(patient_preds)
    patient_probs = np.array(patient_probs)
    
    # Patient-level metrics
    patient_accuracy = 100 * np.mean(patient_preds == patient_labels)
    pat_tn, pat_fp, pat_fn, pat_tp = confusion_matrix(patient_labels, patient_preds).ravel()
    pat_sens = 100 * pat_tp / (pat_tp + pat_fn) if (pat_tp + pat_fn) > 0 else 0
    pat_spec = 100 * pat_tn / (pat_tn + pat_fp) if (pat_tn + pat_fp) > 0 else 0
    
    try:
        pat_auc = roc_auc_score(patient_labels, patient_probs)
    except:
        pat_auc = 0.0
    
    return {
        # Recording-level
        'rec_accuracy': rec_accuracy,
        'rec_auc': rec_auc,
        'rec_sensitivity': rec_sens,
        'rec_specificity': rec_spec,
        'rec_tn': rec_tn,
        'rec_fp': rec_fp,
        'rec_fn': rec_fn,
        'rec_tp': rec_tp,
        # Patient-level
        'patient_accuracy': patient_accuracy,
        'patient_auc': pat_auc,
        'patient_sensitivity': pat_sens,
        'patient_specificity': pat_spec,
        'patient_tn': pat_tn,
        'patient_fp': pat_fp,
        'patient_fn': pat_fn,
        'patient_tp': pat_tp
    }


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda',
    patience=15,
    aggregation_method='average'
):
    """
    Training loop with dynamic learning rate, early stopping based on validation patient-level AUC
    Multi-GPU support with DataParallel
    
    Args:
        aggregation_method: Method for patient-level aggregation ('majority_vote', 'average', 'weighted_average')
    """
    # Multi-GPU setup
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Class weighting for imbalanced data
    # Data: 77% PD (label=1), 23% Control (label=0)
    # Weight minority class (Control) higher: [Control_weight, PD_weight] = [3.25, 1.0]
    class_weights = torch.FloatTensor([3.25, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler: reduce LR when validation AUC plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_val_auc = 0.0
    best_metrics = None
    best_model_state = None
    epochs_without_improvement = 0
    
    # Track training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_rec_auc': [],
        'val_patient_auc': [],
        'learning_rate': []
    }
    
    print(f"\nStarting training (aggregation: {aggregation_method})...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, aggregation_method=aggregation_method)
        
        # Update learning rate based on validation patient AUC
        scheduler.step(val_metrics['patient_auc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_rec_auc'].append(val_metrics['rec_auc'])
        history['val_patient_auc'].append(val_metrics['patient_auc'])
        history['learning_rate'].append(current_lr)
        
        # Print progress every 5 epochs or when best model found
        if epoch % 5 == 0 or val_metrics['patient_auc'] > best_val_auc:
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Recording - AUC: {val_metrics['rec_auc']:.4f}, Acc: {val_metrics['rec_accuracy']:.2f}%, Sens: {val_metrics['rec_sensitivity']:.2f}%, Spec: {val_metrics['rec_specificity']:.2f}%")
            print(f"  Val Patient   - AUC: {val_metrics['patient_auc']:.4f}, Acc: {val_metrics['patient_accuracy']:.2f}%, Sens: {val_metrics['patient_sensitivity']:.2f}%, Spec: {val_metrics['patient_specificity']:.2f}%")
        
        # Save best model based on patient-level validation AUC
        if val_metrics['patient_auc'] > best_val_auc:
            best_val_auc = val_metrics['patient_auc']
            best_metrics = val_metrics.copy()
            if isinstance(model, nn.DataParallel):
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()
            epochs_without_improvement = 0
            print(f"  ✓ Best model saved (Patient AUC: {best_val_auc:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
    
    # Load best model (handle DataParallel)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)
    
    return model, best_metrics, history


def run_5fold_cv(
    waveform_dir,
    label_csv,
    train_split_csv,
    val_test_split_csv,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    patience=15,
    device='cuda',
    output_dir='lstm_results',
    max_length=441000,
    aggregation_method='average'
):
    """
    Run 5-fold cross-validation
    
    Args:
        waveform_dir: Directory containing .npy waveform files
        label_csv: Path to CSV file with healthCode and labels
        train_split_csv: Path to train split CSV
        val_test_split_csv: Path to val/test split CSV
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        patience: Early stopping patience
        device: Device to train on ('cuda' or 'cpu')
        output_dir: Directory to save results and plots
        max_length: Maximum waveform length (samples)
        aggregation_method: Patient-level aggregation ('majority_vote', 'average', 'weighted_average')
    
    Returns:
        DataFrame with results for all folds
    """
    # Create output directory and plots subdirectory
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'Plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    print(f"Plots will be saved to: {plots_dir}")
    
    # Load labels
    print("Loading labels...")
    label_df = pd.read_csv(label_csv, sep=';')
    healthcode_to_label = dict(zip(label_df['healthCode'], label_df['label_PD']))
    print(f"  Total healthCodes with labels: {len(healthcode_to_label)}")
    
    # Load fold splits
    print("\nLoading fold splits...")
    train_splits = pd.read_csv(train_split_csv)
    val_test_splits = pd.read_csv(val_test_split_csv)
    
    # Get all waveform files
    print("\nFinding waveform files...")
    all_files = [f for f in os.listdir(waveform_dir) if f.endswith('.npy')]
    print(f"  Found {len(all_files)} .npy files")
    
    # Parse filenames to extract healthCode
    # Format: healthCode_recordingId_audio_audio_m4a.npy
    file_info = []
    for fname in all_files:
        parts = fname.replace('.npy', '').split('_')
        if len(parts) >= 2:
            healthcode = parts[0]
            if healthcode in healthcode_to_label:
                file_info.append({
                    'filepath': os.path.join(waveform_dir, fname),
                    'healthCode': healthcode,
                    'label': healthcode_to_label[healthcode]
                })
    
    file_df = pd.DataFrame(file_info)
    print(f"  Matched {len(file_df)} files with labels")
    print(f"    Control (label=0): {(file_df['label'] == 0).sum()} recordings")
    print(f"    PD (label=1): {(file_df['label'] == 1).sum()} recordings")
    
    # 5-fold cross-validation
    results_summary = []
    
    for fold in range(5):
        print("\n" + "="*80)
        print(f"FOLD {fold}")
        print("="*80)
        
        # Get train, val, and test healthCodes for this fold
        train_hcs = set(train_splits[train_splits['fold_iteration'] == fold]['healthCode'])
        val_hcs = set(val_test_splits[
            (val_test_splits['fold_iteration'] == fold) & 
            (val_test_splits['subset'] == 'val')
        ]['healthCode'])
        test_hcs = set(val_test_splits[
            (val_test_splits['fold_iteration'] == fold) & 
            (val_test_splits['subset'] == 'test')
        ]['healthCode'])
        
        print(f"\nTrain: {len(train_hcs)} patients")
        print(f"Val:   {len(val_hcs)} patients")
        print(f"Test:  {len(test_hcs)} patients")
        
        # Split data by healthCode
        train_data = file_df[file_df['healthCode'].isin(train_hcs)]
        val_data = file_df[file_df['healthCode'].isin(val_hcs)]
        test_data = file_df[file_df['healthCode'].isin(test_hcs)]
        
        print(f"\nTrain: {len(train_data)} recordings")
        print(f"Val:   {len(val_data)} recordings")
        print(f"Test:  {len(test_data)} recordings")
        
        # Create datasets
        train_dataset = WaveformDataset(
            train_data['filepath'].tolist(),
            train_data['label'].tolist(),
            train_data['healthCode'].tolist(),
            max_length=max_length
        )
        val_dataset = WaveformDataset(
            val_data['filepath'].tolist(),
            val_data['label'].tolist(),
            val_data['healthCode'].tolist(),
            max_length=max_length
        )
        test_dataset = WaveformDataset(
            test_data['filepath'].tolist(),
            test_data['label'].tolist(),
            test_data['healthCode'].tolist(),
            max_length=max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Initialize model
        model = LSTMWaveformClassifier(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            bidirectional=True
        )
        
        # Train
        trained_model, best_val_metrics, training_history = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            patience=patience,
            aggregation_method=aggregation_method
        )
        
        # Save training history
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(f'{output_dir}/fold_{fold}_training_history.csv', index=False)
        
        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Fold {fold} Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(history_df['epoch'], history_df['train_acc'], 'g-', label='Train Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Validation AUC (Recording)
        axes[1, 0].plot(history_df['epoch'], history_df['val_rec_auc'], 'r-', label='Val Recording AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Validation Recording AUC')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Validation AUC (Patient) - used for model selection
        axes[1, 1].plot(history_df['epoch'], history_df['val_patient_auc'], 'purple', label='Val Patient AUC')
        best_epoch = history_df.loc[history_df['val_patient_auc'].idxmax(), 'epoch']
        axes[1, 1].axvline(x=best_epoch, color='orange', linestyle='--', label=f'Best Epoch: {int(best_epoch)}')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Validation Patient AUC (Model Selection Metric)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/fold_{fold}_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Evaluate best model on TEST set
        test_metrics = evaluate(trained_model, test_loader, device, aggregation_method=aggregation_method)
        
        # Print summary for this fold
        print(f"\nFold {fold} - Best Model Performance:")
        print(f"  Val Patient AUC: {best_val_metrics['patient_auc']:.4f} | Test Patient AUC: {test_metrics['patient_auc']:.4f}")
        print(f"  Test Patient - Acc: {test_metrics['patient_accuracy']:.2f}%, Sens: {test_metrics['patient_sensitivity']:.2f}%, Spec: {test_metrics['patient_specificity']:.2f}%")
        
        results_summary.append({
            'fold': fold,
            'num_train_patients': len(train_hcs),
            'num_val_patients': len(val_hcs),
            'val_rec_auc': best_val_metrics['rec_auc'],
            'val_rec_acc': best_val_metrics['rec_accuracy'],
            'val_rec_sens': best_val_metrics['rec_sensitivity'],
            'val_rec_spec': best_val_metrics['rec_specificity'],
            'val_patient_auc': best_val_metrics['patient_auc'],
            'val_patient_acc': best_val_metrics['patient_accuracy'],
            'val_patient_sens': best_val_metrics['patient_sensitivity'],
            'val_patient_spec': best_val_metrics['patient_specificity'],
            'num_test_patients': len(test_hcs),
            'test_rec_auc': test_metrics['rec_auc'],
            'test_rec_acc': test_metrics['rec_accuracy'],
            'test_rec_sens': test_metrics['rec_sensitivity'],
            'test_rec_spec': test_metrics['rec_specificity'],
            'test_patient_auc': test_metrics['patient_auc'],
            'test_patient_acc': test_metrics['patient_accuracy'],
            'test_patient_sens': test_metrics['patient_sensitivity'],
            'test_patient_spec': test_metrics['patient_specificity']
        })
    
    # Final summary across all folds
    results_df = pd.DataFrame(results_summary)
    
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)
    print("\nPer-Fold Results:")
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"TEST SET - AVERAGE PERFORMANCE (Patient-Level {aggregation_method.replace('_', ' ').title()})")
    print("="*80)
    print(f"  AUC:         {results_df['test_patient_auc'].mean():.4f} ± {results_df['test_patient_auc'].std():.4f}")
    print(f"  Accuracy:    {results_df['test_patient_acc'].mean():.2f}% ± {results_df['test_patient_acc'].std():.2f}%")
    print(f"  Sensitivity: {results_df['test_patient_sens'].mean():.2f}% ± {results_df['test_patient_sens'].std():.2f}%")
    print(f"  Specificity: {results_df['test_patient_spec'].mean():.2f}% ± {results_df['test_patient_spec'].std():.2f}%")
    
    # Save detailed results
    results_df.to_csv(f'{output_dir}/detailed_5fold_results.csv', index=False)
    print(f"\n✓ Detailed results saved to '{output_dir}/detailed_5fold_results.csv'")
    
    # Create summary statistics file
    with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("5-FOLD CROSS-VALIDATION SUMMARY STATISTICS - LSTM WAVEFORM MODEL\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Hyperparameters:\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Learning Rate: {learning_rate}\n")
        f.write(f"  Hidden Size: {hidden_size}\n")
        f.write(f"  Num Layers: {num_layers}\n")
        f.write(f"  Dropout: {dropout}\n")
        f.write(f"  Max Epochs: {num_epochs}\n")
        f.write(f"  Early Stopping Patience: {patience}\n")
        f.write(f"  Max Waveform Length: {max_length} samples\n")
        f.write(f"  Aggregation Method: {aggregation_method}\n\n")
        
        f.write("--- VALIDATION SET ---\n\n")
        f.write("Recording-Level:\n")
        f.write(f"  AUC:         {results_df['val_rec_auc'].mean():.4f} ± {results_df['val_rec_auc'].std():.4f}\n")
        f.write(f"  Accuracy:    {results_df['val_rec_acc'].mean():.2f}% ± {results_df['val_rec_acc'].std():.2f}%\n")
        f.write(f"  Sensitivity: {results_df['val_rec_sens'].mean():.2f}% ± {results_df['val_rec_sens'].std():.2f}%\n")
        f.write(f"  Specificity: {results_df['val_rec_spec'].mean():.2f}% ± {results_df['val_rec_spec'].std():.2f}%\n\n")
        
        f.write(f"Patient-Level ({aggregation_method.replace('_', ' ').title()}):\n")
        f.write(f"  AUC:         {results_df['val_patient_auc'].mean():.4f} ± {results_df['val_patient_auc'].std():.4f}\n")
        f.write(f"  Accuracy:    {results_df['val_patient_acc'].mean():.2f}% ± {results_df['val_patient_acc'].std():.2f}%\n")
        f.write(f"  Sensitivity: {results_df['val_patient_sens'].mean():.2f}% ± {results_df['val_patient_sens'].std():.2f}%\n")
        f.write(f"  Specificity: {results_df['val_patient_spec'].mean():.2f}% ± {results_df['val_patient_spec'].std():.2f}%\n\n")
        
        f.write("--- TEST SET (PRIMARY RESULTS) ---\n\n")
        f.write("Recording-Level:\n")
        f.write(f"  AUC:         {results_df['test_rec_auc'].mean():.4f} ± {results_df['test_rec_auc'].std():.4f}\n")
        f.write(f"  Accuracy:    {results_df['test_rec_acc'].mean():.2f}% ± {results_df['test_rec_acc'].std():.2f}%\n")
        f.write(f"  Sensitivity: {results_df['test_rec_sens'].mean():.2f}% ± {results_df['test_rec_sens'].std():.2f}%\n")
        f.write(f"  Specificity: {results_df['test_rec_spec'].mean():.2f}% ± {results_df['test_rec_spec'].std():.2f}%\n\n")
        
        f.write(f"Patient-Level ({aggregation_method.replace('_', ' ').title()}):\n")
        f.write(f"  AUC:         {results_df['test_patient_auc'].mean():.4f} ± {results_df['test_patient_auc'].std():.4f}\n")
        f.write(f"  Accuracy:    {results_df['test_patient_acc'].mean():.2f}% ± {results_df['test_patient_acc'].std():.2f}%\n")
        f.write(f"  Sensitivity: {results_df['test_patient_sens'].mean():.2f}% ± {results_df['test_patient_sens'].std():.2f}%\n")
        f.write(f"  Specificity: {results_df['test_patient_spec'].mean():.2f}% ± {results_df['test_patient_spec'].std():.2f}%\n\n")
        
        f.write("\nPer-Fold Results:\n")
        f.write(results_df.to_string(index=False))
    
    print(f"✓ Summary statistics saved to '{output_dir}/summary_statistics.txt'")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('5-Fold Cross-Validation Performance Summary - LSTM', fontsize=16, fontweight='bold')
    
    # Test AUC comparison (Recording vs Patient)
    folds = results_df['fold'].values
    axes[0, 0].plot(folds, results_df['test_rec_auc'], 'o-', label='Recording-Level', linewidth=2, markersize=8)
    axes[0, 0].plot(folds, results_df['test_patient_auc'], 's-', label='Patient-Level', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=results_df['test_rec_auc'].mean(), color='blue', linestyle='--', alpha=0.5, label=f'Mean Rec: {results_df["test_rec_auc"].mean():.3f}')
    axes[0, 0].axhline(y=results_df['test_patient_auc'].mean(), color='orange', linestyle='--', alpha=0.5, label=f'Mean Pat: {results_df["test_patient_auc"].mean():.3f}')
    axes[0, 0].set_xlabel('Fold', fontsize=12)
    axes[0, 0].set_ylabel('AUC', fontsize=12)
    axes[0, 0].set_title('Test AUC by Fold', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(folds)
    
    # Validation vs Test AUC (Patient-Level)
    axes[0, 1].plot(folds, results_df['val_patient_auc'], 'o-', label='Validation', linewidth=2, markersize=8)
    axes[0, 1].plot(folds, results_df['test_patient_auc'], 's-', label='Test', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=results_df['val_patient_auc'].mean(), color='blue', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=results_df['test_patient_auc'].mean(), color='orange', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Fold', fontsize=12)
    axes[0, 1].set_ylabel('AUC', fontsize=12)
    axes[0, 1].set_title('Patient-Level: Validation vs Test AUC', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(folds)
    
    # Test Metrics Comparison (Patient-Level)
    x = np.arange(len(folds))
    width = 0.2
    axes[1, 0].bar(x - 1.5*width, results_df['test_patient_auc']*100, width, label='AUC×100', alpha=0.8)
    axes[1, 0].bar(x - 0.5*width, results_df['test_patient_acc'], width, label='Accuracy', alpha=0.8)
    axes[1, 0].bar(x + 0.5*width, results_df['test_patient_sens'], width, label='Sensitivity', alpha=0.8)
    axes[1, 0].bar(x + 1.5*width, results_df['test_patient_spec'], width, label='Specificity', alpha=0.8)
    axes[1, 0].set_xlabel('Fold', fontsize=12)
    axes[1, 0].set_ylabel('Score (%)', fontsize=12)
    axes[1, 0].set_title('Test Patient-Level Metrics by Fold', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(folds)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Average Performance Comparison
    metrics = ['AUC×100', 'Accuracy', 'Sensitivity', 'Specificity']
    test_means = [
        results_df['test_patient_auc'].mean()*100,
        results_df['test_patient_acc'].mean(),
        results_df['test_patient_sens'].mean(),
        results_df['test_patient_spec'].mean()
    ]
    test_stds = [
        results_df['test_patient_auc'].std()*100,
        results_df['test_patient_acc'].std(),
        results_df['test_patient_sens'].std(),
        results_df['test_patient_spec'].std()
    ]
    
    x_pos = np.arange(len(metrics))
    axes[1, 1].bar(x_pos, test_means, yerr=test_stds, alpha=0.8, capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1, 1].set_ylabel('Score (%)', fontsize=12)
    axes[1, 1].set_title('Average Test Patient-Level Performance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(test_means, test_stds)):
        axes[1, 1].text(i, mean + std + 2, f'{mean:.1f}±{std:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/performance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Performance summary plot saved to '{plots_dir}/performance_summary.png'")
    print(f"\n" + "="*80)
    print(f"All results saved to: {output_dir}/")
    print(f"  - detailed_5fold_results.csv: Per-fold metrics")
    print(f"  - summary_statistics.txt: Mean ± std for all metrics")
    print(f"  - fold_X_training_history.csv: Training history for each fold")
    print(f"\nAll plots saved to: {plots_dir}/")
    print(f"  - performance_summary.png: Comparison plots")
    print(f"  - fold_X_training_curves.png: Training curves for each fold")
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    """
    5-fold cross-validation with recording and patient-level results
    """
    
    # Paths
    WAVEFORM_DIR = "/mloscratch/users/gnahas/data/waveform_norm"
    LABEL_CSV = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
    TRAIN_SPLIT_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
    VAL_TEST_SPLIT_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
    
    # Hyperparameters
    BATCH_SIZE = 32  # Total batch size (splits across GPUs with DataParallel)
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    PATIENCE = 15
    MAX_LENGTH = 445000  # Safe margin to cover all waveforms (max observed: ~444k)
    AGGREGATION_METHOD = 'weighted_average'  # Options: 'majority_vote', 'average', 'weighted_average'
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Run 5-fold cross-validation
    results_df = run_5fold_cv(
        waveform_dir=WAVEFORM_DIR,
        label_csv=LABEL_CSV,
        train_split_csv=TRAIN_SPLIT_CSV,
        val_test_split_csv=VAL_TEST_SPLIT_CSV,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        patience=PATIENCE,
        device=device,
        output_dir='/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/LSTM/V1',
        max_length=MAX_LENGTH,
        aggregation_method=AGGREGATION_METHOD
    )
