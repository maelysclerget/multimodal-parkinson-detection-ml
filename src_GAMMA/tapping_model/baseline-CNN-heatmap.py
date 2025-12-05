import numpy as np
import pandas as pd
from pathlib import Path
import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

train_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_train.csv"
valtest_split_path = "/mloscratch/users/clerget/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_val_test.csv"
labels_path = "/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/paired_healthcode.csv"

# Heatmap directory
heatmap_base_path = "/mloscratch/users/clerget/data/tapping_heatmaps"

NUM_FOLDS = 5
batch_size = 32

# ===== Image Transform =====
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize heatmap to standard size
    transforms.ToTensor(),  # Convert to tensor (0-1 range)
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# ===== Custom Dataset for Heatmaps =====
class HeatmapDataset(Dataset):
    """Dataset for loading heatmap images per trial"""
    def __init__(self, healthcodes, trial_ids, labels, heatmap_base_path, transform=None):
        self.healthcodes = healthcodes
        self.trial_ids = trial_ids
        self.labels = labels
        self.heatmap_base_path = heatmap_base_path
        self.transform = transform
        self.valid_indices = []
        
        # Check which heatmaps exist
        for idx in range(len(healthcodes)):
            hc = healthcodes[idx]
            trial_id = trial_ids[idx]
            heatmap_path = Path(heatmap_base_path) / hc / f"{trial_id}.png"
            if heatmap_path.exists():
                self.valid_indices.append(idx)
        
        print(f"Found {len(self.valid_indices)} valid heatmaps out of {len(healthcodes)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        hc = self.healthcodes[real_idx]
        trial_id = self.trial_ids[real_idx]
        label = self.labels[real_idx]
        
        # Load heatmap image
        heatmap_path = Path(self.heatmap_base_path) / hc / f"{trial_id}.png"
        
        try:
            image = Image.open(heatmap_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {heatmap_path}: {e}")
            # Return a blank image on error
            blank_image = torch.zeros(1, 224, 224)
            return blank_image, torch.tensor(label, dtype=torch.long)


# ===== Define 2D CNN Model for Heatmaps =====
class CNN2D_Heatmap(nn.Module):
    """2D CNN for heatmap images"""
    def __init__(self, num_channels=32, dropout_rate=0.5):
        super(CNN2D_Heatmap, self).__init__()
        
        # Input: (batch, 1, 224, 224)
        # Conv block 1
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels * 4)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_channels * 4, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 2)  # Binary classification
    
    def forward(self, x):
        """Forward pass
        Input: (batch_size, 1, 224, 224)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # FC layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        return x


def create_balanced_sampler(y_train):
    """
    Create a WeightedRandomSampler for 50-50 balanced batch sampling at SESSION level.
    
    Uses the ratio num_class_0 / num_class_1 as PD weight to achieve 50-50 split:
    - Healthy sessions weight: 1.0
    - PD sessions weight: num_healthy / num_pd
    - Result: ~50% Healthy, ~50% PD in each batch
    
    Math: Expected batch ratio = n_healthy × 1.0 / (n_healthy × 1.0 + n_pd × (n_healthy/n_pd))
                               = n_healthy / (n_healthy + n_healthy) 
                               = 50-50 split
    """
    
    num_class_0 = np.sum(y_train == 0)  # Healthy sessions
    num_class_1 = np.sum(y_train == 1)  # PD sessions
    
    # Calculate weight factor for 50-50 balance
    pd_weight_factor = num_class_0 / num_class_1
    weights = np.where(y_train == 0, 1.0, pd_weight_factor)
    
    print(f"  Using 50-50 balanced sampler:")
    print(f"    - Healthy sessions: {num_class_0} (weight=1.0)")
    print(f"    - PD sessions: {num_class_1} (weight={pd_weight_factor:.4f})")
    print(f"    - Expected batch ratio: ~50% Healthy, ~50% PD")
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def aggregate_predictions(healthcodes, pred_probas, pred_binaries, labels, aggregation_method='mean'):
    """Aggregate trial-level predictions to patient-level predictions"""
    
    results_df = pd.DataFrame({
        'healthCode': healthcodes,
        'pred_proba': pred_probas,
        'pred_binary': pred_binaries,
        'label': labels
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


def train_and_evaluate(data_with_labels, split_info, model_name, model_prefix, fold=0, use_balanced_sampler=True):
    print(f"\n Model: {model_name} | Fold: {fold}")
    print(f" └─ Balanced Sampler (50-50 Healthy/PD): {'✓ Enabled' if use_balanced_sampler else '✗ Disabled'}")
    
    # Get split info for this fold
    train_hc = split_info[(split_info["fold_iteration"] == fold) & (split_info["subset"] == "train")]["healthCode"].unique()
    val_hc = split_info[(split_info["fold_iteration"] == fold) & (split_info["subset"] == "val")]["healthCode"].unique()
    test_hc = split_info[(split_info["fold_iteration"] == fold) & (split_info["subset"] == "test")]["healthCode"].unique()
    
    # Split data
    train_df = data_with_labels[data_with_labels["healthCode"].isin(train_hc)].copy()
    val_df = data_with_labels[data_with_labels["healthCode"].isin(val_hc)].copy()
    test_df = data_with_labels[data_with_labels["healthCode"].isin(test_hc)].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Get class distribution
    y_train = train_df["label_PD"].values
    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1)
    total_samples = len(y_train)
    
    print(f"Class distribution - Healthy (0): {num_class_0}, PD (1): {num_class_1}")
    print(f"Using balanced sampler → No class weights needed (equal weight for both classes)")
    
    # Create datasets
    train_dataset = HeatmapDataset(
        train_df['healthCode'].values,
        train_df['trial_id'].values,
        train_df['label_PD'].values,
        heatmap_base_path,
        transform=image_transform
    )
    
    val_dataset = HeatmapDataset(
        val_df['healthCode'].values,
        val_df['trial_id'].values,
        val_df['label_PD'].values,
        heatmap_base_path,
        transform=image_transform
    )
    
    test_dataset = HeatmapDataset(
        test_df['healthCode'].values,
        test_df['trial_id'].values,
        test_df['label_PD'].values,
        heatmap_base_path,
        transform=image_transform
    )
    
    # ===== Create balanced sampler for training =====
    if use_balanced_sampler:
        print("\n[BALANCED SAMPLING] Per-batch 50-50 balancing of Healthy/PD sessions:")
        train_sampler = create_balanced_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CNN2D_Heatmap(num_channels=32, dropout_rate=0.5).to(device)
    
    # Loss and optimizer
    # Since balanced sampler ensures 50-50 split, use equal weights for both classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    # Training loop
    num_epochs = 50
    best_val_auc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.detach().item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_preds_proba = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                val_preds_proba.extend(probs[:, 1].cpu().numpy().tolist())
                val_labels.extend(labels.numpy().tolist())
        
        if len(val_labels) > 0:
            val_auc = roc_auc_score(val_labels, val_preds_proba)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Test evaluation
    model.eval()
    test_preds_proba = []
    test_preds_binary = []
    test_labels = []
    test_healthcodes = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            probs_class1 = probs[:, 1]
            
            test_preds_proba.extend(probs_class1.cpu().numpy().tolist())
            test_preds_binary.extend((probs_class1 > 0.5).cpu().numpy().tolist())
            test_labels.extend(labels.numpy().tolist())
            
            # Get corresponding healthcodes
            batch_indices = test_dataset.valid_indices[idx * batch_size:(idx + 1) * batch_size]
            for real_idx in batch_indices:
                test_healthcodes.append(test_dataset.healthcodes[real_idx])
    
    # Aggregate to patient-level
    patient_preds = aggregate_predictions(test_healthcodes, test_preds_proba, test_preds_binary, test_labels, aggregation_method='mean')
    
    patient_accuracy = accuracy_score(patient_preds['label'], patient_preds['pred_binary'])
    patient_f1 = f1_score(patient_preds['label'], patient_preds['pred_binary'])
    patient_auc = roc_auc_score(patient_preds['label'], patient_preds['pred_proba'])
    
    print(f"Patient-level Accuracy: {patient_accuracy:.4f}")
    print(f"Patient-level F1-Score: {patient_f1:.4f}")
    print(f"Patient-level AUC-ROC:  {patient_auc:.4f}")
    
    # ===== Generate Per-Fold Visualizations =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{model_name} (CNN Heatmap) - Fold {fold} Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(patient_preds['label'], patient_preds['pred_binary'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False, 
                xticklabels=['Healthy', 'PD'], yticklabels=['Healthy', 'PD'])
    axes[0, 0].set_title(f'Confusion Matrix (Accuracy: {patient_accuracy:.4f})', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(patient_preds['label'], patient_preds['pred_proba'])
    axes[0, 1].plot(fpr, tpr, linewidth=2.5, label=f'AUC = {patient_auc:.4f}', color='#1f77b4')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
    axes[0, 1].set_title('ROC Curve', fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Metrics Bar Plot
    metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
    values = [patient_accuracy, patient_f1, patient_auc]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes[1, 0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Performance Metrics', fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(values):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    
    # 4. Prediction Distribution
    axes[1, 1].hist(patient_preds[patient_preds['label'] == 0]['pred_proba'], bins=15, alpha=0.6, label='Healthy', color='blue')
    axes[1, 1].hist(patient_preds[patient_preds['label'] == 1]['pred_proba'], bins=15, alpha=0.6, label='PD', color='red')
    axes[1, 1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[1, 1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Prediction Distribution', fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/cv_results')
    fig_path = output_dir / f'{model_prefix}_{model_name.replace(" ", "_")}_fold{fold}_results.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Fold visualization saved: {fig_path}")
    
    plt.close()
    
    return {
        'model': model_name,
        'fold': fold,
        'accuracy': patient_accuracy,
        'f1_score': patient_f1,
        'auc': patient_auc,
        'num_patients': len(patient_preds),
        'num_trials': len(test_preds_proba)
    }


def create_cv_visualization(model_name, accuracy, f1_score, auc, accuracy_std, f1_score_std, auc_std, model_prefix):
    """Create visualization for CV mean results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} (CNN Heatmap) - Cross-Validation Results (Mean ± Std across 5 Folds)', 
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
    
    # 2. Comparison Table
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
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1].set_title('Detailed Results', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/cv_results')
    fig_path = output_dir / f'{model_prefix}_{model_name.replace(" ", "_")}_cv_results.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ CV Visualization saved: {fig_path}")
    
    plt.close()


# ===== Main =====
if __name__ == "__main__":
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    labels_df["label_PD"] = labels_df["label_PD"].astype(int)
    
    # Load split info
    split_info = pd.read_csv(train_split_path)
    split_info_valtest = pd.read_csv(valtest_split_path)
    split_info = pd.concat([split_info, split_info_valtest], ignore_index=True)
    
    print("\n===== Heatmap CNN Training =====")
    print(f"Heatmap directory: {heatmap_base_path}")
    
    # Create trial_id from heatmap directories
    heatmap_dir = Path(heatmap_base_path)
    all_trials = []
    
    for hc_dir in heatmap_dir.iterdir():
        if hc_dir.is_dir():
            healthcode = hc_dir.name
            for heatmap_file in hc_dir.glob('*.png'):
                trial_id = heatmap_file.stem
                all_trials.append({
                    'healthCode': healthcode,
                    'trial_id': trial_id
                })
    
    trials_df = pd.DataFrame(all_trials)
    print(f"Found {len(trials_df)} heatmap trials")
    
    # Merge with labels
    data_with_labels = trials_df.merge(labels_df, on='healthCode', how='inner')
    print(f"After merge with labels: {len(data_with_labels)} trials")
    
    # Store results for all folds
    all_results = []
    
    # Loop through all 5 folds
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{NUM_FOLDS - 1}")
        print(f"{'='*70}")
        
        # Train heatmap CNN
        result = train_and_evaluate(data_with_labels, split_info, "Heatmap CNN", "07_cnn_heatmap", fold=fold)
        all_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate mean metrics
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS (Mean ± Std across 5 folds)")
    print("="*70)
    
    mean_accuracy = results_df['accuracy'].mean()
    std_accuracy = results_df['accuracy'].std()
    
    mean_f1 = results_df['f1_score'].mean()
    std_f1 = results_df['f1_score'].std()
    
    mean_auc = results_df['auc'].mean()
    std_auc = results_df['auc'].std()
    
    print(f"\nHeatmap CNN:")
    print(f"  Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  AUC-ROC:   {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Add mean row
    results_df = pd.concat([results_df, pd.DataFrame([{
        'model': 'Heatmap CNN',
        'fold': 'MEAN',
        'accuracy': mean_accuracy,
        'f1_score': mean_f1,
        'auc': mean_auc,
        'num_patients': '',
        'num_trials': '',
        'accuracy_std': std_accuracy,
        'f1_score_std': std_f1,
        'auc_std': std_auc
    }])], ignore_index=True)
    
    # Save results
    output_dir = Path('/mloscratch/users/clerget/NeuroMeditron/src_GAMMA/tapping_model/cv_results')
    csv_path = output_dir / '07_cnn_heatmap_Heatmap_CNN_cv_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate visualization
    print("\n" + "="*70)
    print("Generating CV Mean Visualization...")
    print("="*70)
    
    create_cv_visualization(
        model_name="Heatmap CNN",
        accuracy=mean_accuracy,
        f1_score=mean_f1,
        auc=mean_auc,
        accuracy_std=std_accuracy,
        f1_score_std=std_f1,
        auc_std=std_auc,
        model_prefix='07_cnn_heatmap'
    )
    
    print("\n" + "="*70)
    print("✓ Heatmap CNN Cross-Validation Complete!")
    print("="*70)
