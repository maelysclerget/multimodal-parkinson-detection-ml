"""
InceptionV3-based classifier for mel-spectrogram images (PwPD vs HC)
- Uses pre-trained InceptionV3 as fixed feature extractor
- Custom classifier head: BatchNorm -> Dense(1024, ReLU) -> Dense(1024, ReLU) -> Dense(2, Softmax)
- Only classifier head is trained
- 5-fold cross-validation, same fold logic as ResNet1D
- Plots confusion matrix and saves results per fold
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import Inception_V3_Weights
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Dataset
# =====================
class MelSpecImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.df['label_PD'].values.astype(np.int64)
        self.health_codes = self.df['healthCode'].values
        self.filenames = self.df['filename'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.LongTensor([self.labels[idx]])
        return image, label, self.health_codes[idx]

# =====================
# Model
# =====================
class InceptionV3Classifier(nn.Module):
    def __init__(self, dropout=0.2, num_classes=2):
        super().__init__()
        self.backbone = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        out = self.backbone(x)

        # Handle different torchvision behaviors
        if isinstance(out, tuple):
            x = out[0]               # (logits, aux_logits)
        elif hasattr(out, "logits"):
            x = out.logits           # InceptionOutputs(logits=..., aux_logits=...)
        else:
            x = out                  # fallback, shouldn't really happen

        # Safety: ensure (N, C) before BatchNorm1d
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x

# =====================
# Training & Evaluation
# =====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.squeeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(dataloader), 100 * correct / total

def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels, all_health_codes = [], [], [], []
    with torch.no_grad():
        for images, labels, health_codes in dataloader:
            images, labels = images.to(device), labels.squeeze(1).to(device)
            outputs = model(images)
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
    unique_hcs = np.unique(all_health_codes)
    patient_labels, patient_preds_majority, patient_probs = [], [], []
    for hc in unique_hcs:
        mask = all_health_codes == hc
        patient_labels.append(all_labels[mask][0])
        vote_counts = Counter(all_preds[mask])
        patient_preds_majority.append(vote_counts.most_common(1)[0][0])
        avg_prob = np.mean(all_probs[mask])
        patient_probs.append(avg_prob)
    patient_labels = np.array(patient_labels)
    patient_preds_majority = np.array(patient_preds_majority)
    patient_probs = np.array(patient_probs)
    # Majority vote metrics
    acc_majority = 100 * np.mean(patient_preds_majority == patient_labels)
    cm_majority = confusion_matrix(patient_labels, patient_preds_majority)
    if cm_majority.ravel().shape[0] == 4:
        tn, fp, fn, tp = cm_majority.ravel()
        sens_majority = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_majority = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens_majority = spec_majority = 0
    # Probability-based aggregation metrics
    patient_preds_prob = (patient_probs >= 0.5).astype(int)
    acc_prob = 100 * np.mean(patient_preds_prob == patient_labels)
    cm_prob = confusion_matrix(patient_labels, patient_preds_prob)
    if cm_prob.ravel().shape[0] == 4:
        tn, fp, fn, tp = cm_prob.ravel()
        sens_prob = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_prob = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens_prob = spec_prob = 0

    # Compute AUC for probability-based aggregation
    try:
        auc = roc_auc_score(patient_labels, patient_probs)
    except Exception:
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

def plot_confusion_matrix(cm, filepath, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
# =====================
# 5-FOLD CROSS-VALIDATION
# =====================
def run_5fold_cv_inception(
    melspec_dir,
    label_csv,
    train_split_csv,
    val_test_split_csv,
    batch_size=64,
    num_epochs=50,
    learning_rate=0.001,
    dropout=0.2,
    device='cuda',
    output_dir='inceptionv3_results'):
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Plots'), exist_ok=True)
    # Load label mapping
    label_df = pd.read_csv(label_csv, sep=';')
    healthcode_to_label = dict(zip(label_df['healthCode'], label_df['label_PD']))
    # Get all mel spectrogram files
    print("\nFinding mel spectrogram files...")
    all_files = [f for f in os.listdir(melspec_dir) if f.endswith('.jpg')]
    print(f"  Found {len(all_files)} .jpg files")
    # Parse filenames to extract healthCode and build file info list
    file_info = []
    for fname in all_files:
        parts = fname.replace('.jpg', '').split('_')
        if len(parts) >= 2:
            health_code = parts[0]
            if health_code in healthcode_to_label:
                file_info.append({
                    'filename': fname,
                    'filepath': os.path.join(melspec_dir, fname),
                    'healthCode': health_code,
                    'label_PD': healthcode_to_label[health_code]
                })
    file_info_df = pd.DataFrame(file_info)
    train_splits = pd.read_csv(train_split_csv)
    val_test_splits = pd.read_csv(val_test_split_csv)
    results_summary = []
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for fold in range(5):
        print(f"\n{'='*80}")
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
        train_df = file_info_df[file_info_df['healthCode'].isin(train_hcs)].copy()
        val_df = file_info_df[file_info_df['healthCode'].isin(val_hcs)].copy()
        test_df = file_info_df[file_info_df['healthCode'].isin(test_hcs)].copy()
        train_dataset = MelSpecImageDataset(train_df, melspec_dir, transform)
        val_dataset = MelSpecImageDataset(val_df, melspec_dir, transform)
        test_dataset = MelSpecImageDataset(test_df, melspec_dir, transform)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
        model = InceptionV3Classifier(dropout=dropout, num_classes=2)
        model = model.to(device)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
            model = nn.DataParallel(model)
        # Compute class weights at record level (train set)
        train_labels = train_df['label_PD'].values
        n_class0 = np.sum(train_labels == 0)
        n_class1 = np.sum(train_labels == 1)
        total = n_class0 + n_class1
        weight0 = total / (2 * n_class0) if n_class0 > 0 else 1.0
        weight1 = total / (2 * n_class1) if n_class1 > 0 else 1.0
        class_weights = torch.tensor([weight0, weight1], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # If DataParallel, model.module.classifier; else model.classifier
        classifier_params = model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters()
        optimizer = torch.optim.AdamW(classifier_params, lr=1e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6
        )
        best_val_auc = 0.0
        best_model_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        patience_es = 15  # Early stopping patience (can be parameterized)
        pth_dir = os.path.join(output_dir, 'PTH')
        os.makedirs(pth_dir, exist_ok=True)
        current_checkpoint_path = None
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            val_auc = val_metrics['patient_auc']
            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                if isinstance(model, nn.DataParallel):
                    best_model_state = model.module.state_dict()
                else:
                    best_model_state = model.state_dict()
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                # Save checkpoint (remove previous)
                if current_checkpoint_path and os.path.exists(current_checkpoint_path):
                    os.remove(current_checkpoint_path)
                checkpoint_path = os.path.join(pth_dir, f'fold_{fold}_best_auc_{best_val_auc:.4f}_epoch_{best_epoch}.pth')
                torch.save({
                    'fold': fold,
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'val_auc': best_val_auc,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                current_checkpoint_path = checkpoint_path
            else:
                epochs_without_improvement += 1
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Val AUC: {val_auc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            if epochs_without_improvement >= patience_es:
                print(f"Early stopping at epoch {epoch+1} (no Val AUC improvement for {patience_es} epochs)")
                break
        # Load best model
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        # Test evaluation
        test_metrics = evaluate(model, test_loader, device)
        # Plot confusion matrix
        plots_dir = os.path.join(output_dir, 'Plots')
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            f'{plots_dir}/fold_{fold}_test_cm.png',
            f'Fold {fold} Test Set - InceptionV3'
        )
        # Save results
        results_summary.append({
            'fold': fold,
            'test_auc': test_metrics['patient_auc'],
            'test_acc': test_metrics['patient_accuracy'],
            'test_sens': test_metrics['patient_sensitivity'],
            'test_spec': test_metrics['patient_specificity'],
            'test_f1': f1_score(test_metrics['patient_labels'], test_metrics['patient_preds'], zero_division=0),
            'test_precision': precision_score(test_metrics['patient_labels'], test_metrics['patient_preds'], zero_division=0),
            'test_recall': recall_score(test_metrics['patient_labels'], test_metrics['patient_preds'], zero_division=0)
        })
        print(f"\nFold {fold} Test Results:")
        print(f"  AUC: {test_metrics['patient_auc']:.4f}")
        print(f"  Acc: {test_metrics['patient_accuracy']:.2f}%, F1: {results_summary[-1]['test_f1']:.4f}")
        print(f"  Sens: {test_metrics['patient_sensitivity']:.2f}%, Spec: {test_metrics['patient_specificity']:.2f}%")
    # Final summary
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(f'{output_dir}/results_summary.csv', index=False)
    print(f"\n{'='*80}")
    print(f"5-FOLD CROSS-VALIDATION RESULTS - InceptionV3")
    print(f"{'='*80}")
    print(f"AUC:       {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
    print(f"Accuracy:  {results_df['test_acc'].mean():.2f}% ± {results_df['test_acc'].std():.2f}%")
    print(f"F1-Score:  {results_df['test_f1'].mean():.4f} ± {results_df['test_f1'].std():.4f}")
    print(f"Sens/Spec: {results_df['test_sens'].mean():.2f}% / {results_df['test_spec'].mean():.2f}%")
    print(f"{'='*80}\n")

    # Plot summary bar plot for AUC, F1, Accuracy per fold (all as percentages)
    plt.figure(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.25
    auc_percent = results_df['test_auc'] * 100
    f1_percent = results_df['test_f1'] * 100
    acc_percent = results_df['test_acc']
    plt.bar(x - width, auc_percent, width, label='AUC (%)')
    plt.bar(x, f1_percent, width, label='F1 (%)')
    plt.bar(x + width, acc_percent, width, label='Accuracy (%)')
    plt.xticks(x, [f'Fold {i}' for i in results_df['fold']], fontsize=10)
    plt.ylabel('Percentage', fontsize=12)
    plt.ylim(0, 1.05 * max(auc_percent.max(), f1_percent.max(), acc_percent.max(), 100))
    plt.title('AUC, F1, Accuracy per Fold', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Plots', 'summary_per_fold_auc_f1_acc.png'), dpi=150)
    plt.close()

    return results_df

if __name__ == "__main__":
    LABEL_CSV = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/paired_healthcode.csv"
    MELSPEC_DIR = "/mloscratch/users/gnahas/data/melSpec"
    TRAIN_SPLIT_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_train.csv"
    VAL_TEST_SPLIT_CSV = "/mloscratch/users/gnahas/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_5fold_val_test.csv"
    OUTPUT_DIR = "/mloscratch/users/gnahas/NeuroMeditron/src_GAMMA/audio_model/Results/InceptionV3_MelSpec/V2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_5fold_cv_inception(
        melspec_dir=MELSPEC_DIR,
        label_csv=LABEL_CSV,
        train_split_csv=TRAIN_SPLIT_CSV,
        val_test_split_csv=VAL_TEST_SPLIT_CSV,
        batch_size=64,
        num_epochs=50,
        learning_rate=0.0001,
        dropout=0.2,
        device=device,
        output_dir=OUTPUT_DIR
    )
