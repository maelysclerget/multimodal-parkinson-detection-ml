import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train_split_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/data_paired/10_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_train.csv"
valtest_split_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/data_paired/10_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_val_test.csv"
labels_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/data_paired/csv/paired_healthcode.csv"

# lr ratio, missed taps, right, left, mean x, mean y...
basic_features_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/tapping_statistical_features.csv"

# fatigue slope, variability index, fluctuation index...
advanced_features_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/tapping_advanced_features.csv"

FOLD = 0
batch_size = 64

# Load features and labels
labels_df = pd.read_csv(labels_path)
labels_df["label_PD"] = labels_df["label_PD"].astype(int)

features_df = pd.read_csv(basic_features_path)
#print("Columns in feature file:", features_df.columns.tolist())

data_df = features_df.merge(labels_df, on="healthCode", how="inner")
data_df = data_df.drop(columns=["modality"])

print("Total patients", len(data_df))
print("Final columns:", data_df.columns.tolist())

# Load train/val/test splits
train_split = pd.read_csv(train_split_path)
valtest_split = pd.read_csv(valtest_split_path)

# Get two separate lists of healthCodes for each split
train_hc = train_split[
    (train_split["fold_iteration"] == FOLD) & (train_split["subset"] == "train")
]["healthCode"]

val_hc = valtest_split[
    (valtest_split["fold_iteration"] == FOLD) & (valtest_split["subset"] == "val")
]["healthCode"]

test_hc = valtest_split[
    (valtest_split["fold_iteration"] == FOLD) & (valtest_split["subset"] == "test")
]["healthCode"]

# data_df contains all patients with both features and labels and we filter it using isin to keep only the rows with matching healthCodes
train_df = data_df[data_df["healthCode"].isin(train_hc)].copy()
val_df   = data_df[data_df["healthCode"].isin(val_hc)].copy()
test_df  = data_df[data_df["healthCode"].isin(test_hc)].copy()

print("Train patients:", len(train_df))
print("Val patients:", len(val_df))
print("Test patients:", len(test_df))

feature_cols = [c for c in data_df.columns if c not in ["healthCode", "label_PD"]]
print("MLP input features:", feature_cols)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
X_val   = scaler.transform(val_df[feature_cols].values.astype(np.float32))
X_test  = scaler.transform(test_df[feature_cols].values.astype(np.float32))

y_train = train_df["label_PD"].values.astype(np.float32)
y_val   = val_df["label_PD"].values.astype(np.float32)
y_test  = test_df["label_PD"].values.astype(np.float32)

# Convert to torch tensors
X_train_t = torch.from_numpy(X_train)
X_val_t   = torch.from_numpy(X_val)
X_test_t  = torch.from_numpy(X_test)

y_train_t = torch.from_numpy(y_train)
y_val_t   = torch.from_numpy(y_val)
y_test_t  = torch.from_numpy(y_test)

# Datasets & loaders
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = X_train.shape[1]
print("Input dim:", input_dim)


# ===== Define MLP Model =====
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 64 hidden neurons
            nn.ReLU(),
            nn.Dropout(0.5), #Regularization, randomly drops 50% of neurons to prevent overfitting
            nn.Linear(64, 32), # 64 neurons --> 32 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    
    def forward(self, x): #takes input tensor x and passes it through the network layers defined in __init__
        return torch.sigmoid(self.net(x))

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MLP(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# ===== Training Loop =====
num_epochs = 50
best_val_loss = float('inf')
patience = 10  # Stop if val loss doesn't improve for 10 epochs
patience_counter = 0


for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset counter
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            print(f"  Best val loss: {best_val_loss:.4f}")
            break
        
    

# ===== Test Evaluation =====
print("\n===== Test Evaluation =====")
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch).squeeze()
        test_preds.extend((y_pred > 0.5).cpu().numpy())
        test_labels.extend(y_batch.numpy())

accuracy = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
auc = roc_auc_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)

print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# Plot metrics
metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
values = [accuracy, f1, auc]
axes[1].bar(metrics, values)
axes[1].set_ylim([0, 1])
axes[1].set_title('Test Performance Metrics')

plt.tight_layout()
plt.savefig('src_GAMMA/tapping_model/mlp_results.png', dpi=100)
print("\n✓ Results saved to mlp_results.png")
