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

train_split_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_train.csv"
valtest_split_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/data_paired/5_fold_CV/processed_paired/paired_splits/balanced_train/healthcode_10fold_val_test.csv"
labels_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/data_paired/csv/paired_healthcode.csv"

# lr ratio, missed taps, right, left, mean x, mean y...
basic_features_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/tapping_statistical_features.csv"

# fatigue slope, variability index, fluctuation index...
advanced_features_path = "/Users/maelysclerget/Desktop/CS-433/Project2/NeuroMeditron/src_GAMMA/data/tapping_advanced_features_session.csv"

FOLD = 0
batch_size = 64

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

def train_and_evaluate(features_df, model_name): 
    print(f"\n Model: {model_name}")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    labels_df["label_PD"] = labels_df["label_PD"].astype(int)
    
    # Merge features with labels
    data_df = features_df.merge(labels_df, on="healthCode", how="inner")
    
    # Drop modality if exists
    if "modality" in data_df.columns:
        data_df = data_df.drop(columns=["modality"])   
        
    # if 1 trial, std is NaN --> fill NaN with mean
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns
    data_df[numeric_cols] = data_df[numeric_cols].fillna(data_df[numeric_cols].mean())
    
    print(f"Total patients: {len(data_df)}")
    print(f"Columns: {data_df.columns.tolist()}")
    
    # Load splits
    train_split = pd.read_csv(train_split_path)
    valtest_split = pd.read_csv(valtest_split_path)
    
    # Get healthCodes for each split
    train_hc = train_split[(train_split["fold_iteration"] == FOLD) & (train_split["subset"] == "train")]["healthCode"]
    val_hc = valtest_split[(valtest_split["fold_iteration"] == FOLD) & (valtest_split["subset"] == "val")]["healthCode"]
    test_hc = valtest_split[(valtest_split["fold_iteration"] == FOLD) & (valtest_split["subset"] == "test")]["healthCode"]
    
    # Filter dataframes
    train_df = data_df[data_df["healthCode"].isin(train_hc)].copy()
    val_df = data_df[data_df["healthCode"].isin(val_hc)].copy()
    test_df = data_df[data_df["healthCode"].isin(test_hc)].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Extract features
    feature_cols = [c for c in data_df.columns if c not in ["healthCode", "label_PD"]]
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    X_val = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    
    y_train = train_df["label_PD"].values.astype(np.float32)
    y_val = val_df["label_PD"].values.astype(np.float32)
    y_test = test_df["label_PD"].values.astype(np.float32)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)
    
    y_train_t = torch.from_numpy(y_train)
    y_val_t = torch.from_numpy(y_val)
    y_test_t = torch.from_numpy(y_test)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MLP(len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("\n===== Training =====")
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
    
        # Test evaluation
    print("\n===== Test Evaluation =====")
    model.eval()
    
    test_preds_proba = []
    test_preds_binary = []
    test_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).squeeze()
            test_preds_proba.extend(y_pred.cpu().numpy())
            test_preds_binary.extend((y_pred > 0.5).cpu().numpy())
            test_labels.extend(y_batch.numpy())
    
    accuracy = accuracy_score(test_labels, test_preds_binary)
    f1 = f1_score(test_labels, test_preds_binary)
    auc = roc_auc_score(test_labels, test_preds_proba)
    cm = confusion_matrix(test_labels, test_preds_binary)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    
# Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    
    metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
    values = [accuracy, f1, auc]
    axes[1].bar(metrics, values)
    axes[1].set_ylim([0, 1])
    axes[1].set_title(f'Test Performance - {model_name}')
    
    plt.tight_layout()

    if "Basic" in model_name:
        filename = "src_GAMMA/tapping_model/01_basic_features_results.png"
    else:
        filename = "src_GAMMA/tapping_model/02_advanced_features_results.png"
    
    plt.savefig(filename, dpi=100)
    print(f"\n✓ Results saved to {filename}")
    plt.close()
    
    return {'model': model_name, 'accuracy': accuracy, 'f1': f1, 'auc': auc}

# ===== Main =====
if __name__ == "__main__":
    
    # Load both feature sets
    basic_features = pd.read_csv(basic_features_path)
    advanced_features = pd.read_csv(advanced_features_path)
    
    # Train both models
    results = []
    results.append(train_and_evaluate(basic_features, "Baseline MLP (Basic Features)"))
    results.append(train_and_evaluate(advanced_features, "Advanced MLP (Advanced Features)"))
    
"""     # Compare
    print("COMPARISON")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    if results[1]['auc'] > results[0]['auc']:
        improvement = (results[1]['auc'] - results[0]['auc']) * 100
        print(f"\n✓ Advanced features improve AUC by {improvement:.2f}%")
    else:
        print(f"\n✗ Basic features perform better") """
        
        
    