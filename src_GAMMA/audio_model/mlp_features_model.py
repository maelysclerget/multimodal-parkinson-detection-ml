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
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.squeeze().to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=0.001,
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
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model
    """
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 20
    
    print("="*80)
    print("Starting Training")
    print("="*80)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mlp_features_model.pth')
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    print("="*80)
    print(f"Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")
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


if __name__ == "__main__":
    """
    Example usage (paths and data splits to be provided)
    """
    
    # Example configuration
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    INPUT_SIZE = 57  # Number of extracted acoustic features (adjust based on your features)
    HIDDEN_SIZES = [256, 128, 64]
    DROPOUT = 0.5
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Create dummy data loaders (replace with actual data)
    # train_dataset = AcousticFeaturesDataset(csv_path='train_features.csv')
    # val_dataset = AcousticFeaturesDataset(csv_path='val_features.csv')
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Or with pre-loaded arrays:
    # train_dataset = AcousticFeaturesDataset(features=train_features, labels=train_labels)
    # val_dataset = AcousticFeaturesDataset(features=val_features, labels=val_labels)
    
    # Initialize model
    model = MLPAcousticClassifier(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        num_classes=2,
        dropout=DROPOUT
    )
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example feature normalization
    print("\nFeature Normalization:")
    print("  Use normalize_features() to standardize features before training")
    print("  Example: train_norm, val_norm, mean, std = normalize_features(train_features, val_features)")
    
    # Train model (uncomment when data loaders are ready)
    # trained_model = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     num_epochs=NUM_EPOCHS,
    #     learning_rate=LEARNING_RATE,
    #     device=device
    # )
    
    print("\nModel created successfully!")
    print("Provide train/val data loaders to start training.")
    print("\nExpected input: CSV with 'label' column and acoustic features")
    print("Features should be extracted from normalized waveforms (.npy files)")
