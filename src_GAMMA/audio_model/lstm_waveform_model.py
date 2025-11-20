"""
LSTM model for audio waveform classification
Simple architecture for Parkinson's detection from raw audio waveforms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class WaveformDataset(Dataset):
    """
    Dataset for loading audio waveforms from .npy files
    """
    def __init__(self, file_paths, labels, max_length=441000):
        """
        Args:
            file_paths: List of paths to .npy waveform files
            labels: List of labels (0 or 1 for binary classification)
            max_length: Maximum waveform length (default: 10s at 44100 Hz)
        """
        self.file_paths = file_paths
        self.labels = labels
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
        
        return waveform, label


class LSTMWaveformClassifier(nn.Module):
    """
    LSTM-based classifier for raw audio waveforms
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
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
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
        # Reshape: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        x = x.transpose(1, 2)
        
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
    
    Returns:
        Average loss and accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.squeeze().to(device)
        
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
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.squeeze().to(device)
            
            # Forward pass
            outputs = model(waveforms)
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
    num_epochs=50,
    learning_rate=0.001,
    device='cuda'
):
    """
    Full training loop
    
    Args:
        model: LSTM model instance
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_acc = 0.0
    
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
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        
        print()
    
    print("="*80)
    print(f"Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    return model


if __name__ == "__main__":
    """
    Example usage (paths and data splits to be provided)
    """
    
    # Example configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_LENGTH = 441000  # 10 seconds at 44100 Hz
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Create dummy data loaders (replace with actual data)
    # train_dataset = WaveformDataset(train_files, train_labels, max_length=MAX_LENGTH)
    # val_dataset = WaveformDataset(val_files, val_labels, max_length=MAX_LENGTH)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = LSTMWaveformClassifier(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        bidirectional=True
    )
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
