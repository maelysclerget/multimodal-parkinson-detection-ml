"""
CNN model for mel spectrogram classification
Simple architecture for Parkinson's detection from mel spectrograms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa


class MelSpectrogramDataset(Dataset):
    """
    Dataset for loading audio and converting to mel spectrograms
    """
    def __init__(self, file_paths, labels, sr=44100, n_mels=128, n_fft=2048, hop_length=512, duration=10):
        """
        Args:
            file_paths: List of paths to .npy waveform files
            labels: List of labels (0 or 1 for binary classification)
            sr: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
            duration: Audio duration in seconds
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.max_length = sr * duration
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load waveform
        waveform = np.load(self.file_paths[idx])
        
        # Pad or truncate to max_length
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Convert to tensor and add channel dimension
        mel_spec_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0)  # Shape: (1, n_mels, time)
        label = torch.LongTensor([self.labels[idx]])
        
        return mel_spec_tensor, label


class CNNMelSpectrogramClassifier(nn.Module):
    """
    Simple CNN classifier for mel spectrograms
    """
    def __init__(self, num_classes=2, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout probability
        """
        super(CNNMelSpectrogramClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout * 0.5)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout * 0.5)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(dropout * 0.5)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(dropout * 0.5)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
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
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x


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
    
    for mel_specs, labels in dataloader:
        mel_specs = mel_specs.to(device)
        labels = labels.squeeze().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(mel_specs)
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
        for mel_specs, labels in dataloader:
            mel_specs = mel_specs.to(device)
            labels = labels.squeeze().to(device)
            
            # Forward pass
            outputs = model(mel_specs)
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
        model: CNN model instance
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
            torch.save(model.state_dict(), 'best_cnn_melspec_model.pth')
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        
        print()
    
    print("="*80)
    print(f"Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_cnn_melspec_model.pth'))
    
    return model


if __name__ == "__main__":
    """
    Example usage (paths and data splits to be provided)
    """
    
    # Example configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    SAMPLE_RATE = 44100
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    DURATION = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Create dummy data loaders (replace with actual data)
    # train_dataset = MelSpectrogramDataset(
    #     train_files, train_labels,
    #     sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
    #     hop_length=HOP_LENGTH, duration=DURATION
    # )
    # val_dataset = MelSpectrogramDataset(
    #     val_files, val_labels,
    #     sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
    #     hop_length=HOP_LENGTH, duration=DURATION
    # )
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = CNNMelSpectrogramClassifier(
        num_classes=2,
        dropout=0.5
    )
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example mel spectrogram shape calculation
    # For 10 seconds at 44100 Hz with hop_length=512:
    # time_steps = (44100 * 10) / 512 ≈ 861
    # Input shape: (batch_size, 1, 128, 861)
    print(f"\nExpected input shape: (batch_size, 1, {N_MELS}, ~{(SAMPLE_RATE * DURATION) // HOP_LENGTH})")
    
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
