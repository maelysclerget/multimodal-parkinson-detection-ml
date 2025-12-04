import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


class EarlyFusionCNN(nn.Module):
    def __init__(self, 
                 input_dim: int = 1280,
                 hidden_dims: list = [512, 256],
                 dropout: float = 0.3,
                 pooling_type: str = 'max',
                 verbose: bool = True):
        """
        Early fusion CNN model that takes a single concatenated feature vector,
        reshapes it to 2D, and passes through a CNN classifier.
        
        Args:
            input_dim: Dimension of input features (default: 1280)
            hidden_dims: List of hidden layer dimensions for CNN channels (default: [512, 256])
            dropout: Dropout probability (default: 0.3)
            pooling_type: Type of pooling layer - 'max', 'avg', or 'adaptive_avg' (default: 'max')
            verbose: Whether the constructor should output verbal execution tracing (default: True)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.pooling_type = pooling_type.lower()
        
        # Validate pooling type
        valid_pooling_types = ['max', 'avg', 'adaptive_avg']
        if self.pooling_type not in valid_pooling_types:
            raise ValueError(f"pooling_type must be one of {valid_pooling_types}, got {self.pooling_type}")
        
        # Reshape input to 2D: (batch, 1, H, W)
        # Using square-ish dimensions for the feature map
        self.H = int(self.input_dim ** 0.5)
        self.W = (self.input_dim + self.H - 1) // self.H  # Ceiling division
        self.padded_dim = self.H * self.W
        
        # Create pooling layer based on pooling_type
        if self.pooling_type == 'max':
            pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'avg':
            pooling_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'adaptive_avg':
            pooling_layer = nn.AdaptiveAvgPool2d(output_size=(None, None))
        
        # Build CNN layers
        layers = []
        in_channels = 1
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                pooling_layer,
                nn.Dropout2d(dropout)
            ])
            in_channels = hidden_dim
        
        # Calculate flattened size after convolutions and pooling
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.H, self.W)
            for layer in layers:
                dummy = layer(dummy)
            self.flattened_size = dummy.view(1, -1).size(1)
        
        self.cnn = nn.Sequential(*layers)
        
        # Output layer (single node for binary classification)
        self.fc = nn.Linear(self.flattened_size, 1)

        # Training history (populated by fit)
        self.train_loss_history = []
        self.train_acc_history = []
        
        # Test metrics (populated by test)
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.test_loss = None
        self.test_acc = None
        self.test_auc = None
        self.test_f1 = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if verbose:
            print(f"Input dimension: {self.input_dim}")
            print(f"Reshaped to: (1, {self.H}, {self.W}) [padded_dim={self.padded_dim}]")
            print(f"CNN architecture: {len(hidden_dims)} conv layers with channels {hidden_dims}")
            print(f"Pooling type: {self.pooling_type}")
            print(f"Flattened size before FC: {self.flattened_size}")
            print(f"Output: 1 (binary classification)")
            print(f"Device: {self.device}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, 1)
        """
        # Pad if necessary to match H*W
        if features.size(1) < self.padded_dim:
            padding = self.padded_dim - features.size(1)
            features = torch.cat([features, torch.zeros(features.size(0), padding, device=features.device)], dim=1)
        
        # Reshape to 2D: (batch, 1, H, W)
        features = features.view(-1, 1, self.H, self.W)
        
        # Pass through CNN
        features = self.cnn(features)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Pass through FC layer
        forward_output = self.fc(features)
        
        return forward_output

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Get class predictions (0 or 1)."""
        output = self.forward(features)
        return (output > 0).squeeze().long()
    
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        output = self.forward(features)
        probs = torch.sigmoid(output)
        return torch.cat([1 - probs, probs], dim=1)
    
    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor,
            epochs: int = 50, batch_size: int = 32,
            lr: float = 1e-3, weight_decay: float = 0, verbose: bool = True):
        """
        Fit the model on training data.
        
        Args:
            train_features: Training features (N, input_dim)
            train_labels: Training labels (N,)
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size (default: 32)
            lr: Learning rate (default: 1e-3)
            weight_decay: L2 penalization coefficient (default: 0)
            verbose: Whether the method should output verbal execution tracing (default: True)
        """
        # Reset history
        self.train_loss_history = []
        self.train_acc_history = []
        
        # Create data loader
        dataset = TensorDataset(train_features, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            for feat, lbl in loader:
                feat, lbl = feat.to(self.device), lbl.to(self.device)
                
                optimizer.zero_grad()
                logits = self.forward(feat)
                loss = criterion(logits, lbl.float().unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(lbl)
                all_preds.extend((logits > 0).squeeze().long().cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())
            
            epoch_loss /= len(dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        if verbose:
            print(f"Training complete. Final - Loss: {self.train_loss_history[-1]:.4f}, "
                  f"Acc: {self.train_acc_history[-1]:.4f}")
            
    def test(self, test_features: torch.Tensor, test_labels: torch.Tensor):
        """
        Evaluate the model on test data and return a dictionary of metrics.
        
        Args:
            test_features: Test features of shape (N, input_dim)
            test_labels: Ground truth labels of shape (N,)
        
        Returns:
            Dictionary containing test metrics (loss, accuracy, F1, confusion matrix, ROC AUC, ROC curve)
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(test_features)
            y_true  = test_labels.numpy()
            y_pred  = self.predict(test_features).numpy()
            y_score = self.predict_proba(test_features).numpy()

        result_metrics = {
            "test_loss": nn.BCEWithLogitsLoss()(output, test_labels.float().unsqueeze(1)).item(),
            "test_acc": accuracy_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_conf_mat": confusion_matrix(y_true, y_pred),
            "test_roc_auc": roc_auc_score(y_true, torch.sigmoid(output).detach().cpu().numpy()),
            "test_roc_curve": roc_curve(y_true, torch.sigmoid(output).detach().cpu().numpy())
        }

        return result_metrics
