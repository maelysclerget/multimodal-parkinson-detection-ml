import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


class EarlyFusionMLP(nn.Module):
    def __init__(self, 
                 input_dim: int = 1280,
                 hidden_dims: list = [512, 256],
                 dropout: float = 0.3,
                 verbose: bool = True):
        """
        Early fusion MLP model that takes a single concatenated feature vector,
        and passes it through an MLP classifier.
        
        Args:
            input_dim: Dimension of input features (default: 1280)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            dropout: Dropout probability (default: 0.3)
            verbose: Whether the constructor should output verbal execution tracing (default: True)
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build MLP layers
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer (two nodes for binary classification with CrossEntropyLoss)
        layers.append(nn.Linear(in_dim, 2))
        
        self.mlp = nn.Sequential(*layers)

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
            print(f"MLP architecture: {self.input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 2 (binary classification)")
            print(f"Device: {self.device}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, 2)
        """
        # Pass through MLP
        forward_output = self.mlp(features)

        return forward_output

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Get class predictions (0 or 1)."""
        output = self.forward(features)
        return torch.argmax(output, dim=1)
    
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        output = self.forward(features)
        probs = torch.softmax(output, dim=1)
        return probs
    
    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor,
            epochs: int = 50, batch_size: int = 32, class_weight: float = 2.0, 
            lr: float = 1e-3, weight_decay: float = 0, verbose: bool = True):
        """
        Fit the model on training data.
        
        Args:
            train_features: Training features (N, input_dim)
            train_labels: Training labels (N,)
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size (default: 32)
            class_weight: Weight for class 0 (controls) to handle imbalance (default: 2)
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

        class_weights = torch.FloatTensor([class_weight, 1.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
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
                loss = criterion(logits, lbl)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(lbl)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
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
        
        # Move tensors to the same device as the model
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)
        
        with torch.no_grad():
            output = self.forward(test_features)
            y_true  = test_labels.cpu().numpy()
            y_pred  = self.predict(test_features).cpu().numpy()
            y_score = self.predict_proba(test_features).cpu().numpy()

        result_metrics = {
            "test_loss": nn.CrossEntropyLoss()(output, test_labels).item(),
            "test_acc": accuracy_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_conf_mat": confusion_matrix(y_true, y_pred),
            "test_roc_auc": roc_auc_score(y_true, torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()),
            "test_roc_curve": roc_curve(y_true, torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        }

        return result_metrics
