import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


class IntermediateFusionMLP(nn.Module):
    def __init__(self, 
                 feature_1_dim: int = 512,
                 feature_2_dim: int = 768,
                 hidden_dims: list = [512, 256],
                 dropout: float = 0.3,
                 verbose: bool = True):
        """
        Early fusion model that concatenates two feature embeddings,
        then passes through an MLP classifier.
        
        Args:
            feature_1_dim: Dimension of first feature embeddings (default: 512)
            feature_2_dim: Dimension of second feature embeddings (default: 768)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            dropout: Dropout probability (default: 0.3)
            verbose: Whether the constructor should output verbal execution tracing (default: True)
        """
        super().__init__()
        
        self.feature_1_dim = feature_1_dim
        self.feature_2_dim = feature_2_dim
        self.fused_dim = feature_1_dim + feature_2_dim
        
        # Build MLP layers
        layers = []
        in_dim = self.fused_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer (single node for binary classification)
        layers.append(nn.Linear(in_dim, 1))
        
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
            print(f"Fused input dimension: {self.fused_dim}")
            print(f"MLP architecture: {self.fused_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1 (binary classification)")
            print(f"Device: {self.device}")
    
    def forward(self, feature_1: torch.Tensor, feature_2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feature_1: First feature embeddings of shape (batch, feature_1_dim)
            feature_2: Second feature embeddings of shape (batch, feature_2_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Early fusion: concatenate embeddings
        fused = torch.cat([feature_1, feature_2], dim=1)
        
        # Pass through MLP
        forward_output = self.mlp(fused)
        
        return forward_output

    def predict(self, feature_1: torch.Tensor, feature_2: torch.Tensor) -> torch.Tensor:
        """Get class predictions (0 or 1)."""
        output = self.forward(feature_1, feature_2)
        return (output > 0).squeeze().long()
    
    def predict_proba(self, feature_1: torch.Tensor, feature_2: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        output = self.forward(feature_1, feature_2)
        probs = torch.sigmoid(output)
        return torch.cat([1 - probs, probs], dim=1)
    
    def fit(self, train_feature_1: torch.Tensor, train_feature_2: torch.Tensor,
            train_labels: torch.Tensor, epochs: int = 50, batch_size: int = 32,
            lr: float = 1e-3, weight_decay: float = 0, verbose: bool = True):
        """
        Fit the model on training data.
        
        Args:
            train_feature_1: Training first feature embeddings (N, feature_1_dim)
            train_feature_2: Training second feature embeddings (N, feature_2_dim)
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
        dataset = TensorDataset(train_feature_1, train_feature_2, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            for feat_1, feat_2, lbl in loader:
                feat_1, feat_2, lbl = feat_1.to(self.device), feat_2.to(self.device), lbl.to(self.device)
                
                optimizer.zero_grad()
                logits = self.forward(feat_1, feat_2)
                loss = criterion(logits, lbl.float().unsqueeze(1))
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
            
    def test(self, test_feature_1: torch.Tensor, test_feature_2: torch.Tensor,
            test_labels: torch.Tensor):
        """
        Evaluate the model on test data and return a dictionary of metrics.
        
        Args:
            test_feature_1: First feature embeddings of shape (N, feature_1_dim)
            test_feature_2: Second feature embeddings of shape (N, feature_2_dim)
            test_labels: Ground truth labels of shape (N,)
        
        Returns:
            Dictionary containing test metrics (loss, accuracy, F1, confusion matrix, ROC AUC, ROC curve)
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(test_feature_1, test_feature_2)
            y_true  = test_labels.numpy()
            y_pred  = self.predict(test_feature_1, test_feature_2).numpy()
            y_score = self.predict_proba(test_feature_1, test_feature_2).numpy()

        result_metrics = {
            "test_loss": nn.BCEWithLogitsLoss()(output, test_labels.float().unsqueeze(1)).item(),
            "test_acc": accuracy_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_conf_mat": confusion_matrix(y_true, y_pred),
            "test_roc_auc": roc_auc_score(y_true, torch.sigmoid(output).detach().cpu().numpy()),
            "test_roc_curve": roc_curve(y_true, torch.sigmoid(output).detach().cpu().numpy())
        }

        return result_metrics


