import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


class EarlyFusionMLP(nn.Module):
    def __init__(self, 
                 image_embed_dim: int = 512,
                 audio_embed_dim: int = 768,
                 hidden_dims: list = [512, 256],
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 verbose: bool = True):
        """
        Early fusion model that concatenates image and audio embeddings,
        then passes through an MLP classifier.
        
        Args:
            image_embed_dim: Dimension of image embeddings (default: 512)
            audio_embed_dim: Dimension of audio embeddings (default: 768)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            num_classes: Number of output classes (default: 10)
            dropout: Dropout probability (default: 0.3)
            verbose: Whether the constructor should output verbal execution tracing (default: True)
        """
        super().__init__()
        
        self.image_embed_dim = image_embed_dim
        self.audio_embed_dim = audio_embed_dim
        self.fused_dim = image_embed_dim + audio_embed_dim
        
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
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
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
            print(f"MLP architecture: {self.fused_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
            print(f"Device: {self.device}")
    
    def forward(self, image_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image_embed: Image embeddings of shape (batch, image_embed_dim)
            audio_embed: Audio embeddings of shape (batch, audio_embed_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Early fusion: concatenate embeddings
        fused = torch.cat([image_embed, audio_embed], dim=1)
        
        # Pass through MLP
        forward_output = self.mlp(fused)
        
        return forward_output

    def predict(self, image_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        output = self.forward(image_embed, audio_embed)
        return torch.argmax(output, dim=1)
    
    def predict_proba(self, image_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        output = self.forward(image_embed, audio_embed)
        return torch.softmax(output, dim=1)
    
    def fit(self, train_image_embeds: torch.Tensor, train_audio_embeds: torch.Tensor,
            train_labels: torch.Tensor, epochs: int = 50, batch_size: int = 32,
            lr: float = 1e-3, verbose: bool = True):
        """
        Fit the model on training data.
        
        Args:
            train_image_embeds: Training image embeddings (N, image_embed_dim)
            train_audio_embeds: Training audio embeddings (N, audio_embed_dim)
            train_labels: Training labels (N,)
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size (default: 32)
            lr: Learning rate (default: 1e-3)
            verbose: Whether the method should output verbal execution tracing (default: True)
        """
        # Reset history
        self.train_loss_history = []
        self.train_acc_history = []
        
        # Create data loader
        dataset = TensorDataset(train_image_embeds, train_audio_embeds, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            for img, aud, lbl in loader:
                img, aud, lbl = img.to(self.device), aud.to(self.device), lbl.to(self.device)
                
                optimizer.zero_grad()
                logits = self.forward(img, aud)
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
            
    def test(self, test_image_embeds: torch.Tensor, test_audio_embeds: torch.Tensor,
            test_labels: torch.Tensor):
        """
        Evaluate the model on test data and return a dictionary of metrics.
        
        Args:
            test_image_embeds: Image embeddings of shape (N, image_embed_dim)
            test_audio_embeds: Audio embeddings of shape (N, audio_embed_dim)
            test_labels: Ground truth labels of shape (N,)
        
        Returns:
            Dictionary containing test metrics (loss, accuracy, F1, confusion matrix, ROC AUC, ROC curve)
        """

        self.eval()
        with torch.no_grad():
            output = self.forward(test_image_embeds, test_audio_embeds)
            y_true  = test_labels.numpy()
            y_pred  = self.predict(test_image_embeds, test_audio_embeds).numpy()
            y_score = self.predict_proba(test_image_embeds, test_audio_embeds).numpy()

        result_metrics = {
            "test_loss": nn.CrossEntropyLoss()(output, y_true).item(),
            "test_acc": accuracy_score(y_true, y_pred),
            "test_f1": f1_score(y_true, y_pred),
            "test_conf_mat": confusion_matrix(y_true, y_pred),
            "test_roc_auc": roc_auc_score(y_true, y_score[:, 1]),
            "test_roc_curve": roc_curve(y_true, y_score[:, 1])
        }

        return result_metrics


if __name__ == "__main__":
    # Initialize model
    model = EarlyFusionMLP(
        image_embed_dim=512,
        audio_embed_dim=768,
        hidden_dims=[512, 256],
        num_classes=10,
        dropout=0.3
    )
    
    # Example batch of embeddings
    batch_size = 8
    image_embeddings = torch.randn(batch_size, 512)
    audio_embeddings = torch.randn(batch_size, 768)
    
    # Forward pass
    logits = model(image_embeddings, audio_embeddings)
    print(f"\nInput shapes: image={image_embeddings.shape}, audio={audio_embeddings.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Get predictions
    predictions = model.predict(image_embeddings, audio_embeddings)
    print(f"Predictions shape: {predictions.shape}")
    
    # Example training loop
    print("\n--- Example Training Loop ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    image_embeddings = torch.randn(batch_size, 512).to(device)
    audio_embeddings = torch.randn(batch_size, 768).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    logits = model(image_embeddings, audio_embeddings)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")