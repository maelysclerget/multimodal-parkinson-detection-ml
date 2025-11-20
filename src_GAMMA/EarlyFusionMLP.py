import torch
import torch.nn as nn
from typing import Optional


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
        
        if verbose:
            print(f"Fused input dimension: {self.fused_dim}")
            print(f"MLP architecture: {self.fused_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
    
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


# Example usage
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