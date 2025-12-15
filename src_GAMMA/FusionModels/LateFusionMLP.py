import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-layer Perceptron for binary classification (PD vs Healthy).
    
    Architecture:
    - Input Layer: variable input_dim
    - Hidden Layer 1: hidden_dim_1 neurons with ReLU + Dropout
    - Hidden Layer 2: hidden_dim_2 neurons with ReLU + Dropout
    - Hidden Layer 3: 16 neurons with ReLU + Dropout(0.2)
    - Output Layer: 2 neurons (logits for CrossEntropyLoss)
    
    Args:
        input_dim (int): Number of input features
        hidden_dim_1 (int): Number of neurons in first hidden layer (default: 64)
        hidden_dim_2 (int): Number of neurons in second hidden layer (default: 32)
        dropout_rate (float): Dropout rate for first two hidden layers (default: 0.5)
    """
    def __init__(self, input_dim, hidden_dim_1=64, hidden_dim_2=32, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)  
        )

    def forward(self, x):
        """Forward pass that returns logits for two classes."""
        return self.net(x)
    
