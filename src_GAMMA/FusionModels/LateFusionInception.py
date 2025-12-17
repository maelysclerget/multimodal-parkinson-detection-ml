from torchvision.models import Inception_V3_Weights
from torchvision import models, transforms
import torch
import torch.nn as nn

class InceptionV3Classifier(nn.Module):
    def __init__(self, dropout=0.1, num_classes=2):
        super().__init__()
        self.backbone = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        # Freeze all backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        
    def forward(self, x):
        out = self.backbone(x)

        # Handle different torchvision behaviors
        if isinstance(out, tuple):
            x = out[0]               # (logits, aux_logits)
        elif hasattr(out, "logits"):
            x = out.logits           # InceptionOutputs(logits=..., aux_logits=...)
        else:
            x = out                  # fallback, shouldn't really happen

        # Safety: ensure (N, C) before BatchNorm1d
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x