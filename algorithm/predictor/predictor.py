import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    """
    MLP mapping H = [h || g || horizon_m] -> predicted (lambda, delay_ms).
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        elif num_layers == 2:
            layers = []
            # Input layer
            layers.append(nn.Linear(in_dim, 2))
            self.net = nn.Sequential(*layers)
        else:
            layers = []
            # Input layer
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout)) 
            # Additional hidden layers 
            for _ in range(num_layers - 2): 
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout)) 

            # Output layer 
            layers.append(nn.Linear(hidden_dim, 2))
            self.net = nn.Sequential(*layers)

    def forward(self, H):
        """H: Tensor [D] or [batch, D] -> returns (lambda, delay_ms)"""
        out = self.net(H)
        return out
