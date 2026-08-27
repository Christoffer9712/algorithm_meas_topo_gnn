import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    """
    MLP mapping H = [h || g || horizon_m] -> predicted (lambda, delay_ms).
    """

    def __init__(self, in_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, H):
        """H: Tensor [D] or [batch, D] -> returns (lambda, delay_ms)"""
        out = self.net(H)
        return out
