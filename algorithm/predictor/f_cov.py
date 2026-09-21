import torch
import torch.nn as nn
import torch.nn.functional as F

class f_cov(nn.Module):
    def __init__(self, h_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*h_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),          # -> [s, log_prec]
        )

    def forward(self, h_pred, h_hist, elapsed):
        if h_pred.dim() == 1: h_pred = h_pred.unsqueeze(0)
        if h_hist.dim() == 1: h_hist = h_hist.unsqueeze(0)
        elapsed = elapsed.reshape(-1, 1)
        x = torch.cat([h_pred.expand_as(h_hist), h_hist, elapsed], dim=1)
        out = self.net(x)
        s = out[:, 0]           # unbounded scale (ratio)
        l = out[:, 1]           # log-precision (confidence); exp(l) used as weight
        return s, l.clamp(-4,4)