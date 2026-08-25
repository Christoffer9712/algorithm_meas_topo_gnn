import torch
import torch.nn as nn
import torch.nn.functional as F


class MeasurementEmbedder(nn.Module):
    """
    Embed historical measurements relative to a predicted topology embedding h_pred.

    The API mirrors the LaTeX description: given h_pred (tensor) and a set of
    historical pairs (h_hist, lambda, delta, elapsed), produce a single vector
    g representing the measurement-informed embedding.
    """

    def __init__(self, h_dim, hidden_dim=64, out_dim=64):
        super().__init__()
        # e_ij = MLP(h_pred, h_hist, elapsed)
        self.e_mlp = nn.Sequential(
            nn.Linear(h_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ) # Attention scorer
        
        # MLP to embed (lambda, delta)
        self.y_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h_pred, h_hists, meas_vals, elapsed):
        """
        h_pred: Tensor [D] or [batch, D]
        h_hists: Tensor [N, D]
        meas_vals: Tensor [N, 2] (lambda, delta)
        elapsed: Tensor [N, 1] (time since measurement)

        Returns g: Tensor [out_dim]
        """
        # Ensure 2D
        if h_pred.dim() == 1:
            h_pred = h_pred.unsqueeze(0)  # [1, D]
        D = h_pred.shape[-1]
        # Broadcast h_pred to match history length
        if h_hists.dim() == 1:
            h_hists = h_hists.unsqueeze(0)
        if meas_vals.dim() == 1:
            meas_vals = meas_vals.unsqueeze(0)
        if elapsed.dim() == 1:
            elapsed = elapsed.unsqueeze(1)

        # Concatenate h_pred, h_hist and elapsed to compute attention logits
        N = h_hists.shape[0]
        # Repeat h_pred N times
        h_repeat = h_pred.expand(N, D)
        e_in = torch.cat([h_repeat, h_hists, elapsed], dim=1)  # [N, 2D+1]
        logits = self.e_mlp(e_in).squeeze(1)  # [N]
        alpha = torch.softmax(logits, dim=0)  # attention over history

        # Embed measurements
        y = self.y_mlp(meas_vals)  # [N, out_dim]
        g = (alpha.unsqueeze(1) * y).sum(dim=0)  # [out_dim]
        return g
