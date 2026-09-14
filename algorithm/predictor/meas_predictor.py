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

    def __init__(self, h_dim):
        super().__init__()
        self.proj = nn.Linear(h_dim, h_dim)
        self.log_temp = nn.Parameter(torch.zeros(()))   # temp = exp(0) = 1 initially
        self.log_decay = nn.Parameter(torch.zeros(()))   # temp = exp(0) = 1 initially

    def get_alpha(self, h_pred, h_hists, elapsed, self_mask=None):
        if h_pred.dim() == 1:
            h_pred = h_pred.unsqueeze(0)
        elapsed = elapsed.reshape(-1)

        # learnable projection, then cosine similarity in the projected space
        q = self.proj(h_pred)                   # [1, hidden]
        k = self.proj(h_hists)                  # [N, hidden]

        sim = F.cosine_similarity(q, k, dim=1)    # [N] in [-1, 1]
        temp = self.log_temp.exp()                # learnable, positive

        logits = sim / temp - self.log_decay.exp()*elapsed       # [N]
        if self_mask is not None:
            logits = logits.masked_fill(self_mask, float('-inf'))
        alpha = torch.softmax(logits, dim=0)      # [N]

        return alpha

    def reconstruct_loo(self, h_pred, h_hists, meas_vals_std, elapsed, self_mask):
        """
        Leave-one-out
        self_mask: bool [N], True where the history entry belongs to the query path
                (these get -inf logits, i.e. excluded).
        Returns predicted standardized [2] using only OTHER paths.
        """
        alpha = self.get_alpha(h_pred, h_hists, elapsed, self_mask)
        return (alpha.unsqueeze(1) * meas_vals_std).sum(dim=0)

    def forward(self, h_pred, h_hists, meas_vals, elapsed):
        """
        h_pred:   [D] or [1, D]
        h_hists:  [N, D]
        meas_vals:[N, 2]  standardized (delay, loss)
        elapsed:  [N] or [N, 1]
        Returns:  g [2]   attention-weighted raw standardized measurement
        """
        if meas_vals.dim() == 1:
            meas_vals = meas_vals.unsqueeze(0)    # [1, 2]

        alpha = self.get_alpha(h_pred, h_hists, elapsed)

        g = (alpha.unsqueeze(1) * meas_vals).sum(dim=0)   # [2]
        return g