import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v):
    v = torch.tensor(float(v))
    return v + torch.log(-torch.expm1(-v))         # stable inverse of softplus, also for large v


class f_cov(nn.Module):
    """
    Deep kernel, PSD by construction, non-negative entries:

      k(a, b) = [ psi(h_a)^T psi(h_b) + k_id(a, b) ] * exp(-|t_a - t_b| / ell_t),   psi = softplus(phi(.)) >= 0

    psi(h) acts as a learned soft node-membership vector, so
      - psi_a^T psi_b >= 0           (overlays only covary through shared nodes)
      - k(a, a) = ||psi_a||^2        (variance grows with path length / load)
      - Gram (PSD) * OU kernel (PSD) is PSD by the Schur product theorem.

    k_id adds the observable overlay identity (optional, used when keys are given):
      k_id(a, b) = alpha_id  * 1[same overlay id]
                 + sum_l alpha_l * 1[same waypoint at position l]        (AC, entry SAT, GW, TGT)
                 + alpha_ov * |waypoints(a) ∩ waypoints(b)| / n_wp
    Each term is a Gram matrix of one-hot / count vectors (PSD) with a weight alpha >= 0,
    so the sum stays PSD with non-negative entries.

    The Gram part has rank <= feat_dim, so a learned nugget keeps C invertible.

    fcov(H_hist, elapsed, keys_hist=K_hist)                                -> C, shape (M, M)   (includes nugget)
    fcov(H_hist, elapsed, h_pred=h_pred, keys_hist=K_hist, keys_pred=k*)  -> g, shape (M,)
    fcov.prior_var(h_pred, keys_pred=k*)                                  -> k(h*, h*, 0)

    keys: long tensor (M, 1 + n_wp) = [overlay id, waypoint ids ...], -1 = no waypoint.
    Without keys the identity part is left out (previous behaviour).
    """

    def __init__(self, in_dim, feat_dim=16, hidden_dim=64, dropout=0.0,
                 ell_t_init=100.0, nugget_init=1e-2, n_wp=4, id_weight_init=0.1):
        super().__init__()
        self.phi = nn.Sequential(                      # learned feature map
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
        )
        # unconstrained params -> softplus makes them positive
        self.raw_ell_t = nn.Parameter(_inv_softplus(ell_t_init))    # time length-scale (O-U: 1/theta)
        self.raw_nugget = nn.Parameter(_inv_softplus(nugget_init))  # diagonal noise, normalised units
        # identity weights: [same overlay id, same waypoint at position 1..n_wp, waypoint overlap]
        self.n_wp = n_wp
        self.raw_alpha = nn.Parameter(torch.full((n_wp + 2,), float(_inv_softplus(id_weight_init))))

    def psi(self, h):
        return F.softplus(self.phi(h))                 # (M, D), non-negative features

    def ell_t(self):
        return F.softplus(self.raw_ell_t) + 1e-3

    def nugget(self):
        return F.softplus(self.raw_nugget) + 1e-6

    def alpha(self):
        return F.softplus(self.raw_alpha)              # (n_wp + 2,), >= 0

    def _k_id(self, k_a, k_b, dtype):
        # k_a: (Ma, 1 + n_wp), k_b: (Mb, 1 + n_wp) long -> (Ma, Mb)
        a = self.alpha().to(dtype)
        same_id = (k_a[:, None, 0] == k_b[None, :, 0]).to(dtype)
        w_a, w_b = k_a[:, 1:], k_b[:, 1:]
        v_a, v_b = w_a >= 0, w_b >= 0
        pos_eq = ((w_a[:, None, :] == w_b[None, :, :]) & v_a[:, None, :] & v_b[None, :, :]).to(dtype)
        overlap = ((w_a[:, None, :, None] == w_b[None, :, None, :])
                   & v_a[:, None, :, None] & v_b[None, :, None, :]).to(dtype).sum((-1, -2)) / self.n_wp
        return a[0] * same_id + (pos_eq * a[1:1 + self.n_wp]).sum(-1) + a[-1] * overlap

    def _k(self, p_a, p_b, lag, k_a=None, k_b=None, include_phi=True):
        # p_a: (Ma, D), p_b: (Mb, D), lag: (Ma, Mb) -> (Ma, Mb)
        K = p_a @ p_b.T
        K = 0*K if not include_phi else K
        if k_a is not None and k_b is not None:
            K = K + self._k_id(k_a.to(p_a.device), k_b.to(p_a.device), p_a.dtype)
        return K * torch.exp(-lag / self.ell_t())

    def forward(self, h_hist, elapsed, h_pred=None, keys_hist=None, keys_pred=None, include_phi=True):
        if h_hist.dim() == 1:
            h_hist = h_hist.unsqueeze(0)                          # (M, N)
        h_hist = h_hist.to(self.raw_ell_t.dtype)
        elapsed = elapsed.reshape(-1).to(h_hist)                  # (M,)
        M = h_hist.shape[0]
        assert elapsed.numel() == M, "need one elapsed value per history row"

        p_hist = self.psi(h_hist)                                 # (M, D)

        if h_pred is None:
            lag = (elapsed[:, None] - elapsed[None, :]).abs()     # |t_i - t_j|
            C = self._k(p_hist, p_hist, lag, keys_hist, keys_hist, include_phi=include_phi)  # PSD, entries >= 0
            return C + self.nugget() * torch.eye(M, device=h_hist.device, dtype=h_hist.dtype)

        p_pred = self.psi(h_pred.reshape(1, -1).to(h_hist))       # (1, D)
        k_pred = None if keys_pred is None else keys_pred.reshape(1, -1)
        return self._k(p_pred, p_hist, elapsed[None, :], k_pred, keys_hist, include_phi=include_phi).squeeze(0)  # (M,), lag = t* - t_j

    def prior_var(self, h_pred, keys_pred=None):
        p = self.psi(h_pred.reshape(1, -1).to(self.raw_ell_t.dtype))
        v = (p * p).sum()
        if keys_pred is not None:
            k = keys_pred.reshape(1, -1)
            v = v + self._k_id(k, k, p.dtype).squeeze()
        return v