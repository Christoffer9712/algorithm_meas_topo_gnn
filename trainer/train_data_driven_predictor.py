"""
Data-driven E2E predictor: ablation of the learned method WITHOUT topology.

Same structure, splits, normalisation, loss, batching, early stopping and
self-measurement handling as train_predictor.train(), but no graph encoder
and no fcov:

    out = f_dd(x*) + [ sum_j w_j * z_hat_j , 0 ]

    x         overlay-level geometry only: hop distances between the overlay
              waypoints (ovl['overlay_path']) from node positions data.x[:, 0:2],
              plus the total distance. The underlay graph / routing is not used.
    f_dd      MLP prior (the ftopo counterpart) -> normalised [delay, loss].
    z_hat_j   meas_j - f_dd(x_j)[0] for every overlay measurement in the
              history t, ..., t-Delta+1 (same as hat_z in train()).
    w_j       softmax attention over the history plus a learned "null" slot
              (weight on the null slot = shrink towards the prior). Scores come
              from an MLP on observable pairwise overlay features: both
              geometries, elapsed time, same-overlay flag, per-position
              waypoint equality and waypoint overlap.

Checkpoint: models/data_driven_predictor_models.pth

For evaluation scripts:

    from trainer.train_data_driven_predictor import DataDrivenInference
    dd = DataDrivenInference(model_path, samples)      # samples[t] = dataset[gidx][t]
    dd.predict(t, ovl, m=1, remove_self=True)          # raw [delay, loss] for target ovl at t+m
    dd.predict(t, ovl, m=1, include_meas=False)        # prior only
"""
import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from .dataset import PredictorDataset
from .train_predictor import calculate_mean_std, baselines

INCLUDE_MEAS = True
REMOVE_SELF_MEAS = True
REMOVE_SELF_MEAS_PERC = 0.5

N_HOPS = 3                  # hop distances kept per overlay (padded / truncated)
N_WP = N_HOPS + 1           # overlay waypoints used in the pairwise equality features
FEAT_DIM = N_HOPS + 1       # hop distances + total distance
PAIR_DIM = 3 * FEAT_DIM + N_WP + 3


# ===========================================================================
# Model
# ===========================================================================
class Predictor(nn.Module):
    """MLP prior: overlay geometry features -> normalised (delay, loss)."""

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
        return self.net(H)


class DataDrivenE2E(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, pair_dim=PAIR_DIM, hidden_dim=128,
                 score_dim=64, dropout=0.1, delta=3):
        super().__init__()
        self.delta = delta
        self.prior = Predictor(feat_dim, hidden_dim, dropout)
        self.score = nn.Sequential(
            nn.Linear(pair_dim, score_dim),
            nn.ReLU(),
            nn.Linear(score_dim, score_dim),
            nn.ReLU(),
            nn.Linear(score_dim, 1),
        )
        self.null_logit = nn.Parameter(torch.zeros(()))
        # feature standardisation, set from training data before training
        self.register_buffer('x_mean', torch.zeros(feat_dim))
        self.register_buffer('x_std', torch.ones(feat_dim))

    def norm(self, x):
        return (x - self.x_mean) / (self.x_std + 1e-8)

    def pair_features(self, xa, Wa, IDa, xh, Wh, IDh, E, m):
        """(A, K, PAIR_DIM) features between targets a and history rows j."""
        A, K, Fd = xa.size(0), xh.size(0), xa.size(1)
        ea = xa[:, None, :].expand(A, K, Fd)
        eh = xh[None, :, :].expand(A, K, Fd)
        el = ((E + m) / self.delta)[None, :, None].expand(A, K, 1)
        same_id = (IDa[:, None] == IDh[None, :]).float()[..., None]
        va, vh = Wa >= 0, Wh >= 0
        pos_eq = ((Wa[:, None, :] == Wh[None, :, :]) & va[:, None, :]).float()
        shared = ((Wa[:, None, :, None] == Wh[None, :, None, :])
                  & va[:, None, :, None] & vh[None, :, None, :]).any(-1).float()
        shared = shared.sum(-1, keepdim=True) / Wa.size(1)
        return torch.cat([ea, eh, (ea - eh).abs(), el, same_id, pos_eq, shared], dim=-1)

    def forward(self, xa, Wa, IDa, hist=None, m=1, mask=None):
        """Normalised [delay, loss] for A targets.

        xa (A, F) raw features, Wa (A, N_WP) waypoint ids, IDa (A,) overlay ids.
        hist = (xh, meas_h, Wh, IDh, E) for K history rows, meas_h normalised.
        mask (A, K) bool, True = measurement may be used. hist=None -> prior only.
        """
        xa_n = self.norm(xa)
        out = self.prior(xa_n)                                   # (A, 2)
        if hist is None or hist[0].size(0) == 0:
            return out
        xh, meas_h, Wh, IDh, E = hist
        xh_n = self.norm(xh)
        z = meas_h[:, 0] - self.prior(xh_n)[:, 0]                # (K,) neighbour deviation
        s = self.score(self.pair_features(xa_n, Wa, IDa, xh_n, Wh, IDh, E, m)).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, float('-inf'))
        s = torch.cat([s, self.null_logit.expand(s.size(0), 1)], dim=1)
        w = torch.softmax(s, dim=1)[:, :-1]                      # (A, K)
        corr = w @ z                                             # (A,) delay correction only
        return out + torch.stack([corr, torch.zeros_like(corr)], dim=1)


# ===========================================================================
# Static per-step inputs
# ===========================================================================
def overlay_dist(step, overlay):
    """Hop distances between consecutive overlay waypoints (data.x[:, 0:2] positions)."""
    x = torch.as_tensor(step['x'], dtype=torch.float)
    n2i = step['name_to_idx']
    return [float(torch.norm(x[n2i[a]][0:2] - x[n2i[b]][0:2]))
            for a, b in zip(overlay[:-1], overlay[1:])]


def overlay_features(step, ovl):
    d = overlay_dist(step, ovl['overlay_path'])
    hops = (d + [0.0] * N_HOPS)[:N_HOPS]
    return hops + [sum(d)]


class SimFeatures:
    """Precomputed model inputs for every time step of one simulation.

    steps[t] = dataset[gidx][t]. norm = (mean_tot, std_tot, mean_loss, std_loss).
    """

    def __init__(self, steps, norm, device='cpu'):
        mean_tot, std_tot, mean_loss, std_loss = norm
        self.device = device
        self.steps = [steps[t] for t in range(len(steps))]
        self.name_ids, self.ovl_ids = {}, {}
        self.X, self.MEAS, self.W, self.ID, self.ids = [], [], [], [], []
        for step in self.steps:
            ovls = step['overlay_paths']
            X = [overlay_features(step, o) for o in ovls]
            MEAS = [[(float(o['meas'][0]) - mean_tot) / (std_tot + 1e-8),
                     (float(o['meas'][1]) - mean_loss) / (std_loss + 1e-8)] for o in ovls]
            W = [([self.name_ids.setdefault(n, len(self.name_ids)) for n in o['overlay_path']]
                  + [-1] * N_WP)[:N_WP] for o in ovls]
            ID = [self.ovl_ids.setdefault(o['id'], len(self.ovl_ids)) for o in ovls]
            self.X.append(torch.tensor(X, dtype=torch.float, device=device).reshape(len(ovls), FEAT_DIM))
            self.MEAS.append(torch.tensor(MEAS, dtype=torch.float, device=device).reshape(len(ovls), 2))
            self.W.append(torch.tensor(W, dtype=torch.long, device=device).reshape(len(ovls), N_WP))
            self.ID.append(torch.tensor(ID, dtype=torch.long, device=device))
            self.ids.append([o['id'] for o in ovls])
        self._hist = {}

    def history(self, t, delta):
        """(xh, meas_h, Wh, IDh, E) over t, ..., t-delta+1, plus row keys (id, time)."""
        if (t, delta) not in self._hist:
            rows = [i for i in range(delta) if t - i >= 0]
            hist = (torch.cat([self.X[t - i] for i in rows]),
                    torch.cat([self.MEAS[t - i] for i in rows]),
                    torch.cat([self.W[t - i] for i in rows]),
                    torch.cat([self.ID[t - i] for i in rows]),
                    torch.cat([torch.full((len(self.ids[t - i]),), float(i), device=self.device)
                               for i in rows]))
            keys = [(oid, t - i) for i in rows for oid in self.ids[t - i]]
            self._hist[(t, delta)] = (hist, keys)
        return self._hist[(t, delta)]


def self_mask(target_ids, keys, t, delta, perc, rng=random):
    """(A, K) mask that drops each of a target's own history rows with probability perc."""
    key_idx = {k: j for j, k in enumerate(keys)}
    mask = torch.ones(len(target_ids), len(keys), dtype=torch.bool)
    for a, oid in enumerate(target_ids):
        for d in range(delta):
            j = key_idx.get((oid, t - d))
            if j is not None and (perc >= 1.0 or rng.random() < perc):
                mask[a, j] = False
    return mask


def step_forward(model, sf, t, m, delta, include_meas, remove_self_perc, rng=random):
    """Predictions + normalised targets for all eligible overlays of step t + m."""
    curr = set(sf.ids[t])
    tgt_idx = [k for k, oid in enumerate(sf.ids[t + m]) if oid in curr]
    if not tgt_idx:
        return None, None, []
    idx = torch.tensor(tgt_idx, device=sf.device)
    xa, Wa, IDa = sf.X[t + m][idx], sf.W[t + m][idx], sf.ID[t + m][idx]
    target_ids = [sf.ids[t + m][k] for k in tgt_idx]
    if include_meas:
        hist, keys = sf.history(t, delta)
        mask = self_mask(target_ids, keys, t, delta, remove_self_perc, rng).to(sf.device)
        out = model(xa, Wa, IDa, hist=hist, m=m, mask=mask)
    else:
        out = model(xa, Wa, IDa)
    return out, sf.MEAS[t + m][idx], target_ids


# ===========================================================================
# Training (mirrors train_predictor.train)
# ===========================================================================
def train(dataset_path=None, model_dir=None, epochs=50, batch_size=16, lr=0.5e-3,
          device='cpu', min_delta=1e-4, patience=15):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    dataset = PredictorDataset(dataset_path)
    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 5.0

    nbr_sims = dataset.get_nbr_of_sims()

    # Prediction horizon: M
    M = 1

    # Measurement history depth: Delta
    Delta = 3

    # Contiguous time-based split (identical to train_predictor.train)
    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    gap = M
    train_t = all_t[int(0.10*n)+gap:int(0.30*n)] + all_t[int(0.40*n)+gap:int(0.70*n)] + all_t[int(0.80*n)+gap:]
    val_t   = all_t[:int(0.10*n)-gap] + all_t[int(0.30*n)+gap:int(0.40*n)-gap] + all_t[int(0.70*n)+gap:int(0.80*n)-gap]
    test_t  = all_t # Used on unseen topology (last sim)
    (mean_delay, mean_loss, std_dev_delay, std_dev_delay_tot, std_dev_loss) = calculate_mean_std(train_t, dataset[:-1], nbr_sims-1)

    print(f"Baseline (predict-mean) train = {baselines(train_t, dataset[:-1], nbr_sims-1, mean_delay.sum(), std_dev_delay_tot, c_delta)}")
    print(f"Baseline (predict-mean) val   = {baselines(val_t, dataset[:-1], nbr_sims-1, mean_delay.sum(), std_dev_delay_tot, c_delta)}")
    print(f"Baseline (predict-mean) test  = {baselines(test_t, [dataset[-1]], 1, mean_delay.sum(), std_dev_delay_tot, c_delta)}")

    norm = (float(mean_delay.sum()), float(std_dev_delay_tot), float(mean_loss), float(std_dev_loss))
    sims = [SimFeatures([dataset[g][t] for t in range(len(dataset))], norm, device)
            for g in range(nbr_sims)]
    train_sims = sims[:-1]

    model = DataDrivenE2E(delta=Delta).to(device)
    X_train = torch.cat([sf.X[t] for sf in train_sims for t in train_t])
    model.x_mean.copy_(X_train.mean(0))
    model.x_std.copy_(X_train.std(0))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    def run_t(t, sf):
        loss = torch.zeros((), device=device)
        count = 0
        for m in range(1, M + 1):
            perc = REMOVE_SELF_MEAS_PERC if REMOVE_SELF_MEAS else 0.0
            out, tgt, _ = step_forward(model, sf, t, m, Delta, INCLUDE_MEAS, perc)
            if out is None:
                continue
            z0 = out[:, 0] - tgt[:, 0]
            z1 = out[:, 1] - tgt[:, 1]
            loss = loss + (c_delta * z0 ** 2 + c_lambda * z1 ** 2).sum()
            count += out.size(0)
        return loss, count

    batch_size = batch_size/(nbr_sims-1) # For each step, train over all sims

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for ep in range(epochs):
        # ---- train ----
        model.train()
        train_loss = 0.0
        batch_sample = 1
        loss_batch = torch.zeros((), device=device)
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            for sf in train_sims:
                loss_g, count_g = run_t(t, sf)
                loss = loss + loss_g
                count = count + count_g

            step_loss = loss / count if count > 0 else torch.zeros((), device=device)
            loss_batch = loss_batch + step_loss
            train_loss += float(step_loss.detach())

            if batch_sample >= batch_size:
                opt.zero_grad()
                if loss_batch.requires_grad:
                    loss_batch.backward()
                    opt.step()
                batch_sample = 1
                loss_batch = torch.zeros((), device=device)
            else:
                batch_sample = batch_sample + 1

        train_loss = train_loss / (len(train_t))

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t in val_t:
                loss = torch.zeros((), device=device); count = 0
                for sf in train_sims:
                    loss_g, count_g = run_t(t, sf)
                    loss = loss + loss_g
                    count = count + count_g
                val_loss += float((loss / count).detach()) if count > 0 else 0.0
            val_loss = val_loss / (len(val_t))

        scheduler.step(val_loss)
        # ---- early stopping ----
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best val_loss = {best_val:.6f})")
                break

        msg = f"""
              Epoch: {ep}
              train_loss     = {train_loss}
              val_loss       = {val_loss}
              null logit     = {model.null_logit.item():.3f}
              lr             = {opt.param_groups[0]["lr"]}
              """
        print(msg)

    # restore best weights before test + save
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test on the held-out simulation, all three measurement settings ----
    model.eval()
    for name, include_meas, perc in (("with self-meas", True, 0.0),
                                     ("without self-meas", True, 1.0),
                                     ("no meas", False, 0.0)):
        test_loss = 0.0
        with torch.no_grad():
            for t in test_t:
                out, tgt, _ = step_forward(model, sims[-1], t, 1, Delta, include_meas, perc)
                if out is None:
                    continue
                test_loss += float((c_delta * (out[:, 0] - tgt[:, 0]) ** 2
                                    + c_lambda * (out[:, 1] - tgt[:, 1]) ** 2).mean())
        print(f"Test loss ({name:17s}) = {test_loss / len(test_t):.6f}")

    # Save models
    path = os.path.join(model_dir, 'data_driven_predictor_models.pth')
    torch.save({'model_state': model.state_dict(),
                'config': {'delta': Delta, 'N_HOPS': N_HOPS},
                'std_dev_delay_tot': std_dev_delay_tot,
                'mean_delay': mean_delay, 'mean_loss': mean_loss,
                'std_dev_delay': std_dev_delay, 'std_dev_loss': std_dev_loss}, path)
    print('Saved trained models to', path)
    return path


# ===========================================================================
# Inference for evaluation / visualisation scripts
# ===========================================================================
class DataDrivenInference:
    """Trained data-driven predictor on one simulation, in raw measurement units.

    samples: steps of one simulation, samples[t] = dataset[gidx][t].
    """

    def __init__(self, model_path, samples, device='cpu'):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.delta = ckpt['config']['delta']
        if ckpt['config']['N_HOPS'] != N_HOPS:
            raise ValueError("checkpoint was trained with a different N_HOPS")
        self.mean_tot = float(np.sum(ckpt['mean_delay']))
        self.std_tot = float(ckpt['std_dev_delay_tot'])
        self.mean_loss = float(ckpt['mean_loss'])
        self.std_loss = float(ckpt['std_dev_loss'])
        self.model = DataDrivenE2E(delta=self.delta).to(device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self.sf = SimFeatures(samples, (self.mean_tot, self.std_tot, self.mean_loss, self.std_loss), device)

    def denorm(self, out):
        out = out.detach().cpu().numpy().copy()
        out[..., 0] = out[..., 0] * (self.std_tot + 1e-8) + self.mean_tot
        out[..., 1] = out[..., 1] * (self.std_loss + 1e-8) + self.mean_loss
        return out

    @torch.no_grad()
    def predict(self, t, ovl, m=1, remove_self=True, include_meas=True):
        """Raw [delay, loss] for target overlay `ovl` (as in samples[t+m]) from history up to t."""
        sf = self.sf
        k = sf.ids[t + m].index(ovl['id'])
        xa, Wa, IDa = sf.X[t + m][k:k + 1], sf.W[t + m][k:k + 1], sf.ID[t + m][k:k + 1]
        if include_meas:
            hist, keys = sf.history(t, self.delta)
            mask = self_mask([ovl['id']], keys, t, self.delta, 1.0 if remove_self else 0.0)
            out = self.model(xa, Wa, IDa, hist=hist, m=m, mask=mask.to(sf.device))
        else:
            out = self.model(xa, Wa, IDa)
        return self.denorm(out[0])


if __name__ == '__main__':
    train(epochs=30, batch_size=32)