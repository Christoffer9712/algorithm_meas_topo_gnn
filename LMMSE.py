"""
Analytical LMMSE estimator for overlay E2E delay prediction.

No learning. Uses the true overlay-to-underlay mapping h (ovl['underlay_path'])
and the known M/M/1 + Ornstein-Uhlenbeck parameters:

    q_hat*_{k+m} = f_topo(h*) + g^T (C + R)^{-1} z_hat

    f_topo(h)  = f_prop(h) + h^T r,            r = 1 / (mu (1 - rho_bar))
    C_ij       = Sigma * F^|t_i - t_j| * h_i^T h_j
    g_j        = Sigma * F^|t* - t_j|  * h*^T h_j
    z_hat_j    = q_tot_j - f_topo(h_j)

Because it uses the true h and the true model parameters, this is an
"oracle" reference, not a competitor under the same information.

Use from another script (one simulation = one list of time steps):

    from LMMSE import LMMSEPredictor

    lmmse = LMMSEPredictor(samples)               # samples[t] = dataset[gidx][t]
    ...
    for ovl in eligible:                          # target overlays at step t + 1
        out = lmmse.predict(t, ovl, m=1, remove_self=True)
        out['pred']    # LMMSE total delay, same units as ovl['meas'][0]
        out['prior']   # f_topo only (no measurements)
        out['corr']    # measurement correction g^T (C + R)^{-1} z_hat

Run standalone (loss on the same splits / normalisation as train()):
    python -m LMMSE            # evaluate
    python -m LMMSE inspect    # print overlay/step layout
"""
import os
import sys
import math
import heapq
import random
import statistics

import numpy as np

try:
    import torch
except ImportError:          # the estimator itself only needs numpy
    torch = None

# ---------------------------------------------------------------------------
# Underlay model parameters (simulator ground truth)
# ---------------------------------------------------------------------------
MU = 4.0            # service rate per underlay node
RHO_BAR = 0.5       # O-U point of attraction
THETA = 0.006       # O-U mean-reversion rate
SIGMA = 0.02        # O-U volatility
DT = 1.0            # time between dataset steps, in O-U time units
DELAY_SCALE = 1.0   # model time unit -> measurement unit (e.g. 1000 if meas in ms)

R_VAR = 1e-4        # variance of measurement noise eps, in (measurement unit)^2

# ---------------------------------------------------------------------------
# Same settings as the training script (used by the standalone evaluation)
# ---------------------------------------------------------------------------
M = 1
DELTA = 3
C_LAMBDA = 1.0
C_DELTA = 5.0
INCLUDE_MEAS = True
REMOVE_SELF_MEAS = True
REMOVE_SELF_MEAS_PERC = 1.0
SEED = 0

# Layout of ovl['meas']
IDX_DELAY_TOT, IDX_LOSS, IDX_PROP, IDX_QUEUE = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Overlay -> underlay mapping config
# ---------------------------------------------------------------------------
UDL_PATH_KEY = 'underlay_path'     # ovl key with the underlay nodes actually traversed
OVL_WAYPOINT_KEY = 'overlay_path'  # fallback: overlay waypoints; each hop routed by shortest path
EDGE_WEIGHT_COL = 0                # edge_attr column used as routing weight in the fallback
EXCLUDE_ENDPOINTS = False          # True if the path's first/last node is not an M/M/1 queue

# ---------------------------------------------------------------------------
# Derived constants (scalar / diagonal case)
# ---------------------------------------------------------------------------
A = 1.0 / (MU * (1.0 - RHO_BAR) ** 2)              # d x_bar / d rho
R_NODE = 1.0 / (MU * (1.0 - RHO_BAR))              # r(mu, rho_bar)
F = math.exp(-THETA * DT)                          # xi_{k+1} = F xi_k + w_k
Q = A ** 2 * SIGMA ** 2 / (2 * THETA) * (1 - F ** 2)
SIGMA_XI = Q / (1 - F ** 2)                        # stationary var, Sigma = F Sigma F + Q

R_MEAS = DELAY_SCALE * R_NODE                      # per-node mean delay, meas units
SIGMA_MEAS = DELAY_SCALE ** 2 * SIGMA_XI           # per-node state var, meas units^2


# ===========================================================================
# Small helpers
# ===========================================================================
def _f(v):
    return float(v.item()) if hasattr(v, 'item') else float(v)


def _to_list(v):
    if torch is not None and torch.is_tensor(v):
        return v.detach().cpu().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


# ===========================================================================
# Overlay -> underlay mapping
# ===========================================================================
def _to_names(seq, step):
    """Node names from a sequence of names, local node indices, or (u, v) edges."""
    names = _to_list(step['node_names'])
    seq = _to_list(seq)
    if seq and isinstance(seq[0], (list, tuple)):          # edge list -> node sequence
        if not all(len(e) == 2 for e in seq) or any(
                seq[i][1] != seq[i + 1][0] for i in range(len(seq) - 1)):
            raise ValueError(f"path elements are sequences but not a chained edge list: "
                             f"{seq[:5]}... Adapt _to_names to this format.")
        seq = [seq[0][0]] + [e[1] for e in seq]
    out = []
    for v in seq:
        if isinstance(v, str):
            out.append(v)
        else:
            i = int(v)
            if not 0 <= i < len(names):
                raise IndexError(f"path element {v} is not a valid index into node_names "
                                 f"(len {len(names)}); path = {seq[:10]}")
            out.append(names[i])
    return out


def _shortest_path(step, src, dst):
    """Dijkstra over the step's underlay graph, returns node names src..dst."""
    names = _to_list(step['node_names'])
    name_to_idx = step['name_to_idx']
    ei = np.asarray(_to_list(step['edge_index']))
    w = np.asarray(_to_list(step['edge_attr']), dtype=float)
    w = w[:, EDGE_WEIGHT_COL] if w.ndim == 2 else w

    adj = {}
    for (u, v), c in zip(ei.T, w):
        adj.setdefault(int(u), []).append((int(v), float(c)))

    s, d = int(name_to_idx[src]), int(name_to_idx[dst])
    dist, prev, pq = {s: 0.0}, {}, [(0.0, s)]
    while pq:
        du, u = heapq.heappop(pq)
        if u == d:
            break
        if du > dist[u]:
            continue
        for v, c in adj.get(u, ()):
            nd = du + c
            if nd < dist.get(v, float('inf')):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if d not in dist:
        raise ValueError(f"no underlay path {src} -> {dst}")
    path = [d]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return [names[i] for i in reversed(path)]


def udl_node_names(ovl, step):
    """Ordered underlay node names traversed by overlay `ovl` at this time step."""
    if UDL_PATH_KEY in ovl:
        nodes = _to_names(ovl[UDL_PATH_KEY], step)
    elif OVL_WAYPOINT_KEY in ovl:
        wps = _to_names(ovl[OVL_WAYPOINT_KEY], step)
        nodes = [wps[0]]
        for a, b in zip(wps[:-1], wps[1:]):
            nodes += _shortest_path(step, a, b)[1:]
    else:
        raise KeyError(f"overlay has neither '{UDL_PATH_KEY}' nor '{OVL_WAYPOINT_KEY}'; "
                       f"keys are {list(ovl.keys())}. Set UDL_PATH_KEY / OVL_WAYPOINT_KEY.")
    if EXCLUDE_ENDPOINTS:
        nodes = nodes[1:-1]
    return list(dict.fromkeys(nodes))                     # dedupe, keep order


def build_node_vocab(steps):
    """Global node-name -> column index over a list of time steps (one simulation).

    Local indices in name_to_idx can change between time steps as the
    topology changes, so h vectors from different times are only comparable
    when they share one global column order.
    """
    vocab = {}
    for step in steps:
        for name in _to_list(step['node_names']):
            vocab.setdefault(name, len(vocab))
    return vocab


def ovl_to_udl(ovl, step, vocab):
    """Row h_o of H: 0/1 vector over the global node vocab of the simulation."""
    h = np.zeros(len(vocab))
    for name in udl_node_names(ovl, step):
        if name not in vocab:
            raise KeyError(f"underlay node {name!r} of overlay {ovl['id']} is not in node_names; "
                           f"sample node_names = {_to_list(step['node_names'])[:5]}")
        h[vocab[name]] = 1.0
    return h


# ===========================================================================
# LMMSE
# ===========================================================================
def lmmse_correction(H, E, Z, h_star, m, r_var=R_VAR):
    """g^T (C + R)^{-1} z_hat for one target overlay.

    H: (K, N) history rows h_j, E: (K,) elapsed steps t - t_j,
    Z: (K,) z_hat, h_star: (N,) target row, m: steps ahead of t.
    """
    lag = np.abs(E[:, None] - E[None, :])
    C = SIGMA_MEAS * F ** lag * (H @ H.T)
    g = SIGMA_MEAS * F ** (E + m) * (H @ h_star)
    # lstsq = minimum-norm solve; stays well defined when r_var = 0 and C is singular
    w = np.linalg.lstsq(C + r_var * np.eye(len(Z)), Z, rcond=None)[0]
    return float(g @ w)


class LMMSEPredictor:
    """LMMSE delay predictor for one simulation.

    steps:   sequence of time steps, steps[t] = dataset[gidx][t]
             (a list, or anything indexable by t with len()).
    delta:   measurement history depth (t, t-1, ..., t-delta+1).
    r_var:   measurement-noise variance, (measurement unit)^2.
    prop_fn: optional callable (t, ovl) -> propagation delay. Default is the
             simulator's ovl['meas'][IDX_PROP] (exact, oracle). Pass e.g. a
             geometry-based estimate to remove that advantage.
    """

    def __init__(self, steps, delta=DELTA, r_var=R_VAR, include_meas=INCLUDE_MEAS,
                 prop_fn=None):
        self.steps = [steps[t] for t in range(len(steps))]
        self.delta = delta
        self.r_var = r_var
        self.include_meas = include_meas
        self.prop_fn = prop_fn if prop_fn is not None else (
            lambda t, ovl: _f(ovl['meas'][IDX_PROP]))
        self.vocab = build_node_vocab(self.steps)
        self._h = {}
        self._hist = {}

    def h(self, t, ovl):
        """h vector of overlay `ovl` as it appears at step t (cached)."""
        key = (t, ovl['id'])
        if key not in self._h:
            self._h[key] = ovl_to_udl(ovl, self.steps[t], self.vocab)
        return self._h[key]

    def f_topo(self, t, ovl):
        """f_prop(h, G) + h^T r for overlay `ovl` at step t."""
        return self.prop_fn(t, ovl) + R_MEAS * self.h(t, ovl).sum()

    def history(self, t):
        """(H, E, Z, keys) for all overlay measurements at t, ..., t-delta+1.

        keys[j] = (overlay id, time step) of row j.
        """
        if t not in self._hist:
            H, E, Z, keys = [], [], [], []
            for i in range(self.delta):
                if t - i < 0:
                    break
                for ovl in self.steps[t - i]['overlay_paths']:
                    H.append(self.h(t - i, ovl))
                    E.append(float(i))
                    Z.append(_f(ovl['meas'][IDX_DELAY_TOT]) - self.f_topo(t - i, ovl))
                    keys.append((ovl['id'], t - i))
            H = np.array(H).reshape(len(H), len(self.vocab))
            self._hist[t] = (H, np.array(E), np.array(Z), keys)
        return self._hist[t]

    def predict(self, t, ovl, m=1, remove_self=True, exclude=None):
        """Predict the total delay of target overlay `ovl` at step t + m.

        ovl:         the overlay dict as it appears in steps[t + m]['overlay_paths'].
        t:           last time step with measurements available.
        remove_self: drop all of the target overlay's own measurements
                     (same as REMOVE_SELF_MEAS in the visualisation script).
        exclude:     extra (overlay id, time step) keys to drop.

        Returns dict with 'pred', 'prior' (f_topo only) and 'corr', in the same
        units as ovl['meas'][0].
        """
        prior = self.f_topo(t + m, ovl)
        corr = 0.0
        if self.include_meas:
            H, E, Z, keys = self.history(t)
            drop = set(exclude) if exclude else set()
            if remove_self:
                drop |= {(ovl['id'], t - d) for d in range(self.delta)}
            keep = np.array([k not in drop for k in keys], dtype=bool)
            if keep.any():
                corr = lmmse_correction(H[keep], E[keep], Z[keep],
                                        self.h(t + m, ovl), m, self.r_var)
        return {'pred': prior + corr, 'prior': prior, 'corr': corr}


def steps_of(dataset, gidx):
    """All time steps of simulation gidx as a list."""
    return [dataset[gidx][t] for t in range(len(dataset))]


# ===========================================================================
# Standalone evaluation (same splits / normalisation / loss as train())
# ===========================================================================
def calculate_mean_std(train_t, dataset, nbr_sims):
    """Copy of the training script's normalisation statistics."""
    meas_delay, meas_loss = [], []
    for gidx in range(nbr_sims):
        for t in train_t:
            for ovl in dataset[gidx][t]['overlay_paths']:
                meas = ovl['meas']
                meas_delay.append([_f(meas[IDX_PROP]), _f(meas[IDX_QUEUE])])
                meas_loss.append(_f(meas[IDX_LOSS]))
    mean_loss = statistics.mean(meas_loss)
    mean_delay = np.mean(meas_delay, axis=0)
    std_dev_loss = statistics.stdev(meas_loss)
    std_dev_delay_tot = statistics.stdev([sum(x) for x in meas_delay])
    std_dev_delay = np.std(meas_delay, axis=0)
    return mean_delay, mean_loss, std_dev_delay, std_dev_delay_tot, std_dev_loss


def evaluate(ts, predictors, stats, seed=SEED):
    """Mirror of the per-epoch loss in the training script.

    predictors: list of LMMSEPredictor, one per simulation in the split.
    """
    rng = random.Random(seed)   # own RNG per split -> reproducible self-meas mask
    mean_delay, mean_loss, _, std_tot, std_loss = stats
    keys = ("lmmse", "lmmse_delay", "prior", "prior_delay")
    tot = dict.fromkeys(keys, 0.0)

    for t in ts:
        acc = dict.fromkeys(keys, 0.0)
        count = 0
        for lm in predictors:
            hist_keys = set(lm.history(t)[3]) if INCLUDE_MEAS else set()
            curr_ids = {o['id'] for o in lm.steps[t]['overlay_paths']}
            for m in range(1, M + 1):
                for f_ovl in lm.steps[t + m]['overlay_paths']:
                    if f_ovl['id'] not in curr_ids:
                        continue
                    exclude = set()
                    if REMOVE_SELF_MEAS:
                        for d in range(DELTA):
                            k = (f_ovl['id'], t - d)
                            if k in hist_keys and rng.random() < REMOVE_SELF_MEAS_PERC:
                                exclude.add(k)
                    out = lm.predict(t, f_ovl, m, remove_self=False, exclude=exclude)

                    tgt_delay = _f(f_ovl['meas'][IDX_DELAY_TOT])
                    tgt_loss = _f(f_ovl['meas'][IDX_LOSS])
                    # Normalised errors (offset mean_delay.sum() cancels).
                    z0_lmmse = (out['pred'] - tgt_delay) / (std_tot + 1e-8)
                    z0_prior = (out['prior'] - tgt_delay) / (std_tot + 1e-8)
                    # Packet loss is not modelled: predict the training mean (0 normalised).
                    z1 = (mean_loss - tgt_loss) / (std_loss + 1e-8)

                    acc["lmmse_delay"] += C_DELTA * z0_lmmse ** 2
                    acc["prior_delay"] += C_DELTA * z0_prior ** 2
                    acc["lmmse"] += C_DELTA * z0_lmmse ** 2 + C_LAMBDA * z1 ** 2
                    acc["prior"] += C_DELTA * z0_prior ** 2 + C_LAMBDA * z1 ** 2
                    count += 1

        if count > 0:
            for k in keys:
                tot[k] += acc[k] / count

    # Same averaging as training: divide by all steps, including empty ones.
    return {k: v / len(ts) for k, v in tot.items()}


def sanity_check(ts, predictors):
    """Checks meas layout, units and path lengths against the model."""
    split_err, queue_meas, queue_model, hops = [], [], [], []
    for lm in predictors:
        for t in ts:
            for ovl in lm.steps[t]['overlay_paths']:
                meas = ovl['meas']
                h = lm.h(t, ovl)
                split_err.append(abs(_f(meas[IDX_DELAY_TOT])
                                     - _f(meas[IDX_PROP]) - _f(meas[IDX_QUEUE])))
                queue_meas.append(_f(meas[IDX_QUEUE]))
                queue_model.append(R_MEAS * h.sum())
                hops.append(h.sum())
    ratio = np.mean(queue_meas) / (np.mean(queue_model) + 1e-12)
    print(f"underlay nodes per overlay     = {np.mean(hops):.2f} (min {min(hops):.0f}, max {max(hops):.0f})")
    print(f"mean |tot - prop - queue|      = {np.mean(split_err):.4g}  (expect ~noise level)")
    print(f"mean queue meas / mean h^T r   = {ratio:.3f}  "
          f"(expect ~1, slightly above due to convexity of 1/(1-rho))")


def inspect(dataset):
    """Print the layout of one time step and one overlay."""
    step = dataset[0][0]
    print("step keys:", list(step.keys()))
    print("node_names[:5]:", _to_list(step['node_names'])[:5])
    for ovl in step['overlay_paths'][:2]:
        print("overlay:", {k: (_to_list(v)[:10] if hasattr(v, '__len__') and not isinstance(v, str) else v)
                           for k, v in ovl.items()})
        print("  -> underlay node names:", udl_node_names(ovl, step))


def main(dataset_path=None):
    from trainer.dataset import PredictorDataset

    random.seed(SEED)
    np.random.seed(SEED)

    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)

    dataset = PredictorDataset(dataset_path)
    if len(sys.argv) > 1 and sys.argv[1] == 'inspect':
        inspect(dataset)
        return
    nbr_sims = dataset.get_nbr_of_sims()

    # Identical split to the training script
    lo, hi = DELTA - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    gap = M
    train_t = (all_t[int(0.10 * n) + gap:int(0.30 * n)]
               + all_t[int(0.40 * n) + gap:int(0.70 * n)]
               + all_t[int(0.80 * n) + gap:])
    val_t = (all_t[:int(0.10 * n) - gap]
             + all_t[int(0.30 * n) + gap:int(0.40 * n) - gap]
             + all_t[int(0.70 * n) + gap:int(0.80 * n) - gap])
    test_t = all_t

    stats = calculate_mean_std(train_t, dataset, nbr_sims - 1)
    train_pred = [LMMSEPredictor(steps_of(dataset, g)) for g in range(nbr_sims - 1)]
    test_pred = [LMMSEPredictor(steps_of(dataset, nbr_sims - 1))]

    print(f"F = {F:.6f}, Q = {Q:.4e}, Sigma = {SIGMA_XI:.4e}, r = {R_NODE:.4f}")
    sanity_check(train_t, train_pred)

    for name, ts, preds in (("train", train_t, train_pred),
                            ("val", val_t, train_pred),
                            ("test", test_t, test_pred)):
        res = evaluate(ts, preds, stats)
        print(f"\n[{name}]")
        print(f"  LMMSE       loss = {res['lmmse']:.6f}   (delay term {res['lmmse_delay']:.6f})")
        print(f"  f_topo only loss = {res['prior']:.6f}   (delay term {res['prior_delay']:.6f})")


if __name__ == '__main__':
    main()