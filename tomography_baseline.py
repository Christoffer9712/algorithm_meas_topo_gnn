"""
tomography_baseline.py
======================

A *non-learned* tomography baseline for the overlay-delay prediction problem,
built to be directly comparable with the GATv2 predictor.

What this script does, in order:

  1. Loads the generated dataset and prints its structure (`--inspect`).
  2. Plots queue delay over time for a few random series, plus the empirical
     distribution against its Gaussian approximation (skew / excess kurtosis).
  3. Estimates node-level statistics mu, Sigma (=diag sigma^2), R (=diag rho)
     *from end-to-end measurements only*, via non-negative least squares.
  4. Builds the routing matrix A(t) from a SHORTEST-PATH assumption over the
     underlay graph (the "misspecified oracle" O3 - it does not peek at the
     true routing).
  5. Runs the inversion estimator in two modes - one-shot (prior reset to Sigma
     each step) and recursive (posterior carried forward, i.e. a Kalman filter).
  6. Reports the loss in the SAME standardised units as train_predictor.py,
     alongside the predict-mean and persistence baselines.

MULTI-SIM NOTE
--------------
`sim_index` accepts "all", a single int, or a list of ints.

Different sims are different NETWORKS, so the two kinds of statistic are
handled differently and this distinction matters for correctness:

  * node statistics (mu, sigma^2, rho) are fitted PER SIM - node "SAT0-1" in
    sim 3 is not the same physical queue as in sim 7, so pooling them would be
    meaningless;
  * normalisation constants (mean_total, std_total) are pooled ACROSS sims,
    because that is exactly what calculate_mean_std() in train_predictor.py
    does. Matching them is what makes the loss numbers comparable at all.

Use --ckpt to read mean/std straight out of your trained checkpoint; that
removes any doubt that the two scripts standardise identically.

Run:
    python tomography_baseline.py --inspect              # dataset structure
    python tomography_baseline.py --sim all              # all sims
    python tomography_baseline.py --sim all --ckpt ../models/predictor_models.pth
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # write PNGs, no interactive window needed
import matplotlib.pyplot as plt

from scipy.optimize import nnls
from scipy import stats

import networkx as nx


# ---------------------------------------------------------------------------
# CONFIG - adjust these to match your dataset / project layout
# ---------------------------------------------------------------------------
CONFIG = dict(
    dataset_path=None,          # None -> default location
    out_dir="tomography_out",

    # --- which simulations to use ------------------------------------------
    #   "all"      every sim in the dataset  (use this to compare with training)
    #   3          a single sim
    #   [0, 2, 5]  an explicit subset
    sim_index="all",

    # --- optional: take normalisation constants from a trained checkpoint ----
    # Strongly recommended when comparing against train_predictor.py, since it
    # guarantees mean_total/std_total are bit-identical to the ones the model
    # was trained with.
    ckpt_path=None,

    # --- field names inside each overlay record -----------------------------
    meas_key="meas",
    id_key="id",
    path_key=None,
    path_key_candidates=("overlay_path", "path", "nodes", "node_names", "hops"),

    # If your overlay records store the TRUE underlay route, naming it here
    # switches the script from the shortest-path assumption (O3) to the true
    # routing (O1). Running both and differencing isolates the cost of routing
    # misspecification, which is usually the most interesting number.
    true_path_key=None,
    true_path_key_candidates=("underlay_path",),
    use_true_routing=False,     # set True (or --true-routing) for the O1 oracle

    # --- indices inside 'meas' ---------------------------------------------
    IDX_TOTAL=0,
    IDX_LOSS=1,
    IDX_PROP=2,      # propagation / topology delay
    IDX_QUEUE=3,     # node / queueing delay  <- this is what we model

    # --- optional ground-truth per-node queue delay -------------------------
    node_delay_key=None,
    node_delay_key_candidates=("node_delay", "queue_delay", "node_queue", "delays"),

    # --- how to estimate the AR(1) coefficients -----------------------------
    # Per-node rho is often NOT identifiable (see the rank diagnostic printed at
    # fit time). Unlike mu, a wrong per-node rho does NOT cancel when projected
    # back onto a route - it enters nonlinearly through rho^m - so a bad
    # estimate actively hurts the recursive filter.
    #   "auto"     per-node when the route system is full rank, else global
    #   "global" / "per_node" / "shrunk"
    rho_mode="auto",

    # --- evaluation ---------------------------------------------------------
    train_frac=0.70,        # matches train_predictor.py
    M=1,                    # prediction horizon, must match train_predictor.py
    c_delta=5.0,            # delay loss weight, must match train_predictor.py
    n_plot_nodes=4,
    eps_obs=1e-6,
    max_targets=None,       # cap LOO targets per timestep (None = all)
    seed=0,
)


# ---------------------------------------------------------------------------
# 1. LOADING + INSPECTION
# ---------------------------------------------------------------------------
def load_dataset(path=None):
    """Import the project's PredictorDataset and open the generated .pt file."""
    try:
        from trainer.dataset import PredictorDataset
    except ImportError:
        from trainer.dataset import PredictorDataset

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data",
                            "predictor_dataset.pt")
        path = os.path.abspath(path)
    print(f"[load] dataset: {path}")
    return PredictorDataset(path)


def inspect_dataset(dataset):
    """Print the structure of one snapshot so you can fill in CONFIG."""
    n_sims = dataset.get_nbr_of_sims()
    T = len(dataset)
    print(f"\n[inspect] n_sims={n_sims}  n_timesteps={T}")

    snap = dataset[0][0]
    print(f"[inspect] snapshot keys: {list(snap.keys())}")

    ovls = snap["overlay_paths"]
    print(f"[inspect] n_overlays at t=0: {len(ovls)}")
    print(f"[inspect] overlay record keys: {list(ovls[0].keys())}")
    for k, v in ovls[0].items():
        prev = v if not hasattr(v, "__len__") or isinstance(v, str) else list(v)[:6]
        print(f"           {k!r:>16} : {type(v).__name__:<10} {prev}")

    if "node_names" in snap:
        print(f"[inspect] n_underlay_nodes: {len(snap['node_names'])}")
    if "edge_index" in snap:
        print(f"[inspect] edge_index shape: {np.asarray(snap['edge_index']).shape}")
    if "edge_attr" in snap:
        print(f"[inspect] edge_attr  shape: {np.asarray(snap['edge_attr']).shape}")
    print()


def autodetect_key(sample_dict, candidates, what, quiet=False):
    """Return the first candidate key present in sample_dict, else None."""
    for k in candidates:
        if k in sample_dict:
            return k
    if not quiet:
        print(f"[warn] could not auto-detect the {what} key among {candidates}; "
              f"available keys are {list(sample_dict.keys())}")
    return None


def resolve_sims(dataset, spec):
    """Turn CONFIG['sim_index'] ('all' | int | list) into a list of sim indices."""
    n = dataset.get_nbr_of_sims()
    if isinstance(spec, str) and spec.lower() == "all":
        sims = list(range(n))
    elif isinstance(spec, (list, tuple)):
        sims = [int(s) for s in spec]
    else:
        sims = [int(spec)]
    bad = [s for s in sims if s < 0 or s >= n]
    if bad:
        raise ValueError(f"sim indices {bad} out of range (dataset has {n} sims)")
    return sims


# ---------------------------------------------------------------------------
# 2. UNDERLAY GRAPH + ROUTING MATRIX
# ---------------------------------------------------------------------------
def build_underlay_graph(snap):
    """
    Build a networkx graph of the underlay at one timestep.

    Edge weights: edge_attr when it is a single scalar per edge (typically the
    link distance), otherwise the Euclidean distance between node positions in
    snap['x'][:, :2]. This weight is what the shortest-path assumption uses.
    """
    names = list(snap["node_names"])
    ei = np.asarray(snap["edge_index"])           # [2, E]
    x = np.asarray(snap["x"])                     # [N, F]; first 2 cols = position

    ea = snap.get("edge_attr", None)
    ea = np.asarray(ea) if ea is not None else None

    G = nx.Graph()
    G.add_nodes_from(names)
    for e in range(ei.shape[1]):
        u, v = names[int(ei[0, e])], names[int(ei[1, e])]
        if ea is not None and ea.ndim == 1:
            w = float(ea[e])
        elif ea is not None and ea.ndim == 2 and ea.shape[1] >= 1:
            w = float(ea[e, 0])                   # column 0 = distance
        else:
            w = float(np.linalg.norm(x[int(ei[0, e]), :2] - x[int(ei[1, e]), :2]))
        if G.has_edge(u, v):
            G[u][v]["w"] = min(G[u][v]["w"], w)
        else:
            G.add_edge(u, v, w=w)
    return G


def route_vector_shortest(G, node_index, endpoints):
    """
    Binary route vector a_i from SHORTEST PATHS between consecutive endpoints.
    This is the misspecified-routing oracle (O3): the true route is never read.
    Returns None if any leg is unreachable.
    """
    a = np.zeros(len(node_index))
    for u, v in zip(endpoints[:-1], endpoints[1:]):
        if u not in G or v not in G:
            return None
        try:
            seg = nx.shortest_path(G, u, v, weight="w")
        except nx.NetworkXNoPath:
            return None
        for n in seg:
            if n in node_index:
                a[node_index[n]] = 1.0
    return a


def route_vector_true(node_index, true_path):
    """Binary route vector straight from the stored underlay route (O1 oracle)."""
    a = np.zeros(len(node_index))
    for n in true_path:
        if n in node_index:
            a[node_index[n]] = 1.0
    return a if a.sum() else None


def build_all_routes(dataset, sims, times, cfg):
    """
    Precompute routing vectors and measurements for every (sim, timestep, overlay).

    Returns
    -------
    routes : dict[(gidx, t, ovl_id)] -> binary route vector for that sim
    meas   : dict[(gidx, t, ovl_id)] -> raw meas array
    node_index : dict[gidx] -> {node_name: column}
    """
    routes, meas, node_index = {}, {}, {}
    n_fail = 0

    for g in sims:
        # Node ordering is fixed per sim from its first timestep, so a column
        # means the same node at every t (required by the recursive filter).
        snap0 = dataset[g][times[0]]
        node_index[g] = {n: i for i, n in enumerate(snap0["node_names"])}

        for t in times:
            snap = dataset[g][t]
            G = None
            for ovl in snap["overlay_paths"]:
                if cfg["use_true_routing"] and cfg["true_path_key"]:
                    a = route_vector_true(node_index[g],
                                          list(ovl[cfg["true_path_key"]]))
                else:
                    if G is None:                 # build lazily, once per t
                        G = build_underlay_graph(snap)
                    a = route_vector_shortest(G, node_index[g],
                                              list(ovl[cfg["path_key"]]))
                if a is None:
                    n_fail += 1
                    continue
                key = (g, t, ovl[cfg["id_key"]])
                routes[key] = a
                meas[key] = np.asarray(ovl[cfg["meas_key"]], dtype=float)

    mode = "TRUE routing (O1)" if cfg["use_true_routing"] else "shortest-path (O3)"
    print(f"[routes] mode: {mode}")
    print(f"[routes] built {len(routes)} route vectors across {len(sims)} sim(s); "
          f"nodes per sim: {[len(node_index[g]) for g in sims[:6]]}"
          f"{' ...' if len(sims) > 6 else ''}")
    if n_fail:
        print(f"[routes] {n_fail} overlay/timestep pairs had no route (skipped)")
    return routes, meas, node_index


# ---------------------------------------------------------------------------
# 3. PLOTS
# ---------------------------------------------------------------------------
def plot_series_and_distribution(routes, meas, cfg):
    """
    (a) queue-delay time series for a few randomly chosen overlays
    (b) empirical distribution vs fitted Gaussian, plus a Q-Q plot

    Figure (b) decides how you must label the bound: heavy skew means the
    linear-Gaussian estimator is the best LINEAR estimator (BLUE), not the MMSE,
    so a nonlinear model may legitimately beat it.
    """
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    IQ = cfg["IDX_QUEUE"]

    # Series are keyed per (sim, overlay) - never merge overlays across sims.
    series = {}
    for (g, t, oid), m in meas.items():
        series.setdefault((g, oid), {})[t] = float(m[IQ])

    keys = [k for k, v in series.items()
            if len(v) > 20 and np.std(list(v.values())) > 1e-9]
    if not keys:
        print("[plot] no usable series; skipping plots")
        return None
    pick = [keys[i] for i in rng.choice(len(keys),
                                        size=min(cfg["n_plot_nodes"], len(keys)),
                                        replace=False)]

    # ---- (a) time series ---------------------------------------------------
    fig, axes = plt.subplots(len(pick), 1, figsize=(10, 2.1 * len(pick)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (g, oid) in zip(axes, pick):
        ts = sorted(series[(g, oid)])
        v = np.asarray([series[(g, oid)][t] for t in ts])
        ax.plot(ts, v, lw=0.9)
        ax.axhline(v.mean(), color="crimson", ls="--", lw=0.8,
                   label=f"mean={v.mean():.3g}")
        # lag-1 autocorrelation = rho: decides whether history is worth carrying
        r = np.corrcoef(v[1:], v[:-1])[0, 1] if len(v) > 2 else np.nan
        tau = -1 / np.log(r) if (np.isfinite(r) and 0 < r < 1) else np.nan
        ax.set_title(f"sim {g}, overlay {oid}   rho(lag1)={r:.3f}   tau={tau:.1f}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel("queue delay")
    axes[-1].set_xlabel("timestep")
    fig.tight_layout()
    p1 = os.path.join(cfg["out_dir"], "delay_series.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- (b) distribution vs Gaussian -------------------------------------
    # Pool DEVIATIONS (value minus that series' own mean): the estimator models
    # deviations, not levels, and this makes sims/overlays commensurable.
    dev = np.concatenate([np.asarray(list(series[k].values()))
                          - np.mean(list(series[k].values())) for k in keys])
    mu_d, sd_d = dev.mean(), dev.std()
    skew, exk = stats.skew(dev), stats.kurtosis(dev)

    fig, (axh, axq) = plt.subplots(1, 2, figsize=(11, 4))
    axh.hist(dev, bins=80, density=True, alpha=0.55, label="empirical")
    xs = np.linspace(dev.min(), dev.max(), 400)
    axh.plot(xs, stats.norm.pdf(xs, mu_d, sd_d), "r-", lw=1.6,
             label=f"Gaussian N({mu_d:.2g}, {sd_d:.3g}$^2$)")
    axh.set_title(f"queue-delay deviations (all sims)\n"
                  f"skew={skew:.2f}  excess kurtosis={exk:.2f}")
    axh.set_xlabel("deviation from mean"); axh.legend(fontsize=8)
    stats.probplot(dev, dist="norm", plot=axq)
    axq.set_title("Q-Q plot vs Normal")
    fig.tight_layout()
    p2 = os.path.join(cfg["out_dir"], "delay_distribution.png")
    fig.savefig(p2, dpi=130); plt.close(fig)

    print(f"[plot] wrote {p1}\n[plot] wrote {p2}")
    print(f"[plot] skew={skew:.3f}  excess kurtosis={exk:.3f}  -> "
          + ("Gaussian approx reasonable" if abs(skew) < 1 and abs(exk) < 2
             else "NOT Gaussian: present the bound as a LINEAR (BLUE) bound"))

    # How concentrated is the squared error? Heavy tails mean your MSE training
    # signal is dominated by a few timesteps.
    sq = np.sort(dev ** 2)[::-1]
    top1 = sq[:max(1, len(sq) // 100)].sum() / sq.sum()
    print(f"[plot] top 1% of samples carry {100*top1:.1f}% of the squared error "
          + ("(outlier-dominated: consider a Huber loss)" if top1 > 0.25 else ""))
    return dict(skew=skew, exkurt=exk, top1=top1)


# ---------------------------------------------------------------------------
# 4. PER-SIM NODE STATISTICS FROM END-TO-END MEASUREMENTS
# ---------------------------------------------------------------------------
def estimate_node_stats_one_sim(g, routes, meas, n_nodes, train_times, cfg,
                                verbose=True):
    """
    Recover per-node statistics for ONE sim from its path-level measurements.

    Using meas[IDX_QUEUE] is legitimate, not cheating: propagation is computable
    from geometry, so q = total - prop is available at inference too.

    Three moment identities, each with the binary route vector as its design row:
        mean      E[q_i(t)]              = sum_{n in path_i} mu_n
        variance  E[r_i(t)^2]            = sum_{n in path_i} sigma^2_n
        lag-1     E[r_i(t) r_i(t-1)]     = sum_{n in both} rho_n sigma^2_n
    A time-varying A(t) is handled automatically: every (overlay, timestep) pair
    adds an equation. Pooling over time is exactly why mu is well identified even
    though the instantaneous state x(t) is not.

    NNLS rather than lstsq enforces the physics: all three are non-negative.
    """
    tset = set(train_times)
    IQ = cfg["IDX_QUEUE"]

    rows, ys, keys = [], [], []
    for (gg, t, oid), a in routes.items():
        if gg == g and t in tset:
            rows.append(a); ys.append(meas[(gg, t, oid)][IQ]); keys.append((t, oid))
    if len(rows) < 10:
        return None

    A = np.asarray(rows); y = np.asarray(ys)

    # Identifiability: what matters is RANK, not row count. Repeating the same
    # route adds equations but no information.
    rank = np.linalg.matrix_rank(A)
    covered = int((A.sum(0) > 0).sum())

    mu, _ = nnls(A, y)
    resid = {k: y[i] - A[i] @ mu for i, k in enumerate(keys)}
    r = np.asarray([resid[k] for k in keys])
    sigma2 = np.maximum(nnls(A, r ** 2)[0], 1e-9)

    # ---- rho ---------------------------------------------------------------
    rows2, ys2, pn, pp = [], [], [], []
    for (t, oid) in keys:
        prev = (t - 1, oid)
        if prev in resid and (g, t - 1, oid) in routes:
            inter = routes[(g, t, oid)] * routes[(g, t - 1, oid)]
            if inter.sum() == 0:
                continue
            rows2.append(inter); ys2.append(resid[(t, oid)] * resid[prev])
            pn.append(resid[(t, oid)]); pp.append(resid[prev])

    if rows2:
        R2 = np.asarray(rows2)
        # Pooled path-level estimate: always well determined, even at low rank.
        rho_global = float(np.clip(np.corrcoef(np.asarray(pn),
                                               np.asarray(pp))[0, 1], 0.0, 0.995))
        rho_node = np.clip(nnls(R2, np.asarray(ys2))[0] / (sigma2 + 1e-12),
                           0.0, 0.995)
        rank2 = np.linalg.matrix_rank(R2)
        cov2 = int((R2.sum(0) > 0).sum())
        ident = rank2 / max(cov2, 1)

        mode = cfg.get("rho_mode", "auto")
        used = ("per_node" if ident >= 0.999 else "global") if mode == "auto" else mode
        if used == "global":
            rho = np.full(n_nodes, rho_global)
        elif used == "per_node":
            rho = rho_node
        else:
            rho = ident * rho_node + (1.0 - ident) * rho_global
    else:
        rho_global, ident, used = 0.0, 0.0, "none"
        rho = np.zeros(n_nodes)

    if verbose:
        seen = A.sum(0) > 0
        tau = -1 / np.log(max(rho[seen].mean(), 1e-6)) if seen.any() else np.nan
        print(f"[fit] sim {g:>2}: {A.shape[0]:>5} eq, {n_nodes:>3} nodes, "
              f"rank={rank:>3}, covered={covered:>3} | "
              f"rho_mode={used}, rho_global={rho_global:.3f}, tau={tau:.1f}")
    return dict(mu=mu, sigma2=sigma2, rho=rho, rank=rank, covered=covered,
                rho_global=rho_global)


def estimate_node_stats(routes, meas, sims, node_index, train_times, cfg):
    """Fit statistics independently for each sim (they are different networks)."""
    print(f"[fit] fitting node statistics per sim ({len(sims)} sims)")
    stats_by_sim, skipped = {}, []
    for g in sims:
        st = estimate_node_stats_one_sim(g, routes, meas, len(node_index[g]),
                                         train_times, cfg,
                                         verbose=(len(sims) <= 20))
        if st is None:
            skipped.append(g)
        else:
            stats_by_sim[g] = st
    if skipped:
        print(f"[fit] skipped sims with too little data: {skipped}")
    if len(sims) > 20:
        ranks = [s["rank"] / max(s["covered"], 1) for s in stats_by_sim.values()]
        rgs = [s["rho_global"] for s in stats_by_sim.values()]
        print(f"[fit] identifiability rank/covered: mean={np.mean(ranks):.2f} "
              f"min={np.min(ranks):.2f}")
        print(f"[fit] rho_global across sims: mean={np.mean(rgs):.3f} "
              f"min={np.min(rgs):.3f} max={np.max(rgs):.3f}")
    return stats_by_sim


# ---------------------------------------------------------------------------
# 5. ESTIMATORS
# ---------------------------------------------------------------------------
def predict_oneshot(A_S, z_S, a_i, Sigma_diag, rho_diag, m, eps):
    """
    One-shot inversion: the prior entering each update is the stationary Sigma,
    i.e. everything before this timestep is deliberately forgotten. This is the
    information regime your model is in with Delta=1.
    """
    Sig = np.diag(Sigma_diag)
    M = A_S @ Sig @ A_S.T + eps * np.eye(A_S.shape[0])
    G = Sig @ A_S.T @ np.linalg.inv(M)
    dhat = G @ z_S
    P = Sig - G @ A_S @ Sig

    Rm = np.diag(rho_diag ** m)                        # horizon decay
    P_m = Rm @ P @ Rm + np.diag(Sigma_diag * (1 - rho_diag ** (2 * m)))
    return float(a_i @ (Rm @ dhat)), float(a_i @ P_m @ a_i)


class RecursiveFilter:
    """
    Kalman filter over per-node delay deviations: the posterior is carried
    forward, so a node observed several steps ago still has reduced uncertainty
    now. That is what lets accumulated history resolve nodes no single snapshot
    covers - valuable exactly when coverage rotates over time.
    """

    def __init__(self, Sigma_diag, rho_diag, eps):
        self.Sigma_diag = Sigma_diag
        self.rho = rho_diag
        self.R = np.diag(rho_diag)
        self.Q = np.diag(Sigma_diag * (1 - rho_diag ** 2))
        self.eps = eps
        self.dhat = np.zeros(len(Sigma_diag))
        self.P = np.diag(Sigma_diag.copy())            # start at the prior

    def step(self, A_S, z_S):
        self.dhat = self.R @ self.dhat                 # predict
        self.P = self.R @ self.P @ self.R + self.Q
        if A_S is not None and len(A_S):               # update
            S = A_S @ self.P @ A_S.T + self.eps * np.eye(A_S.shape[0])
            K = self.P @ A_S.T @ np.linalg.inv(S)
            self.dhat = self.dhat + K @ (z_S - A_S @ self.dhat)
            self.P = self.P - K @ A_S @ self.P

    def predict(self, a_i, m):
        Rm = np.diag(self.rho ** m)
        Pm = Rm @ self.P @ Rm + np.diag(self.Sigma_diag * (1 - self.rho ** (2 * m)))
        return float(a_i @ (Rm @ self.dhat)), float(a_i @ Pm @ a_i)


# ---------------------------------------------------------------------------
# 6. EVALUATION - identical standardisation to train_predictor.py
# ---------------------------------------------------------------------------
def evaluate(routes, meas, stats_by_sim, sims, eval_times, cfg,
             mean_total, std_total):
    """
    For each sim, timestep and target overlay:
      - observe all OTHER overlays at t (leave-one-out, matching your masking)
      - predict the target's delay at t+M
      - score c_delta * z^2 with z the standardised error

    Returns pooled results and a per-sim breakdown.
    """
    IQ, IP, IT = cfg["IDX_QUEUE"], cfg["IDX_PROP"], cfg["IDX_TOTAL"]
    M, eps, c = cfg["M"], cfg["eps_obs"], cfg["c_delta"]
    rng = np.random.default_rng(cfg["seed"])

    by_gt = {}
    for (g, t, oid) in routes:
        by_gt.setdefault((g, t), []).append(oid)

    pooled = dict(oneshot=[], recursive=[], mean=[], persist=[], bound=[])
    per_sim = {}

    for g in sims:
        if g not in stats_by_sim:
            continue
        st = stats_by_sim[g]
        mu, sigma2, rho = st["mu"], st["sigma2"], st["rho"]

        # One independent filter per target: the filter for target i must never
        # see i's own measurements, or the LOO comparison is unfair.
        filters = {}
        local = dict(oneshot=[], recursive=[], mean=[], persist=[], bound=[])

        for t in sorted(eval_times):
            ids_t = by_gt.get((g, t), [])
            ids_next = set(by_gt.get((g, t + M), []))
            targets = [i for i in ids_t if i in ids_next]
            if cfg["max_targets"] and len(targets) > cfg["max_targets"]:
                targets = list(rng.choice(targets, cfg["max_targets"], replace=False))

            for i in targets:
                obs = [j for j in ids_t if j != i]
                if not obs:
                    continue
                A_S = np.stack([routes[(g, t, j)] for j in obs])
                # innovation = measured queue minus its route-implied mean
                z_S = np.array([meas[(g, t, j)][IQ] - routes[(g, t, j)] @ mu
                                for j in obs])

                a_i = routes[(g, t + M, i)]
                tgt = meas[(g, t + M, i)]

                dev_os, mse_bound = predict_oneshot(A_S, z_S, a_i, sigma2, rho,
                                                    M, eps)
                if i not in filters:
                    filters[i] = RecursiveFilter(sigma2, rho, eps)
                filters[i].step(A_S, z_S)
                dev_rec, _ = filters[i].predict(a_i, M)

                # Total-delay prediction: route-implied queue mean + borrowed
                # deviation + propagation (computable from geometry).
                base = a_i @ mu + tgt[IP]
                true_tot = tgt[IT]
                zs = lambda p: ((p - true_tot) / (std_total + 1e-8)) ** 2

                local["oneshot"].append(c * zs(base + dev_os))
                local["recursive"].append(c * zs(base + dev_rec))
                local["bound"].append(c * mse_bound / (std_total ** 2 + 1e-8))
                local["mean"].append(
                    c * ((true_tot - mean_total) / (std_total + 1e-8)) ** 2)
                if (g, t, i) in meas:
                    local["persist"].append(
                        c * ((true_tot - meas[(g, t, i)][IT]) /
                             (std_total + 1e-8)) ** 2)

        if local["oneshot"]:
            per_sim[g] = {k: float(np.mean(v)) if v else float("nan")
                          for k, v in local.items()}
            per_sim[g]["n"] = len(local["oneshot"])
            for k in pooled:
                pooled[k].extend(local[k])

    out = {k: float(np.mean(v)) if v else float("nan") for k, v in pooled.items()}
    return out, per_sim, len(pooled["oneshot"])


# ---------------------------------------------------------------------------
# NORMALISATION - must match calculate_mean_std() in train_predictor.py
# ---------------------------------------------------------------------------
def compute_norm(meas, train_times, cfg):
    """
    Pool the TOTAL delay over all selected sims and the train timesteps.

    Computed as prop + queue rather than reading meas[0], so it is bit-identical
    to the trainer, which builds meas_delay = [[meas[2], meas[3]], ...] and takes
    stdev of the row sums. A consistency check against meas[0] is printed.
    """
    tset = set(train_times)
    tot = np.array([m[cfg["IDX_PROP"]] + m[cfg["IDX_QUEUE"]]
                    for (g, t, o), m in meas.items() if t in tset])
    direct = np.array([m[cfg["IDX_TOTAL"]]
                       for (g, t, o), m in meas.items() if t in tset])
    if len(tot) and np.max(np.abs(tot - direct)) > 1e-6:
        print(f"[warn] meas[0] != meas[2]+meas[3] (max diff "
              f"{np.max(np.abs(tot - direct)):.3g}); using the component sum")
    # ddof=1 matches statistics.stdev() used by the trainer.
    return float(tot.mean()), float(tot.std(ddof=1))


def norm_from_checkpoint(path):
    """Read mean_total/std_total straight out of a trained checkpoint."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mean_total = float(np.asarray(ck["mean_delay"]).sum())
    std_total = ck.get("std_dev_delay_tot", None)
    if std_total is None:
        print("[norm] checkpoint has no 'std_dev_delay_tot' -> "
              "recomputing from the dataset instead")
        return None
    return mean_total, float(std_total)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--sim", default=None,
                    help="'all', an int, or comma-separated ints (e.g. 0,2,5)")
    ap.add_argument("--ckpt", default=None,
                    help="trained checkpoint to take mean/std from")
    ap.add_argument("--true-routing", action="store_true",
                    help="use the stored underlay route (O1) instead of "
                         "shortest path (O3)")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.dataset:
        cfg["dataset_path"] = args.dataset
    if args.ckpt:
        cfg["ckpt_path"] = args.ckpt
    if args.true_routing:
        cfg["use_true_routing"] = True
    if args.sim is not None:
        cfg["sim_index"] = (args.sim if args.sim.lower() == "all"
                            else [int(s) for s in args.sim.split(",")])

    dataset = load_dataset(cfg["dataset_path"])
    if args.inspect:
        inspect_dataset(dataset)
        return

    sims = resolve_sims(dataset, cfg["sim_index"])
    T = len(dataset)
    times = list(range(T - cfg["M"]))
    print(f"[sims] using {len(sims)} simulation(s): "
          f"{sims if len(sims) <= 12 else str(sims[:12]) + ' ...'}")

    # ---- resolve field names ------------------------------------------------
    sample_ovl = dataset[sims[0]][times[0]]["overlay_paths"][0]
    if cfg["path_key"] is None:
        cfg["path_key"] = autodetect_key(sample_ovl, cfg["path_key_candidates"],
                                         "overlay endpoint-list")
        if cfg["path_key"] is None:
            print("[fatal] set CONFIG['path_key'] manually; use --inspect.")
            sys.exit(1)
        print(f"[cfg] path_key={cfg['path_key']!r}")
    if cfg["true_path_key"] is None:
        cfg["true_path_key"] = autodetect_key(
            sample_ovl, cfg["true_path_key_candidates"], "true underlay route",
            quiet=True)
        if cfg["true_path_key"]:
            print(f"[cfg] true underlay route available as "
                  f"{cfg['true_path_key']!r} -> run with --true-routing for the "
                  f"O1 oracle; differencing O1 and O3 isolates the cost of "
                  f"routing misspecification")
    if cfg["use_true_routing"] and not cfg["true_path_key"]:
        print("[fatal] --true-routing requested but no stored route field found.")
        sys.exit(1)

    # ---- split, matching train_predictor.py ---------------------------------
    n = len(times)
    train_times = times[: int(cfg["train_frac"] * n)]
    eval_times = times[int(cfg["train_frac"] * n):]
    print(f"[split] train={len(train_times)} steps, eval={len(eval_times)} steps")

    # ---- routes -------------------------------------------------------------
    routes, meas, node_index = build_all_routes(dataset, sims, times, cfg)

    # ---- plots --------------------------------------------------------------
    plot_series_and_distribution(routes, meas, cfg)

    # ---- per-sim node statistics (train split only) -------------------------
    stats_by_sim = estimate_node_stats(routes, meas, sims, node_index,
                                       train_times, cfg)

    # ---- normalisation ------------------------------------------------------
    norm = norm_from_checkpoint(cfg["ckpt_path"]) if cfg["ckpt_path"] else None
    if norm is not None:
        mean_total, std_total = norm
        print(f"[norm] from checkpoint: mean_total={mean_total:.6g}  "
              f"std_total={std_total:.6g}")
    else:
        mean_total, std_total = compute_norm(meas, train_times, cfg)
        print(f"[norm] from dataset (pooled over {len(sims)} sims): "
              f"mean_total={mean_total:.6g}  std_total={std_total:.6g}")
        print( "[norm] CHECK: this must equal the trainer's mean_delay.sum() and "
               "std_dev_delay_tot, or the losses below are not comparable.")

    # ---- evaluate -----------------------------------------------------------
    res, per_sim, n_pred = evaluate(routes, meas, stats_by_sim, sims,
                                    eval_times, cfg, mean_total, std_total)

    mode = "O1 true routing" if cfg["use_true_routing"] else "O3 shortest path"
    print("\n" + "=" * 66)
    print(f"  Loss comparison  ({mode}, c_delta={cfg['c_delta']}, M={cfg['M']})")
    print(f"  {len(sims)} sim(s), {n_pred} predictions")
    print("  Same standardised units as train_predictor.py val_loss")
    print("=" * 66)
    print(f"  predict-mean baseline          : {res['mean']:.4f}")
    print(f"  persistence baseline           : {res['persist']:.4f}")
    print(f"  tomography, one-shot           : {res['oneshot']:.4f}")
    print(f"  tomography, recursive (KF)     : {res['recursive']:.4f}")
    print(f"  analytic BLUE bound            : {res['bound']:.4f}")
    print("=" * 66)

    if len(per_sim) > 1:
        vals = np.array([v["oneshot"] for v in per_sim.values()])
        print(f"  per-sim one-shot: mean={vals.mean():.3f}  sd={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")
        worst = sorted(per_sim.items(), key=lambda kv: -kv[1]["oneshot"])[:3]
        print("  hardest sims (one-shot / mean-baseline):")
        for g, v in worst:
            print(f"     sim {g:>2}: {v['oneshot']:.3f} / {v['mean']:.3f}  "
                  f"(n={v['n']})")
    print()
    print("  Compare your model's val_loss against the ONE-SHOT row - same")
    print("  information regime as Delta=1. The recursive row shows what")
    print("  carrying history would add. A gap to the bound means the fitted")
    print("  statistics, not the method, are the limiting factor. Note the")
    print("  bound is only a true ceiling under CORRECT routing: with --true-")
    print("  routing it is one, with shortest-path assumed it is not.\n")


if __name__ == "__main__":
    main()