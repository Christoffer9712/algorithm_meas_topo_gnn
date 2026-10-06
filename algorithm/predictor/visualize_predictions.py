#!/usr/bin/env python3
"""
Standalone visualization script that mirrors visualize_predictions.ipynb.
Saves scatter plots of predicted vs true packet-loss and delay to outputs/.
"""
import sys, os
import statistics
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from trainer.dataset import PredictorDataset
from algorithm.predictor.path_encoder import GATv2Encoder, GraphEncoder
from algorithm.predictor.hetero_encoder import HeteroGATv2Encoder
from algorithm.predictor.f_cov import f_cov
from algorithm.predictor.f_topo import f_topo

from environment.generate_dataset import generate, generate_nets
from trainer.train_predictor import train

from trainer.train_data_driven_predictor import DataDrivenInference

from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx

# LMMSE.py lives in the project root (two levels up, same place as data/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')))
from LMMSE import LMMSEPredictor

R_VAR = 0.0  # must match R_VAR in train()
INCLUDE_MEAS = True
REMOVE_SELF_MEAS = True
Delta = 3    # measurement history depth, must match Delta in train()

def path_distance_from_data(data, underlay_path, dist_scale=300.0):
    # data: torch_geometric.data.Data produced by snapshot_to_pyg
    # underlay_path: list/tuple of node names (strings) like ["AC-0", "SAT0-1", ...]
    name_to_idx = data.name_to_idx  # mapping node name -> integer index used in Data
    edge_index = data.edge_index.cpu().numpy()   # shape (2, E)
    edge_attr = data.edge_attr.cpu().numpy()     # shape (E, F)
    # Build mapping once: (src_idx, dst_idx) -> position index into edge_attr
    mapping = { (int(s), int(t)): i for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])) }

    total_dist = 0.0
    for n1, n2 in zip(underlay_path[:-1], underlay_path[1:]):
        if n1 not in name_to_idx or n2 not in name_to_idx:
            raise ValueError(f"Node name not in data.name_to_idx: {n1} or {n2}")
        a = name_to_idx[n1]
        b = name_to_idx[n2]
        pos = mapping.get((a, b))
        if pos is None:
            # fallback: maybe only reversed direction exists (shouldn't on PyG snapshot_to_pyg),
            # try reversed or search (slow)
            pos = mapping.get((b, a))
            if pos is None:
                raise ValueError(f"No edge between {n1} ({a}) and {n2} ({b}) in data.edge_index")
        # distance is in column 0 (snapshot_to_pyg uses edge_attrs=("distance",) by default)
        d_norm = float(edge_attr[pos, 0]) if edge_attr.ndim > 1 else float(edge_attr[pos])
        d_raw = d_norm * dist_scale
        total_dist += d_raw
    return total_dist

def deterministic_meas(edge_attr, ovl):

    underlay_path = ovl['underlay_path']
    dist = 0
    #print(f'edge_attr={edge_attr}')
    #for node1, node2 in zip(underlay_path[:-1], underlay_path[1:]):
    #    if edge_data is None:
    #        raise ValueError(f"No edge between {node1} and {node2}")
    #    print(edge_data)
    #    dist += edge_data['dist']
    #print(f'dist={dist}')
    #return (dist / 100, 0)

# Ensure repo root is on path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
# Paths
dataset_path = os.path.join(repo_root, '../../data', 'predictor_dataset.pt')
model_path = os.path.join(repo_root, '../../models', 'predictor_models.pth')
output_dir = os.path.join(repo_root, '../../outputs')
os.makedirs(output_dir, exist_ok=True)

print('Dataset path:', dataset_path)
print('Model path:', model_path)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run trainer.generate_dataset.generate first.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run trainer.train_predictor.train first.")

# create new validation data
nbrSims = 5
if True: #NOTE LMMSE is only valid when data is generated with all sats the same parameters!!!
    T = 300
    seeds = range(100,100+nbrSims)
    nets = generate_nets(seeds)
    for idx in range(len(nets)):
        print(
            f'net-{idx} has '
            f'sats_per_ring = {nets[idx].sats_per_ring}, '
            f'n_rings = {nets[idx].n_rings}, '
            f'n_gateways = {nets[idx].n_gateways}, '
            f'n_targets = {nets[idx].n_targets}, '
            f'ring_radii = {nets[idx].ring_radii}, '
            f'ring_speeds = {nets[idx].ring_speeds}, '
            f'orbit_center = {nets[idx].orbit_center}, '
            f'aircraft_pos = {nets[idx].aircraft_pos}, '
            f'range_limit = {nets[idx].range_limit}'
        )
        print('-----------------------------')
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualiser_dataset.pt')
    dataset_path = os.path.abspath(dataset_path)
    dataset_path = generate(T=T, seeds=seeds, nets=nets, dataset_path=dataset_path)

dataset = PredictorDataset(dataset_path)
N = min(400, len(dataset))
print(f'Loading {N} samples from dataset (total {len(dataset)})')

# Build models and load checkpoint
TOPO_DIM = 64
device = 'cpu'
encoder = HeteroGATv2Encoder(node_in=10, virtual_in=1, hidden_dim=128, out_dim=TOPO_DIM, heads=2, n_layers=3, dropout=0.0).to(device)
graph_encoder = GraphEncoder(encoder, device=device)
fcov = f_cov(in_dim=TOPO_DIM, dropout=0.0).to(device)
ftopo = f_topo(in_dim=TOPO_DIM, hidden_dim=64, dropout=0.0).to(device)

# The checkpoint holds numpy arrays (normalization stats), so weights_only must be False
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

encoder.load_state_dict(ckpt['encoder_state'])
_missing, _ = fcov.load_state_dict(ckpt['fcov_state'], strict=False)   # identity weights may be new
if _missing:
    print(f'fcov: parameters not in checkpoint, left at their initial values: {_missing}')
ftopo.load_state_dict(ckpt['ftopo_state'])
mean_delay = ckpt['mean_delay']
mean_loss = ckpt['mean_loss']
std_dev_delay = ckpt['std_dev_delay']
std_dev_loss = ckpt['std_dev_loss']
graph_encoder = GraphEncoder(encoder, device=device)
encoder.eval()
fcov.eval()
ftopo.eval()

# Observable overlay identity for f_cov: [overlay id, waypoint ids of ovl['overlay_path']], as in train()
name_vocab, id_vocab = {}, {}
def ovl_keys(ovl):
    wps = [name_vocab.setdefault(nm, len(name_vocab)) for nm in ovl['overlay_path']][:fcov.n_wp]
    wps += [-1] * (fcov.n_wp - len(wps))
    return torch.tensor([id_vocab.setdefault(ovl['id'], len(id_vocab))] + wps, dtype=torch.long, device=device)

def plot_and_calc(gidx, plot=True):
    samples = [dataset[gidx][i] for i in range(N)]
    print(f'Sample len = {len(samples)}')

    # Analytical LMMSE on the same simulation (oracle: true underlay path + true O-U parameters)
    lmmse = LMMSEPredictor(samples, delta=Delta)
    dd = DataDrivenInference(os.path.join(repo_root, '../../models', 'data_driven_predictor_models.pth'), samples)

    # Std of the total delay (normalizes meas[0]). Older checkpoints don't store it, so
    # recompute it exactly as train() does: all sims, first 70% of time steps.
    # NOTE: this is only exact if this is the dataset the checkpoint was trained on.
    std_dev_delay_tot = ckpt.get('std_dev_delay_tot')
    if std_dev_delay_tot is None:
        n_t = len(dataset) - 1                      # M = 1, Delta = 1
        train_t = range(int(0.70 * n_t))
        tot = [o['meas'][2] + o['meas'][3]
            for g in range(dataset.get_nbr_of_sims())
            for t in train_t
            for o in dataset[g][t]['overlay_paths']]
        std_dev_delay_tot = statistics.stdev(tot)
        print(f'std_dev_delay_tot not in checkpoint, recomputed = {std_dev_delay_tot:.4f}')

    def norm_meas(meas):
        # Same standardization as train(): [total delay, packet loss]
        m = torch.tensor(meas[:2], dtype=torch.float32)
        return torch.stack([(m[0] - mean_delay.sum()) / (std_dev_delay_tot + 1e-8),
                            (m[1] - mean_loss) / (std_dev_loss + 1e-8)])


    def denorm(out):
        # normalised [delay, loss] -> raw units, same as labels
        out = out.numpy().copy()
        out[0] = out[0] * (std_dev_delay_tot + 1e-8) + mean_delay.sum()
        out[1] = out[1] * (std_dev_loss + 1e-8) + mean_loss
        return out

    def model_pred(out_topo, enc, hist, mask, keys_pred):
        """Learned prediction for one target. mask=None -> no measurements (f_topo only)."""
        H_hist, Elapsed, Meas_hist, Topo_hist, Keys_hist = hist
        if mask is not None and bool(mask.any()):
            K = int(mask.sum())
            C = fcov(H_hist[mask], Elapsed[mask] + 1, keys_hist=Keys_hist[mask]).reshape(K, K)
            g = fcov(H_hist[mask], Elapsed[mask] + 1, h_pred=enc,
                     keys_hist=Keys_hist[mask], keys_pred=keys_pred).reshape(-1)
            hat_z = Meas_hist[mask, 0] - Topo_hist[mask, 0]      # neighbour deviation (delay)
            R = R_VAR * torch.eye(K)
            corr = (g @ torch.linalg.solve(C + R, hat_z)).reshape(())   # delay correction only
        else:
            corr = torch.zeros(())
        return denorm(out_topo + torch.stack([corr, torch.zeros(())]))  # no correction for loss

    def model_pred_no_phi(out_topo, enc, hist, mask, keys_pred):
        """Learned prediction for one target. mask=None -> no measurements (f_topo only)."""
        H_hist, Elapsed, Meas_hist, Topo_hist, Keys_hist = hist
        if mask is not None and bool(mask.any()):
            K = int(mask.sum())
            C = fcov(H_hist[mask], Elapsed[mask] + 1, keys_hist=Keys_hist[mask], include_phi=False).reshape(K, K)
            g = fcov(H_hist[mask], Elapsed[mask] + 1, h_pred=enc,
                     keys_hist=Keys_hist[mask], keys_pred=keys_pred, include_phi=False).reshape(-1)
            hat_z = Meas_hist[mask, 0] - Topo_hist[mask, 0]      # neighbour deviation (delay)
            R = R_VAR * torch.eye(K)
            corr = (g @ torch.linalg.solve(C + R, hat_z)).reshape(())   # delay correction only
        else:
            corr = torch.zeros(())
        return denorm(out_topo + torch.stack([corr, torch.zeros(())]))

    preds_self = []    # model, all measurements incl. the target overlay's own
    preds_noself = []  # model, without the target overlay's own measurements
    preds_noself_no_phi = [] 
    preds_nomeas = []  # model, no measurements at all (ftopo only)
    labels = []
    determ = []
    prevs = []        # one-step persistence: the overlay's own measurement at time t
    lm_self = []      # LMMSE using all measurements, incl. the target overlay's own
    lm_noself = []    # LMMSE without the target overlay's own measurements
    lm_prior = []     # LMMSE prior only: f_topo = true propagation + h^T r
    data_driven_self = []
    data_driven_noself = []
    data_driven_nomeas = []

    datas = [Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'],
                node_names=d['node_names'], name_to_idx=d['name_to_idx']) for d in samples]

    with torch.no_grad():
        for t in range(Delta - 1, len(samples) - 1):   # need Delta steps of history, as in train()
            hist_ovls = samples[t]['overlay_paths']     # overlays at t (candidates for targets)
            if not hist_ovls:
                continue

            # --- history at t, t-1, ..., t-Delta+1: same structure as run_t() in train()
            H_hist = []
            Meas_hist = []
            Elapsed = []
            Hist_ids = []
            Topo_hist = []
            Keys_hist = []
            for i in range(0, Delta):
                hist_overlay = samples[t - i]['overlay_paths']
                if not hist_overlay:
                    continue
                # encode the whole graph for time (t-i) ONCE, for all overlays
                enc_i = graph_encoder.encode_overlays_pyg_batched(datas[t - i], hist_overlay)
                topo_i = ftopo(torch.stack([enc_i[o['id']] for o in hist_overlay]))
                for idx, ovl_h in enumerate(hist_overlay):
                    H_hist.append(enc_i[ovl_h['id']])
                    Meas_hist.append(norm_meas(ovl_h['meas']))   # standardized [delay, loss]
                    Topo_hist.append(topo_i[idx])
                    Elapsed.append(float(i))
                    Hist_ids.append((ovl_h['id'], t - i))
                    Keys_hist.append(ovl_keys(ovl_h))

            H_hist = torch.stack(H_hist)
            Meas_hist = torch.stack(Meas_hist)
            Elapsed = torch.tensor(Elapsed).unsqueeze(1)
            Topo_hist = torch.stack(Topo_hist)
            Keys_hist = torch.stack(Keys_hist)
            hist = (H_hist, Elapsed, Meas_hist, Topo_hist, Keys_hist)

            # raw measurement at t, per overlay id -> the persistence prediction for t+1
            prev_meas = {o['id']: o['meas'] for o in hist_ovls}

            # --- targets at t+1 (M = 1): overlays that also existed at t
            hist_ids = {o['id'] for o in hist_ovls}
            eligible = [o for o in samples[t + 1]['overlay_paths'] if o['id'] in hist_ids]
            if not eligible:
                continue
            enc_fut = graph_encoder.encode_overlays_pyg_batched(datas[t + 1], eligible)

            for ovl in eligible:
                enc = enc_fut[ovl['id']]
                out_topo = ftopo(enc)


                # masks over the history rows
                mask_self = torch.ones(len(H_hist), dtype=torch.bool)      # everything
                mask_noself = mask_self.clone()                            # minus own measurements
                for d in range(Delta):
                    if (ovl['id'], t - d) in Hist_ids:
                        mask_noself[Hist_ids.index((ovl['id'], t - d))] = False

                k_pred = ovl_keys(ovl)
                preds_self.append(model_pred(out_topo, enc, hist, mask_self, k_pred))
                preds_noself.append(model_pred(out_topo, enc, hist, mask_noself, k_pred))
                preds_noself_no_phi.append(model_pred_no_phi(out_topo, enc, hist, mask_noself, k_pred))
                preds_nomeas.append(model_pred(out_topo, enc, hist, None, k_pred))

                labels.append(ovl['meas'])
                prevs.append(prev_meas[ovl['id']])

                determ.append(path_distance_from_data(datas[t + 1], ovl['underlay_path'])/100)

                # LMMSE for the same target, same history window t, ..., t-Delta+1
                lm = lmmse.predict(t, ovl, m=1, remove_self=False)
                lm_self.append(lm['pred'])
                lm_prior.append(lm['prior'])
                lm_noself.append(lmmse.predict(t, ovl, m=1, remove_self=True)['pred'])

                data_driven_self.append(dd.predict(t, ovl, remove_self=False)[0])
                data_driven_noself.append(dd.predict(t, ovl, remove_self=True)[0])
                data_driven_nomeas.append(dd.predict(t, ovl, include_meas=False)[0])

    preds_self = np.array(preds_self)
    preds_noself = np.array(preds_noself)
    preds_noself_no_phi = np.array(preds_noself_no_phi)
    preds_nomeas = np.array(preds_nomeas)
    labels = np.array(labels)
    prevs = np.array(prevs)
    determ = np.array(determ)
    lm_self = np.array(lm_self)
    lm_noself = np.array(lm_noself)
    lm_prior = np.array(lm_prior)
    data_driven_self = np.array(data_driven_self)
    data_driven_noself = np.array(data_driven_noself)
    data_driven_nomeas = np.array(data_driven_nomeas)

    # --- delay MSE summary, all methods on exactly the same targets ---
    methods = [
        ('persistence (y_t)',              prevs[:, 0]),
        ('topology (dist/100)',            determ),
        ('model with self-meas',           preds_self[:, 0]),
        ('model without self-meas',        preds_noself[:, 0]),
        ('model without self-meas -- no phi',        preds_noself_no_phi[:, 0]),
        ('model no meas (ftopo only)',     preds_nomeas[:, 0]),
        ('LMMSE with self-meas',           lm_self),
        ('LMMSE without self-meas',        lm_noself),
        ('LMMSE no meas (f_topo, oracle)', lm_prior),
        ('Datadriven with self-meas',      data_driven_self),
        ('Datadriven without self-meas',   data_driven_noself),
        ('Datadriven no meas',             data_driven_nomeas)
    ]
    MSE = {}
    print(f'\nDelay MSE, sim {gidx}, n = {len(labels)} targets')
    pers_mse = np.mean((prevs[:, 0] - labels[:, 0]) ** 2)
    for name, p in methods:
        mse = np.mean((p - labels[:, 0]) ** 2)
        MSE[name] = mse
        print(f'  {name:34s} {mse:10.4f}   (/ persistence = {mse / pers_mse:.3f})')

    # which variant to plot: INCLUDE_MEAS / REMOVE_SELF_MEAS pick the model run and the
    # LMMSE that uses the same information
    if not INCLUDE_MEAS:
        preds, lm_match, variant = preds_nomeas, lm_prior, 'no meas'
    elif REMOVE_SELF_MEAS:
        preds, lm_match, variant = preds_noself, lm_noself, 'without self-meas'
    else:
        preds, lm_match, variant = preds_self, lm_self, 'with self-meas'
    if plot:
        # Scatter plots
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.scatter(labels[:,0], preds[:,0], s=8, label=f'model ({variant})')

        plt.scatter(labels[:,0], determ[:], c='red', label='topology (dist/100)')
        plt.scatter(labels[:,0], prevs[:,0], c='yellow', marker='x', s=20, label='persistence (y_t)')
        plt.scatter(labels[:,0], lm_match, c='green', marker='+', s=20,
                    label=f'LMMSE ({variant})')
        plt.xlabel('True delay')
        plt.ylabel('Predicted delay')
        plt.title('Delay: true vs predicted')
        mn, mx = min(labels[:,0].min(), preds[:,0].min()), max(labels[:,0].max(), preds[:,0].max())
        plt.plot([mn,mx],[mn,mx],'r--')
        plt.legend(fontsize=7)

        plt.subplot(1,2,2)
        plt.scatter(labels[:,1], preds[:,1], s=8, label='model')
        plt.scatter(labels[:,1], prevs[:,1], c='yellow', marker='x', s=20, label='persistence (y_t)')
        plt.xlabel('Packet-loss')
        plt.ylabel('Packet-loss')
        plt.title('Packet-loss')
        mn, mx = min(labels[:,1].min(), preds[:,1].min()), max(labels[:,1].max(), preds[:,1].max())
        plt.plot([mn,mx],[mn,mx],'r--')
        plt.legend(fontsize=7)
        plt.tight_layout()

        plt.show()
    
    return(MSE)

mse = []
for gidx in range(nbrSims):
    mse.append(plot_and_calc(gidx, plot=False))

avgs = {
    key: sum(d[key] for d in mse) / len(mse)
    for key in mse[0]
}
print(f"-----Averages over {len(mse)} sims----")
for name, p in avgs.items():
    print(f'  {name:34s} {p:10.4f}')

# model vs the LMMSE that uses the same information (ratio of the averaged MSEs)
print("-----Model / LMMSE (same information)----")
for variant in ('with self-meas', 'without self-meas'):
    print(f'  {variant:34s} {avgs[f"model {variant}"] / avgs[f"LMMSE {variant}"]:10.3f}')
print(f'  {"no meas":34s} '
      f'{avgs["model no meas (ftopo only)"] / avgs["LMMSE no meas (f_topo, oracle)"]:10.3f}')