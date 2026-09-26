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

from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx


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
    print(f'edge_attr={edge_attr}')
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

# Load dataset
dataset = PredictorDataset(dataset_path)
N = min(400, len(dataset))
print(f'Loading {N} samples from dataset (total {len(dataset)})')

gidx = 12 #OBS!
samples = [dataset[gidx][i] for i in range(N)]
print(f'Sample len = {len(samples)}')

# Build models and load checkpoint
TOPO_DIM = 64
device = 'cpu'
fcov = f_cov(h_dim=TOPO_DIM, dropout=0.2).to(device)
ftopo = f_topo(in_dim=TOPO_DIM, hidden_dim=64, dropout=0.2).to(device)
encoder = HeteroGATv2Encoder(node_in=10, virtual_in=1, hidden_dim=128, out_dim=TOPO_DIM, heads=2, n_layers=3, dropout=0.2).to(device)

# The checkpoint holds numpy arrays (normalization stats), so weights_only must be False
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

encoder.load_state_dict(ckpt['encoder_state'])
fcov.load_state_dict(ckpt['fcov_state'])
ftopo.load_state_dict(ckpt['ftopo_state'])
mean_delay = ckpt['mean_delay']
mean_loss = ckpt['mean_loss']
std_dev_delay = ckpt['std_dev_delay']
std_dev_loss = ckpt['std_dev_loss']

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

graph_encoder = GraphEncoder(encoder, device=device)

encoder.eval()
fcov.eval()
ftopo.eval()


def norm_meas(meas):
    # Same standardization as train(): [total delay, packet loss]
    m = torch.tensor(meas[:2], dtype=torch.float32)
    return torch.stack([(m[0] - mean_delay.sum()) / (std_dev_delay_tot + 1e-8),
                        (m[1] - mean_loss) / (std_dev_loss + 1e-8)])


preds = []
labels = []
determ = []
prevs = []        # one-step persistence: the overlay's own measurement at time t

datas = [Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'],
              node_names=d['node_names'], name_to_idx=d['name_to_idx']) for d in samples]

with torch.no_grad():
    for t in range(len(samples) - 1):
        # --- history at t (Delta = 1): encodings, topology predictions, standardized measurements
        hist_ovls = samples[t]['overlay_paths']
        if not hist_ovls:
            continue
        enc_hist = graph_encoder.encode_overlays_pyg_batched(datas[t], hist_ovls)
        H_hist = torch.stack([enc_hist[o['id']] for o in hist_ovls])
        Topo_hist = ftopo(H_hist)
        Meas_hist = torch.stack([norm_meas(o['meas']) for o in hist_ovls])
        Elapsed = torch.zeros(len(hist_ovls), 1)

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

            s, l = fcov(enc.unsqueeze(0).expand(len(H_hist), -1), H_hist, Elapsed + 1)
            w = torch.exp(l)
            est = s.unsqueeze(1) * (Meas_hist - Topo_hist)
            out_meas = (w.unsqueeze(1) * est).sum(0) / (w.sum() + 1e-8)

            out = (out_topo + out_meas).numpy()
            out[0] = out[0] * (std_dev_delay_tot + 1e-8) + mean_delay.sum()
            out[1] = out[1] * (std_dev_loss + 1e-8) + mean_loss
            preds.append(out)
            labels.append(ovl['meas'])
            prevs.append(prev_meas[ovl['id']])

            determ.append(path_distance_from_data(datas[t + 1], ovl['underlay_path'])/100)

preds = np.array(preds)
labels = np.array(labels)
prevs = np.array(prevs)

# compute simple metrics
mse_delay = np.mean((preds[:,0] - labels[:,0])**2)
mse_loss = np.mean((preds[:,1] - labels[:,1])**2)
print(f'MSE packet-loss: {mse_loss:.6e}, MSE delay(ms): {mse_delay:.6e}')

# --- model vs one-step persistence (predict y_{t+1} = y_t) ---
pers_delay = np.mean((prevs[:,0] - labels[:,0])**2)
pers_loss  = np.mean((prevs[:,1] - labels[:,1])**2)
print(f'Persistence  MSE delay(ms): {pers_delay:.6e}  (model / persistence = {mse_delay/pers_delay:.3f})')
print(f'Persistence  MSE packet-loss: {pers_loss:.6e}  (model / persistence = {mse_loss/pers_loss:.3f})')

# slope < 1 => predictions shrunk toward the mean; slope ~1 vs prev => model copies y_t
print(f'slope pred~true (delay): {np.polyfit(labels[:,0], preds[:,0], 1)[0]:.3f}')
print(f'slope pred~prev (delay): {np.polyfit(prevs[:,0],  preds[:,0], 1)[0]:.3f}')

# same split as the tail discussion: bulk vs spikes
tail = labels[:,0] > 17
for name, m in [('bulk', ~tail), ('tail', tail)]:
    if m.sum():
        print(f'  {name} (n={m.sum():4d})  model={np.mean((preds[m,0]-labels[m,0])**2):8.4f}'
              f'  persistence={np.mean((prevs[m,0]-labels[m,0])**2):8.4f}')

# Scatter plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(labels[:,0], preds[:,0], s=8, label='model')

print(f'label_len = {len(labels[:,0])}, determ_len = {len(determ[:])}')
plt.scatter(labels[:,0], determ[:], c='red', label='topology (dist/100)')
plt.scatter(labels[:,0], prevs[:,0], c='yellow', marker='x', s=20, label='persistence (y_t)')
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

'''
out_file = os.path.join(output_dir, 'pred_vs_true.png')
plt.savefig(out_file)
print('Saved plot to', out_file)

# Save separate plots too
plt.figure()
plt.scatter(labels[:,0], preds[:,0], s=8)
plt.xlabel('True packet-loss (fraction)')
plt.ylabel('Predicted packet-loss')
plt.title('Packet-loss: true vs predicted')
plt.plot([mn,mx],[mn,mx],'r--')
plt.savefig(os.path.join(output_dir, 'pred_vs_true_loss.png'))

plt.figure()
plt.scatter(labels[:,1], preds[:,1], s=8)
plt.xlabel('True delay (ms)')
plt.ylabel('Predicted delay (ms)')
plt.title('Delay (ms): true vs predicted')
plt.plot([mn,mx],[mn,mx],'r--')
plt.savefig(os.path.join(output_dir, 'pred_vs_true_delay.png'))

print('Saved individual plots to outputs/')
'''