#!/usr/bin/env python3
"""
Standalone visualization script that mirrors visualize_predictions.ipynb.
Saves scatter plots of predicted vs true packet-loss and delay to outputs/.
"""
import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from trainer.dataset import PredictorDataset
from algorithm.predictor.f_cov import f_cov
from algorithm.predictor.f_topo import f_topo
from algorithm.predictor.path_encoder import GATv2, GraphEncoder


def path_distance_from_data(data, underlay_path, dist_scale=300.0):
    # data: the object produced by snapshot_to_pyg (carries name_to_local + a
    # single ("node","link","node") relation, or a homogeneous edge_index).
    name_to_local = data.name_to_local          # name -> ("node", idx)

    # pull the single relation's edges (homogeneous under the hood)
    if hasattr(data, "edge_index_dict") and len(data.edge_index_dict) > 0:
        rel = next(iter(data.edge_index_dict))
        edge_index = data.edge_index_dict[rel].cpu().numpy()      # (2, E)
        edge_attr = data.edge_attr_dict[rel].cpu().numpy()        # (E, F)
    else:
        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()

    mapping = {(int(s), int(t)): i
               for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1]))}

    total_dist = 0.0
    for n1, n2 in zip(underlay_path[:-1], underlay_path[1:]):
        if n1 not in name_to_local or n2 not in name_to_local:
            raise ValueError(f"Node name not in name_to_local: {n1} or {n2}")
        a = name_to_local[n1][1]      # ("node", idx) -> idx
        b = name_to_local[n2][1]
        pos = mapping.get((a, b))
        if pos is None:
            pos = mapping.get((b, a))
            if pos is None:
                raise ValueError(f"No edge between {n1} ({a}) and {n2} ({b})")
        d_norm = float(edge_attr[pos, 0]) if edge_attr.ndim > 1 else float(edge_attr[pos])
        total_dist += d_norm * dist_scale
    return total_dist


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

gidx = 13 #OBS!
samples = [dataset[gidx][i] for i in range(N)]
print(f'Sample len = {len(samples)}')

OUT_DIM = 64
TOPO_DIM = 4 * OUT_DIM          # 256
device = 'cpu'
# Build models and load checkpoint
encoder = GATv2(hidden_dim=64, out_dim=OUT_DIM, heads=2,
                      n_layers=3, edge_dim=1, dropout=0.0).to(device)
graph_encoder = GraphEncoder(encoder, device=device)

fcov = f_cov(h_dim=TOPO_DIM).to(device)
ftopo = f_topo(in_dim=TOPO_DIM, hidden_dim=64, dropout=0.0).to(device)
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

if 'fcov_state' in ckpt and 'ftopo_state' in ckpt and 'encoder_state' in ckpt:
    #fcov.load_state_dict(ckpt['fcov_state'])
    ftopo.load_state_dict(ckpt['ftopo_state'])
    encoder.load_state_dict(ckpt['encoder_state'])
    mean_delay = ckpt['mean_delay']
    mean_loss = ckpt['mean_loss']
    std_dev_delay = ckpt['std_dev_delay']
    std_dev_loss = ckpt['std_dev_loss']

encoder.eval()
ftopo.eval()
#fcov.eval()

graph_encoder = GraphEncoder(encoder, device='cpu')

preds = []
labels = []
determ = []

with torch.no_grad():
    for d in samples:
        data = d['data'].to(device)          # already the snapshot_to_pyg graph
        ovls = d['overlay_paths']
        ovls_encodings = graph_encoder.encode_overlays_pyg_batched(data, ovls)
        for ovl in ovls:
            out = ftopo(ovls_encodings[ovl['id']]).clone()
            out[0] = out[0] * std_dev_delay + mean_delay
            out[1] = out[1] * std_dev_loss + mean_loss
            preds.append(out.numpy())
            labels.append(ovl['meas'])
            determ.append(path_distance_from_data(data, ovl['underlay_path']) / 100)

preds = np.array(preds)
labels = np.array(labels)

# compute simple metrics
mse_delay = np.mean((preds[:, 0] - labels[:, 0]) ** 2)
mse_loss = np.mean((preds[:, 1] - labels[:, 1]) ** 2)
print(f'MSE packet-loss: {mse_loss:.6e}, MSE delay(ms): {mse_delay:.6e}')

# Scatter plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(labels[:, 0], preds[:, 0], s=8)

print(f'label_len = {len(labels[:,0])}, determ_len = {len(determ[:])}')
plt.scatter(labels[:, 0], determ[:], c='red')
plt.xlabel('True delay')
plt.ylabel('Predicted delay')
plt.title('Delay: true vs predicted')
mn, mx = min(labels[:, 0].min(), preds[:, 0].min()), max(labels[:, 0].max(), preds[:, 0].max())
plt.plot([mn, mx], [mn, mx], 'r--')

plt.subplot(1, 2, 2)
plt.scatter(labels[:, 1], preds[:, 1], s=8)
plt.xlabel('Packet-loss')
plt.ylabel('Packet-loss')
plt.title('Packet-loss')
mn, mx = min(labels[:, 1].min(), preds[:, 1].min()), max(labels[:, 1].max(), preds[:, 1].max())
plt.plot([mn, mx], [mn, mx], 'r--')
plt.tight_layout()

plt.show()