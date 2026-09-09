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
from algorithm.predictor.meas_predictor import MeasurementEmbedder
from algorithm.predictor.predictor import Predictor
from algorithm.predictor.path_encoder import GATv2Encoder, GraphEncoder
from algorithm.predictor.hetero_encoder import HeteroGATv2Encoder

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

gidx = 13 #OBS!
samples = [dataset[gidx][i] for i in range(N)]
print(f'Sample len = {len(samples)}')

# Build models and load checkpoint
encoder = HeteroGATv2Encoder(node_in=10, virtual_in=1, hidden_dim=128, out_dim=64, heads=2, n_layers=3, dropout=0.0)
graph_encoder = GraphEncoder(encoder)

embedder = g = torch.zeros(64) #MeasurementEmbedder(h_dim=64, hidden_dim=128, out_dim=64)
# Predictor input: h_topo (D) + g (64) + horizon_m (1)
predictor = Predictor(in_dim=64 + 64 + 1, hidden_dim=128, dropout=0.0)
ckpt = torch.load(model_path, map_location='cpu')

if 'embedder_state' in ckpt and 'predictor_state' in ckpt and 'encoder_state' in ckpt:
    #embedder.load_state_dict(ckpt['embedder_state'])
    predictor.load_state_dict(ckpt['predictor_state'])
    encoder.load_state_dict(ckpt['encoder_state'])
    mean_delay = ckpt['mean_delay']
    mean_loss = ckpt['mean_loss']
    std_dev_delay = ckpt['std_dev_delay']
    std_dev_loss = ckpt['std_dev_loss']

#embedder.eval()
encoder.eval()
predictor.eval()

graph_encoder = GraphEncoder(encoder, device='cpu')

preds = []
labels = []
determ = []

with torch.no_grad():
    for d in samples:
        data = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'])
        ovls = d['overlay_paths']
        for ovl in ovls:
            ovls_encodings = graph_encoder.encode_overlays_pyg(data, [ovl])
            horizon = 1

            h_topo = ovls_encodings[ovl['id']]
            H = torch.cat([h_topo, embedder, torch.tensor([horizon], dtype=torch.float32)], dim=0).unsqueeze(0)
            out = predictor(H).squeeze(0).numpy()
            out[0] = out[0] * std_dev_delay + mean_delay
            out[1] = out[1] * std_dev_loss + mean_loss
            preds.append(out)
            labels.append(ovl['meas'])

            determ.append(path_distance_from_data(data, ovl['underlay_path'])/100)

'''
    # s = (h_pred, h_hists, meas_vals, elapsed, label, horizon)
    h_pred, h_hists, meas_vals, elapsed, label, horizon = s
    with torch.no_grad():
        #g = embedder(h_pred, h_hists, meas_vals, elapsed)
        H = torch.cat([h_pred, g, torch.tensor([horizon], dtype=torch.float32)], dim=0).unsqueeze(0)
        out = predictor(H).squeeze(0).numpy()
    preds.append(out)
    labels.append(label.numpy())
'''
preds = np.array(preds)
labels = np.array(labels)

# compute simple metrics
mse_delay = np.mean((preds[:,0] - labels[:,0])**2)
mse_loss = np.mean((preds[:,1] - labels[:,1])**2)
print(f'MSE packet-loss: {mse_loss:.6e}, MSE delay(ms): {mse_delay:.6e}')

# Scatter plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(labels[:,0], preds[:,0], s=8)

print(f'label_len = {len(labels[:,0])}, determ_len = {len(determ[:])}')
plt.scatter(labels[:,0], determ[:], c='red')
plt.xlabel('True delay')
plt.ylabel('Predicted delay')
plt.title('Delay: true vs predicted')
mn, mx = min(labels[:,0].min(), preds[:,0].min()), max(labels[:,0].max(), preds[:,0].max())
plt.plot([mn,mx],[mn,mx],'r--')

plt.subplot(1,2,2)
plt.scatter(labels[:,1], preds[:,1], s=8)
plt.xlabel('Packet-loss')
plt.ylabel('Packet-loss')
plt.title('Packet-loss')
mn, mx = min(labels[:,1].min(), preds[:,1].min()), max(labels[:,1].max(), preds[:,1].max())
plt.plot([mn,mx],[mn,mx],'r--')
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