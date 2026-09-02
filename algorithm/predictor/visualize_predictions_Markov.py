#!/usr/bin/env python3
"""
Visualize the Markov/GAT estimator: predicted vs measured delay and loss.

Loads the SatNetEstimator checkpoint, runs each overlay through the GAT +
absorbing-chain read-out, and scatters predictions against ovl['meas'].
A distance-based delay baseline is drawn for reference.
"""
import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt

from trainer.dataset import PredictorDataset
from algorithm.predictor.hetero_snapshot import hetero_from_record
from algorithm.predictor.hetero_model import SatNetEstimator, absorbing_endtoend


def path_distance(d, underlay_path, dist_scale=300.0):
    """Raw summed distance along an overlay's underlay path (reference baseline)."""
    name_to_idx = d["name_to_idx"]
    edge_index = np.asarray(d["edge_index"])
    edge_attr = np.asarray(d["edge_attr"])
    mapping = {(int(s), int(t)): i
               for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1]))}

    total = 0.0
    for n1, n2 in zip(underlay_path[:-1], underlay_path[1:]):
        a, b = name_to_idx[n1], name_to_idx[n2]
        pos = mapping.get((a, b), mapping.get((b, a)))
        if pos is None:
            raise ValueError(f"No edge between {n1} and {n2}")
        d_norm = float(edge_attr[pos, 0]) if edge_attr.ndim > 1 else float(edge_attr[pos])
        total += d_norm * dist_scale
    return total


# ---- paths ----
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
dataset_path = os.path.join(repo_root, "../../data", "predictor_dataset.pt")
model_path = os.path.join(repo_root, "../../models", "predictor_models.pth")
output_dir = os.path.join(repo_root, "../../outputs")
os.makedirs(output_dir, exist_ok=True)

print("Dataset path:", dataset_path)
print("Model path:", model_path)
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")

# ---- load dataset ----
dataset = PredictorDataset(dataset_path)
gidx = 0                                   # OBS! which simulation to visualize
N = min(400, len(dataset))
samples = [dataset[gidx][i] for i in range(N)]
print(f"Loading {N} timesteps from sim {gidx} (dataset len {len(dataset)})")

# ---- build model, size it from one promoted snapshot, load weights ----
device = "cpu"
probe = hetero_from_record(samples[0], device=device)
model = SatNetEstimator(
    node_in=probe["node"].x.shape[1],
    edge_in=probe["edge"].x.shape[1],
    hidden_dim=128, emb_dim=64, heads=2, n_layers=3,
).to(device)

ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
mean_delay, std_delay = ckpt["mean_delay"], ckpt["std_delay"]
mean_loss, std_loss = ckpt["mean_loss"], ckpt["std_loss"]

# ---- run predictions ----
preds, labels, determ = [], [], []

with torch.no_grad():
    for d in samples:
        data = hetero_from_record(d, device=device)
        pred = model(data)
        P = absorbing_endtoend.build_P(pred, data, device)

        for ovl in d["overlay_paths"]:
            path = ovl["underlay_path"]
            s = data.node_name_to_idx[path[0]]
            a = data.node_name_to_idx[path[-1]]
            if s == a:
                continue

            delay, survival = absorbing_endtoend.endtoend(P, pred, data, s, a, device)
            pred_loss = 1.0 - survival

            preds.append([float(delay), float(pred_loss)])
            labels.append(ovl["meas"])
            determ.append(path_distance(d, path) / 100.0)

preds = np.array(preds)
labels = np.array(labels)
determ = np.array(determ)

# ---- metrics ----
mse_delay = np.mean((preds[:, 0] - labels[:, 0]) ** 2)
mse_loss = np.mean((preds[:, 1] - labels[:, 1]) ** 2)
print(f"MSE delay = {mse_delay:.6e}   MSE loss = {mse_loss:.6e}")
print(f"(delay mean/std = {mean_delay:.3f}/{std_delay:.3f}, "
      f"loss mean/std = {mean_loss:.3f}/{std_loss:.3f})")

# ---- scatter plots ----
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(labels[:, 0], preds[:, 0], s=8, label="GAT+Markov")
plt.scatter(labels[:, 0], determ, s=8, c="red", alpha=0.4, label="distance baseline")
plt.xlabel("True delay")
plt.ylabel("Predicted delay")
plt.title("Delay: true vs predicted")
mn = min(labels[:, 0].min(), preds[:, 0].min())
mx = max(labels[:, 0].max(), preds[:, 0].max())
plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(labels[:, 1], preds[:, 1], s=8)
plt.xlabel("True packet-loss")
plt.ylabel("Predicted packet-loss")
plt.title("Packet-loss: true vs predicted")
mn = min(labels[:, 1].min(), preds[:, 1].min())
mx = max(labels[:, 1].max(), preds[:, 1].max())
plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)

plt.tight_layout()
out_file = os.path.join(output_dir, "pred_vs_true_markov.png")
plt.savefig(out_file, dpi=120)
print("Saved plot to", out_file)
plt.show()