#!/usr/bin/env python3
"""
Standalone visualization script that mirrors visualize_predictions.ipynb.
Saves scatter plots of predicted vs true packet-loss and delay to outputs/.
"""
import sys, os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure repo root is on path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

from trainer.dataset import PredictorDataset
from meas_predictor import MeasurementEmbedder, Predictor

# Paths
dataset_path = os.path.join(repo_root, 'data', 'predictor_dataset.pt')
model_path = os.path.join(repo_root, 'models', 'predictor_models.pth')
output_dir = os.path.join(repo_root, 'outputs')
os.makedirs(output_dir, exist_ok=True)

print('Dataset path:', dataset_path)
print('Model path:', model_path)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run trainer.generate_dataset.generate first.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run trainer.train_predictor.train first.")

# Load dataset
dataset = PredictorDataset(dataset_path)
N = min(200, len(dataset))
print(f'Loading {N} samples from dataset (total {len(dataset)})')

samples = [dataset[i] for i in range(N)]

# Infer embedding size
h_pred = samples[0][0]
D = h_pred.shape[0]
print('Embedding dim D =', D)

# Build models and load checkpoint
embedder = MeasurementEmbedder(h_dim=D, hidden_dim=128, out_dim=64)
predictor = Predictor(in_dim=D + 64 + 1, hidden_dim=128)
ckpt = torch.load(model_path, map_location='cpu')
if 'embedder_state' in ckpt and 'predictor_state' in ckpt:
    embedder.load_state_dict(ckpt['embedder_state'])
    predictor.load_state_dict(ckpt['predictor_state'])
else:
    # try single-dict
    try:
        embedder.load_state_dict(ckpt)
    except Exception:
        raise RuntimeError('Checkpoint format not recognized; ensure models/predictor_models.pth created by trainer.')

embedder.eval(); predictor.eval()

preds = []
labels = []

for s in samples:
    # s = (h_pred, h_hists, meas_vals, elapsed, label, horizon)
    h_pred, h_hists, meas_vals, elapsed, label, horizon = s
    with torch.no_grad():
        g = embedder(h_pred, h_hists, meas_vals, elapsed)
        H = torch.cat([h_pred, g, torch.tensor([horizon], dtype=torch.float32)], dim=0).unsqueeze(0)
        out = predictor(H).squeeze(0).numpy()
    preds.append(out)
    labels.append(label.numpy())

preds = np.array(preds)
labels = np.array(labels)

# compute simple metrics
mse_loss = np.mean((preds[:,0] - labels[:,0])**2)
mse_delay = np.mean((preds[:,1] - labels[:,1])**2)
print(f'MSE packet-loss: {mse_loss:.6e}, MSE delay(ms): {mse_delay:.6e}')

# Scatter plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(labels[:,0], preds[:,0], s=8)
plt.xlabel('True packet-loss (fraction)')
plt.ylabel('Predicted packet-loss')
plt.title('Packet-loss: true vs predicted')
mn, mx = min(labels[:,0].min(), preds[:,0].min()), max(labels[:,0].max(), preds[:,0].max())
plt.plot([mn,mx],[mn,mx],'r--')

plt.subplot(1,2,2)
plt.scatter(labels[:,1], preds[:,1], s=8)
plt.xlabel('True delay (ms)')
plt.ylabel('Predicted delay (ms)')
plt.title('Delay (ms): true vs predicted')
mn, mx = min(labels[:,1].min(), preds[:,1].min()), max(labels[:,1].max(), preds[:,1].max())
plt.plot([mn,mx],[mn,mx],'r--')
plt.tight_layout()

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
