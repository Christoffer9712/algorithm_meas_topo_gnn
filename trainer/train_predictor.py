"""
Train MeasurementEmbedder + Predictor on generated dataset.

Saves the trained weights to models/predictor_embedder.pth and
models/predictor_predictor.pth (two modules).
"""
import os
import torch
from torch.utils.data import DataLoader

from meas_predictor import MeasurementEmbedder, Predictor
from trainer.dataset import PredictorDataset
from pathEncoder import GATv2Encoder, overlay_encodings
from torch_geometric.data import Data


def train(dataset_path=None, model_dir=None, epochs=20, batch_size=64, lr=3e-4, device='cpu'):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    dataset = PredictorDataset(dataset_path)

    encoder = GATv2Encoder(in_dim=8, hidden_dim=32, out_dim=32, edge_dim=1).to(device)
    embedder = MeasurementEmbedder(h_dim=32, hidden_dim=32, out_dim=32).to(device)
    # Predictor input: h_topo (D) + g (64) + horizon_m (1)
    predictor = Predictor(in_dim=32 + 32 + 1, hidden_dim=32).to(device) # Input is H = [topo_embed || meas_embed || m]

    opt = torch.optim.Adam(list(encoder.parameters()) + list(embedder.parameters()) + list(predictor.parameters()), lr=lr)

    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 1.0

    # Prediction horizon: M
    M = 5

    # Measurement history depth: Delta
    Delta = 3

    FIXED = {'t', 'x', 'edge_index', 'edge_attr', 'node_names', 'name_to_idx'}

    # Contiguous time-based split (no shuffling across splits -> no leakage)
    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    train_t = all_t[: int(0.70 * n)]
    val_t   = all_t[int(0.70 * n) : int(0.85 * n)]
    test_t  = all_t[int(0.85 * n) :]

    def run_t(t):
        curr_overlay = [{k:dataset[t][k]} for k in dataset[t] if k not in FIXED]

        H_hist = []
        Meas_hist = []
        Elapsed = []
        for i in range(0, Delta):
            d = dataset[t - i]
            data = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'])
            hist_overlay = [{k:d[k]} for k in d if k not in FIXED]
            h_hist = overlay_encodings(data, hist_overlay, encoder, device="cpu")
            for ovl_dict in hist_overlay:
                id, ovl = next(iter(ovl_dict.items()))
                H_hist.append(h_hist[id])
                Meas_hist.append(torch.tensor(ovl['meas'], dtype=torch.float))
                Elapsed.append(float(i))

        H_hist = torch.stack(H_hist)
        Meas_hist = torch.stack(Meas_hist)
        Elapsed = torch.tensor(Elapsed).unsqueeze(1)

        loss = torch.zeros((), device=device)
        count = 0
        for m in range(1, M+1):
            d = dataset[t + m]
            data = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'])
            # Using overlays from time t since it's unknown what future overlays will exist
            h_topo = overlay_encodings(data, curr_overlay, encoder, device="cpu")
            for ovl in curr_overlay:
                ovl_id, _ = next(iter(ovl.items()))
                g = embedder(h_topo[ovl_id], H_hist, Meas_hist, Elapsed)
                H = torch.cat([h_topo[ovl_id], g, torch.tensor([float(m)])], dim=0).unsqueeze(0)
                out = predictor(H)
                if ovl_id in d.keys():
                    tgt = torch.tensor(d[ovl_id]['meas'], dtype=torch.float)
                    loss = loss + c_lambda * 100 * (out[0,0] - tgt[0])**2 + c_delta * (0.01*out[0,1] - 0.01*tgt[1])**2
                    count += 1
                else:
                    pass #(ToDO later)
        return loss, count

    for ep in range(1, epochs + 1):
        # ---- train ----
        encoder.train(); embedder.train(); predictor.train()
        total_loss = 0.0
        for t in train_t:
            loss, count = run_t(t)
            if count == 0:
                continue
            loss = loss / count
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())

        # ---- validate ----
        encoder.eval(); embedder.eval(); predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t in val_t:
                loss, count = run_t(t)
                if count == 0:
                    continue
                val_loss += float((loss / count).detach())

        print(f"Epoch: {ep} - total_loss = {total_loss} - val_loss = {val_loss}")

    # ---- test (once, after training) ----
    encoder.eval(); embedder.eval(); predictor.eval()
    test_loss = 0.0
    with torch.no_grad():
        for t in test_t:
            loss, count = run_t(t)
            if count == 0:
                continue
            test_loss += float((loss / count).detach())
    print(f"Test loss = {test_loss}")

    # Save models
    torch.save({'encoder_state': encoder.state_dict(), 'embedder_state': embedder.state_dict(), 'predictor_state': predictor.state_dict()},
                os.path.join(model_dir, 'predictor_models.pth'))
    print('Saved trained models to', os.path.join(model_dir, 'predictor_models.pth'))
    return os.path.join(model_dir, 'predictor_models.pth')

if __name__ == '__main__':
    train(epochs=10, batch_size=32)
