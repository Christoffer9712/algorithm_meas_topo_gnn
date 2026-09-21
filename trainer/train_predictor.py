"""
Train MeasurementEmbedder + Predictor on generated dataset.

"""
import os
import torch
import statistics
from torch.utils.data import DataLoader
import copy

from algorithm.predictor.f_cov import f_cov
from algorithm.predictor.f_topo import f_topo
from .dataset import PredictorDataset
from algorithm.predictor.path_encoder import GraphEncoder, GATv2
import random

INCLUDE_MEAS = False

# Each overlay embedding is the concatenation of its 4 path-node embeddings
# [AC, SA, GW, TG], so the topology embedding width is 4 * out_dim.
OUT_DIM = 64
TOPO_DIM = 4 * OUT_DIM          # 256

def train(dataset_path=None, model_dir=None, epochs=60, batch_size=16, lr=0.5e-3,
          device='cpu', min_delta=1e-4, patience=15, transfer_learning_model=None):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    dataset = PredictorDataset(dataset_path)

    encoder = GATv2(hidden_dim=64, out_dim=OUT_DIM, heads=2,
                            n_layers=3, edge_dim=1, dropout=0.1).to(device)
    graph_encoder = GraphEncoder(encoder, device=device)

    # measurement embedder now operates on the concatenated topology embedding
    fcov = f_cov(h_dim=TOPO_DIM).to(device)
    # Predictor input: h_enc (TOPO_DIM) + g (64) + horizon_m (1)
    ftopo = f_topo(in_dim=TOPO_DIM, hidden_dim=64, dropout=0.1).to(device)

    # Count only trainable parameters
    print(f"Encoder trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
    print(f"Fcov trainable parameters: {sum(p.numel() for p in fcov.parameters() if p.requires_grad)}")
    print(f"Ftopo trainable parameters: {sum(p.numel() for p in ftopo.parameters() if p.requires_grad)}")

    opt = torch.optim.Adam(list(encoder.parameters()) + list(fcov.parameters())
                           + list(ftopo.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.75, patience=6
    )

    if transfer_learning_model:
        model_path = os.path.join(model_dir, transfer_learning_model)
        ckpt = torch.load(model_path, map_location='cpu')
        if 'fcov_state' in ckpt and 'ftopo_state' in ckpt and 'encoder_state' in ckpt:
            fcov.load_state_dict(ckpt['fcov_state'])
            ftopo.load_state_dict(ckpt['ftopo_state'])
            encoder.load_state_dict(ckpt['encoder_state'])
            print(f"Loaded model at {model_path}")
            opt = torch.optim.Adam([
                {'params': encoder.parameters(), 'lr': lr},   # gentle since the encoder carries mostly topology info already trained
                {'params': fcov.parameters(), 'lr': lr},
                {'params': ftopo.parameters(), 'lr': lr},
            ])
            
        else:
            print(f"Could not load model at {model_path}")


    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 5.0

    nbr_sims = dataset.get_nbr_of_sims()

    # Prediction horizon: M
    M = 1
    # Measurement history depth: Delta
    Delta = 3

    # Contiguous time-based split (no shuffling across splits -> no leakage)
    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    train_t = all_t[: int(0.70 * n)]
    val_t   = all_t[int(0.70 * n):]
    test_t  = all_t[int(0.85 * n):]

    meas_delay = []
    meas_loss = []
    for gidx in range(nbr_sims - 1):
        for t in train_t:
            curr_overlay = dataset[gidx][t]['overlay_paths']
            for ovl in curr_overlay:
                ovl_meas = ovl['meas']
                meas_delay.append(ovl_meas[0])
                meas_loss.append(ovl_meas[1])

    mean_loss = statistics.mean(meas_loss)
    print(f"Mean of the loss is {mean_loss}")
    mean_delay = statistics.mean(meas_delay)
    print(f"Mean of the delay is {mean_delay}")

    std_dev_loss = statistics.stdev(meas_loss) if len(set(meas_loss)) > 1 else 1.0
    print(f"Standard Deviation of the loss is {std_dev_loss}")
    std_dev_delay = statistics.stdev(meas_delay)
    print(f"Standard Deviation of the delay is {std_dev_delay}")

    print(f"Baseline (predict-mean) train = {baseline_loss_mean(train_t, dataset, 0, M, mean_loss, std_dev_loss, mean_delay, std_dev_delay, c_lambda, c_delta):.4f}")
    print(f"Baseline (predict-mean) val   = {baseline_loss_mean(val_t, dataset, 0, M, mean_loss, std_dev_loss, mean_delay, std_dev_delay, c_lambda, c_delta):.4f}")
    print(f"Baseline (predict-mean) test  = {baseline_loss_mean(test_t, dataset, 0, M, mean_loss, std_dev_loss, mean_delay, std_dev_delay, c_lambda, c_delta):.4f}")

    def run_t(t, gidx, data, return_attention, ep, h_enc, h_topo):
        curr_overlay = data[(gidx, t)]['overlay_paths']
        curr_overlay_ids = [ovl['id'] for ovl in curr_overlay]

        H_hist = []
        Meas_hist = []
        Elapsed = []
        Hist_ids = []
        Topo_hist = []

        sim_loss = torch.zeros((), device=device)
        sim_count = 0
        loss = torch.zeros((), device=device)
        count = 0
    
        if INCLUDE_MEAS:
            for i in range(0, Delta):
                d = dataset[gidx][t - i]
                hist_overlay = d['overlay_paths']
                # encode the whole graph for time (t-i) ONCE, for all overlays
                missing = [ovl for ovl in hist_overlay if (ovl['id'], t - i) not in h_enc]
                if missing:
                    enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t - i)], missing)
                    topo = ftopo(torch.stack([enc[i] for i in list(enc.keys())]))
                    for idx, entry in enumerate(enc.keys()):
                        h_enc[(entry, t - i)] = enc[entry]
                        h_topo[(entry, t - i)] = topo[idx]
                        
                for ovl in hist_overlay:
                    H_hist.append(h_enc[(ovl['id'], t - i)])
                    Meas_hist.append(torch.tensor(ovl['meas'][:2], dtype=torch.float, device=device))
                    Topo_hist.append(h_topo[(ovl['id'], t-i)])
                        
                    Elapsed.append(float(i))
                    Hist_ids.append((ovl['id'], t - i))

            if len(H_hist) > 0:  # Can be 0 if there are no overlays at given time
                H_hist = torch.stack(H_hist)
                Meas_hist = torch.stack(Meas_hist)
                Meas_hist = torch.stack([
                    (Meas_hist[:, 0] - mean_delay) / (std_dev_delay + 1e-8),
                    (Meas_hist[:, 1] - mean_loss) / (std_dev_loss + 1e-8),
                ], dim=1)
                Elapsed = torch.tensor(Elapsed, device=device).unsqueeze(1)
                Topo_hist = torch.stack(Topo_hist)

        for m in range(1, M + 1):
            fut_overlay = data[(gidx, t + m)]['overlay_paths']

            eligible = [f_ovl for f_ovl in fut_overlay
                        if f_ovl['id'] in curr_overlay_ids]
            if not eligible:
                continue

            # one encode of the whole underlay graph, then per-overlay lookup
            enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t + m)], eligible)
            for entry in enc.keys():
                h_enc[(entry, t + m)] = enc[entry]

            for f_ovl in eligible:
                tgt = torch.tensor(f_ovl['meas'], dtype=torch.float, device=device)
                tgt[0] = (tgt[0] - mean_delay) / (std_dev_delay + 1e-8)
                tgt[1] = (tgt[1] - mean_loss) / (std_dev_loss + 1e-8)

                ovl_id = f_ovl['id']
                enc = h_enc[(ovl_id, t + m)]
                out_topo = ftopo(enc)
                h_topo[(ovl_id, t+m)] = out_topo

                if INCLUDE_MEAS:
                    enc_rep = enc.unsqueeze(0).expand(len(H_hist), -1)
                    (s, l) = fcov(enc_rep, H_hist, Elapsed + m)

                    d_i0 = tgt[0] - out_topo[0]                        # query deviation
                    d_j0 = Meas_hist[:,0] - Topo_hist[:,0]  # neighbor deviation
                    resid = d_i0 - d_j0 * s
                    sim_loss = sim_loss + (resid**2 * torch.exp(l) - l).sum()
                    sim_count = sim_count + len(H_hist)

                    w = torch.exp(l)                                    # [N] precision weights
                    dev_hist = Meas_hist - Topo_hist                    # [N,2] neighbor deviations (see below)
                    est = s.unsqueeze(1) * dev_hist                     # [N,2] each neighbor's estimate of d_i
                    out_meas = (w.unsqueeze(1) * est).sum(0) / (w.sum() + 1e-8)   # [2]
                else:
                    out_meas = torch.zeros((2), dtype=torch.float, device=device)

                out = out_topo + out_meas

                z0 = out[0] - tgt[0]
                z1 = out[1] - tgt[1]

                #cheating = True
                #if cheating:
                #    loss_cheating = (out_topo - tgt[2])**2 + (out_meas - tgt[3])**2 
                #    loss = loss + loss_cheating

                loss = loss + (c_delta * z0 ** 2 + c_lambda * z1 ** 2)
                count += 1

        return loss, count, sim_loss, sim_count, h_enc, h_topo

    batch_size = batch_size / (nbr_sims - 1)  # For each step, train over all sims
    beta = 0.2

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    # ---- load the pre-built HeteroData graphs once ----
    data = {}
    for gidx in range(nbr_sims):
        for t in range(len(dataset)):
            d = dataset[gidx][t]
            tmp = d['data'].to(device)               # already a HeteroData
            tmp.overlay_paths = d['overlay_paths']   # attach measurements for lookup
            data[(gidx, t)] = tmp

    for ep in range(epochs):
        # ---- train ----
        encoder.train(); fcov.train(); ftopo.train()
        train_loss = 0.0
        train_loss_sim = 0.0
        batch_sample = 1
        loss_batch = torch.zeros((), device=device)
        loss_batch_sim = torch.zeros((), device=device)
        h_enc = {}
        h_topo = {}
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            loss_sim = torch.zeros((), device=device); count_sim = 0
            for gidx in range(nbr_sims):
                loss_gidx, count_gidx, sim_loss_gidx, sim_count_gidx, h_enc, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_enc=h_enc, h_topo=h_topo)
                loss = loss + loss_gidx
                count = count + count_gidx
                loss_sim = loss_sim + sim_loss_gidx
                count_sim = count_sim + sim_count_gidx

            step_loss = loss / count if count > 0 else torch.zeros((), dtype=torch.float, device=device)
            loss_batch = loss_batch + step_loss
            train_loss += float((step_loss).detach())

            step_loss_sim = loss_sim / count_sim if count_sim > 0 else torch.zeros((), dtype=torch.float, device=device)
            loss_batch_sim = loss_batch_sim + step_loss_sim
            train_loss_sim += float((step_loss_sim).detach())

            if batch_sample >= batch_size:
                opt.zero_grad()
                full_loss = loss_batch + beta * loss_batch_sim
                full_loss.backward()
                opt.step()
                batch_sample = 1
                loss_batch = torch.zeros((), device=device)
                loss_batch_sim = torch.zeros((), device=device)
                h_enc = {}
                h_topo = {}
            else:
                batch_sample = batch_sample + 1

        train_loss = train_loss / (len(train_t))
        train_loss_sim = train_loss_sim / (len(train_t))
        train_combined = train_loss + beta * train_loss_sim

        # ---- validate ----
        encoder.eval(); fcov.eval(); ftopo.eval()
        val_loss = 0.0
        val_loss_sim = 0.0
        h_enc = {}
        h_topo = {}
        with torch.no_grad():
            for t in val_t:
                loss = torch.zeros((), device=device); count = 0
                loss_sim = torch.zeros((), device=device); count_sim = 0
                for gidx in range(nbr_sims):
                    loss_gidx, count_gidx, sim_loss_gidx, sim_count_gidx, h_enc, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_enc=h_enc, h_topo=h_topo)
                    loss = loss + loss_gidx
                    count = count + count_gidx
                    loss_sim = loss_sim + sim_loss_gidx
                    count_sim = count_sim + sim_count_gidx

                val_loss += float((loss / count).detach()) if count > 0 else 0.0
                val_loss_sim += float((loss_sim / count_sim).detach()) if count_sim > 0 else 0.0

            val_loss = val_loss / (len(val_t))
            val_loss_sim = val_loss_sim / (len(val_t))
            val_combined = val_loss + beta * val_loss_sim

        scheduler.step(val_loss)
        # ---- early stopping ----
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {
                'encoder_state': copy.deepcopy(encoder.state_dict()),
                'fcov_state': copy.deepcopy(fcov.state_dict()),
                'ftopo_state': copy.deepcopy(ftopo.state_dict()),
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best val_loss = {best_val:.6f})")
                break

        msg = f"""
              Epoch: {ep}
              train_loss     = {train_loss}
              train_loss_loo = {train_loss_sim}
              train_combined = {train_combined}
              val_loss       = {val_loss}
              val_loss_loo   = {val_loss_sim}
              val_combined   = {val_combined}
              lr             = {opt.param_groups[0]["lr"]}
              """
        print(msg)

    # restore best weights before save
    if best_state is not None:
        encoder.load_state_dict(best_state['encoder_state'])
        fcov.load_state_dict(best_state['fcov_state'])
        ftopo.load_state_dict(best_state['ftopo_state'])

    # Save models
    torch.save({'encoder_state': encoder.state_dict(), 'fcov_state': fcov.state_dict(),
                'ftopo_state': ftopo.state_dict(),
                'mean_delay': mean_delay, 'mean_loss': mean_loss,
                'std_dev_delay': std_dev_delay, 'std_dev_loss': std_dev_loss},
               os.path.join(model_dir, 'predictor_models.pth'))
    print('Saved trained models to', os.path.join(model_dir, 'predictor_models.pth'))
    return os.path.join(model_dir, 'predictor_models.pth')


def baseline_loss_mean(split_t, dataset, gidx, M, mean_loss, std_dev_loss, mean_delay, std_dev_delay, c_lambda, c_delta):
    total, n_t = 0.0, 0
    for t in split_t:
        curr_overlay = dataset[gidx][t]['overlay_paths']
        loss, count = 0.0, 0
        for m in range(1, M + 1):
            d = dataset[gidx][t + m]
            for ovl in curr_overlay:
                ovl_id = ovl['id']
                tgt = next((path['meas'] for path in d['overlay_paths'] if path['id'] == ovl_id), None)
                if tgt is not None:
                    z0 = (tgt[0] - mean_delay) / (std_dev_delay + 1e-8)
                    z1 = (tgt[1] - mean_loss) / (std_dev_loss + 1e-8)
                    loss += c_delta * z0 ** 2 + c_lambda * z1 ** 2
                    count += 1
        if count:
            total += loss / count
            n_t += 1
    return total / n_t if n_t else float('nan')


if __name__ == '__main__':
    train(epochs=10, batch_size=32)