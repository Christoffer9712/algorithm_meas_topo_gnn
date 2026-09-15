"""
Train MeasurementEmbedder + Predictor on generated dataset.

"""
import os
import torch
import statistics
from torch.utils.data import DataLoader
import copy

from algorithm.predictor.meas_predictor import MeasurementEmbedder
from algorithm.predictor.predictor import Predictor
from .dataset import PredictorDataset
from algorithm.predictor.path_encoder import GraphEncoder, HeteroGATv2Encoder
import random

INCLUDE_MEAS = True

# Each overlay embedding is the concatenation of its 4 path-node embeddings
# [AC, SA, GW, TG], so the topology embedding width is 4 * out_dim.
OUT_DIM = 64
TOPO_DIM = 4 * OUT_DIM          # 256


def train(dataset_path=None, model_dir=None, epochs=100, batch_size=32, lr=5e-4,
          device='cpu', min_delta=1e-4, patience=15, transfer_learning_model=None):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    dataset = PredictorDataset(dataset_path)

    encoder = HeteroGATv2Encoder(hidden_dim=64, out_dim=OUT_DIM, heads=2,
                                 n_layers=3, edge_dim=1, dropout=0.2).to(device)
    graph_encoder = GraphEncoder(encoder, device=device)

    # measurement embedder now operates on the concatenated topology embedding
    embedder = MeasurementEmbedder(h_dim=TOPO_DIM).to(device)
    # Predictor input: h_topo (TOPO_DIM) + g (64) + horizon_m (1)
    predictor_topo = Predictor(in_dim=TOPO_DIM, hidden_dim=128, num_layers=3, dropout=0.2).to(device)
    predictor_meas = Predictor(in_dim=2, hidden_dim=2, num_layers=1, dropout=0.2).to(device)

    params = list(encoder.parameters()) + list(embedder.parameters()) + list(predictor_topo.parameters()) + list(predictor_meas.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3
    )


    if transfer_learning_model:
        model_path = os.path.join(model_dir, transfer_learning_model)
        ckpt = torch.load(model_path, map_location='cpu')
        if 'embedder_state' in ckpt and 'predictor_topo_state' in ckpt and 'predictor_topo_meas' in ckpt and 'encoder_state' in ckpt:
            embedder.load_state_dict(ckpt['embedder_state'])
            predictor_topo.load_state_dict(ckpt['predictor_topo_state'])
            predictor_meas.load_state_dict(ckpt['predictor_topo_meas'])
            encoder.load_state_dict(ckpt['encoder_state'])
            mean_delay = ckpt['mean_delay']
            mean_loss = ckpt['mean_loss']
            std_dev_delay = ckpt['std_dev_delay']
            std_dev_loss = ckpt['std_dev_loss']
            print(f"Loaded model at {model_path}")
            opt = torch.optim.Adam([
                {'params': encoder.parameters(),   'lr': lr/10},   # gentle since the encoder carries mostly topology info already trained
                {'params': embedder.parameters(),  'lr': lr},
                {'params': predictor_topo.parameters(), 'lr': lr},
                {'params': predictor_meas.parameters(), 'lr': lr},
            ])
        else:
            print(f'Could not load model = {transfer_learning_model}')

    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 5.0

    nbr_sims = dataset.get_nbr_of_sims()

    # Prediction horizon: M
    M = 1
    # Measurement history depth: Delta
    Delta = 2

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

    def run_t(t, gidx, data, return_attention, ep, h_topo):
        curr_overlay = data[(gidx, t)]['overlay_paths']
        curr_overlay_ids = [ovl['id'] for ovl in curr_overlay]

        H_hist = []
        Meas_hist = []
        Elapsed = []
        hist_ids = []

        sim_loss = torch.zeros((), device=device)
        sim_count = 0
        if INCLUDE_MEAS:
            for i in range(0, Delta):
                d = dataset[gidx][t - i]
                hist_overlay = d['overlay_paths']
                # encode the whole graph for time (t-i) ONCE, for all overlays
                missing = [ovl for ovl in hist_overlay if (ovl['id'], t - i) not in h_topo]
                if missing:
                    enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t - i)], missing)
                    for entry in enc.keys():
                        h_topo[(entry, t - i)] = enc[entry]
                for ovl in hist_overlay:
                    H_hist.append(h_topo[(ovl['id'], t - i)])
                    Meas_hist.append(torch.tensor(ovl['meas'][:2], dtype=torch.float, device=device))
                    Elapsed.append(float(i))
                    hist_ids.append((ovl['id'], t - i))

            if len(H_hist) > 0:  # Can be 0 if there are no overlays at given time
                H_hist = torch.stack(H_hist)
                Meas_hist = torch.stack(Meas_hist)
                Meas_hist = torch.stack([
                    (Meas_hist[:, 0] - mean_delay) / (std_dev_delay + 1e-8),
                    (Meas_hist[:, 1] - mean_loss) / (std_dev_loss + 1e-8),
                ], dim=1)
                Elapsed = torch.tensor(Elapsed, device=device).unsqueeze(1)

                # ---- LOO similarity objective on a random subset ----
                K = min(5, len(hist_ids))
                if K >= 2:  # need at least a query + one other
                    sample_idx = random.sample(range(len(hist_ids)), K)  # no replacement, in range
                    Meas_sim = Meas_hist[sample_idx]
                    Elapsed_sim = Elapsed[sample_idx]
                    H_sim = H_hist[sample_idx]

                    for i in range(K):
                        self_mask = torch.zeros(K, dtype=torch.bool, device=device)
                        self_mask[i] = True                         # leave out entry i
                        # recency relative to the query being reconstructed
                        rel_elapsed = (Elapsed_sim - Elapsed_sim[i]).abs().reshape(-1)
                        meas_hat = embedder.reconstruct_loo(H_sim[i], H_sim, Meas_sim, rel_elapsed, self_mask)
                        tgt_std = Meas_sim[i].detach()
                        sim_loss = sim_loss + ((meas_hat - tgt_std) ** 2).sum()
                        sim_count += 1

        loss = torch.zeros((), device=device)
        count = 0
        for m in range(1, M + 1):
            fut_overlay = data[(gidx, t + m)]['overlay_paths']

            eligible = [f_ovl for f_ovl in fut_overlay
                        if f_ovl['id'] in curr_overlay_ids]
            if not eligible:
                continue

            # one encode of the whole underlay graph, then per-overlay lookup
            enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t + m)], eligible)
            for entry in enc.keys():
                h_topo[(entry, t + m)] = enc[entry]

            for f_ovl in eligible:
                ovl_id = f_ovl['id']
                if INCLUDE_MEAS:
                    g = embedder(h_topo[(ovl_id, t + m)], H_hist, Meas_hist, Elapsed+m)
                else:
                    g = torch.zeros(2)
                out_topo = predictor_topo(h_topo[(ovl_id, t+m)])
                out_meas = predictor_meas(g)
                out = out_topo + out_meas
                tgt = torch.tensor(f_ovl['meas'], dtype=torch.float, device=device)

                z0 = out[0] - (tgt[0] - mean_delay) / (std_dev_delay + 1e-8)
                z1 = out[1] - (tgt[1] - mean_loss) / (std_dev_loss + 1e-8)

                loss = loss + (c_delta * z0 ** 2 + c_lambda * z1 ** 2)
                count += 1

        return loss, count, sim_loss, sim_count, h_topo

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
        encoder.train(); embedder.train(); predictor_topo.train(); predictor_meas.train()
        train_loss = 0.0
        train_loss_sim = 0.0
        batch_sample = 1
        loss_batch = torch.zeros((), device=device)
        loss_batch_sim = torch.zeros((), device=device)
        h_topo = {}
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            loss_sim = torch.zeros((), device=device); count_sim = 0
            for gidx in range(nbr_sims):
                loss_gidx, count_gidx, sim_loss_gidx, sim_count_gidx, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_topo=h_topo)
                loss = loss + loss_gidx
                count = count + count_gidx
                loss_sim = loss_sim + sim_loss_gidx
                count_sim = count_sim + sim_count_gidx

            if count == 0 or count_sim == 0:
                continue
            step_loss = loss / count
            loss_batch = loss_batch + step_loss
            train_loss += float((step_loss).detach())

            step_loss_sim = loss_sim / count_sim
            loss_batch_sim = loss_batch_sim + step_loss_sim
            train_loss_sim += float((step_loss_sim).detach())

            if batch_sample >= batch_size:
                opt.zero_grad()
                full_loss = loss_batch + beta * loss_batch_sim
                full_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                batch_sample = 1
                loss_batch = torch.zeros((), device=device)
                loss_batch_sim = torch.zeros((), device=device)
                h_topo = {}
            else:
                batch_sample = batch_sample + 1

        train_loss = train_loss / (len(train_t))
        train_loss_sim = train_loss_sim / (len(train_t))
        train_combined = train_loss + beta * train_loss_sim

        # ---- validate ----
        encoder.eval(); embedder.eval(); predictor_topo.eval(); predictor_meas.eval()
        val_loss = 0.0
        val_loss_sim = 0.0
        h_topo = {}
        with torch.no_grad():
            for t in val_t:
                loss = torch.zeros((), device=device); count = 0
                loss_sim = torch.zeros((), device=device); count_sim = 0
                for gidx in range(nbr_sims):
                    loss_gidx, count_gidx, sim_loss_gidx, sim_count_gidx, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_topo=h_topo)
                    loss = loss + loss_gidx
                    count = count + count_gidx
                    loss_sim = loss_sim + sim_loss_gidx
                    count_sim = count_sim + sim_count_gidx

                if count == 0 or count_sim == 0:
                    continue
                val_loss += float((loss / count).detach())
                val_loss_sim += float((loss_sim / count_sim).detach())

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
                'embedder_state': copy.deepcopy(embedder.state_dict()),
                'predictor_topo_state': copy.deepcopy(predictor_topo.state_dict()),
                'predictor_meas_state': copy.deepcopy(predictor_meas.state_dict()),
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
        embedder.load_state_dict(best_state['embedder_state'])
        predictor_topo.load_state_dict(best_state['predictor_topo_state'])
        predictor_meas.load_state_dict(best_state['predictor_meas_state'])

    # Save models
    torch.save({'encoder_state': encoder.state_dict(), 'embedder_state': embedder.state_dict(),
                'predictor_topo_state': predictor_topo.state_dict(),
                'predictor_meas_state': predictor_meas.state_dict(),
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