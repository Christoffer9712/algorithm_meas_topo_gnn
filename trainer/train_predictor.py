"""
Train MeasurementEmbedder + Predictor on generated dataset.

Saves the trained weights to models/predictor_embedder.pth and
models/predictor_predictor.pth (two modules).
"""
import os
import torch
import torch.nn as nn
import statistics
from torch.utils.data import DataLoader
import copy
import numpy as np
import random

from algorithm.predictor.hetero_encoder import HeteroGATv2Encoder
from .dataset import PredictorDataset
from algorithm.predictor.path_encoder import GATv2Encoder, GraphEncoder
from torch_geometric.data import Data
from environment.environment import RoutingEnvironment
from algorithm.predictor.f_cov import f_cov
from algorithm.predictor.f_topo import f_topo

INCLUDE_MEAS = True
REMOVE_SELF_MEAS = True
REMOVE_SELF_MEAS_PERC = 1.0

# Path encoder: 'gat' = HeteroGATv2 over the full underlay graph,
#               'mlp' = small MLP on the overlay waypoint nodes only (no underlay topology)
ENCODER_TYPE = 'gat'
GRAD_CLIP = 1.0

class OverlayMLPEncoder(nn.Module):
    """Encodes an overlay path from its overlay nodes only (ovl['overlay_path']).

    Input per overlay: node features data.x of each waypoint (+ a valid flag),
    padded to n_wp slots, the hop distances between consecutive waypoints
    (positions data.x[:, 0:2]) and the total distance. Nothing from the underlay
    graph (edges, intermediate satellites, routing) is used.

    Same interface as GraphEncoder, so it can replace graph_encoder directly.
    """

    def __init__(self, node_in=10, n_wp=4, hidden_dim=128, out_dim=64, dropout=0.0, device='cpu'):
        super().__init__()
        self.node_in, self.n_wp, self.device = node_in, n_wp, device
        in_dim = n_wp * (node_in + 1) + (n_wp - 1) + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def overlay_input(self, data, ovl):
        idx = [data.name_to_idx[n] for n in ovl['overlay_path']]
        feats = data.x[idx].float()                            # (L, node_in)
        pos = feats[:, 0:2]
        hops = (pos[1:] - pos[:-1]).norm(dim=1)                # (L-1,)
        L = min(len(idx), self.n_wp)
        nodes = torch.zeros(self.n_wp, self.node_in + 1, device=feats.device)
        nodes[:L, :self.node_in] = feats[:L]
        nodes[:L, self.node_in] = 1.0                          # slot is a real waypoint
        hop_block = torch.zeros(self.n_wp - 1, device=feats.device)
        k = min(len(hops), self.n_wp - 1)
        hop_block[:k] = hops[:k]
        return torch.cat([nodes.flatten(), hop_block, hops.sum().reshape(1)])

    def encode_overlays_pyg_batched(self, data, overlays):
        X = torch.stack([self.overlay_input(data, o) for o in overlays])
        H = self.net(X)
        return {o['id']: H[k] for k, o in enumerate(overlays)}


def build_encoder(encoder_type, topo_dim, device='cpu'):
    """(encoder module with the parameters, object with encode_overlays_pyg_batched)."""
    if encoder_type == 'gat':
        encoder = HeteroGATv2Encoder(node_in=10, virtual_in=1, hidden_dim=128, out_dim=topo_dim, heads=2, n_layers=3, dropout=0.0).to(device)
        return encoder, GraphEncoder(encoder, device=device)
    if encoder_type == 'mlp':
        encoder = OverlayMLPEncoder(node_in=8, n_wp=4, hidden_dim=128, out_dim=topo_dim, dropout=0.0, device=device).to(device)
        return encoder, encoder
    raise ValueError(f"unknown encoder_type {encoder_type!r}, use 'gat' or 'mlp'")


def model_filename(encoder_type):
    return 'predictor_models.pth' if encoder_type == 'gat' else f'predictor_models_{encoder_type}.pth'


def train(dataset_path=None, model_dir=None, epochs=10, batch_size=16, lr=0.5e-3,
          device='cpu', min_delta=1e-4, patience=4, transfer_learning_model=None, encoder_type=None):
    encoder_type = encoder_type or ENCODER_TYPE
    print(f"Path encoder: {encoder_type}")
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    dataset = PredictorDataset(dataset_path)
    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 5.0

    nbr_sims = dataset.get_nbr_of_sims()

    # Prediction horizon: M
    M = 1 # NOTE!!!

    # Measurement history depth: Delta
    Delta = 3

    # Contiguous time-based split (no shuffling across splits -> no leakage)
    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    gap = M
    train_t = all_t[int(0.10*n)+gap:int(0.30*n)] + all_t[int(0.40*n)+gap:int(0.70*n)] + all_t[int(0.80*n)+gap:]
    val_t   = all_t[:int(0.10*n)-gap] + all_t[int(0.30*n)+gap:int(0.40*n)-gap] + all_t[int(0.70*n)+gap:int(0.80*n)-gap]

    test_t  = all_t # Used on unseen topology
    #train_t = val_t = test_t = all_t #OBS, use all data when validation is done on different simulation
    (mean_delay, mean_loss, std_dev_delay, std_dev_delay_tot, std_dev_loss) = calculate_mean_std(train_t, dataset[:-1], nbr_sims-1)

    print(f"Baseline (predict-mean) train = {baselines(train_t, dataset[:-1], nbr_sims-1, mean_delay.sum(), std_dev_delay_tot, c_delta)}")
    print(f"Baseline (predict-mean) val   = {baselines(val_t, dataset[:-1], nbr_sims-1, mean_delay.sum(), std_dev_delay_tot, c_delta)}")
    print(f"Baseline (predict-mean) test  = {baselines(test_t, [dataset[-1]], 1, mean_delay.sum(), std_dev_delay_tot, c_delta)}")

    TOPO_DIM = 64
    encoder, graph_encoder = build_encoder(encoder_type, TOPO_DIM, device)
    fcov = f_cov(in_dim=TOPO_DIM, dropout=0.0).to(device)
    ftopo = f_topo(in_dim=TOPO_DIM, hidden_dim=64, dropout=0.0).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(ftopo.parameters()) + list(fcov.parameters()), lr=lr)
    
    if transfer_learning_model:
        model_path = os.path.join(model_dir, transfer_learning_model)
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if ('fcov_state' in ckpt and 'ftopo_state' in ckpt and 'encoder_state' in ckpt
                and ckpt.get('encoder_type', 'gat') == encoder_type):
            #missing_keys, _ = fcov.load_state_dict(ckpt['fcov_state'], strict=False)  # identity weights may be new
            #if missing_keys:
            #    print(f"fcov: initialised new parameters {missing_keys}")
            ftopo.load_state_dict(ckpt['ftopo_state'])
            encoder.load_state_dict(ckpt['encoder_state'])
            print(f"Loaded model at {model_path}")
            opt = torch.optim.Adam([
                {'params': encoder.parameters(), 'lr': 0.0*lr},   # gentle since the encoder carries mostly topology info already trained
                {'params': fcov.parameters(), 'lr': lr},
                {'params': ftopo.parameters(), 'lr': 0.0*lr},
            ])
            #mean_delay, std_dev_delay = ckpt['mean_delay'], ckpt['std_dev_delay']
            #mean_loss, std_dev_loss = ckpt['mean_loss'], ckpt['std_dev_loss']
            #std_dev_delay_tot = ckpt['std_dev_delay_tot']
            
        else:
            print(f"Could not load model at {model_path}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=3)

    R_VAR = 0.0  # measurement-noise variance in normalised delay units (tune / estimate)

    # Observable overlay identity for f_cov: [overlay id, waypoint ids of ovl['overlay_path']]
    name_vocab, id_vocab = {}, {}
    def ovl_keys(ovl):
        wps = [name_vocab.setdefault(nm, len(name_vocab)) for nm in ovl['overlay_path']][:fcov.n_wp]
        wps += [-1] * (fcov.n_wp - len(wps))
        return torch.tensor([id_vocab.setdefault(ovl['id'], len(id_vocab))] + wps, dtype=torch.long, device=device)

    def run_t(t, gidx, data, return_attention, ep, h_enc, h_topo):
        curr_overlay = data[(gidx,t)]['overlay_paths']
        curr_overlay_ids = [ovl['id'] for ovl in curr_overlay]
        H_hist = []
        Meas_hist = []
        Elapsed = []
        Hist_ids = []
        Topo_hist = []
        Keys_hist = []

        sim_loss = torch.zeros((), device=device)
        sim_count = 0
        loss = torch.zeros((), device=device)
        count = 0

        if INCLUDE_MEAS:
            for i in range(0, Delta):
                d = dataset[gidx][t - i]
                hist_overlay = d['overlay_paths']
                # encode the whole graph for time (t-i) ONCE, for all overlays
                missing = [ovl for ovl in hist_overlay if (gidx, ovl['id'], t - i) not in h_enc]
                if missing:
                    enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t - i)], missing)
                    topo = ftopo(torch.stack([enc[i] for i in list(enc.keys())]))
                    for idx, entry in enumerate(enc.keys()):
                        h_enc[(gidx, entry, t - i)] = enc[entry]
                        h_topo[(gidx, entry, t - i)] = topo[idx]
                        
                for ovl in hist_overlay:
                    H_hist.append(h_enc[(gidx, ovl['id'], t - i)])
                    Meas_hist.append(torch.tensor(ovl['meas'][:2], dtype=torch.float, device=device))
                    Topo_hist.append(h_topo[(gidx, ovl['id'], t-i)])
                        
                    Elapsed.append(float(i))
                    Hist_ids.append((ovl['id'], t - i))
                    Keys_hist.append(ovl_keys(ovl))

            if len(H_hist) > 0:  # Can be 0 if there are no overlays at given time
                H_hist = torch.stack(H_hist)
                Meas_hist = torch.stack(Meas_hist)
                Meas_hist = torch.stack([
                    (Meas_hist[:, 0] - mean_delay.sum()) / (std_dev_delay_tot + 1e-8),
                    (Meas_hist[:, 1] - mean_loss) / (std_dev_loss + 1e-8),
                ], dim=1)
                Elapsed = torch.tensor(Elapsed, device=device).unsqueeze(1)
                Topo_hist = torch.stack(Topo_hist)
                Keys_hist = torch.stack(Keys_hist)

        loss = torch.zeros((), device=device)
        loss_cheating = torch.zeros((), device=device)
        count = 0
        for m in range(1, M+1):         
            fut_overlay = data[(gidx, t + m)]['overlay_paths']

            # overlays we actually train on this step
            eligible = [f_ovl for f_ovl in fut_overlay
                        if f_ovl['id'] in curr_overlay_ids]
            if not eligible:
                continue

            enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t + m)], eligible)
            for entry in enc.keys():
                h_enc[(gidx, entry, t + m)] = enc[entry]

            for f_ovl in eligible:
                tgt = torch.tensor(f_ovl['meas'], dtype=torch.float, device=device)
                tgt[0] = (tgt[0] - mean_delay.sum()) / (std_dev_delay_tot + 1e-8)
                tgt[1] = (tgt[1] - mean_loss) / (std_dev_loss + 1e-8)

                ovl_id = f_ovl['id']
                enc = h_enc[(gidx, ovl_id, t + m)]
                out_topo = ftopo(enc)
                h_topo[(gidx, ovl_id, t+m)] = out_topo
                

                if INCLUDE_MEAS:
                    mask = torch.ones((len(H_hist)), dtype=torch.bool, device=device).detach()
                    if REMOVE_SELF_MEAS:
                        for d in range(Delta):  #Don't use previous measurements from the same overlay
                            if (f_ovl['id'], t-d) in Hist_ids:
                                if random.random() < REMOVE_SELF_MEAS_PERC:
                                    idx = Hist_ids.index((f_ovl['id'], t-d))
                                    mask[idx] = False

                    Topo_hist_tmp = Topo_hist[mask]
                    Elapsed_tmp = Elapsed[mask]
                    H_hist_tmp = H_hist[mask]
                    Meas_hist_tmp = Meas_hist[mask]
                    Keys_hist_tmp = Keys_hist[mask]

                    C = (fcov(H_hist_tmp, Elapsed_tmp + m, keys_hist=Keys_hist_tmp)).reshape(H_hist_tmp.size(0), H_hist_tmp.size(0))
                    g = (fcov(H_hist_tmp, Elapsed_tmp + m, h_pred=enc,
                              keys_hist=Keys_hist_tmp, keys_pred=ovl_keys(f_ovl))).reshape(H_hist_tmp.size(0), 1)
                    hat_z = Meas_hist_tmp[:,0] - Topo_hist_tmp[:,0]  # neighbor deviation
                    R = R_VAR * torch.eye(H_hist_tmp.size(0), device=device)

                    tmp = torch.linalg.cond(C + R)
                    if tmp > 10**5:
                        print(f'C+R might be ill-conditioned! Cond = {tmp}')
                    out_meas = g.T @ torch.linalg.solve(C + R, hat_z)  # [1], delay correction only
                    out_meas = torch.cat([out_meas, torch.zeros(1, device=device)])  # [2], no correction for loss
                else:
                    out_meas = torch.zeros((2), dtype=torch.float, device=device)
                
                out = out_topo + out_meas
                z0 = out[0] - tgt[0]
                z1 = out[1] - tgt[1]
                loss = loss + (c_delta * z0 ** 2 + c_lambda * z1 ** 2)

                count += 1

        return loss, count, sim_loss, sim_count, loss_cheating, h_enc, h_topo

    
    batch_size = batch_size/(nbr_sims-1) #For each step, train over all sims
    beta = 0 #0.1 OBS!!!!

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    data = {}
    for gidx in range(nbr_sims-1):
        for t in range(len(dataset)):
            d = dataset[gidx][t]
            tmp = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'], overlay_paths=d['overlay_paths']).to(device)
            data[(gidx, t)] = tmp

    for ep in range(epochs):
        # ---- train ----
        encoder.train(); fcov.train(); ftopo.train()
        train_loss = 0.0
        train_loss_sim = 0.0
        train_loss_cheating = 0.0
        batch_sample = 1
        loss_batch = torch.zeros((), device=device)
        loss_batch_sim = torch.zeros((), device=device)
        loss_batch_cheating = torch.zeros((), device=device)
        h_enc = {}
        h_topo = {}
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            loss_cheating = torch.zeros((), device=device);
            loss_sim = torch.zeros((), device=device); count_sim = 0
            for gidx in range(nbr_sims-1):
                loss_gidx, count_gidx, sim_loss_gidx, sim_count_gidx, cheating_loss, h_enc, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_enc=h_enc, h_topo=h_topo)
                loss = loss + loss_gidx
                loss_cheating = loss_cheating + cheating_loss
                count = count + count_gidx
                loss_sim = loss_sim + sim_loss_gidx
                count_sim = count_sim + sim_count_gidx

            step_loss = loss / count if count > 0 else torch.zeros((), dtype=torch.float, device=device)
            loss_batch = loss_batch + step_loss
            train_loss += float((step_loss).detach())

            step_loss_cheating = loss_cheating / count if count > 0 else torch.zeros((), dtype=torch.float, device=device)
            loss_batch_cheating = loss_batch_cheating + step_loss_cheating
            train_loss_cheating += float((step_loss_cheating).detach())

            step_loss_sim = loss_sim / count_sim if count_sim > 0 else torch.zeros((), dtype=torch.float, device=device)
            loss_batch_sim = loss_batch_sim + step_loss_sim
            train_loss_sim += float((step_loss_sim).detach())

            if batch_sample >= batch_size:
                opt.zero_grad()
                full_loss = loss_batch + beta * (loss_batch_sim + loss_batch_cheating)
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], GRAD_CLIP)
                full_loss.backward()
                opt.step()
                batch_sample = 1
                loss_batch = torch.zeros((), device=device)
                loss_batch_sim = torch.zeros((), device=device)
                loss_batch_cheating = torch.zeros((), device=device)
                h_enc = {}
                h_topo = {}
            else:
                batch_sample = batch_sample + 1

        train_loss = train_loss / (len(train_t))
        train_loss_cheating = train_loss_cheating / (len(train_t))
        train_loss_sim = train_loss_sim / (len(train_t))
        train_combined = train_loss + beta * (train_loss_sim + train_loss_cheating)

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
                for gidx in range(nbr_sims-1):
                    loss_gidx, count_gidx, sim_loss_gidx, sim_count_gidx, cheating_loss, h_enc, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_enc=h_enc, h_topo=h_topo)
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
              train_loss_cheating = {train_loss_cheating}
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
    model_path = os.path.join(model_dir, model_filename(encoder_type))
    torch.save({'encoder_type': encoder_type,
                'encoder_state': encoder.state_dict(), 'fcov_state': fcov.state_dict(),
                'ftopo_state': ftopo.state_dict(), 'std_dev_delay_tot': std_dev_delay_tot,
                'mean_delay': mean_delay, 'mean_loss': mean_loss,
                'std_dev_delay': std_dev_delay, 'std_dev_loss': std_dev_loss},
               model_path)
    print('Saved trained models to', model_path)
    return model_path

def calculate_mean_std(train_t, dataset, nbr_sims):
    meas_delay = []
    meas_loss = []
    for gidx in range(nbr_sims):
        for t in train_t:
            curr_overlay = dataset[gidx][t]['overlay_paths']
            for ovl in curr_overlay:
                ovl_meas = ovl['meas']
                meas_delay.append([ovl_meas[2], ovl_meas[3]])
                meas_loss.append(ovl_meas[1])

    # Calculate mean
    mean_loss = statistics.mean(meas_loss)
    print(f"Mean of the loss is {mean_loss}")

    mean_delay = np.mean(meas_delay, axis=0)
    print(f"Mean of the delay is {mean_delay}")

    # Calculate standard deviation
    std_dev_loss= statistics.stdev(meas_loss)
    print(f"Standard Deviation of the loss is {std_dev_loss}")
    
    std_dev_delay_tot = statistics.stdev([sum(x) for x in meas_delay])
    std_dev_delay = np.std(meas_delay, axis=0)
    print(f"Standard Deviation of the delay is {std_dev_delay}")
    return(mean_delay, mean_loss, std_dev_delay, std_dev_delay_tot, std_dev_loss)

def baselines(split_t, dataset, nbr_sims, mu, sd, c_delta):
    mean_tot, pers_tot, n_t = 0.0, 0.0, 0
    for t in split_t[:-1]:
        ms, ps, n = 0.0, 0.0, 0
        for gidx in range(nbr_sims):
            prev = {o['id']: o['meas'][0] for o in dataset[gidx][t]['overlay_paths']}
            for o in dataset[gidx][t + 1]['overlay_paths']:
                if o['id'] in prev:
                    y = o['meas'][0]
                    ms += ((y - mu) / sd) ** 2
                    ps += ((y - prev[o['id']]) / sd) ** 2
                    n += 1
        if n:
            mean_tot += ms / n; pers_tot += ps / n; n_t += 1
    return c_delta * mean_tot / n_t, c_delta * pers_tot / n_t

if __name__ == '__main__':
    train(epochs=10, batch_size=32)