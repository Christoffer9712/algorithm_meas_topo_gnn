"""
Train MeasurementEmbedder + Predictor on generated dataset.

Saves the trained weights to models/predictor_embedder.pth and
models/predictor_predictor.pth (two modules).
"""
import os
import torch
import statistics
from torch.utils.data import DataLoader
import copy

from algorithm.predictor.meas_predictor import MeasurementEmbedder 
from algorithm.predictor.predictor import Predictor
from algorithm.predictor.hetero_encoder import HeteroGATv2Encoder
from .dataset import PredictorDataset
from algorithm.predictor.path_encoder import GATv2Encoder, GraphEncoder
from torch_geometric.data import Data
from environment.environment import RoutingEnvironment
import random

INCLUDE_MEAS = True

def train(dataset_path=None, model_dir=None, epochs=10, batch_size=16, lr=1e-3, device='cpu', min_delta=1e-4, patience=15, transfer_learning_model=None):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    
    dataset = PredictorDataset(dataset_path)    
    encoder = HeteroGATv2Encoder(node_in=10, virtual_in=1, hidden_dim=128, out_dim=64, heads=2, n_layers=3, dropout=0.0).to(device)
    graph_encoder = GraphEncoder(encoder, device=device)

    embedder = MeasurementEmbedder(h_dim=64).to(device)
    # Predictor input: h_topo (D) + g (64) + horizon_m (1)
    predictor = Predictor(in_dim=64 + 2 + 1, hidden_dim=128, dropout=0.0).to(device) # Input is H = [topo_embed || meas_embed || m]

    opt = torch.optim.Adam(list(encoder.parameters()) + list(embedder.parameters()) + list(predictor.parameters()), lr=lr)

    if transfer_learning_model:
        model_path = os.path.join(model_dir, transfer_learning_model)
        ckpt = torch.load(model_path, map_location='cpu')
        if 'embedder_state' in ckpt and 'predictor_state' in ckpt and 'encoder_state' in ckpt:
            embedder.load_state_dict(ckpt['embedder_state'])
            predictor.load_state_dict(ckpt['predictor_state'])
            encoder.load_state_dict(ckpt['encoder_state'])
            mean_delay = ckpt['mean_delay']
            mean_loss = ckpt['mean_loss']
            std_dev_delay = ckpt['std_dev_delay']
            std_dev_loss = ckpt['std_dev_loss']
            print(f"Loaded model at {model_path}")
            opt = torch.optim.Adam([
                {'params': encoder.parameters(),   'lr': lr/10},   # gentle since the encoder carries mostly topology info already trained
                {'params': embedder.parameters(),  'lr': lr},
                {'params': predictor.parameters(), 'lr': lr},
            ])
            
        else:
            print(f"Could not load model at {model_path}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.75, patience=3
    )
    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 5.0

    nbr_sims = dataset.get_nbr_of_sims()

    # Prediction horizon: M
    M = 1 # NOTE!!!

    # Measurement history depth: Delta
    Delta = 5

    # Contiguous time-based split (no shuffling across splits -> no leakage)
    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    train_t = all_t[: int(0.70 * n)]
    val_t   = all_t[int(0.70 * n) :] # OBS!!! made this to be larger!!!
    test_t  = all_t[int(0.85 * n) :] # Currently not really in use
    #train_t = val_t = test_t = all_t #OBS, use all data when validation is done on different simulation

    meas_delay = []
    meas_loss = []
    for gidx in range(nbr_sims-1): #OBS -1
        for t in train_t:
            curr_overlay = dataset[gidx][t]['overlay_paths']
            for ovl in curr_overlay:
                ovl_meas = ovl['meas']
                meas_delay.append(ovl_meas[0])
                meas_loss.append(ovl_meas[1])

    # Calculate mean
    mean_loss = statistics.mean(meas_loss)
    print(f"Mean of the loss is {mean_loss}")

    mean_delay = statistics.mean(meas_delay)
    print(f"Mean of the delay is {mean_delay}")

    # Calculate standard deviation
    std_dev_loss= statistics.stdev(meas_loss)
    print(f"Standard Deviation of the loss is {std_dev_loss}")
    
    std_dev_delay = statistics.stdev(meas_delay)
    print(f"Standard Deviation of the delay is {std_dev_delay}")

    print(f"Baseline (predict-mean) train = {baseline_loss_mean(train_t, dataset, 0, M, mean_loss, std_dev_loss, mean_delay, std_dev_delay, c_lambda, c_delta):.4f}")
    print(f"Baseline (predict-mean) val   = {baseline_loss_mean(val_t, dataset, 0, M, mean_loss, std_dev_loss, mean_delay, std_dev_delay, c_lambda, c_delta):.4f}")
    print(f"Baseline (predict-mean) test  = {baseline_loss_mean(test_t, dataset, 0, M, mean_loss, std_dev_loss, mean_delay, std_dev_delay, c_lambda, c_delta):.4f}")


    def run_t(t, gidx, data, return_attention, ep, h_topo={}):
        curr_overlay = data[(gidx,t)]['overlay_paths']
        curr_overlay_ids = [ovl['id'] for ovl in curr_overlay]
        H_hist = []
        Meas_hist = []
        Elapsed = []
        hist_ids = []

        sim_loss = 0
        sim_count = 0
        if INCLUDE_MEAS:
            for i in range(0, Delta):
                d = dataset[gidx][t - i]       
                hist_overlay = d['overlay_paths']
                for ovl in hist_overlay:
                    if not (ovl['id'],t-i) in h_topo.keys():
                        h = graph_encoder.encode_overlays_pyg(data[gidx, t-i], [ovl], return_attention=False, ep=ep)
                        assert(len(h) == 1)
                        for entry in h.keys():
                            h_topo.update({(entry, t-i): h[entry]})
                        #print(f'Needed to recalculate h for id {ovl['id']} for i = {i}')
                    H_hist.append(h_topo[(ovl['id'], t-i)].detach())
                    Meas_hist.append(torch.tensor(ovl['meas'][:2], dtype=torch.float, device=device))
                    Elapsed.append(float(i+1)) #Time to next prediction at time t+1
                    hist_ids.append((ovl['id'],t-i))

            if len(H_hist) > 0: # Can be 0 if there are no overlays at given time
                H_hist = torch.stack(H_hist)
                Meas_hist = torch.stack(Meas_hist)
                Meas_hist = torch.stack([
                    (Meas_hist[:, 0] - mean_delay) / (std_dev_delay + 1e-8),
                    (Meas_hist[:, 1] - mean_loss)  / (std_dev_loss + 1e-8),
                ], dim=1)
                Elapsed = torch.tensor(Elapsed, device=device).unsqueeze(1)

                # ---- LOO similarity objective on a random subset ----
                K = min(5, len(hist_ids))
                if K >= 2:  # need at least a query + one other
                    sample_idx = random.sample(range(len(hist_ids)), K)  # no replacement, in range
                    Meas_sim = Meas_hist[sample_idx]
                    Elapsed_sim = Elapsed[sample_idx]

                    if True:
                        # re-encode the sampled paths FRESH (grad-carrying, this-step graph)
                        H_sim = []
                        for j in sample_idx:
                            oid, ot = hist_ids[j]
                            ovl_j = next(o for o in dataset[gidx][ot]['overlay_paths'] if o['id'] == oid)
                            h_j = graph_encoder.encode_overlays_pyg(data[(gidx, ot)], [ovl_j],
                                                                    return_attention=False, ep=ep)[oid]
                            H_sim.append(h_j)
                        H_sim = torch.stack(H_sim)                      # [K, D], grad-carrying

                        for i in range(K):
                            self_mask = torch.zeros(K, dtype=torch.bool, device=device)
                            self_mask[i] = True                         # leave out entry i
                            # recency relative to the query being reconstructed
                            rel_elapsed = (Elapsed_sim - Elapsed_sim[i]).abs().reshape(-1)
                            meas_hat = embedder.reconstruct_loo(H_sim[i], H_sim, Meas_sim, rel_elapsed, self_mask)
                            tgt_std = Meas_sim[i].detach()
                            sim_loss = sim_loss + ((meas_hat - tgt_std) ** 2).sum()
                            sim_count += 1
                    else: #Just use the mean (gives around 0.8-1 in loss)
                        for i in range(K):
                            mask = [j != i for j in range(K)]
                            meas_hat = Meas_sim[mask].mean(dim=0)      # uniform average of the OTHER K-1, per channel
                            tgt_std = Meas_sim[i].detach()
                            sim_loss = sim_loss + ((meas_hat - tgt_std) ** 2).sum()
                            sim_count += 1


        loss = torch.zeros((), device=device)
        count = 0

        for m in range(1, M+1):
            fut_overlay = data[(gidx, t + m)]['overlay_paths']

            # overlays we actually train on this step
            eligible = [f_ovl for f_ovl in fut_overlay]
                        #if f_ovl['id'] in curr_overlay_ids]
            if not eligible:
                continue

            # --- attention plotting path: unchanged, single-graph, one overlay ---
            if return_attention:
                first = eligible[0]
                h_first = graph_encoder.encode_overlays_pyg(
                    data[(gidx, t+m)], [first], return_attention=True, ep=ep)
                return_attention = False   # only plot one overlay
                h_topo.update({(first['id'], t+m): h_first[first['id']]})
                # encode the rest in one batched launch
                rest = eligible[1:]
                if rest:
                    tmp = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t+m)], rest)
                    for entry in tmp.keys():
                        h_topo.update({(entry, t+m): tmp[entry]})
            else:
                # --- normal path: one batched forward for all eligible overlays ---
                tmp = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t+m)], eligible)
                for entry in tmp.keys():
                    h_topo.update({(entry, t+m): tmp[entry]})

            # --- heads + loss: per overlay, identical to before ---
            for f_ovl in eligible:
                ovl_id = f_ovl['id']
                if INCLUDE_MEAS and len(H_hist) > 0:
                    g = embedder(h_topo[(ovl_id, t+m)], H_hist, Meas_hist, Elapsed)
                else:
                    g = torch.zeros(2, device=device) #OBS!!!!


                H = torch.cat([h_topo[(ovl_id, t+m)], g,
                               torch.tensor([float(m)], dtype=torch.float, device=device)],
                              dim=0).unsqueeze(0)
                out = predictor(H)
                tgt = torch.tensor(f_ovl['meas'], dtype=torch.float, device=device)

                z0 = out[0,0] - (tgt[0]-mean_delay)/(std_dev_delay+1e-8)
                z1 = out[0,1] - (tgt[1]-mean_loss)/(std_dev_loss+1e-8)

                loss = loss + (c_delta * z0**2 + c_lambda * z1**2)


                    
                count += 1

        return (loss,sim_loss), (count, sim_count) , h_topo

    
    batch_size = batch_size/(nbr_sims-1) #For each step, train over all sims

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    data = {}
    for gidx in range(nbr_sims):
        for t in range(len(dataset)):
            d = dataset[gidx][t]
            tmp = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'], overlay_paths=d['overlay_paths']).to(device)
            data[(gidx, t)] = tmp
            
    beta = 0.2
    for ep in range(epochs):
        # ---- train ----
        encoder.train(); embedder.train(); predictor.train()
        train_loss = [0.0, 0.0]
        batch_sample = 1
        loss_batch = 0.0

        h_topo = {} #Keep a record of the most recently calculated path embedding to not have to recalculate when adding the measuements

        for t in train_t:
            loss = torch.zeros(2, device=device); count = [0,0]
            for gidx in range(nbr_sims): #OBS! add -1 for leaving 1 sim out for validation
                if ep%10 == 0 and gidx == 0 and t == 0: # This is only to visualise the attention
                    encoder.eval(); embedder.eval(); predictor.eval()
                    with torch.no_grad():
                        loss_gidx, count_gidx, h_topo = run_t(t, gidx, data, return_attention=True, ep=ep, h_topo=h_topo) #Start with gidx == 0
                    encoder.train(); embedder.train(); predictor.train()
                else:
                    loss_gidx, count_gidx, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_topo=h_topo)
                loss[0] = loss[0] + loss_gidx[0]
                loss[1] = loss[1] + loss_gidx[1]
                count[0] = count[0] + count_gidx[0]
                count[1] = count[1] + count_gidx[1]

            pred_step = loss[0]/count[0] if count[0] else torch.zeros((), device=device)
            sim_step  = loss[1]/count[1] if count[1] else torch.zeros((), device=device)
            step_loss = pred_step + beta * sim_step
            loss_batch = loss_batch + step_loss
            train_loss[0] += float(pred_step.detach()); train_loss[1] += float(sim_step.detach())
            if batch_sample >= batch_size:
                opt.zero_grad(); loss_batch.backward(); opt.step()
                batch_sample = 1; loss_batch = 0.0
            else:
                batch_sample += 1

        train_loss = [t/(len(train_t)) for t in train_loss] #OBS!!!

        # ---- validate ----
        encoder.eval(); embedder.eval(); predictor.eval()
        val_loss = [0.0, 0.0]
        with torch.no_grad():
            for t in val_t:
                loss = torch.zeros(2, device=device); count = [0,0]
                for gidx in range(nbr_sims): #range(1): #OBS! leaving 1 sim out for validation
                    loss_gidx, count_gidx, h_topo = run_t(t, gidx, data, return_attention=False, ep=ep, h_topo=h_topo) #run_t(t, nbr_sims-1, dataset) #validating using unseen topology (last one)
                    loss[0] = loss[0] + loss_gidx[0]
                    loss[1] = loss[1] + loss_gidx[1]
                    count[0] = count[0] + count_gidx[0]
                    count[1] = count[1] + count_gidx[1]
   
                for i in range(len(val_loss)):
                    if count[i] == 0:
                        continue
                    val_loss[i] += float((loss[i] / count[i]).detach()) / len(val_t)
            

        scheduler.step(sum(val_loss)) # This reduces lr if the learning stalls
        # ---- early stopping ----
        if val_loss[0] + beta*val_loss[1] < best_val - min_delta:
            best_val = val_loss[0] + beta*val_loss[1]
            epochs_no_improve = 0
            best_state = {
                'encoder_state': copy.deepcopy(encoder.state_dict()),
                'embedder_state': copy.deepcopy(embedder.state_dict()),
                'predictor_state': copy.deepcopy(predictor.state_dict()),
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best val_loss = {best_val:.6f})")
                break

        msg = f"""
              Epoch: {ep}
              train_loss = {train_loss}
              Weighted Sum(train_loss) = {train_loss[0] + beta*train_loss[1]}
              val_loss   = {val_loss}
              Weighted Sum(val_loss) = {val_loss[0] + beta*val_loss[1]}
              lr         = {opt.param_groups[0]["lr"]}
              encoder    = {next(encoder.parameters()).device}
              embedder   = {next(embedder.parameters()).device}
              predictor  = {next(predictor.parameters()).device}
              """

        print(msg)

    # restore best weights before test + save
    if best_state is not None:
        encoder.load_state_dict(best_state['encoder_state'])
        embedder.load_state_dict(best_state['embedder_state'])
        predictor.load_state_dict(best_state['predictor_state'])

    # ---- test (once, after training) ----
    '''
    encoder.eval(); embedder.eval(); predictor.eval()
    test_loss = 0.0
    with torch.no_grad():
        for t in test_t:
            for gidx in range(nbr_sims):
                loss, count = run_t(t, gidx, dataset)
                if count == 0:
                    continue
                test_loss += float((loss / count).detach())
    test_loss = test_loss/(len(test_t))
    print(f"Test loss = {test_loss}")
    '''

    # Save models
    if INCLUDE_MEAS:
        save_model = 'predictor_models_with_meas.pth'
    else:
        save_model = 'predictor_models.pth'

    torch.save({'encoder_state': encoder.state_dict(), 'embedder_state': embedder.state_dict(), 'predictor_state': predictor.state_dict(),
                'mean_delay':mean_delay, 'mean_loss':mean_loss, 'std_dev_delay':std_dev_delay, 'std_dev_loss':std_dev_loss},
                os.path.join(model_dir, save_model))
    print('Saved trained models to', os.path.join(model_dir, save_model))
    return os.path.join(model_dir, save_model)

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
                    z0 = (tgt[0]-mean_delay)/(std_dev_delay+1e-8)
                    z1 = (tgt[1]-mean_loss)/(std_dev_loss+1e-8)
                    # prediction = 0 in standardized space
                    loss += c_delta * z0**2 + c_lambda * z1**2
                    count += 1
        if count:
            total += loss / count
            n_t += 1
    return total / n_t if n_t else float('nan')

if __name__ == '__main__':
    train(epochs=10, batch_size=32)
