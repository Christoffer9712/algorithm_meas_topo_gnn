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

INCLUDE_MEAS = False

def train(dataset_path=None, model_dir=None, epochs=100, batch_size=16, lr=5e-4, device='cpu', min_delta=1e-4, patience=15):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    
    #env = RoutingEnvironment()
    dataset = PredictorDataset(dataset_path)
    
    #encoder = GATv2Encoder(in_dim=8, hidden_dim=128, out_dim=64, edge_dim=1).to(device)
    
    encoder = HeteroGATv2Encoder(node_in=9, virtual_in=1, hidden_dim=128, out_dim=64, heads=2, n_layers=3, dropout=0.1)
    graph_encoder = GraphEncoder(encoder, device=device)

    embedder = MeasurementEmbedder(h_dim=64, hidden_dim=128, out_dim=64).to(device)
    # Predictor input: h_topo (D) + g (64) + horizon_m (1)
    predictor = Predictor(in_dim=64 + 64 + 1, hidden_dim=128, dropout=0.1).to(device) # Input is H = [topo_embed || meas_embed || m]

    opt = torch.optim.Adam(list(encoder.parameters()) + list(embedder.parameters()) + list(predictor.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=8
    )
    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 5.0

    nbr_sims = dataset.get_nbr_of_sims()

    # Prediction horizon: M
    M = 1 # NOTE!!!

    # Measurement history depth: Delta
    Delta = 1

    # Contiguous time-based split (no shuffling across splits -> no leakage)
    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    train_t = all_t[: int(0.70 * n)]
    val_t   = all_t[int(0.70 * n) : int(0.85 * n)]
    test_t  = all_t[int(0.85 * n) :]
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


    def run_t(t, gidx, dataset, return_attention, ep):
        curr_overlay = dataset[gidx][t]['overlay_paths']
        curr_overlay_ids = [ovl['id'] for ovl in curr_overlay]
        H_hist = []
        Meas_hist = []
        Elapsed = []
        if INCLUDE_MEAS:
            for i in range(0, Delta):
                d = dataset[gidx][t - i]
                #G = env.snapshot_at_time_t(t-i)
                data = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'])            
                hist_overlay = d['overlay_paths']
                h_hist = graph_encoder.encode_overlays_pyg(data, hist_overlay, return_attention)
                for ovl in hist_overlay:
                    H_hist.append(h_hist[ovl['id']])
                    Meas_hist.append(torch.tensor(ovl['meas'], dtype=torch.float, device=device))
                    Elapsed.append(float(i))

            H_hist = torch.stack(H_hist)
            Meas_hist = torch.stack(Meas_hist)
            Elapsed = torch.tensor(Elapsed, device=device).unsqueeze(1)

        loss = torch.zeros((), device=device)
        count = 0
        for m in range(1, M+1):
            d = dataset[gidx][t + m]
            #G = env.snapshot_at_time_t(t+m)
            # Using overlays from time t since it's unknown what future overlays will exist
            data = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'])            
            fut_overlay = d['overlay_paths']
            for f_ovl in fut_overlay:
                if f_ovl['id'] in curr_overlay_ids:
                    if return_attention:
                        h_topo = graph_encoder.encode_overlays_pyg(data, [f_ovl], return_attention=return_attention, ep=ep)
                        return_attention = False # Let's only plot one overlay attention plot
                    else:
                        h_topo = graph_encoder.encode_overlays_pyg(data, [f_ovl], return_attention=return_attention)
                    ovl_id = f_ovl['id']
                    if INCLUDE_MEAS:
                        g = embedder(h_topo[ovl_id], H_hist, Meas_hist, Elapsed)
                    else:
                        g = torch.zeros(64, device=device)
                    H = torch.cat([h_topo[ovl_id], g, torch.tensor([float(m)], dtype=torch.float, device=device)], dim=0).unsqueeze(0)
                    out = predictor(H)
                    tgt = torch.tensor(f_ovl['meas'], dtype=torch.float, device=device)

                    z0 = out[0,0] - (tgt[0]-mean_delay)/(std_dev_delay+1e-8)
                    z1 = out[0,1] - (tgt[1]-mean_loss)/(std_dev_loss+1e-8)

                    loss = loss + (c_delta * z0**2 + c_lambda * z1**2)
                    count += 1
                else:
                    pass
        return loss, count

    
    batch_size = batch_size/(nbr_sims-1) #For each step, train over all sims

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    for ep in range(epochs):
        # ---- train ----
        encoder.train(); embedder.train(); predictor.train()
        train_loss = 0.0
        batch_sample = 1
        loss_batch = 0.0
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            for gidx in range(nbr_sims): #OBS! add -1 for leaving 1 sim out for validation
                if ep%10 == 0 and gidx == 0 and t == 0: # This is only to visualise the attention
                    encoder.eval(); embedder.eval(); predictor.eval()
                    with torch.no_grad():
                        loss_gidx, count_gidx = run_t(t, gidx, dataset, return_attention=True, ep=ep) #Start with gidx == 0
                    encoder.train(); embedder.train(); predictor.train()
                else:
                    loss_gidx, count_gidx = run_t(t, gidx, dataset, return_attention=False, ep=ep)
                loss = loss + loss_gidx
                count = count + count_gidx

            if count == 0:
                continue
            step_loss = loss / count
            loss_batch = loss_batch + step_loss
            train_loss += float((step_loss).detach())
            if batch_sample >= batch_size:
                opt.zero_grad()
                loss_batch.backward()
                opt.step()
                batch_sample = 1
                loss_batch = 0.0
            else:
                batch_sample = batch_sample + 1

        train_loss = train_loss/(len(train_t)) #OBS!!!

        # ---- validate ----
        encoder.eval(); embedder.eval(); predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t in val_t:
                loss = torch.zeros((), device=device); count = 0
                for gidx in range(nbr_sims): #range(1): #OBS! leaving 1 sim out for validation
                    loss_gidx, count_gidx = run_t(t, gidx, dataset, return_attention=False, ep=ep) #run_t(t, nbr_sims-1, dataset) #validating using unseen topology (last one)
                    loss = loss + loss_gidx
                    count = count + count_gidx

                if count == 0:
                    continue
   
                val_loss += float((loss / count).detach())
            
            val_loss = val_loss/(len(val_t)) #OBS!!!

        scheduler.step(val_loss) # This reduces lr if the learning stalls
        # ---- early stopping ----
        if val_loss < best_val - min_delta:
            best_val = val_loss
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

        print(f"Epoch: {ep} - train_loss = {train_loss} - val_loss = {val_loss} - lr = {opt.param_groups[0]["lr"]}")

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
    torch.save({'encoder_state': encoder.state_dict(), 'embedder_state': embedder.state_dict(), 'predictor_state': predictor.state_dict(),
                'mean_delay':mean_delay, 'mean_loss':mean_loss, 'std_dev_delay':std_dev_delay, 'std_dev_loss':std_dev_loss},
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
