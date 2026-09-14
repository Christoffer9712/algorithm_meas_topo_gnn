import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data

import statistics
import copy
from environment.environment import RoutingEnvironment
from .dataset import PredictorDataset



class Predictor(nn.Module):
    """
    MLP mapping ovl distance -> predicted (lambda, delay_ms).
    """

    def __init__(self, in_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, H):
        """H: Tensor [D] or [batch, D] -> returns (lambda, delay_ms)"""
        out = self.net(H)
        return out


def overlay_dist(data, overlay):
    dist = []
    for node_src, node_dst in zip(overlay[:-1], overlay[1:]):
        src_pos = data.x[data.name_to_idx[node_src]][0:2]
        dst_pos = data.x[data.name_to_idx[node_dst]][0:2]
        dist.append(torch.norm(src_pos - dst_pos))
    return dist

def train(dataset_path=None, model_dir=None, epochs=100, batch_size=16, lr=1e-3, device='cpu', min_delta=1e-4, patience=15):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    
    dataset = PredictorDataset(dataset_path)

    data_driven_predictor = Predictor(in_dim=3, hidden_dim=128, dropout=0.1)

    opt = torch.optim.Adam(list(data_driven_predictor.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.75, patience=3
    )
    # Loss weights: c_lambda, c_delta
    c_lambda = 1.0
    c_delta = 5.0
    nbr_sims = dataset.get_nbr_of_sims()

    # Contiguous time-based split (no shuffling across splits -> no leakage)
    Delta = M = 1
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


    def run_t(t, gidx, data, ep):
        curr_overlay = data[(gidx,t)]['overlay_paths']
        curr_overlay_ids = [ovl['id'] for ovl in curr_overlay]
        H_hist = []
        Meas_hist = []
        Elapsed = []
       
        loss = torch.zeros((), device=device)
        count = 0
        for m in range(1, M+1):
            fut_overlay = data[(gidx, t + m)]['overlay_paths']

            # overlays we actually train on this step
            eligible = [f_ovl for f_ovl in fut_overlay
                        if f_ovl['id'] in curr_overlay_ids]
            if not eligible:
                continue

            # --- heads + loss: per overlay, identical to before ---
            for f_ovl in eligible:
                H = torch.tensor(overlay_dist(data[(gidx,t+m)], f_ovl['overlay_path']), dtype=torch.float, device=device)
                out = data_driven_predictor(H)
                tgt = torch.tensor(f_ovl['meas'], dtype=torch.float, device=device)

                z0 = out[0] - (tgt[0]-mean_delay)/(std_dev_delay+1e-8)
                z1 = out[1] - (tgt[1]-mean_loss)/(std_dev_loss+1e-8)

                loss = loss + (c_delta * z0**2 + c_lambda * z1**2)
                count += 1

        return loss, count

    
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
            
    
    for ep in range(epochs):
        # ---- train ----
        data_driven_predictor.train()
        train_loss = 0.0
        batch_sample = 1
        loss_batch = 0.0
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            for gidx in range(nbr_sims): #OBS! add -1 for leaving 1 sim out for validation
                loss_gidx, count_gidx = run_t(t, gidx, data, ep=ep)
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
        data_driven_predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t in val_t:
                loss = torch.zeros((), device=device); count = 0
                for gidx in range(nbr_sims): #range(1): #OBS! leaving 1 sim out for validation
                    loss_gidx, count_gidx = run_t(t, gidx, data, ep=ep) #run_t(t, nbr_sims-1, dataset) #validating using unseen topology (last one)
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
                'data_driven_predictor_state': copy.deepcopy(data_driven_predictor.state_dict()),
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best val_loss = {best_val:.6f})")
                break

        msg = f"""
                Epoch: {ep}
                train_loss = {train_loss}
                val_loss   = {val_loss}
                lr         = {opt.param_groups[0]["lr"]}
                data_driven_predictor  = {next(data_driven_predictor.parameters()).device}
                """

        print(msg)

    # restore best weights before test + save
    if best_state is not None:
        data_driven_predictor.load_state_dict(best_state['data_driven_predictor_state'])

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
    torch.save({'data_driven_predictor_state': data_driven_predictor.state_dict(),
                'mean_delay':mean_delay, 'mean_loss':mean_loss, 'std_dev_delay':std_dev_delay, 'std_dev_loss':std_dev_loss},
                os.path.join(model_dir, 'data_driven_predictor_models.pth'))
    
    print('Saved trained models to', os.path.join(model_dir, 'data_driven_predictor_models.pth'))
    return os.path.join(model_dir, 'data_driven_predictor_models.pth')

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
    train()