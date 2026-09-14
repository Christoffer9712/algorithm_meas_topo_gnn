"""
Train MeasurementEmbedder + Predictor on generated dataset.

Saves the trained weights to models/predictor_embedder.pth and
models/predictor_predictor.pth (two modules).
"""
import os
import statistics
from .dataset import PredictorDataset
from torch_geometric.data import Data


def train(dataset_path=None, model_dir=None, device=None, transfer_learning_model=None):
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
    Delta = 1

    # Contiguous time-based split (no shuffling across splits -> no leakage)
    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    train_t = all_t[: int(0.70 * n)]
    val_t   = all_t[int(0.70 * n) :] # OBS!!! made this to be larger!!!

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

    def run_t(t, gidx, data):
        curr_overlay = data[(gidx,t)]['overlay_paths']
        curr_overlay_ids = [ovl['id'] for ovl in curr_overlay]
        prev_meas = {}

        for ovl in curr_overlay_ids:
            for t_back in range(t+1):
                ovl_id_list = [tmp['id'] for tmp in data[(gidx,t-t_back)]['overlay_paths']]
                if ovl in ovl_id_list:
                    idx = ovl_id_list.index(ovl)
                    prev_meas[ovl] = data[(gidx,t-t_back)]['overlay_paths'][idx]['meas'][0:2]
                    break

        loss = 0
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
                ovl_id = f_ovl['id']
                if ovl_id not in prev_meas:
                    continue
                out = [0,0]
                out[0] = (prev_meas[ovl_id][0] - mean_delay) / (std_dev_delay + 1e-8)
                out[1] = (prev_meas[ovl_id][1] - mean_loss)  / (std_dev_loss  + 1e-8)

                tgt = f_ovl['meas']

                z0 = out[0] - (tgt[0]-mean_delay)/(std_dev_delay+1e-8)
                z1 = out[1] - (tgt[1]-mean_loss)/(std_dev_loss+1e-8)

                loss = loss + (c_delta * z0**2 + c_lambda * z1**2)
                count += 1

        return loss, count


    data = {}
    for gidx in range(nbr_sims):
        for t in range(len(dataset)):
            d = dataset[gidx][t]
            tmp = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'], overlay_paths=d['overlay_paths'])
            data[(gidx, t)] = tmp
            
    # ---- validate ----
    val_loss = 0.0
    for t in val_t:
        loss = 0; count = 0
        for gidx in range(nbr_sims): #range(1): #OBS! leaving 1 sim out for validation
            loss_gidx, count_gidx = run_t(t, gidx, data)
            loss = loss + loss_gidx
            count = count + count_gidx

        if count == 0:
            continue

        val_loss += float((loss / count))
        
    val_loss = val_loss/(len(val_t)) #OBS!!!
    print(f"Baseline (persistence) val = {val_loss:.4f}")
    return None
    
    
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
