"""
Train the Markov/GAT estimator against end-to-end (delay, loss) measurements.

The GAT predicts latent per-node and per-edge quantities; the absorbing
chain maps them to an end-to-end (delay, loss) per overlay, supervised
against ovl['meas']. Source/target of each overlay are the first/last node
of its underlay_path.
"""
import os
import copy
import statistics
import torch

from algorithm.predictor.hetero_snapshot import hetero_from_record
from algorithm.predictor.hetero_model import SatNetEstimator, absorbing_endtoend
from .dataset import PredictorDataset


def train(dataset_path=None, model_dir=None, epochs=10, batch_size=1,
          lr=5e-4, device="cpu", min_delta=1e-4, patience=10):
    if dataset_path is None:
        dataset_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "data", "predictor_dataset.pt"))
    if model_dir is None:
        model_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    
    dataset = PredictorDataset(dataset_path)
    nbr_sims = dataset.get_nbr_of_sims()

    # build the model, sizing its inputs from one promoted snapshot
    probe = hetero_from_record(dataset[0][0], device=device)
    model = SatNetEstimator(
        node_in=probe["node"].x.shape[1],
        edge_in=probe["edge"].x.shape[1],
        hidden_dim=128, emb_dim=64, heads=2, n_layers=3,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5)

    c_lambda = 1.0
    c_delta = 5.0
    M = 1
    Delta = 1 #OBS!!!

    lo, hi = Delta - 1, len(dataset) - M
    all_t = list(range(lo, hi))
    train_t = val_t = all_t          # validation uses a held-out sim, not held-out time
    
    # ---- standardization stats over the training sims --------------------
    meas_delay, meas_loss = [], []
    for gidx in range(nbr_sims-1):
        for t in train_t:
            for ovl in dataset[gidx][t]["overlay_paths"]:
                meas_delay.append(ovl["meas"][0])
                meas_loss.append(ovl["meas"][1])
    mean_delay, std_delay = statistics.mean(meas_delay), statistics.stdev(meas_delay)
    mean_loss, std_loss = statistics.mean(meas_loss), statistics.stdev(meas_loss)
    print(f"delay mean/std = {mean_delay:.4f} / {std_delay:.4f}")
    print(f"loss  mean/std = {mean_loss:.4f} / {std_loss:.4f}")

    print(f"Baseline (predict-mean) train = {baseline_loss_mean(train_t, dataset, 0, M, mean_loss, std_loss, mean_delay, std_delay, c_lambda, c_delta):.4f}")
    print(f"Baseline (predict-mean) val   = {baseline_loss_mean(val_t, dataset, 0, M, mean_loss, std_loss, mean_delay, std_delay, c_lambda, c_delta):.4f}")
    #print(f"Baseline (predict-mean) test  = {baseline_loss_mean(test_t, dataset, 0, M, mean_loss, std_loss, mean_delay, std_delay, c_lambda, c_delta):.4f}")

    def run_t(t, gidx):
        """Summed loss over all overlays at (gidx, t)."""
        curr_overlay = dataset[gidx][t]["overlay_paths"]
        loss = torch.zeros((), device=device)
        count = 0

        for m in range(1, M + 1):
            d = dataset[gidx][t + m]
            data = hetero_from_record(d, device=device)

            future = {p["id"]: p for p in d["overlay_paths"]}
            for ovl in curr_overlay:
                if ovl["id"] not in future:
                    continue
                path = future[ovl["id"]]["overlay_path"]
                node_idx_paths = [data['node_name_to_idx'][node] for node in path]

                x = data['node']['x']
                flag = torch.zeros((x.shape[0], 1), device=device)
                flag[node_idx_paths] = 1.0
                data['node']['x'] = torch.cat([x[:, :-1], flag], dim=1)

                pred = model(data)
                P = absorbing_endtoend.build_P(pred, data, device)

                s = data.node_name_to_idx[path[0]]
                a = data.node_name_to_idx[path[-1]]
                if s == a:
                    continue

                delay, survival = absorbing_endtoend.endtoend(
                    P, pred, data, s, a, device)
                pred_loss = 1.0 - survival

                tgt = torch.as_tensor(ovl["meas"], dtype=torch.float, device=device)
                z0 = (delay - tgt[0]) / (std_delay + 1e-8)
                z1 = (pred_loss - tgt[1]) / (std_loss + 1e-8)
                loss = loss + (c_delta * z0**2 + c_lambda * z1**2)
                count += 1
        return loss, count

    batch_size = max(1, batch_size // (nbr_sims - 1))

    best_val, best_state, no_improve = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        batch_sample = 1
        loss_batch = torch.zeros((), device=device)
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            for gidx in range(nbr_sims - 1):      # last sim held out
                lg, cg = run_t(t, gidx)
                loss = loss + lg; count += cg
            if count == 0:
                continue
            step_loss = loss / count
            loss_batch = loss_batch + step_loss
            train_loss += float(step_loss.detach())
            if batch_sample >= batch_size:
                opt.zero_grad(); loss_batch.backward(); opt.step()
                batch_sample = 1
                loss_batch = torch.zeros((), device=device)
            else:
                batch_sample += 1
        train_loss /= len(train_t)

        # ---- validate on the held-out (last) sim ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t in val_t:
                lg, cg = run_t(t, nbr_sims - 1)
                if cg == 0:
                    continue
                val_loss += float((lg / cg).detach())
            val_loss /= len(val_t)

        scheduler.step(val_loss)

        # ---- early stopping ----
        if val_loss < best_val - min_delta:
            best_val = val_loss
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best val_loss = {best_val:.6f})")
                break

        print(f"Epoch {ep} - train_loss = {train_loss:.6f} - val_loss = {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = os.path.join(model_dir, "predictor_models.pth")
    torch.save({"model_state": model.state_dict(),
                "mean_delay": mean_delay, "mean_loss": mean_loss,
                "std_delay": std_delay, "std_loss": std_loss},
               save_path)
    print("Saved trained model to", save_path)
    return save_path


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

if __name__ == "__main__":
    train(epochs=10, batch_size=32)