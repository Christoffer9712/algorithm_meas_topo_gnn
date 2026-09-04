import os
import algorithm.predictor.path_encoder as path_encoder
from environment.network import LayeredOrbitNetwork
import numpy as np

# Build the routing environment and the joint encoder/scheduler.
# RUN_MODE selects which chain to run:
#   'full'            : existing DQN-based mapping loop (unchanged)
#   'predictor_only'  : run predictor inference on environment snapshots (requires trained models)
#   'train_predictor' : generate dataset and train the measurement/predictor models
RUN_MODE = 'train_predictor'  # change to 'train_predictor' or 'predictor_only' as needed

#env = environment.RoutingEnvironment(seed=42, queue_seed=1, dt=1.0)
encoder = path_encoder.GraphEncoder(
    path_encoder.GATv2Encoder(in_dim=8, hidden_dim=64, out_dim=64, edge_dim=1),
    device='cpu',
)



if RUN_MODE == 'train_predictor':
    # Generate dataset and train predictor models
    from environment.generate_dataset import generate, generate_nets
    from trainer.train_predictor import train

    re_calculate_data = False
    if re_calculate_data:
        T = 200
        seeds = range(20)
        nets = generate_nets(seeds)
        for idx in range(len(nets)):
            print(
                f'net-{idx} has '
                f'sats_per_ring = {nets[idx].sats_per_ring}, '
                f'n_rings = {nets[idx].n_rings}, '
                f'n_gateways = {nets[idx].n_gateways}, '
                f'n_targets = {nets[idx].n_targets}, '
                f'ring_radii = {nets[idx].ring_radii}, '
                f'ring_speeds = {nets[idx].ring_speeds}, '
                f'orbit_center = {nets[idx].orbit_center}, '
                f'aircraft_pos = {nets[idx].aircraft_pos}, '
                f'range_limit = {nets[idx].range_limit}'
            )
            print('-----------------------------')

        dataset_path= generate(T=T, seeds=seeds, nets=nets)

    else:
        dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)

    model_path = train(dataset_path)
    print('Training complete. Models saved to', model_path)
    raise SystemExit(0)

'''
snapshots, times = [], []
prev_flow_map = {}
optimal_prev_flow_map = {}
flows = [['F1', 'TGT-0', 10, 15, 0.2, 5]]

if RUN_MODE == 'predictor_only':
    # Run predictor on snapshots using saved models
    import torch
    model_file = 'models/predictor_models.pth'
    if not os.path.exists(model_file):
        raise RuntimeError('Model file not found: ' + model_file)
    model_dict = torch.load(model_file, map_location='cpu')
    from algorithm.meas_predictor import MeasurementEmbedder, Predictor
    # infer dimensions from encoder
    dummy_H = env.snapshot()
    overlays = env.get_overlays()
    ovl_enc = encoder.encode_overlays(dummy_H, overlays)
    # pick any embedding
    any_h = next(iter(ovl_enc.values()))
    D = any_h.shape[0]
    embedder = MeasurementEmbedder(h_dim=D, hidden_dim=128, out_dim=64)
    predictor = Predictor(in_dim=D + 64 + 1, hidden_dim=128)
    embedder.load_state_dict(model_dict['embedder_state'])
    # load predictor safely (handle older checkpoints with different input dims)
    try:
        predictor.load_state_dict(model_dict['predictor_state'])
    except RuntimeError:
        old_sd = model_dict['predictor_state']
        new_sd = predictor.state_dict()
        if 'net.0.weight' in old_sd and 'net.0.weight' in new_sd:
            old_w = old_sd['net.0.weight']
            new_w = new_sd['net.0.weight']
            cols = min(old_w.shape[1], new_w.shape[1])
            new_w[:, :cols] = old_w[:, :cols]
            new_sd['net.0.weight'] = new_w
        if 'net.0.bias' in old_sd and 'net.0.bias' in new_sd:
            new_sd['net.0.bias'] = old_sd['net.0.bias']
        for k, v in old_sd.items():
            if k in new_sd and new_sd[k].shape == v.shape:
                new_sd[k] = v
        predictor.load_state_dict(new_sd)
    embedder.eval(); predictor.eval()

    for k in range(20):
        H = env.snapshot()
        overlays = env.get_overlays()
        ovl_enc = encoder.encode_overlays(H, overlays)
        for o in overlays:
            oid = o['id']
            h = ovl_enc[oid]
            # no history available here; use a dummy small history equal to current
            h_hists = torch.stack([h for _ in range(10)], dim=0)
            meas_vals = torch.zeros((10,2))
            elapsed = torch.arange(10).unsqueeze(1).float()
            g = embedder(h, h_hists, meas_vals, elapsed)
            horizon_t = torch.tensor([0.0], dtype=torch.float32)
            Hcat = torch.cat([h, g, horizon_t], dim=0).unsqueeze(0)
            out = predictor(Hcat).squeeze(0)
            lam_hat, del_hat = float(out[0].item()), float(out[1].item())
            print(f"Overlay {oid}: pred_loss={lam_hat:.4f}, pred_delay_ms={del_hat:.2f}")
        env.step()
    raise SystemExit(0)


if RUN_MODE == 'full':
    # Full predictor -> MILP scheduling loop
    import torch
 
    model_file = os.path.join(os.path.dirname(__file__), 'models', 'predictor_models.pth')
    if not os.path.exists(model_file):
        raise RuntimeError('Model file not found: ' + model_file + '. Please run with RUN_MODE=\'train_predictor\' first to produce models.')

    ckpt = torch.load(model_file, map_location='cpu')
    from algorithm.meas_predictor import MeasurementEmbedder, Predictor

    # infer sizes
    dummy_H = env.snapshot()
    overlays = env.get_overlays()
    if not overlays:
        raise RuntimeError('No overlays available in environment snapshot')
    ovl_enc = encoder.encode_overlays(dummy_H, overlays)
    any_h = next(iter(ovl_enc.values()))
    D = any_h.shape[0]

    embedder = MeasurementEmbedder(h_dim=D, hidden_dim=128, out_dim=64)
    predictor_model = Predictor(in_dim=D + 64 + 1, hidden_dim=128)
    embedder.load_state_dict(ckpt['embedder_state'])
    # Handle potential mismatch if models were trained with the old predictor input size
    try:
        predictor_model.load_state_dict(ckpt['predictor_state'])
    except RuntimeError:
        # Attempt to merge old weights into new model by copying existing compatible slices
        old_sd = ckpt['predictor_state']
        new_sd = predictor_model.state_dict()
        # Copy layer 0 weights/bias if shapes compatible on shared dims
        if 'net.0.weight' in old_sd and 'net.0.weight' in new_sd:
            old_w = old_sd['net.0.weight']
            new_w = new_sd['net.0.weight']
            # Copy shared columns
            cols = min(old_w.shape[1], new_w.shape[1])
            new_w[:, :cols] = old_w[:, :cols]
            new_sd['net.0.weight'] = new_w
        if 'net.0.bias' in old_sd and 'net.0.bias' in new_sd:
            new_sd['net.0.bias'] = old_sd['net.0.bias']
        # Copy remaining layers if shapes match
        for k, v in old_sd.items():
            if k in new_sd and new_sd[k].shape == v.shape:
                new_sd[k] = v
        predictor_model.load_state_dict(new_sd)

    embedder.eval(); predictor_model.eval()

    # history by timestep: dict[tstep -> dict[overlay id -> (h_tensor, (lam, delta_ms))]]
    history = {}
    all_oids = set()
    M = 5
    Delta = 10
    prev_flow_map = {}
    flows = [['F1', 'TGT-0', 10, 15, 0.2, 5]]

    for k in range(50):
        t = env.time
        H = env.snapshot()
        overlays = env.get_overlays()
        ovl_enc = encoder.encode_overlays(H, overlays)

        # update history for all overlays at this timestep
        current_history = {}
        for o in overlays:
            oid = o['id']
            h = ovl_enc[oid].detach().cpu()
            meas = env.path_metrics_for_overlay(o, H)  # (delay_seconds, loss)
            lam = meas[1]
            delta_ms = meas[0] * 1000.0
            current_history[oid] = (h, (lam, delta_ms))
            all_oids.add(oid)
        history[k] = current_history
        if len(history) > Delta:
            oldest = min(history.keys())
            del history[oldest]
 
        # build preds for horizon M by predicting current step and repeating
        preds = {}
        ordered_oids = sorted(all_oids)
        history_times = sorted(history.keys())
        pad_n = max(0, Delta - len(history_times))
        padded_history_times = []
        if history_times:
            first_tau = history_times[0]
            for i in range(pad_n):
                padded_history_times.append(first_tau - (pad_n - i))
        padded_history_times.extend(history_times)
 
        # build future embeddings by stepping a copy of the environment
        import copy
        clone_env = copy.deepcopy(env)
        future_ovl_encs = []  # list of dicts per future step
        for mstep in range(M):
            clone_env.step()
            H_future = clone_env.snapshot()
            overlays_future = clone_env.get_overlays()
            ovl_enc_f = encoder.encode_overlays(H_future, overlays_future)
            future_ovl_encs.append(ovl_enc_f)
 
        for o in overlays:
            oid = o['id']
            h_pred = ovl_enc[oid].detach()
            preds_list = []
            for m in range(M):
                # h_pred at future time k+m
                # future ovl enc may not contain the same overlay id (topology changes); fallback to current
                h_pred_m = future_ovl_encs[m].get(oid, ovl_enc[oid]).detach()
                target_time = k + m
 
                if not padded_history_times or not ordered_oids:
                    h_hists_m = torch.stack([h_pred_m for _ in range(Delta)], dim=0)
                    meas_vals_m = torch.zeros((Delta, 2))
                    elapsed_m = torch.arange(Delta).unsqueeze(1).float()
                else:
                    h_hists_m = []
                    meas_vals_m = []
                    elapsed_m = []
                    for tau in padded_history_times:
                        step_history = history.get(tau, {})
                        for hist_oid in ordered_oids:
                            entry = step_history.get(hist_oid)
                            if entry is not None:
                                h_hists_m.append(entry[0])
                                meas_vals_m.append(torch.tensor(entry[1], dtype=torch.float32))
                            else:
                                h_hists_m.append(torch.zeros_like(h_pred_m))
                                meas_vals_m.append(torch.zeros(2, dtype=torch.float32))
                            elapsed_m.append(float(target_time - tau))
                    h_hists_m = torch.stack(h_hists_m, dim=0)
                    meas_vals_m = torch.stack(meas_vals_m, dim=0)
                    elapsed_m = torch.tensor(elapsed_m, dtype=torch.float32).unsqueeze(1)
 
                with torch.no_grad():
                    g_m = embedder(h_pred_m, h_hists_m, meas_vals_m, elapsed_m)
                    # horizon scalar m as staleness indicator
                    horizon_t = torch.tensor([float(m)], dtype=torch.float32)
                    Hcat = torch.cat([h_pred_m, g_m, horizon_t], dim=0).unsqueeze(0)
                    out = predictor_model(Hcat).squeeze(0)
                    lam_hat_m = float(out[0].item())
                    del_hat_m = float(out[1].item())
                preds_list.append((lam_hat_m, del_hat_m))
 
            preds[oid] = preds_list

        # schedule using MILP
        sched = scheduler_module.Scheduler()
        selected_overlays, flow_map = sched.map_flows(env, flows, preds=preds, prev_flow_map=prev_flow_map, M=M)

        # compute metrics for selected overlays (convert delay to ms)
        metrics = {}
        for overlay in selected_overlays:
            meas = env.path_metrics_for_overlay(overlay, H)
            metrics[overlay['id']] = (meas[0]*1000.0, meas[1])

        cost = sched.cost(flows, metrics, flow_map, prev_flow_map)
        print(f"Step {k}: Cost = {cost}, assigned = {flow_map}")

        prev_flow_map = flow_map
        env.step()

    raise SystemExit(0)

for k in range(50):
    t = env.time
    H = env.snapshot()
    snapshots.append(H.copy())

    selected_overlays, flow_map = sched.map_flows(env, flows, rand=False, epsilon=0.1)
    metrics = {
        overlay['id']: env.path_metrics_for_overlay(overlay, H)
        for overlay in selected_overlays
    }
    cost = sched.cost(flows, metrics, flow_map, prev_flow_map)
    print(f"Step {k}: Cost = {cost}")

    optimal_selected_overlays, optimal_flow_map = sched.optimal_map_flows(env, flows, optimal_prev_flow_map)
    optimal_metrics = {
        overlay['id']: env.path_metrics_for_overlay(overlay, H)
        for overlay in optimal_selected_overlays
    }
    optimal_cost = sched.cost(flows, optimal_metrics, optimal_flow_map, optimal_prev_flow_map)
    print(f"Step {k}: Optimal Cost = {optimal_cost}")

    prev_flow_map = flow_map
    optimal_prev_flow_map = optimal_flow_map
    env.step()
    times.append(t)

x_bound = abs(env.net.ring_radii[0]) + abs(env.net.aircraft_pos[0])
y_bound = abs(env.net.ring_radii[0]) + abs(env.net.aircraft_pos[1])
x_lim = (-x_bound, x_bound)
y_lim = (-y_bound, y_bound)
network.plot_grid(env.net, times=times, x_lim=x_lim, y_lim=y_lim)

'''