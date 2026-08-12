"""
Generate a dataset for training the measurement+predictor models.

Runs the existing environment for `T` timesteps, and for each overlay at each
step records:
    - h_embedding (from existing pathEncoder.GraphEncoder)
    - measured metrics (lambda, delay)

Then constructs samples where the input is:
    h_pred (embedding at the future prediction time for a target overlay),
    h_hists (all overlay embeddings from the previous Delta time steps),
    meas_vals (all overlay measurements from the previous Delta time steps),
    elapsed (elapsed time from each history entry to the prediction),
and label is the measured metrics at the future prediction time for the same
overlay. This makes historical measurements from different overlays available
for every prediction because overlays share the same underlying network.

Saves dataset to data/predictor_dataset.pt
"""
import torch
import os
import numpy as np
 
import environment
import pathEncoder

import matplotlib.pyplot as plt

def generate(dataset_path=None, T=400, Delta=10, seed=42, device='cpu', animate=False, anim_path=None, attr='queue_delay', interval=80):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)

    # Ensure data dir exists
    data_dir = os.path.dirname(dataset_path)
    os.makedirs(data_dir, exist_ok=True)

    env = environment.RoutingEnvironment(seed=seed, queue_seed=seed, dt=1.0)

    # history per time-step: dict[overlay id -> (h_vec (numpy), (lambda, delay))]
    history = {}
    snapshots = []

    delay_queue_vs_path = {}
    # Run full simulation to collect history
    for tstep in range(T):
        H = env.snapshot()
        overlays = env.get_overlays()
        snapshots.append(H.copy())

        data = pathEncoder.snapshot_to_pyg(H)
        history[tstep] = {}
        history[tstep]['t'] = tstep
        history[tstep]['x'] = data.x
        history[tstep]['edge_index'] = data.edge_index
        history[tstep]['edge_attr'] = data.edge_attr
        history[tstep]['node_names'] = data.node_names
        history[tstep]['name_to_idx'] = data.name_to_idx

        for ov in overlays:
            oid = ov['id']
            underlay, meas = env.path_metrics_for_overlay(ov, H)  # (delay, loss, delay_queue, delay_path)
            lam = meas[1]
            delta = meas[0] * 1000.0  # convert to ms
            history[tstep][oid] = {'underlay_path': underlay, 'path': ov['path'], 'meas':(lam, delta)}
            if oid not in delay_queue_vs_path:                      # init on first sighting
                delay_queue_vs_path[oid] = {'t': [], 'meas_delay': [], 'meas_loss': [], 'path_change': []}

            if tstep > 1 and oid in history[tstep-1].keys():
                changed = not (history[tstep][oid]['underlay_path'] == history[tstep-1][oid]['underlay_path'])
            else:
                changed = True
            delay_queue_vs_path[oid]['path_change'].append(changed)

            delay_queue_vs_path[oid]['t'].append(tstep)
            delay_queue_vs_path[oid]['meas_delay'].append((meas[2], meas[3]))
            delay_queue_vs_path[oid]['meas_loss'].append(lam)

        env.step()

    ovl_ids = list(delay_queue_vs_path.keys())
    plot_overlay_delays_loss(delay_queue_vs_path, overlay_ids=ovl_ids[0:10], max_overlays=6)

    print(f"Generated {len(history)} steps, saving to {dataset_path}")
    torch.save({'history': history}, dataset_path)

    # Optional: animate queue states across snapshots
    if animate:
        try:
            import network as netmod
            save_path = anim_path or os.path.join(os.path.dirname(dataset_path), '..', 'outputs', 'sim_animation.gif')
            save_path = os.path.abspath(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Animating {len(snapshots)} frames, saving to {save_path} (attr={attr})")
            anim = netmod.animate_queue_states(snapshots, attr=attr, interval=interval, save_path=save_path, show=False)
            print('Animation saved to', save_path)
        except Exception as e:
            print('Failed to create animation:', e)



def plot_overlay_delays_loss(data, overlay_ids=None, max_overlays=6):
    """
    data: {ovl_id: {'t': [...], 'meas_delay': [(path, queue), ...],
                    'meas_loss': [float, ...], 'path_change': [bool, ...]}, ...}
    """
    if overlay_ids is None:
        overlay_ids = list(data.keys())[:max_overlays]

    n = len(overlay_ids)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.8 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, oid in zip(axes, overlay_ids):
        t = np.asarray(data[oid]['t'])
        meas_delay = np.asarray(data[oid]['meas_delay'], dtype=float)   # [T, 2]
        p, q = meas_delay[:, 0], meas_delay[:, 1]                        # path, queue

        # --- delay on the left axis (stacked) ---
        ax.stackplot(t, p, q, labels=['path delay', 'queue delay'], alpha=0.5)
        ax.plot(t, p + q, color='black', linewidth=1, label='total delay')
        ax.set_ylabel('delay (ms)')

        # --- loss rate on the right axis ---
        ax2 = ax.twinx()
        loss = np.asarray(data[oid]['meas_loss'], dtype=float)
        ax2.plot(t, loss, color='purple', linewidth=1.5, label='loss rate')
        ax2.set_ylabel('loss rate')
        ax2.set_ylim(0, max(loss.max() * 1.1, 1e-3))                     # loss is a probability

        # --- mark underlay changes (full-height, spans both axes) ---
        changes = np.asarray(data[oid].get('path_change', []), dtype=bool)
        change_t = t[changes] if len(changes) == len(t) else []
        for i, ct in enumerate(change_t):
            ax.axvline(ct, color='red', linestyle='--', linewidth=1, alpha=0.6,
                       label='underlay change' if i == 0 else None)

        # --- combined legend (both axes into one box) ---
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

        ax.set_title(f'overlay {oid}')

    axes[-1].set_xlabel('time')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path = generate(T=400, seed=1)
    print('Done, dataset at', path)
