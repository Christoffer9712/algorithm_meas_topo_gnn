from .generate_dataset import generate
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from .environment import RoutingEnvironment

def plot_overlay_delays_loss(data, overlay_ids=None, max_overlays=6):
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


dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
dataset_path = os.path.abspath(dataset_path)

print(f'Data set is stored at path {dataset_path}')

T = 400
generate(dataset_path, T, seeds=(42,43))

try:
    print('before')
    data = torch.load(dataset_path, weights_only=False)
    print('after')
except TypeError:
    # Older PyTorch versions don't support the keyword; fall back.
    data = torch.load(dataset_path)
    print('type_error')

print('Before printing')
print(f'len(data)={len(data)}')
print(f'data.keys()={data.keys()}')
print('after Printing')
env = RoutingEnvironment()
snapshots = []
delay_queue_vs_path = {}

for gidx in range(len(data)):
    print(f'gidx={gidx}')
    for tstep in range(T):
        snapshots.append(env.snapshot_at_time_t(tstep))
        #print(f'data[history_list][gidx] = {data['history_list'][gidx]}')
        for ovl in data['history_list'][gidx][tstep]['overlay_paths']:
            oid = ovl['id']
            underlay = ovl['underlay_path']  # (delay, loss, delay_queue, delay_path)
            meas = ovl['meas']
            ovl_path = ovl['overlay_path']

            if oid not in delay_queue_vs_path:                      # init on first sighting
                delay_queue_vs_path[oid] = {'t': [], 'meas_delay': [], 'meas_loss': [], 'path_change': []}

            changed = True
            if tstep > 1:
                for d in data['history_list'][gidx][tstep-1]['overlay_paths']:
                    if oid == d['id']:
                        changed = not (d['underlay_path'] == underlay)

            delay_queue_vs_path[oid]['path_change'].append(changed)
            delay_queue_vs_path[oid]['t'].append(tstep)
            delay_queue_vs_path[oid]['meas_delay'].append((meas[2], meas[3]))
            delay_queue_vs_path[oid]['meas_loss'].append(meas[1])
        

ovl_ids = list(delay_queue_vs_path.keys())
plot_overlay_delays_loss(delay_queue_vs_path, overlay_ids=ovl_ids[0:10], max_overlays=6)
plot_overlay_delays_loss(delay_queue_vs_path, overlay_ids=ovl_ids[-2:], max_overlays=6)

# Optional: animate queue states across snapshots
if True:
    try:
        import environment.network as netmod
        save_path = os.path.join(os.path.dirname(dataset_path), '..', 'outputs', 'sim_animation.gif')
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Animating {len(snapshots)} frames, saving to {save_path} (attr={'queue_delay'})")
        anim = netmod.animate_queue_states(snapshots, attr='queue_delay', interval=80, save_path=save_path, show=False)
        print('Animation saved to', save_path)
    except Exception as e:
        print('Failed to create animation:', e)
