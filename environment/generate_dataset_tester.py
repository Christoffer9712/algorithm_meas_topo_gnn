from .generate_dataset import generate, generate_nets
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from .environment import RoutingEnvironment
from .network import LayeredOrbitNetwork

def plot_covariance(data, overlay_ids=None, max_overlays=None, use='total', min_overlap=5):
    """
    Correlation of per-overlay delay time series.

    data: dict {oid: {'t':[...], 'meas_delay':[(path,queue),...], ...}}  (as built above)
    use:  'total' (path+queue), 'path', or 'queue'
    min_overlap: minimum shared timesteps required to compute a pair's correlation
    Returns the correlation matrix (np.ndarray) and the oid order.
    """
    if overlay_ids is None:
        overlay_ids = list(data.keys())
    if max_overlays is not None:
        overlay_ids = overlay_ids[:max_overlays]

    # build a time-indexed series per overlay
    series = {}   # oid -> dict {t: value}
    for oid in overlay_ids:
        t = np.asarray(data[oid]['t'])
        md = np.asarray(data[oid]['meas_delay'], dtype=float)   # [T,2] (path, queue)
        if use == 'total':
            v = md[:, 0] + md[:, 1]
        elif use == 'path':
            v = md[:, 0]
        elif use == 'queue':
            v = md[:, 1]
        else:
            raise ValueError(use)
        series[oid] = dict(zip(t.tolist(), v.tolist()))

    n = len(overlay_ids)
    corr = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            sa, sb = series[overlay_ids[a]], series[overlay_ids[b]]
            common = sorted(set(sa) & set(sb))          # shared timesteps
            if len(common) < min_overlap:
                continue
            va = np.array([sa[t] for t in common])
            vb = np.array([sb[t] for t in common])
            # correlation is undefined if either series is constant over the overlap
            if va.std() < 1e-9 or vb.std() < 1e-9:
                corr[a, b] = np.nan
                continue
            corr[a, b] = np.corrcoef(va, vb)[0, 1]

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(0.5 * n + 3, 0.5 * n + 3))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(overlay_ids, rotation=90, fontsize=7)
    ax.set_yticklabels(overlay_ids, fontsize=7)
    ax.set_title(f'overlay delay correlation ({use})')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')

    # annotate if small enough to read
    if n <= 15:
        for a in range(n):
            for b in range(n):
                if not np.isnan(corr[a, b]):
                    ax.text(b, a, f'{corr[a,b]:.2f}', ha='center', va='center',
                            fontsize=6,
                            color='white' if abs(corr[a, b]) > 0.5 else 'black')

    plt.tight_layout()
    plt.show()

    # ---- summary: this is the number that decides your architecture question ----
    off = corr[~np.eye(n, dtype=bool)]
    off = off[~np.isnan(off)]
    if len(off):
        print(f'off-diagonal |r|: mean={np.nanmean(np.abs(off)):.3f}  '
              f'median={np.nanmedian(np.abs(off)):.3f}  '
              f'frac|r|>0.3={np.mean(np.abs(off) > 0.3):.2f}  '
              f'frac|r|>0.5={np.mean(np.abs(off) > 0.5):.2f}')
    return corr, overlay_ids
    
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


dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset_test.pt')
dataset_path = os.path.abspath(dataset_path)

print(f'Data set is stored at path {dataset_path}')

T = 2000
seeds = (42,43)
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
generate(dataset_path, T, seeds=seeds, nets=nets, include_queue_delay=True, include_queue_loss=False)

try:
    data = torch.load(dataset_path, weights_only=False)
except TypeError:
    # Older PyTorch versions don't support the keyword; fall back.
    data = torch.load(dataset_path)
    print('type_error')

snapshots_list = []
for gidx in range(len(data['history_list'])):
    print(f'gidx={gidx}')
    env = RoutingEnvironment(net=nets[gidx],  include_queue_delay=True, include_queue_loss=False)
    snapshots = []
    delay_queue_vs_path = {}
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
            if tstep > 0:
                for d in data['history_list'][gidx][tstep-1]['overlay_paths']:
                    if oid == d['id']:
                        changed = not (d['underlay_path'] == underlay)

            delay_queue_vs_path[oid]['path_change'].append(changed)
            delay_queue_vs_path[oid]['t'].append(tstep)
            delay_queue_vs_path[oid]['meas_delay'].append((meas[2], meas[3]))
            delay_queue_vs_path[oid]['meas_loss'].append(meas[1])
        
    snapshots_list.append(snapshots)
    ovl_ids = list(delay_queue_vs_path.keys())
    plot_covariance(delay_queue_vs_path, overlay_ids=ovl_ids, use='queue')
    plot_covariance(delay_queue_vs_path, overlay_ids=ovl_ids, use='total')

    plot_overlay_delays_loss(delay_queue_vs_path, overlay_ids=ovl_ids[0:10], max_overlays=6)
    plot_overlay_delays_loss(delay_queue_vs_path, overlay_ids=ovl_ids[-2:], max_overlays=6)

# Optional: animate queue states across snapshots
if False:
    try:
        import environment.network as netmod
        for idx in range(len(snapshots_list)):
            save_path = os.path.join(os.path.dirname(dataset_path), '..', 'outputs',f'sim_animation_{idx}.gif')
            save_path = os.path.abspath(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Animating {len(snapshots_list[idx])} frames, saving to {save_path} (attr={'queue_delay'})")
            anim = netmod.animate_queue_states(snapshots_list[idx], attr='queue_delay', interval=80, save_path=save_path, show=False)
            print(f'Animation nr. {idx} saved to', save_path)
    except Exception as e:
        print('Failed to create animation:', e)
