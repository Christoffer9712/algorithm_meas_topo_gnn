import torch
import os 
from .environment import RoutingEnvironment
from algorithm.predictor.path_encoder import snapshot_to_pyg

def generate(dataset_path=None, T=400, seeds=(42,)):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)

    # Ensure data dir exists
    data_dir = os.path.dirname(dataset_path)
    os.makedirs(data_dir, exist_ok=True)

    history_list = []
    history = {}
    
    # Run full simulation to collect history
    for seed in seeds:
        env = RoutingEnvironment(seed=seed, queue_seed=seed, dt=1.0)
        for tstep in range(T):
            H = env.snapshot()
            overlays = env.get_overlays()

            history[tstep] = {}
            history[tstep]['t'] = tstep

            data = snapshot_to_pyg(H)
            history[tstep] = {}
            history[tstep]['x'] = data.x
            history[tstep]['edge_index'] = data.edge_index
            history[tstep]['edge_attr'] = data.edge_attr
            history[tstep]['node_names'] = data.node_names
            history[tstep]['name_to_idx'] = data.name_to_idx

            ovl_list = []
            for ovl in overlays:
                oid = ovl['id']
                underlay, meas = env.path_metrics_for_overlay(ovl, H)  # (delay, loss, delay_queue, delay_path)
                ovl_list.append({'id': oid, 'underlay_path': underlay, 'overlay_path': ovl['overlay_path'], 'meas':meas})

            history[tstep]['overlay_paths'] = ovl_list
            env.step()
            history_list.append(history)

    print(f"Generated {len(seeds)} x {len(history)} steps, saving to {dataset_path}")
    #history_list = [history[t] for t in sorted(history.keys())] 
    #torch.save(history_list, dataset_path)
    torch.save({'history_list': history_list}, dataset_path)

if __name__ == '__main__':
    path = generate(T=400, seeds=(1))
    print('Done, dataset at', path)
