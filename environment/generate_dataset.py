import torch
import os 
from .environment import RoutingEnvironment
from algorithm.predictor.path_encoder import snapshot_to_pyg
from .network import LayeredOrbitNetwork
import numpy as np

def generate_nets(seeds):
    nets = []
    for seed in seeds:
        np.random.seed(seed)
        rand_arr = np.random.rand(12)
        net = LayeredOrbitNetwork(
                    sats_per_ring=(3+round(rand_arr[0]*5), 5+round(rand_arr[1]*5), 5+round(rand_arr[2]*5)),
                    n_gateways=1+round(rand_arr[3]*3),
                    n_targets=1,
                    ring_radii=(340.0+round(rand_arr[4]*50), 200.0+round(rand_arr[5]*50), 120.0+round(rand_arr[6]*50)),
                    ring_speeds=(0.010+rand_arr[7]/100, -0.035+rand_arr[8]/10, 0.015+rand_arr[9]/50),   # rad per unit time; sign = direction
                    orbit_center=(0.0, 0.0),
                    aircraft_pos=(-40.0+round(rand_arr[10]*10), -70.0+round(rand_arr[11]*10)),
                    gateway_positions=None,
                    target_positions=None,
                    range_limit=350.0,
                    seed=seed)
        nets.append(net)
    return nets

def generate(dataset_path=None, T=400, seeds=(42,), nets=None):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)

    # Ensure data dir exists
    data_dir = os.path.dirname(dataset_path)
    os.makedirs(data_dir, exist_ok=True)

    history_list = []
        
    # Run full simulation to collect history
    for idx in range(len(seeds)):
        history = {}
        seed = seeds[idx]
        if nets is None:
            print('CHRISTOFFER NET IS NONE')
            net = LayeredOrbitNetwork(
                sats_per_ring=(5, 10, 5),
                n_gateways=3,
                n_targets=1,
                ring_radii=(340.0, 250.0, 170.0),
                ring_speeds=(0.010, -0.035, 0.015),   # rad per unit time; sign = direction
                orbit_center=(0.0, 0.0),
                aircraft_pos=(-40.0, -70.0),
                gateway_positions=None,
                target_positions=None,
                range_limit=300.0,
                seed=seed,
            )
        else:
            net = nets[idx]

        env = RoutingEnvironment(net=net, seed=seed, queue_seed=seed, dt=1.0)
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
        print(f'Finished simulation sim {idx}')
        history_list.append(history)

    print(f"Generated {len(seeds)} x {len(history)} steps, saving to {dataset_path}")
    #history_list = [history[t] for t in sorted(history.keys())] 
    #torch.save(history_list, dataset_path)
    torch.save({'history_list': history_list}, dataset_path)
    return(dataset_path)

if __name__ == '__main__':
    path = generate(T=400, seeds=(1))
    print('Done, dataset at', path)
