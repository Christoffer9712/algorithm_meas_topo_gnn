import algorithm.scheduler_milp as scheduler_milp


class Scheduler:
    """MILP-based scheduler wrapper.

    This class exposes a simple interface the rest of the codebase expects:
      - map_flows(env, flows, ...): returns selected_overlays, flow_map for the
        current step. Internally it calls the MILP scheduler and commits to the
        first-step decisions.
      - cost(...) : retains the existing cost function for evaluation.
    """

    def __init__(self):
        # Stateless wrapper; the MILP scheduler encapsulates hyperparameters.
        pass

    def map_flows(self, env, flows, preds=None, prev_flow_map=None, M=10, params=None):
        """Schedule flows using the MILP scheduler.

        preds: optional dict overlay_id -> list of M (lambda, delay) tuples. If
               not provided, current measured metrics are used for all horizon
               steps.
        """
        overlays = env.get_overlays()
        if preds is None:
            preds = {}
            H = env.snapshot()
            for o in overlays:
                meas = env.path_metrics_for_overlay(o, H)
                preds[o['id']] = [meas for _ in range(M)]

        if params:
            ms = scheduler_milp.MILPScheduler(
                M=M,
                gamma=params.get('gamma', 0.95),
                c_l=params.get('c_l', 0.5),
                c_d=params.get('c_d', 0.5),
                c_s=params.get('c_s', 1.0),
                drop_cost=params.get('drop_cost', 1000.0),
            )
        else:
            ms = scheduler_milp.MILPScheduler(M=M)

        selected_overlays, flow_map = ms.schedule(flows, overlays, preds, prev_flow_map=prev_flow_map)
        # schedule() returns first_map for the committed step as the first element
        # but to keep compatibility we transform to the original return types.
        # Note: scheduler_milp.MILPScheduler.schedule returns (first_map, horizon_map)
        first_map, _ = selected_overlays, flow_map  # pass through if wrapped
        # However the MILP wrapper already returns correct types, so call it directly
        first_map, horizon_map = ms.schedule(flows, overlays, preds, prev_flow_map=prev_flow_map)

        selected_overlay_ids = set([v for v in first_map.values() if v is not None])
        selected_overlays = [o for o in overlays if o['id'] in selected_overlay_ids]
        return selected_overlays, first_map

    def cost(self, flows, metrics, flow_map, prev_flow_map):
        cost = 0
        for flow in flows:
            flow_id = flow[0]
            # switching cost
            cost_switch = 0
            if flow_id in prev_flow_map:
                if prev_flow_map[flow_id] != flow_map.get(flow_id, None):
                    cost_switch = 1

            required_delay = flow[3]
            required_loss = flow[4]
            priority = flow[5]

            meas = metrics.get(flow_map.get(flow_id, None), (0.0, 0.0))
            meas_delay = meas[0]
            meas_loss = meas[1]

            cost_delay = max((meas_delay - required_delay) / max(required_delay, 1e-6), 0)
            cost_loss = max((meas_loss - required_loss) / max(required_loss, 1e-6), 0)
            # Normalize individual penalties by capping at 1.0 and give equal weight
            cost_delay = min(cost_delay, 1.0)
            cost_loss = min(cost_loss, 1.0)

            cost += priority * ((cost_delay + cost_loss) / 2.0) + cost_switch

        return cost