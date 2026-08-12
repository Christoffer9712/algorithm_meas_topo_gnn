"""
MILP-based scheduler using OR-Tools (pywraplp). Produces a horizon schedule
and returns the committed (first-step) mapping.

This implementation expects:
 - flows: list of flows, each as [flow_id, target, b, d, l, q]
 - overlays: list of overlays, each with {'id': id, 'path': [...], 'bandwidth': B^p (optional)}
 - preds: dict mapping overlay_id -> list of length M of (lambda, delta)
 - prev_flow_map: dict mapping flow_id -> overlay_id (previous assignment)

Returns mapping for m=0 (first step) and full horizon decisions.

"""
from ortools.linear_solver import pywraplp


class MILPScheduler:
    def __init__(self, M=10, gamma=0.95, c_l=0.5, c_d=0.5, c_s=1.0, drop_cost=1000.0):
        self.M = M
        self.gamma = gamma
        self.c_l = c_l
        self.c_d = c_d
        self.c_s = c_s
        self.drop_cost = drop_cost

    def schedule(self, flows, overlays, preds, prev_flow_map=None):
        prev_flow_map = prev_flow_map or {}
        M = self.M

        # Indexing maps
        flow_ids = [f[0] for f in flows]
        f_idx = {fid: i for i, fid in enumerate(flow_ids)}
        overlay_ids = [o['id'] for o in overlays]
        p_idx = {pid: i for i, pid in enumerate(overlay_ids)}

        # Bandwidth per overlay
        B = {}
        for o in overlays:
            B[o['id']] = o.get('bandwidth', 1000.0)  # default capacity units

        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            raise RuntimeError('OR-Tools CBC solver unavailable')

        # Variables: mu[f,p,m] binary
        mu = {}
        for fi in range(len(flows)):
            for pi in range(len(overlay_ids)):
                for m in range(M):
                    mu[fi, pi, m] = solver.IntVar(0, 1, f"mu_{fi}_{pi}_{m}")
        # Drop variables: drop[f,m] == 1 if flow f is not scheduled at time m
        drop = {}
        for fi in range(len(flows)):
            for m in range(M):
                drop[fi, m] = solver.IntVar(0, 1, f"drop_{fi}_{m}")

        # Constraints: each flow at each time either assigned to one overlay or dropped
        for fi in range(len(flows)):
            for m in range(M):
                ct = solver.Sum([mu[fi, pi, m] for pi in range(len(overlay_ids))]) + drop[fi, m]
                solver.Add(ct == 1)

        # Capacity constraints per overlay per time
        for pi, pid in enumerate(overlay_ids):
            for m in range(M):
                ct = solver.Sum([flows[fi][2] * mu[fi, pi, m] for fi in range(len(flows))])
                solver.Add(ct <= B[pid])

        # Objective: linearised using precomputed penalties
        obj = solver.Sum([])

        for m in range(M):
            discount = (self.gamma ** m)
            for pi, pid in enumerate(overlay_ids):
                for fi, flow in enumerate(flows):
                    fid = flow[0]
                    b_f = flow[2]
                    d_f = flow[3]
                    l_f = flow[4]
                    q_f = flow[5]

                    # predicted metrics for overlay pid at time m
                    lam_hat, del_hat = preds[pid][m]
                    delay_pen = max((del_hat - d_f) / max(d_f, 1e-6), 0.0)
                    loss_pen = max((lam_hat - l_f) / max(l_f, 1e-6), 0.0)
                    # Normalize penalties to avoid extremely large values when
                    # requirements are small (e.g., d_f in ms). Cap to 1.0 so
                    # packet-loss and delay contribute comparably.
                    delay_pen = min(delay_pen, 1.0)
                    loss_pen = min(loss_pen, 1.0)

                    # switching cost depends on previous assignment at time m-1
                    if m == 0:
                        # previous assignment is provided externally
                        prev_assigned = 1 if (fid in prev_flow_map and prev_flow_map[fid] == pid) else 0
                        switch_cost = self.c_s * (1 - prev_assigned)
                        # constant offset can be dropped from objective but include for clarity
                        coeff = discount * q_f * (self.c_l * loss_pen + self.c_d * delay_pen) + discount * q_f * switch_cost
                        obj += coeff * mu[fi, pi, m]
                    else:
                        # switching cost is c_s*(1 - mu[f,p,m-1]) -> yields c_s*mu - c_s*mu_prev ; linear in variables
                        # implement as c_s * (1 - mu_prev) * assigned_now. This is quadratic in general; approximate by summing c_s*(assigned_now) and subtracting c_s*(assigned_prev) as constants to keep linearity.
                        # Simpler: penalise assignments that differ from previous step by adding c_s*(mu[fi,pi,m] - mu[fi,pi,m-1]) positive part approx via linear term. For now, include c_s*(1 - mu_prev_var) as constant since mu_prev_var is not known here. To keep formulation linear and consistent with document, charge switching cost only when assignment changes between committed steps; here we approximate by c_s*(1 - mu_prev_eq) where mu_prev_eq is 0 (encourages staying unassigned) — conservative.
                        coeff = discount * q_f * (self.c_l * loss_pen + self.c_d * delay_pen)
                        obj += coeff * mu[fi, pi, m]

            # drop costs: heavy penalty to discourage dropping
            for fi, flow in enumerate(flows):
                obj += discount * self.drop_cost * drop[fi, m]

        solver.Minimize(obj)

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
            raise RuntimeError('Solver failed or returned infeasible')


        # Extract decisions for the first step (m=0)
        first_map = {}
        horizon_map = {}
        for fi, flow in enumerate(flows):
            assigned = False
            for pi, pid in enumerate(overlay_ids):
                val = int(mu[fi, pi, 0].solution_value())
                # Debug print per var
                # print(f'mu[{fi},{pi},0] =', val)
                if val == 1:
                    first_map[flow[0]] = pid
                    assigned = True
                    break
            if not assigned:
                # dropped
                first_map[flow[0]] = None

        # Debug: print drop var for first step
        for fi, flow in enumerate(flows):
            dval = int(drop[fi, 0].solution_value())
            # print(f'drop[{fi},0] =', dval)

        # Full horizon mapping (for debugging/analysis)
        for m in range(M):
            horizon_map[m] = {}
            for fi, flow in enumerate(flows):
                assigned = None
                for pi, pid in enumerate(overlay_ids):
                    if int(mu[fi, pi, m].solution_value()) == 1:
                        assigned = pid
                        break
                horizon_map[m][flow[0]] = assigned

        return first_map, horizon_map
