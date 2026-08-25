import math
import random

import numpy as np


class NodeQueues:
    """
    Per-node M/M/1 queueing state with temporal memory.

    Two sources of memory:
      1. Offered load follows an Ornstein-Uhlenbeck process (mean-reverting,
         autocorrelated) -> congested now implies probably congested next step.
      2. Queue occupancy relaxes toward its M/M/1 equilibrium with a finite
         time constant, so it lags sudden load changes.

    Attributes written onto each node:
      service_rate     mu, packets per unit time  (static)
      load             lambda, current offered rate
      utilisation      rho = lambda / mu
      queue_len        L, current occupancy (packets)
      queue_delay      W, waiting + service time
      loss             packet-loss probability
    """

    def __init__(
        self,
        graph,
        service_rate_range=(3, 5),      # mu, per node type below
        base_load_range=(0.25, 0.7),    # mean rho each node sits at
        ou_theta= 0.006,                # mean-reversion rate (1/timescale)
        ou_sigma=0.03,                  # driving noise magnitude
        queue_tau=1.0,                  # queue relaxation time constant
        buffer_size=40.0,               # packets; sets loss curve
        rho_max=0.985,                  # clamp to keep M/M/1 finite
        seed=0,
    ):
        self.G = graph
        self.rng = np.random.default_rng(seed)
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.queue_tau = queue_tau
        self.buffer_size = buffer_size
        self.rho_max = rho_max

        # Node-type multipliers: satellites are the scarce, contended resource;
        # gateways less so; the wired target is effectively uncongested.
        self.type_mu = {
            "Aircraft": 4.0,
            "Satellite": 1.0,
            "Gateway": 2.0,
            "Target": 8.0,
        }

        for n, d in self.G.nodes(data=True):
            mult = self.type_mu.get(d["node_type"], 1.0)
            mu = self.rng.uniform(*service_rate_range) * mult
            rho0 = self.rng.uniform(*base_load_range)

            d["service_rate"] = mu
            d["base_rho"] = rho0          # OU mean, in utilisation units
            d["rho_state"] = rho0         # OU state
            d["utilisation"] = rho0
            d["load"] = rho0 * mu         # lambda
            d["queue_len"] = self._equilibrium_len(rho0)
            d["queue_delay"] = self._equilibrium_delay(rho0, mu)
            d["loss"] = self._loss(d["queue_len"])

    # ------------------------------------------------------------- M/M/1 core

    def _equilibrium_len(self, rho):
        """L = rho / (1 - rho)."""
        rho = min(rho, self.rho_max)
        return rho / (1.0 - rho)

    def _equilibrium_delay(self, rho, mu):
        """W = 1 / (mu - lambda) = 1 / (mu (1 - rho))."""
        rho = min(rho, self.rho_max)
        return 1.0 / (mu * (1.0 - rho))

    def _loss(self, queue_len):
        """
        Finite-buffer loss. Smooth, monotone, ~0 when the queue is short and
        rising sharply as occupancy approaches the buffer size.
        """
        return 1.0 - math.exp(-queue_len / self.buffer_size)

    # ------------------------------------------------------------------ step

    def step(self, dt):
        """
        Advance all node queues by dt.

        """

        for n, d in self.G.nodes(data=True):
            mu = d["service_rate"]

            # --- 1. Offered load: Ornstein-Uhlenbeck, mean-reverting ---------
            mean = d["base_rho"]
            drift = self.ou_theta * (mean - d["rho_state"]) * dt
            noise = self.ou_sigma * math.sqrt(dt) * self.rng.standard_normal()
            rho = d["rho_state"] + drift + noise
            rho = float(np.clip(rho, 0.02, self.rho_max))
            d["rho_state"] = rho
            d["utilisation"] = rho
            d["load"] = rho * mu

            # --- 2. Queue relaxes toward equilibrium, not instantly ----------
            target_len = self._equilibrium_len(rho)
            alpha = 1.0 - math.exp(-dt / self.queue_tau)
            d["queue_len"] += alpha * (target_len - d["queue_len"])

            # --- 3. Derived metrics ------------------------------------------
            # Delay from actual occupancy via Little's law: W = L / lambda
            d["queue_delay"] = d["queue_len"] / max(d["load"], 1e-6)
            d["loss"] = self._loss(d["queue_len"])

    # -------------------------------------------------------- path aggregation

    def path_metrics(self, path, snapshot, prop_speed=100.0):
        """
        Delay and loss for an overlay path.
        Delay = sum of per-node queueing + per-edge propagation.
        Loss  = 1 - prod(1 - per-node loss).
        """
        delay_queue = 0.0
        delay_path = 0.0
        keep = 1.0
        for n in path:
            delay_queue += 0*self.G.nodes[n]["queue_delay"] #TMP!!!!!!!!!!!
            keep *= 1.0 #- self.G.nodes[n]["loss"]         #TMP!!!!!!!!!!!
        for u, v in zip(path[:-1], path[1:]):
            if snapshot.has_edge(u, v):
                delay_path += snapshot[u][v]["distance"] / prop_speed

        return delay_queue+delay_path, 1.0 - keep, delay_path, delay_queue