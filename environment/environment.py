from .network import LayeredOrbitNetwork
from .node_queues import NodeQueues


class RoutingEnvironment:
    """Encapsulates a dynamic network, queue state, and current time."""

    def __init__(self, net=None, queue_seed=0, dt=1.0, seed=42):
        self.dt = dt
        self.t = 0.0
        self.seed = seed
        self.net = net if net is not None else LayeredOrbitNetwork(seed=seed)
        self.queues = NodeQueues(self.net.G, seed=queue_seed)
        self._snapshot = None

    def snapshot(self):
        self._snapshot = self.net.snapshot(self.t)
        return self._snapshot

    def snapshot_at_time_t(self, t):
        return self.net.snapshot(t)

    def get_overlays(self):
        H = self.snapshot()
        return self.net.get_all_overlays(H)

    def overlay_to_underlay(self, overlay):
        return self.net.overlay_to_underlay(self.t, overlay['overlay_path'])

    def path_metrics(self, path, H=None):
        if H is None:
            H = self.snapshot()
        return self.queues.path_metrics(path, H)

    def path_metrics_for_overlay(self, overlay, H=None):
        underlay = self.overlay_to_underlay(overlay)
        return (underlay, self.path_metrics(underlay, H))

    def step(self):
        self.queues.step(self.dt)
        self.t += self.dt

    @property
    def time(self):
        return self.t
