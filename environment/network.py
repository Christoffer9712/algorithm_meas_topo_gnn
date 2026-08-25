import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

class LayeredOrbitNetwork:
    """
    2D testbed network.

      - Satellites orbit a fixed centre (the "Earth"), in concentric rings,
        each ring rotating at its own angular velocity.
      - Aircraft, gateways, targets are stationary at fixed absolute positions.
      - Node *features* are expressed relative to the aircraft: (dx, dy, range).
        Geometry is absolute; only the representation is aircraft-centric.
      - Edges are recomputed at each time t from Euclidean distance, so the
        topology is genuinely dynamic.

    Layer indices:  0 = aircraft, 1..n_rings = satellite rings,
                    n_rings+1 = gateways, n_rings+2 = targets.
    """

    def __init__(
        self,
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
        seed=42,
    ):
        self.rng = random.Random(seed)
        self.sats_per_ring = sats_per_ring
        self.n_rings = len(sats_per_ring)
        self.n_gateways = n_gateways
        self.n_targets = n_targets
        self.ring_radii = ring_radii
        self.ring_speeds = ring_speeds
        self.orbit_center = tuple(float(c) for c in orbit_center)
        self.aircraft_pos = tuple(float(c) for c in aircraft_pos)
        self.range_limit = range_limit

        assert len(ring_radii) == self.n_rings
        assert len(ring_speeds) == self.n_rings

        cx, cy = self.orbit_center
        if gateway_positions is None:
            span = 260.0
            gateway_positions = []
            for i in range(n_gateways):
                frac = 0.5 if n_gateways == 1 else i / (n_gateways - 1)
                gateway_positions.append((cx - span / 2 + frac * span, cy + 110.0))
        if target_positions is None:
            target_positions = [(cx + 40.0 * (i - (n_targets - 1) / 2), cy + 190.0)
                                for i in range(n_targets)]

        self.G = nx.Graph()
        
        self._build_nodes(gateway_positions, target_positions)

    # ------------------------------------------------------------------ nodes

    def _build_nodes(self, gateway_positions, target_positions):
        """Create all nodes with static attributes. Positions come later."""
        self.G.add_node("AC-0", node_type="Aircraft", layer=0,
                        static_pos=self.aircraft_pos)

        for r, n in enumerate(self.sats_per_ring):
            radius = self.ring_radii[r]
            speed = self.ring_speeds[r]
            jitter = self.rng.random() * 2 * math.pi
            for i in range(n):
                phase = jitter + 2 * math.pi * i / n
                self.G.add_node(
                    f"SAT{r}-{i}", node_type="Satellite", layer=1 + r,
                    ring=r, radius=radius, phase=phase, speed=speed,
                )

        gw_layer = 1 + self.n_rings
        for i, p in enumerate(gateway_positions):
            self.G.add_node(f"GW-{i}", node_type="Gateway",
                            layer=gw_layer, static_pos=tuple(p))

        tgt_layer = gw_layer + 1
        for i, p in enumerate(target_positions):
            self.G.add_node(f"TGT-{i}", node_type="Target",
                            layer=tgt_layer, static_pos=tuple(p))

    # -------------------------------------------------------------- positions

    def positions_at(self, t):
        """Absolute positions. Satellites orbit `orbit_center`."""
        cx, cy = self.orbit_center
        pos = {}
        for n, d in self.G.nodes(data=True):
            if d["node_type"] == "Satellite":
                ang = d["phase"] + d["speed"] * t
                pos[n] = (cx + d["radius"] * math.cos(ang),
                          cy + d["radius"] * math.sin(ang))
            else:
                pos[n] = d["static_pos"]
        return pos

    def offsets_at(self, t):
        """{node: (dx, dy)} relative to the aircraft — the GAT node features."""
        pos = self.positions_at(t)
        ax, ay = pos["AC-0"]
        return {n: (x - ax, y - ay) for n, (x, y) in pos.items()}

    def ranges_at(self, t):
        """Slant range from the aircraft to every node."""
        return {n: math.hypot(dx, dy) for n, (dx, dy) in self.offsets_at(t).items()}

    # ------------------------------------------------------------------ edges

    def _layer_range(self, la, lb):
        gw_layer = 1 + self.n_rings
        if {la, lb} == {gw_layer, gw_layer + 1}:
            return math.inf          # gateway<->target is wired
        return self.range_limit

    def snapshot(self, t):
        """
        Graph at time t: same nodes, edges recomputed by proximity.
        Node attrs carry `offset` (dx, dy from aircraft) and `range` as the GAT
        features; `pos` is absolute, for drawing.
        """
        pos = self.positions_at(t)
        off = self.offsets_at(t)
        rng_ = self.ranges_at(t)

        H = nx.Graph()
        for n, d in self.G.nodes(data=True):
            attrs = {k: v for k, v in d.items() if k != "static_pos"}
            H.add_node(n, **attrs, offset=off[n], range=rng_[n], pos=pos[n], t=t)

        nodes = list(self.G.nodes(data=True))
        for i, (u, du) in enumerate(nodes):
            for v, dv in nodes[i + 1:]:
                if abs(du["layer"] - dv["layer"]) != 1:
                    continue
                dist = math.dist(pos[u], pos[v])
                if dist <= self._layer_range(du["layer"], dv["layer"]):
                    H.add_edge(u, v, distance=dist)
        return H

    # --------------------------------------------------------- feature export

    def node_features(self, t):
        """{node: (dx, dy, range, type_index)} in the aircraft frame."""
        types = ["Aircraft", "Satellite", "Gateway", "Target"]
        off, rng_ = self.offsets_at(t), self.ranges_at(t)
        return {
            n: (off[n][0], off[n][1], rng_[n], types.index(d["node_type"]))
            for n, d in self.G.nodes(data=True)
        }

    # ------------------------------------------------------- path enumeration

    def underlay_paths(self, t, max_paths=None):
        """All AC -> ... -> TGT paths taking one node per layer."""
        H = self.snapshot(t)
        paths = []
        for tgt in [n for n, d in H.nodes(data=True)
                    if d["node_type"] == "Target"]:
            for p in nx.all_simple_paths(H, "AC-0", tgt,
                                         cutoff=2 + self.n_rings):
                if len(p) == 3 + self.n_rings:
                    paths.append(tuple(p))
                    if max_paths and len(paths) >= max_paths:
                        return paths
        return paths

    def distance(self, path, t):
        pos = []
        cx, cy = self.orbit_center
        for d in path:
            if self.G.nodes[d]["node_type"] == "Satellite":
                ang = self.G.nodes[d]["phase"] + self.G.nodes[d]["speed"] * t
                pos.append((cx + self.G.nodes[d]["radius"] * math.cos(ang),
                           cy + self.G.nodes[d]["radius"] * math.sin(ang)))
            else:
                pos.append(self.G.nodes[d]["static_pos"])

        dist = 0
        for p in range(len(pos) - 1):
            dist = dist + math.dist(pos[p+1], pos[p]) 

        return dist

    def overlay_to_underlay(self, t, overlay):
        underlays = self.underlay_paths(t) #All paths from aircraft to target
        possible_paths = []
        for underlay in underlays:
             if set(overlay).issubset(underlay):
                possible_paths.append((underlay, self.distance(underlay, t)))
                
        possible_paths.sort(key=lambda item: item[1])
        return(possible_paths[0][0])

    def get_all_overlays(self, H):
        overlays = []
        ac_sat_connections = H.neighbors('AC-0')
   
        for con in ac_sat_connections:
            for i in range(self.n_gateways):
                overlays.append({'overlay_path': ['AC-0', con, f'GW-{i}', 'TGT-0'], 'id': f'0-{con[-1]}-{i}-0'})

        return overlays
# ----------------------------------------------------------------- plotting

_COLORS = {
    "Aircraft": "#d62728",
    "Satellite": "#1f77b4",
    "Gateway": "#2ca02c",
    "Target": "#9467bd",
}


def _draw(ax, net, t, x_lim, y_lim, highlight_path=None, show_range=True):
    ax.clear()
    H = net.snapshot(t)
    pos = {n: d["pos"] for n, d in H.nodes(data=True)}
    acx, acy = pos["AC-0"]
    cx, cy = net.orbit_center

    # Orbit guides about the orbit centre
    for radius in net.ring_radii:
        ax.add_patch(plt.Circle((cx, cy), radius, fill=False,
                                color="0.88", lw=0.8, zorder=0))
    ax.plot([cx], [cy], marker="+", color="0.6", ms=9, mew=1.2, zorder=0)

    # Coverage circle about the aircraft
    if show_range:
        ax.add_patch(plt.Circle((acx, acy), net.range_limit, fill=False,
                                color="#d62728", lw=0.9, ls="--",
                                alpha=0.45, zorder=0))

    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="0.75",
                           width=0.7, alpha=0.7)

    if highlight_path:
        edges = [e for e in zip(highlight_path[:-1], highlight_path[1:])
                 if H.has_edge(*e)]
        nx.draw_networkx_edges(H, pos, edgelist=edges, ax=ax,
                               edge_color="#ff7f0e", width=2.5)

    for ntype, color in _COLORS.items():
        nodes = [n for n, d in H.nodes(data=True) if d["node_type"] == ntype]
        if not nodes:
            continue
        nx.draw_networkx_nodes(H, pos, nodelist=nodes, ax=ax,
                               node_color=color, node_size=90,
                               edgecolors="white", linewidths=0.8,
                               label=ntype)

    xs = [p[0] for p in pos.values()] + [acx - net.range_limit, acx + net.range_limit]
    ys = [p[1] for p in pos.values()] + [acy - net.range_limit, acy + net.range_limit]
    pad = 40.0
    ax.set_xlim(x_lim)#min(xs) - pad, max(xs) + pad)
    ax.set_ylim(y_lim)#min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_title(f"t = {t:.1f}   |E| = {H.number_of_edges()}   "
                 f"paths = {len(net.underlay_paths(t))}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    return H




def plot_grid(net, times, x_lim, y_lim,):
    n = len(times)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, t in zip(axes, times):
        _draw(ax, net, t, x_lim, y_lim)
        ax.get_legend().remove()
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()



_NODE_MARKERS = {
    "Aircraft": "^",
    "Satellite": "o",
    "Gateway": "s",
    "Target": "*",
}
_MARKER_SIZES = {
    "Aircraft": 190,
    "Satellite": 130,
    "Gateway": 150,
    "Target": 320,
}


def animate_queue_states(
    snapshots,
    times=None,
    attr="queue_delay",
    cmap="viridis",
    log_scale=True,
    percentile_clip=(2, 98),
    bounds=None,
    highlight_paths=None,
    interval=80,
    figsize=(8.5, 7.5),
    save_path=None,
    show=False,
):
    """
    Animate a sequence of network snapshots, colouring nodes by queueing delay.

    Parameters
    ----------
    snapshots : list[nx.Graph]
        One graph per timestep, e.g. [H.copy() for each t]. Each node must
        carry 'pos', 'node_type', and the attribute named by `attr`.
    times : array-like, optional
        Timestamp per snapshot. Falls back to node attr 't', then to index.
    attr : str
        Node attribute to colour by ('queue_delay', 'queue_len', 'loss', ...).
    log_scale : bool
        Log colour normalisation. Appropriate for queueing delay, which is
        heavy-tailed near rho -> 1.
    percentile_clip : (lo, hi)
        Percentiles of the pooled values defining the colour range. Clipping
        stops one congestion spike from flattening the rest of the scale.
    bounds : (x0, x1, y0, y1), optional
        Fixed viewport. Computed from all snapshots if omitted.
    highlight_paths : list[tuple] | None
        Optional path to trace per frame (same length as snapshots), or a
        single path applied to every frame.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Keep a reference to this, or it will be garbage-collected.
    """
    if not snapshots:
        raise ValueError("snapshots is empty")

    n_frames = len(snapshots)

    # ---- timestamps ------------------------------------------------------
    if times is None:
        times = []
        for i, H in enumerate(snapshots):
            t_vals = [d.get("t") for _, d in H.nodes(data=True) if "t" in d]
            times.append(t_vals[0] if t_vals else i)
    times = list(times)

    # ---- colour normalisation, pooled over ALL frames --------------------
    pooled = np.array([
        d[attr] for H in snapshots for _, d in H.nodes(data=True) if attr in d
    ], dtype=float)
    if pooled.size == 0:
        raise ValueError(f"no node carries attribute {attr!r}")

    lo, hi = np.percentile(pooled, percentile_clip)
    if log_scale:
        lo = max(lo, np.min(pooled[pooled > 0]) if np.any(pooled > 0) else 1e-6)
        hi = max(hi, lo * 1.01)
        norm = LogNorm(vmin=lo, vmax=hi)
    else:
        hi = max(hi, lo + 1e-9)
        norm = Normalize(vmin=lo, vmax=hi)

    # ---- fixed viewport --------------------------------------------------
    if bounds is None:
        xs, ys = [], []
        for H in snapshots:
            for _, d in H.nodes(data=True):
                x, y = d["pos"]
                xs.append(x)
                ys.append(y)
        pad = 0.06 * max(max(xs) - min(xs), max(ys) - min(ys))
        bounds = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    x0, x1, y0, y1 = bounds

    # ---- highlight paths -------------------------------------------------
    if highlight_paths is not None and highlight_paths:
        first = highlight_paths[0]
        if isinstance(first, str):                    # single path given
            highlight_paths = [tuple(highlight_paths)] * n_frames
        elif len(highlight_paths) != n_frames:
            highlight_paths = list(highlight_paths) + \
                [None] * (n_frames - len(highlight_paths))

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.93, bottom=0.02)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    label = {"queue_delay": "queueing + processing time",
             "queue_len": "queue occupancy (packets)",
             "loss": "packet-loss probability"}.get(attr, attr)
    cbar.set_label(label, rotation=270, labelpad=18)

    shape_handles = [
        Line2D([0], [0], marker=m, ls="", color="0.35",
               markersize=math.sqrt(_MARKER_SIZES[k]) * 0.9, label=k)
        for k, m in _NODE_MARKERS.items()
    ]

    def draw(idx):
        ax.clear()
        H = snapshots[idx]
        pos = {n: d["pos"] for n, d in H.nodes(data=True)}

        nx.draw_networkx_edges(H, pos, ax=ax, edge_color="0.78",
                               width=0.7, alpha=0.75)

        hp = highlight_paths[idx] if highlight_paths else None
        if hp:
            edges = [e for e in zip(hp[:-1], hp[1:]) if H.has_edge(*e)]
            nx.draw_networkx_edges(H, pos, edgelist=edges, ax=ax,
                                   edge_color="#ff7f0e", width=2.6, alpha=0.9)

        # Node type -> marker shape; queueing delay -> fill colour.
        for ntype, marker in _NODE_MARKERS.items():
            nodes = [n for n, d in H.nodes(data=True)
                     if d.get("node_type") == ntype]
            if not nodes:
                continue
            vals = [H.nodes[n].get(attr, np.nan) for n in nodes]
            ax.scatter(
                [pos[n][0] for n in nodes],
                [pos[n][1] for n in nodes],
                c=vals, cmap=cmap, norm=norm,
                marker=marker, s=_MARKER_SIZES[ntype],
                edgecolors="white", linewidths=0.9, zorder=3,
            )

        vals = np.array([d[attr] for _, d in H.nodes(data=True) if attr in d])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"t = {times[idx]:7.1f}    |E| = {H.number_of_edges():3d}    "
            f"{attr}: med {np.median(vals):6.2f}  max {vals.max():7.2f}",
            fontsize=10, family="monospace",
        )
        ax.legend(handles=shape_handles, loc="upper right",
                  fontsize=8, framealpha=0.9, labelspacing=0.8)
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames, interval=interval,
                         blit=False, repeat=True, cache_frame_data=False)

    if save_path:
        fps = max(1, int(1000 / interval))
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps, dpi=110)
    if show:
        plt.show()
    return anim