import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

# All nodes share features [x, y, vx, vy]
NODE_IN_DIM = 4


# ---------------------------------------------------------------------------
# Encoder: plain edge-enhanced GATv2 stack (single node type, one relation)
# ---------------------------------------------------------------------------
class GATv2(nn.Module):
    """
    Edge-enhanced GATv2 over the underlay graph (homogeneous, self-loops).

    Name kept as HeteroGATv2 for drop-in compatibility with the training code,
    but this is now a plain (non-hetero) GATv2 stack over a single node type.

    forward(x, edge_index, edge_attr) -> node embeddings [num_nodes, out_dim].
    """

    def __init__(self, hidden_dim=64, out_dim=64, heads=4, n_layers=3,
                 edge_dim=1, dropout=0.1):
        super().__init__()
        assert hidden_dim % heads == 0 and out_dim % heads == 0, \
            "hidden_dim and out_dim must be divisible by heads (concat=True)."
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_dim = out_dim

        self.input_proj = nn.Linear(NODE_IN_DIM, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(n_layers):
            last = layer == n_layers - 1
            width = out_dim if last else hidden_dim
            per_head = width // heads          # concat=True -> heads*per_head = width
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    per_head,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(width))

    def forward(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        for layer in range(self.n_layers):
            h = self.convs[layer](h, edge_index, edge_attr)
            # h = self.norms[layer](h)
            if layer < self.n_layers - 1:
                h = F.dropout(F.elu(h), p=self.dropout, training=self.training)
        return h


# ---------------------------------------------------------------------------
# Snapshot -> PyG Data (single node type, homogeneous)
# ---------------------------------------------------------------------------
def snapshot_to_pyg(H, pos_scale=300.0, dist_scale=300.0, vel_scale=1.0):
    """
    Convert one networkx snapshot to a homogeneous PyG Data.

    Node features (all nodes): [x, y, vx, vy]
        position from d["offset"], velocity from d["velocity"] (defaults 0).
    Edge features: [distance / dist_scale], edges built bidirectionally.

    Records name_to_local: node name -> ("node", global index), so the
    (type, local_idx) unpacking in the training code keeps working unchanged.
    """
    nodes = sorted(H.nodes())
    name_to_local = {n: ("node", i) for i, n in enumerate(nodes)}

    rows = []
    for n in nodes:
        d = H.nodes[n]
        x0, y0 = d["offset"][0] / pos_scale, d["offset"][1] / pos_scale
        vel = d.get("velocity", (0.0, 0.0))
        vx, vy = vel[0] / vel_scale, vel[1] / vel_scale
        rows.append([x0, y0, vx, vy])

    x = torch.tensor(rows, dtype=torch.float32) if rows \
        else torch.empty((0, NODE_IN_DIM), dtype=torch.float32)

    src, dst, attr = [], [], []
    for u, v, ed in H.edges(data=True):
        ef = [ed["distance"] / dist_scale]
        (_, ui), (_, vi) = name_to_local[u], name_to_local[v]
        for a_i, b_i in ((ui, vi), (vi, ui)):   # bidirectional
            src.append(a_i)
            dst.append(b_i)
            attr.append(ef)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.name_to_local = name_to_local
    return data


# ---------------------------------------------------------------------------
# Graph encoder wrapper
# ---------------------------------------------------------------------------
class GraphEncoder:
    def __init__(self, encoder, device="cpu"):
        self.encoder = encoder
        self.device = device

    def encode_overlays_pyg_batched(self, data, overlays):
        """
        Encode the whole underlay graph ONCE, then for each overlay concatenate
        the embeddings of its path nodes in the path's order [AC, SA, GW, TG].

        `data` is a Data from snapshot_to_pyg (carrying name_to_local).
        Each overlay has an 'overlay_path' list of node names ordered AC->SA->GW->TG.

        Returns {overlay_id: concat_embedding} with concat_embedding of shape
        [4 * out_dim].
        """
        data = data.to(self.device)
        h = self.encoder(data.x, data.edge_index, data.edge_attr)   # [num_nodes, out_dim]

        name_to_local = data.name_to_local
        out = {}
        for overlay in overlays:
            parts = []
            for name in overlay["overlay_path"]:      # AC, SA, GW, TG in order
                _, local_idx = name_to_local[name]    # ("node", idx)
                parts.append(h[local_idx])
            out[overlay["id"]] = torch.cat(parts, dim=0)   # [4 * out_dim]
        return out