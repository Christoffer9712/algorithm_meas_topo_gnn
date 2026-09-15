import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# Node types and their raw input feature dimensions.
#   SA (satellite): [x, y, layer_norm, idx_norm]   -> 4
#   AC / GW / TG  : [x, y]                          -> 2
# ---------------------------------------------------------------------------
NODE_IN_DIM = {"AC": 2, "SA": 4, "GW": 2, "TG": 2}

# networkx snapshot node_type -> our short type string
_SNAPSHOT_TYPE = {
    "Aircraft":  "AC",
    "Satellite": "SA",
    "Gateway":   "GW",
    "Target":    "TG",
}

# Directed relations we build message passing over. Each is added in BOTH
# directions (see _all_relations) so information flows both ways.
_BASE_RELATIONS = [
    ("SA", "SA"),   # satellite mesh
    ("SA", "GW"),
    ("GW", "TG"),
    ("AC", "SA"),
]


def _all_relations():
    """Return every (src, 'link', dst) relation, both directions."""
    rels = []
    for a, b in _BASE_RELATIONS:
        rels.append((a, "link", b))
        if a != b:                      # SA-SA is its own reverse
            rels.append((b, "link", a))
    return rels


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class HeteroGATv2Encoder(nn.Module):
    """
    Heterogeneous, edge-enhanced GATv2 over the underlay graph.

    Four node types (AC, SA, GW, TG) with per-type input projections, and a
    stack of HeteroConv layers, each holding one GATv2Conv per relation. Every
    GATv2Conv uses the physical-distance edge feature (edge_dim=1).

    forward() returns a dict {node_type: [num_nodes_of_type, out_dim]}.
    """

    def __init__(self, hidden_dim=64, out_dim=64, heads=2, n_layers=3,
                 edge_dim=1, dropout=0.1):
        super().__init__()
        assert hidden_dim % heads == 0 and out_dim % heads == 0, \
            "hidden_dim and out_dim must be divisible by heads (concat=True)."
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_dim = out_dim

        node_types = list(NODE_IN_DIM.keys())
        relations = _all_relations()

        # Per-type input projection to a common hidden width, so each
        # per-relation GATv2Conv sees uniform in-channels on both endpoints.
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(NODE_IN_DIM[nt], hidden_dim) for nt in node_types
        })

        # Conv + norm per layer (built together so the per-layer width is
        # computed once).
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(n_layers):
            last = layer == n_layers - 1
            width = out_dim if last else hidden_dim
            per_head = width // heads          # concat=True -> heads*per_head = width
            conv = HeteroConv(
                {
                    rel: GATv2Conv(
                        hidden_dim,            # uniform in-channels after input_proj
                        per_head,
                        heads=heads,
                        concat=True,
                        edge_dim=edge_dim,
                        add_self_loops=False,  # self-loops don't apply across types
                        dropout=dropout,
                    )
                    for rel in relations
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(width) for nt in node_types
            }))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # project each type to the common hidden width
        h = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

        for layer in range(self.n_layers):
            h = self.convs[layer](h, edge_index_dict, edge_attr_dict)
            h = {nt: self.norms[layer][nt](v) for nt, v in h.items()}
            if layer < self.n_layers - 1:
                h = {nt: F.dropout(F.elu(v), p=self.dropout, training=self.training)
                     for nt, v in h.items()}
        return h


# ---------------------------------------------------------------------------
# Snapshot -> HeteroData
# ---------------------------------------------------------------------------
def snapshot_to_pyg(H, pos_scale=300.0, dist_scale=300.0,
                    n_layers_sat=1.0, n_sats_per_ring=1.0):
    """
    Convert one networkx snapshot to a HeteroData with four node types.

    Node features:
        SA: [x, y, layer / n_layers_sat, idx / n_sats_per_ring]
        AC/GW/TG: [x, y]
    Edge features: [distance / dist_scale], relations built bidirectionally.

    `n_layers_sat` and `n_sats_per_ring` normalise the satellite layer and idx
    so they sit on a comparable scale to the (already position-normalised)
    coordinates. Pass your constellation constants; defaults of 1.0 leave them
    unnormalised.

    Records name_to_local: node name -> (type, local index within that type).
    """
    # group node names by type, sorted for stable ordering across timesteps
    names_by_type = {t: [] for t in NODE_IN_DIM}
    for n in sorted(H.nodes()):
        nt = _SNAPSHOT_TYPE[H.nodes[n]["node_type"]]
        names_by_type[nt].append(n)

    name_to_local = {}
    for nt, names in names_by_type.items():
        for i, name in enumerate(names):
            name_to_local[name] = (nt, i)

    data = HeteroData()

    # node features per type
    for nt, names in names_by_type.items():
        rows = []
        for name in names:
            d = H.nodes[name]
            x0, y0 = d["offset"][0] / pos_scale, d["offset"][1] / pos_scale
            if nt == "SA":
                rows.append([
                    x0, y0,
                    float(d["layer"]) / n_layers_sat,
                    float(d["idx"]) / n_sats_per_ring,
                ])
            else:
                rows.append([x0, y0])
        data[nt].x = torch.tensor(rows, dtype=torch.float32) if rows \
            else torch.empty((0, NODE_IN_DIM[nt]), dtype=torch.float32)

    # which unordered type pairs carry links (both directions built)
    linked_pairs = {frozenset(p) for p in _BASE_RELATIONS}

    src = {rel: [] for rel in _all_relations()}
    dst = {rel: [] for rel in _all_relations()}
    attr = {rel: [] for rel in _all_relations()}

    for u, v, ed in H.edges(data=True):
        (ut, ui), (vt, vi) = name_to_local[u], name_to_local[v]
        if frozenset((ut, vt)) not in linked_pairs:
            continue  # ignore edges between type pairs we don't model
        ef = [ed["distance"] / dist_scale]
        for (a_t, a_i, b_t, b_i) in ((ut, ui, vt, vi), (vt, vi, ut, ui)):
            rel = (a_t, "link", b_t)
            if rel in src:
                src[rel].append(a_i)
                dst[rel].append(b_i)
                attr[rel].append(ef)

    for rel in _all_relations():
        if src[rel]:
            data[rel].edge_index = torch.tensor([src[rel], dst[rel]], dtype=torch.long)
            data[rel].edge_attr = torch.tensor(attr[rel], dtype=torch.float32)
        else:
            data[rel].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[rel].edge_attr = torch.empty((0, 1), dtype=torch.float32)

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

        `data` is a HeteroData from snapshot_to_pyg (carrying name_to_local).
        Each overlay has an 'overlay_path' list of node names ordered AC->SA->GW->TG.

        Returns {overlay_id: concat_embedding} with concat_embedding of shape
        [4 * out_dim].
        """
        data = data.to(self.device)
        h_dict = self.encoder(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

        name_to_local = data.name_to_local
        out = {}
        for overlay in overlays:
            parts = []
            for name in overlay["overlay_path"]:      # AC, SA, GW, TG in order
                nt, local_idx = name_to_local[name]
                parts.append(h_dict[nt][local_idx])
            out[overlay["id"]] = torch.cat(parts, dim=0)   # [4 * out_dim]
        return out