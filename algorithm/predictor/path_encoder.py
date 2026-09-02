import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

NODE_TYPES = ["Aircraft", "Satellite", "Gateway", "Target", "Virtual"]
_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

class GATv2Encoder(nn.Module):
    """
    Stack of edge-conditioned GATv2 layers producing node embeddings.

    GATv2 (Brody et al., 2022) computes
        e_ij = a^T LeakyReLU( W [h_i || h_j || W_e e_ij] )
    i.e. the linear layer is applied *before* the nonlinearity, unlike GAT.
    That ordering is what makes the attention dynamic rather than static.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        out_dim=64,
        edge_dim=1,
        heads=2,
        n_layers=3,
        dropout=0.1,
        residual=True,
        share_weights=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.out_dim = out_dim

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for l in range(n_layers):
            d_in, d_out = dims[l], dims[l + 1]
            last = l == n_layers - 1
            self.convs.append(
                GATv2Conv(
                    d_in,
                    d_out if last else d_out // heads,
                    heads=heads,
                    concat=not last,          # average heads on the last layer
                    edge_dim=edge_dim,
                    add_self_loops=True,
                    dropout=dropout,
                    share_weights=share_weights,
                    residual=residual
                )
            )
            self.norms.append(nn.LayerNorm(d_out))

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        attns = []
        for l, (conv, norm) in enumerate(
            zip(self.convs, self.norms)
        ):

            if return_attention:
                x, att = conv(x, edge_index, edge_attr=edge_attr,
                              return_attention_weights=True)
                attns.append(att)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)

            x = norm(x)
            if l < self.n_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return (x, attns) if return_attention else x


def snapshot_to_pyg(
    H,
    node_attrs=("pos_to_ac", "in_overlay"),      # which attrs to keep
    edge_attrs=("distance",),
    include_type_onehot=True,
    pos_scale=300.0,                     # normalisation constants
    dist_scale=300.0,
    device="cpu",
):
    """
    Convert one snapshot to a PyG Data object, selecting only the listed
    attributes. Node ordering is fixed by sorted node name so that embeddings
    are comparable across timesteps.
    """
    nodes = sorted(H.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    # ---- node features ---------------------------------------------------
    rows = []
    for n in nodes:
        d = H.nodes[n]
        feat = []
        for a in node_attrs:
            v = d[a]
            if a == "pos_to_ac":                     # (dx, dy) tuple
                feat.extend([v[0] / pos_scale, v[1] / pos_scale])
            elif a == "in_overlay":
                feat.append(0)
            elif isinstance(v, (tuple, list, np.ndarray)):
                feat.extend(np.asarray(v, dtype=float).ravel())
            else:
                feat.append(float(v))
        if include_type_onehot:
            oh = [0.0] * len(NODE_TYPES)
            oh[_TYPE_IDX[d["node_type"]]] = 1.0
            feat.extend(oh)
        rows.append(feat)
    x = torch.tensor(rows, dtype=torch.float32, device=device)

    # ---- edges: undirected -> both directions ----------------------------
    src, dst, eattr = [], [], []
    for u, v, ed in H.edges(data=True):
        ef = []
        for a in edge_attrs:
            val = ed[a]
            ef.append(val / dist_scale if a == "distance" else float(val))
        for a_, b_ in ((u, v), (v, u)):
            src.append(idx[a_])
            dst.append(idx[b_])
            eattr.append(ef)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_attr = torch.tensor(eattr, dtype=torch.float32, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, len(edge_attrs)), dtype=torch.float32,
                                device=device)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_names = nodes          # keep the mapping for later lookup
    data.name_to_idx = idx
    return data

def _add_overlay_nodes_pyg(data, overlays, virtual_type="Virtual", device="cpu"):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    name_to_idx = dict(data.name_to_idx)
    node_names = list(data.node_names)

    ovl_feat = [-1.0, -1.0, -1.0] + [1.0 if t == virtual_type else 0.0 for t in NODE_TYPES]

    new_rows, src, dst, eattr = [], [], [], []
    for i, overlay in enumerate(overlays):
        ovl_idx = x.size(0) + len(new_rows)
        name_to_idx[f"OVL-{i}"] = ovl_idx
        node_names.append(f"OVL-{i}")
        new_rows.append(ovl_feat)
        for node in overlay["underlay_path"]:
            tgt = name_to_idx[node]
            src += [ovl_idx, tgt]
            dst += [tgt, ovl_idx]
            eattr += [[-1.0] * edge_attr.size(1)] * 2

    x = torch.cat([x, torch.tensor(new_rows, dtype=x.dtype, device=device)], dim=0)
    if src:
        edge_index = torch.cat([edge_index, torch.tensor([src, dst], dtype=torch.long, device=device)], dim=1)
        edge_attr = torch.cat([edge_attr, torch.tensor(eattr, dtype=edge_attr.dtype, device=device)], dim=0)

    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    out.node_names = node_names
    out.name_to_idx = name_to_idx
    return out

class GraphEncoder:
    def __init__(self, encoder, device="cpu"):
        self.encoder = encoder
        self.device = device

    def _add_path_nodes(self, H, overlays):
        H = H.copy()
        for idx, overlay in enumerate(overlays):
            H.add_node(
                f"OVL-{idx}",
                node_type="Virtual",
                layer=-1,
                offset=[-1, -1],
                range=-1,
            )
            for node in overlay["overlay_path"]:
                H.add_edge(f"OVL-{idx}", node, distance=-1)
        return H

    def encode_overlays(self, H, overlays):
        H = self._add_path_nodes(H, overlays)
        graph = snapshot_to_pyg(H, device=self.device)
        encoding = self.encoder.forward(
            graph.x, graph.edge_index, graph.edge_attr, return_attention=False
        )

        ovl_encodings = {}
        for idx, overlay in enumerate(overlays):
            node_idx = graph.name_to_idx[f"OVL-{idx}"]
            ovl_encodings[overlay["id"]] = encoding[node_idx]

        return ovl_encodings
    
    def encode_overlays_pyg(self, data, overlays):
        graph = _add_overlay_nodes_pyg(data, overlays, device=self.device)
        encoding = self.encoder.forward(
            graph.x, graph.edge_index, graph.edge_attr, return_attention=False
        )
        ovl_encodings = {}
        for idx, overlay in enumerate(overlays):
            node_idx = graph.name_to_idx[f"OVL-{idx}"]
            ovl_id = overlay['id']
            ovl_encodings[ovl_id] = encoding[node_idx]
        return ovl_encodings
    