import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.data import HeteroData
from torch_geometric.data import Batch
import re

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
    node_attrs=("offset", "range"),      # which attrs to keep
    edge_attrs=("distance",),
    include_type_onehot=True,
    pos_scale=300.0,                     # normalisation constants
    dist_scale=300.0,
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
            if a == "offset":                     # (dx, dy) tuple
                feat.extend([v[0] / pos_scale, v[1] / pos_scale])
            elif a == "range":
                feat.append(v / pos_scale)
            elif isinstance(v, (tuple, list, np.ndarray)):
                feat.extend(np.asarray(v, dtype=float).ravel())
            else:
                feat.append(float(v))
        if include_type_onehot:
            oh = [0.0] * len(NODE_TYPES)
            oh[_TYPE_IDX[d["node_type"]]] = 1.0
            feat.extend(oh)
        rows.append(feat)
    x = torch.tensor(rows, dtype=torch.float32)

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
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_attrs)), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_names = nodes          # keep the mapping for later lookup
    data.name_to_idx = idx
    return data

def overlay_dist(data, overlay):
    dist = 0.0
    for node_src, node_dst in zip(overlay[:-1], overlay[1:]):
        src_pos = data.x[data.name_to_idx[node_src]][0:2]
        dst_pos = data.x[data.name_to_idx[node_dst]][0:2]
        dist = dist + torch.norm(src_pos - dst_pos)
    return dist

def _add_overlay_nodes_pyg(data, overlays):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    name_to_idx = dict(data.name_to_idx)
    node_names = list(data.node_names)

    #ovl_feat = [0] #[-1.0, -1.0, -1.0] + [1.0 if t == virtual_type else 0.0 for t in NODE_TYPES]

    virtual, src_virtual, dst_node, src_node, dst_virtual = [], [], [], [], []
    for i, overlay in enumerate(overlays):
        ovl_idx = len(virtual) #x.size(0) + len(virtual)
        name_to_idx[f"OVL-{i}"] = ovl_idx
        node_names.append(f"OVL-{i}")
        virtual.append([overlay_dist(data, overlay['overlay_path'])])
        for node in overlay["overlay_path"]: #REMEMBER, WAS A MISSTAKE EARLIER
            tgt = name_to_idx[node]
            src_virtual += [ovl_idx]
            src_node += [tgt]
            dst_virtual += [ovl_idx]
            dst_node += [tgt]

    hetero = HeteroData()
    hetero['node'].x = x
    hetero['virtual'].x = torch.tensor(virtual, dtype=torch.float32)

    hetero['node', 'n-v', 'virtual'].edge_index = torch.tensor((src_node, dst_virtual), dtype=torch.long)

    hetero['virtual', 'v-n', 'node'].edge_index = torch.tensor((src_virtual, dst_node), dtype=torch.long)

    #hetero['node', 'n-v-rev', 'virtual'].edge_index = (dst_virtual, src_node)
    #hetero['node', 'n-v-rev', 'virtual'].edge_attr = [-1]*len(dst_virtual)

    #hetero['virtual', 'v-n-rev', 'node'].edge_index = (dst_node, src_virtual)
    #hetero['virtual', 'v-n-rev', 'node'].edge_attr = [-1]*len(dst_node)

    hetero['node', 'n-n', 'node'].edge_index = edge_index
    hetero['node', 'n-n', 'node'].edge_attr = edge_attr
    
    
    #out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    hetero.node_names = node_names
    hetero.name_to_idx = name_to_idx
    return hetero

def _add_underlay_path_feature(graph, overlays):
    """
    Append one column to node features encoding each node's normalized
    position along the (single) overlay path. Non-path nodes get -1.
    Returns the graph (features replaced, not mutated in place).
    """
    assert len(overlays) == 1, "single-overlay only"
    path = overlays[0]["overlay_path"]
    n_nodes = graph['node'].x.size(0)
    extra_feature = torch.full((n_nodes, 2), -1.0,
                    dtype=graph['node'].x.dtype,
                    device=graph['node'].x.device)
    map_dict = {'AC': (0,1), 'SA': (1,2), 'GW': (2,3), 'TG': (3,3)}

    for node in graph.name_to_idx:
        idx = graph.name_to_idx[node]
        node_type = node[0:2]
        if node_type == "OV":
            continue
        (ovl_src, ovl_dest) = map_dict[node_type]

        pos = graph['node'].x[idx][0:2]
        ovl_src_pos = graph['node'].x[graph.name_to_idx[path[ovl_src]]][0:2]
        ovl_dest_pos = graph['node'].x[graph.name_to_idx[path[ovl_dest]]][0:2]
   
        dist_to_src = torch.norm(ovl_src_pos - pos)
        dist_to_dest = torch.norm(ovl_dest_pos - pos)
        extra_feature[idx, 0] = dist_to_src
        extra_feature[idx, 1] = dist_to_dest

    graph['node'].x = torch.cat([graph['node'].x, extra_feature], dim=1)

    return graph
    """
    assert len(overlays) == 1, "single-overlay only"
    path = overlays[0]["overlay_path"]
    n_nodes = graph['node'].x.size(0)

    # position feature, -1 sentinel for off-path nodes
    pos = torch.full((n_nodes, 1), -1.0,
                     dtype=graph['node'].x.dtype,
                     device=graph['node'].x.device)
    L = max(len(path) - 1, 1)          # avoid div-by-zero on length-1 paths
    for rank, name in enumerate(path):
        node_i = graph.name_to_idx[name]
        pos[node_i, 0] = rank / L      # normalized position in [0, 1]

    # build a NEW tensor rather than editing the shared/cached one
    graph['node'].x = torch.cat([graph['node'].x, pos], dim=1)
    return graph
    """

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
        graph = snapshot_to_pyg(H)
        encoding = self.encoder.forward(
            graph.x, graph.edge_index, graph.edge_attr, return_attention=False
        )

        ovl_encodings = {}
        for idx, overlay in enumerate(overlays):
            node_idx = graph.name_to_idx[f"OVL-{idx}"]
            ovl_encodings[overlay["id"]] = encoding[node_idx]

        return ovl_encodings

    def encode_overlays_pyg(self, data, overlays, return_attention=False, ep=None):
        graph = _add_overlay_nodes_pyg(data, overlays)
        graph = _add_underlay_path_feature(graph, overlays) #This only works if overlay is a single path
        graph = graph.to(self.device)
        encoding = self.encoder.forward(
            graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict, return_attention, [data.name_to_idx[name] for name in overlays[0]['overlay_path']], ep=ep
        )
        ovl_encodings = {}
        for idx, overlay in enumerate(overlays):
            ovl_idx = graph.name_to_idx[f"OVL-{idx}"]
            ovl_id = overlay['id']
            ovl_encodings[ovl_id] = encoding['virtual'][ovl_idx]
        return ovl_encodings

    def encode_overlays_pyg_batched(self, data, overlays):
        """
        Encode several overlays that share the same base topology `data`.

        Each overlay is built into its OWN single-overlay HeteroData (identical
        base graph, its own virtual node + path column), so attention stays
        scoped to one overlay. The graphs are collated into one disjoint batch
        and run through the encoder in a single forward pass.

        Returns {overlay_id: virtual_embedding}, same shape as encode_overlays_pyg.
        """
        graphs = []
        for overlay in overlays:
            g = _add_overlay_nodes_pyg(data, [overlay])  # still single-overlay
            g = _add_underlay_path_feature(g, [overlay]) # assert len==1 still holds
            graphs.append(g.to(self.device))

        # Disjoint batch: no edges between sub-graphs, so overlay i's virtual
        # node only attends within overlay i's own topology copy.
        batch = Batch.from_data_list(graphs)

        encoding = self.encoder.forward(
            batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict,
            return_attention=False,
        )

        # Every sub-graph had exactly one virtual node, so row i == overlay i.
        return {overlay['id']: encoding['virtual'][i]
                for i, overlay in enumerate(overlays)}