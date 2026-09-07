"""
Heterogeneous GATv2 encoder + prediction heads + absorbing-chain read-out.

The encoder gives each of the four relations its own GATv2Conv, so
node->edge and edge->node never share weights. The heads predict per-node
(proc_time, drop) and per-edge (prop_delay, drop, routing_prob). The
absorbing chain turns those into an end-to-end (delay, loss) per overlay.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.utils import softmax
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

RELATIONS = [
    ("node", "n-v", "virtual"),
    ("virtual", "v-n", "node"),
    ("node", "n-n", "node"),
    #("virtual", "rev_tail", "node"),
    #("node", "rev_head", "virtual"),
]

def plot_attention(node_x, edge_index, alpha, relation_name,
                   node_names=None, ovl_path=None,
                   drop_self_loops=True, max_edges=None, ep=None):
    ei = edge_index.numpy()
    a = alpha.numpy()

    if drop_self_loops:
        keep = ei[0] != ei[1]
        ei, a = ei[:, keep], a[keep]

    coords = node_x.detach().cpu().numpy()[:, :2]
    pos = {i: (float(coords[i, 0]), float(coords[i, 1]))
           for i in range(coords.shape[0])}

    overlay_nodes = set(ovl_path or [])
    H = a.shape[1]
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, H, figsize=(6 * H, 5))
    if H == 1:
        axes = [axes]

    for h in range(H):
        ax = axes[h]
        w = a[:, h]
        order = w.argsort()
        if max_edges:
            order = order[-max_edges:]

        G = nx.DiGraph()
        G.add_nodes_from(range(coords.shape[0]))
        for e in order:
            G.add_edge(int(ei[0, e]), int(ei[1, e]), weight=float(w[e]))

        node_colors = ["tomato" if n in overlay_nodes else "lightsteelblue"
                       for n in G.nodes()]
        weights = [G[u][v]["weight"] for u, v in G.edges()]

        vmax = max(weights) if weights else 1.0
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200,
                               node_color=node_colors)
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=weights, edge_cmap=cmap,
            edge_vmin=0.0, edge_vmax=vmax,
            width=[2 + 4 * x for x in weights], arrows=True)
        lbl = {i: (node_names[i] if node_names else i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=lbl, font_size=7)
        ax.set_title(f"{relation_name} — head {h}")
        ax.set_aspect("equal")
        ax.axis("on")

        # colorbar: maps edge color -> attention weight
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("attention weight")

    plt.tight_layout()
    plt.savefig(f"//wsl.localhost/Ubuntu-22.04/home/chris/algorithm_test_v2/outputs/attention_ep{ep}.png", dpi=300, bbox_inches="tight")
    return fig

class HeteroGATv2Encoder(nn.Module):
    def __init__(self, node_in, virtual_in, hidden_dim=128, out_dim=64,
                 heads=2, n_layers=3, dropout=0.1):
        assert(hidden_dim % heads == 0)
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # project both node types to a common width first
        self.lin_node = nn.Linear(node_in, hidden_dim)
        self.lin_virtual = nn.Linear(virtual_in, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [hidden_dim] * n_layers + [out_dim]
        for l in range(n_layers):
            d_in, d_out = dims[l], dims[l + 1]
            last = l == n_layers - 1
            per_head = d_out if last else d_out // heads
            conv = HeteroConv(
                {
                    ("node", "n-v", "virtual"): GATv2Conv(d_in, per_head, heads=heads, edge_dim=None,
                                concat=not last, add_self_loops=False,
                                dropout=dropout),
                    ("virtual", "v-n", "node"): GATv2Conv(d_in, per_head, heads=heads, edge_dim=None,
                                concat=not last, add_self_loops=False,
                                dropout=dropout),
                    ("node", "n-n", "node"): GATv2Conv(d_in, per_head, heads=heads, edge_dim=1,
                                                    concat=not last, add_self_loops=True,
                                                    dropout=dropout)
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                "node": nn.LayerNorm(d_out),
                "virtual": nn.LayerNorm(d_out),
            }))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, return_attention=False, ovl_path=None, ep=None):
        x = {"node": self.lin_node(x_dict["node"]),
             "virtual": self.lin_virtual(x_dict["virtual"])} #torch.zeros(x_dict["virtual"].shape[0], 128, dtype=x_dict["virtual"].dtype, device=x_dict["virtual"].device)}
        attn_per_layer = []
        for l in range(self.n_layers):
            last = l == self.n_layers - 1
            if not return_attention:
                x = self.convs[l](x, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                conv_dict = self.convs[l].convs            # sub-convs, keyed by relation
                out, layer_attn = {}, {}
                for rel in RELATIONS:                       # (src_type, rel_name, dst_type)
                    src, _, dst = rel
                    subconv = conv_dict[rel]
                    ei = edge_index_dict[rel]
                    ea = edge_attr_dict.get(rel) if edge_attr_dict else None

                    res, (ei_out, alpha) = subconv(
                        (x[src], x[dst]), ei, edge_attr=ea,
                        return_attention_weights=True,
                    )
                    out.setdefault(dst, []).append(res)     # collect messages by destination
                    layer_attn[rel] = (ei_out.detach().cpu(), alpha.detach().cpu())

                #if l == 0:
                #    plot_attention(x_dict['node'], layer_attn[("node", "n-n", "node")][0], layer_attn[("node", "n-n", "node")][1], ("node", "n-n", "node"), ovl_path=ovl_path, ep=ep)
                x = {k: torch.stack(v).sum(0) for k, v in out.items()}
                attn_per_layer.append(layer_attn)
            
            new_x = {}
            for k, v in x.items():
                v = self.norms[l][k](v)
                if not last:
                    v = F.dropout(F.elu(v), p=self.dropout, training=self.training)
                new_x[k] = v
            x = new_x
        return x

'''
import torch
from torch_geometric.data import HeteroData


def hetero_from_record(d, device="cpu"):
    x = torch.as_tensor(d["x"], dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(d["edge_index"], dtype=torch.long, device=device)
    edge_attr = torch.as_tensor(d["edge_attr"], dtype=torch.float32, device=device)
    name_to_idx = dict(d["name_to_idx"])

    # Each COLUMN of edge_index is one directed edge-node. The record is
    # already bidirectional (u->v and v->u are separate columns), so we do
    # not add reverse links ourselves here.
    src = edge_index[0]          # tail node index of each edge-node
    dst = edge_index[1]          # head node index of each edge-node
    E = edge_index.size(1)
    ei = torch.arange(E, device=device)

    data = HeteroData()
    data["node"].x = x
    data["edge"].x = edge_attr   # edge-node features = the link's own attributes

    # forward flow: tail -> edge, edge -> head
    data["node", "tail", "edge"].edge_index = torch.stack([src, ei])
    data["edge", "head", "node"].edge_index = torch.stack([ei, dst])
    # backward flow: same links, swapped, so messages travel both ways
    data["edge", "rev_tail", "node"].edge_index = torch.stack([ei, src])
    data["node", "rev_head", "edge"].edge_index = torch.stack([dst, ei])

    # bookkeeping
    data.node_name_to_idx = name_to_idx
    data.edge_src = src          # tail node index per edge-node
    data.edge_dst = dst          # head node index per edge-node
    data.tail_of_edge = src      # grouping key for the routing softmax

    return data
'''