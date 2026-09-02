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


RELATIONS = [
    ("node", "tail", "edge"),
    ("edge", "head", "node"),
    ("edge", "rev_tail", "node"),
    ("node", "rev_head", "edge"),
]


class HeteroGATv2Encoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim=128, out_dim=64,
                 heads=2, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # project both node types to a common width first
        self.lin_node = nn.Linear(node_in, hidden_dim)
        self.lin_edge = nn.Linear(edge_in, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [hidden_dim] * n_layers + [out_dim]
        for l in range(n_layers):
            d_in, d_out = dims[l], dims[l + 1]
            last = l == n_layers - 1
            per_head = d_out if last else d_out // heads
            conv = HeteroConv(
                {rel: GATv2Conv(d_in, per_head, heads=heads,
                                concat=not last, add_self_loops=False,
                                dropout=dropout)
                 for rel in RELATIONS},
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                "node": nn.LayerNorm(d_out),
                "edge": nn.LayerNorm(d_out),
            }))

    def forward(self, x_dict, edge_index_dict):
        x = {"node": self.lin_node(x_dict["node"]),
             "edge": self.lin_edge(x_dict["edge"])}
        for l in range(self.n_layers):
            x = self.convs[l](x, edge_index_dict)
            last = l == self.n_layers - 1
            new_x = {}
            for k, v in x.items():
                v = self.norms[l][k](v)
                if not last:
                    v = F.dropout(F.elu(v), p=self.dropout, training=self.training)
                new_x[k] = v
            x = new_x
        return x


class SatNetEstimator(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim=128, emb_dim=64,
                 heads=2, n_layers=3, dropout=0.1):
        super().__init__()
        self.encoder = HeteroGATv2Encoder(
            node_in, edge_in, hidden_dim=hidden_dim, out_dim=emb_dim,
            heads=heads, n_layers=n_layers, dropout=dropout)
        self.node_head = nn.Linear(emb_dim, 2)   # proc_time, drop
        self.edge_head = nn.Linear(emb_dim, 3)   # prop_delay, drop, routing_logit

    def forward(self, data):
        z = self.encoder(data.x_dict, data.edge_index_dict)
        node_raw = self.node_head(z["node"])
        edge_raw = self.edge_head(z["edge"])

        routing_prob = softmax(edge_raw[:, 2], data.tail_of_edge)  # sums to 1 per tail
        return {
            "proc_time":    F.softplus(node_raw[:, 0]),
            "node_drop":    torch.sigmoid(node_raw[:, 1]),
            "prop_delay":   F.softplus(edge_raw[:, 0]),
            "edge_drop":    torch.sigmoid(edge_raw[:, 1]),
            "routing_prob": routing_prob,
        }


class absorbing_endtoend:
    """Absorbing-Markov-chain read-out (all differentiable)."""

    @staticmethod
    def build_P(pred, data, device):
        """Pure routing matrix. P[i, j] = prob a packet at i is routed to j.

        No drop, no delay — rows sum to 1 (per the routing softmax).
        Survival and delay are applied in endtoend().
        """
        n = data["node"].x.size(0)
        P = torch.zeros(n, n, device=device)
        for ei in range(len(data.edge_src)):
            u = int(data.edge_src[ei])
            v = int(data.edge_dst[ei])
            if P[u, v] != 0:
                raise Exception(f"P[{u},{v}] already assigned")
            P[u, v] = pred["routing_prob"][ei]
        return P

    @staticmethod
    def _node_out_delay(pred, data, device):
        """Expected propagation delay incurred departing each node:
        sum over outgoing edges of routing_prob * prop_delay."""
        n = data["node"].x.size(0)
        out_delay = torch.zeros(n, device=device)
        for ei in range(len(data.edge_src)):
            u = int(data.edge_src[ei])
            out_delay[u] = out_delay[u] + pred["routing_prob"][ei] * pred["prop_delay"][ei]
        return out_delay

    @staticmethod
    def endtoend(P, pred, data, s, a, device):
        """Expected delay and survival from source s to absorbing target a."""
        n = P.size(0)

        # fold drop into the transition mass: a hop survives only if the
        # departing node doesn't drop AND the chosen link doesn't drop.
        node_survive = 1.0 - pred["node_drop"]              # [n]
        edge_survive = torch.zeros(n, n, device=device)     # per-link survival, routed
        for ei in range(len(data.edge_src)):
            u = int(data.edge_src[ei])
            v = int(data.edge_dst[ei])
            edge_survive[u, v] = 1.0 - pred["edge_drop"][ei]
        # substochastic transition: route * link-survival * node-survival
        S = P * edge_survive * node_survive.unsqueeze(1)    # [n, n]

        transient = [i for i in range(n) if i != a]
        idx = torch.tensor(transient, device=device)

        Q = S[idx][:, idx]                                  # substochastic
        I = torch.eye(len(transient), device=device)
        N = torch.linalg.solve(I - Q, I)                    # fundamental matrix

        pos = {node: k for k, node in enumerate(transient)}
        visits = N[pos[s]]                                  # expected visits from s

        # survival = expected absorption into a
        survival = (visits * S[idx, a]).sum()

        # delay = expected visits * (processing + departing propagation) per node
        out_delay = absorbing_endtoend._node_out_delay(pred, data, device)
        node_delay = pred["proc_time"] + out_delay
        delay = (visits * node_delay[idx]).sum()

        return delay, 1 #survival #TMP!!!!!!!!!!!!!!!!!!!!
