"""
Promote a PyG-style dataset record into a heterogeneous graph.

Two node types:
  'node' : the original graph nodes (satellites, gateways, ...)
  'edge' : one node per DIRECTED link u->v (edges promoted to nodes)

Four incidence relations wire nodes to edge-nodes, both directions, so the
GAT can pass messages upstream and downstream. The incidence links are
unweighted; the link's own feature (distance, ...) lives on the edge-node.
"""
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