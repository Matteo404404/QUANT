"""
penn.py
=======
Permutation-Equivariant Neural Network (PENN)
----------------------------------------------
State-of-the-art architecture for systemic risk approximation
following the theoretical framework of:

    Gonon, L., Herrera, C., Kruse, T., & Ritter, G. (2024).
    "Computing Systemic Risk Measures with Graph Neural Networks."
    arXiv:2410.07222.

Motivation
----------
Standard GNNs process graphs via message passing, but do not
explicitly exploit the permutation symmetry of the problem:
if we relabel the banks, the aggregate systemic risk measure
should not change (permutation invariance for scalars) or
should permute accordingly (permutation equivariance for
node-level outputs like DebtRank).

A PENN enforces this by construction using:
1. Shared weights across all node pairs (equivariant linear layers)
2. Symmetric aggregation (sum/mean over neighbours)
3. A readout that is provably invariant/equivariant

Architecture
------------
    Node embeddings h_i^{(0)} = MLP_node(x_i)
    Edge embeddings e_ij^{(0)} = MLP_edge(x_i, x_j, L_ij)

    For l = 1..L:
        m_ij = MLP_msg(h_i || h_j || e_ij)           (message)
        h_i  = MLP_upd(h_i || Σ_{j∈N(i)} m_ij)      (update, equivariant)

    Graph readout (invariant):
        z = Σ_i h_i^{(L)}   (sum pooling)
        output = MLP_out(z)

    Node readout (equivariant):
        output_i = MLP_node_out(h_i^{(L)})

This is a standard MPNN (Gilmer et al. 2017) with specific
design choices enforcing equivariance via shared weights
and symmetric aggregation.

Reference:
    Gilmer, J., Schütt, S., Riley, P., Vinyals, O., & Dahl, G. (2017).
    "Neural Message Passing for Quantum Chemistry." ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter

def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    """Build a simple MLP with BatchNorm and ReLU activations."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class PENNLayer(nn.Module):
    """
    Single Permutation-Equivariant Message Passing Layer.

    For each node i:
        1. Compute messages from each neighbour j:
               m_ij = MLP_msg(h_i || h_j || e_ij)
        2. Aggregate:
               agg_i = Σ_{j∈N(i)} m_ij
        3. Update node embedding:
               h_i' = MLP_upd(h_i || agg_i)

    The use of SHARED MLP_msg and MLP_upd weights across ALL
    node pairs enforces permutation equivariance by construction.

    Parameters
    ----------
    node_dim  : dimension of node embeddings
    edge_dim  : dimension of edge embeddings
    hidden_dim: hidden dimension in MLPs
    dropout   : dropout rate
    """

    def __init__(
        self,
        node_dim:   int,
        edge_dim:   int,
        hidden_dim: int,
        dropout:    float = 0.1,
    ):
        super().__init__()
        # Message MLP: (h_i || h_j || e_ij) → message
        self.msg_mlp = _mlp(
            [node_dim * 2 + edge_dim, hidden_dim, hidden_dim],
            dropout=dropout,
        )
        # Update MLP: (h_i || agg_i) → h_i'
        self.upd_mlp = _mlp(
            [node_dim + hidden_dim, hidden_dim, node_dim],
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h:          torch.Tensor,    # (N, node_dim)
        edge_index: torch.Tensor,    # (2, E)
        edge_attr:  torch.Tensor,    # (E, edge_dim)
        num_nodes:  int,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]

        # Messages
        msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        messages  = self.msg_mlp(msg_input)                            # (E, hidden)

        # Aggregate: sum over incoming messages per node
        agg = scatter(messages, dst, dim=0, dim_size=num_nodes, reduce="sum")   # (N, hidden)
        # Update with residual connection
        upd_input = torch.cat([h, agg], dim=-1)                        # (N, node+hidden)
        h_new     = self.upd_mlp(upd_input)                            # (N, node_dim)
        h_new     = self.norm(h + h_new)                               # residual + LayerNorm

        return h_new


class PENN(nn.Module):
    """
    Full Permutation-Equivariant Neural Network for systemic risk.

    Parameters
    ----------
    in_channels  : node feature dimension (F)
    edge_in_dim  : edge feature dimension
    hidden_dim   : hidden dimension throughout
    out_dim      : output dimension (1 for scalar, n for node-level)
    n_layers     : number of message passing rounds
    dropout      : dropout rate
    node_level   : if True, return node-level outputs (for DebtRank)
    pooling      : "add" or "mean" for graph-level invariant readout
    """

    def __init__(
        self,
        in_channels: int   = 4,
        edge_in_dim: int   = 1,
        hidden_dim:  int   = 64,
        out_dim:     int   = 1,
        n_layers:    int   = 4,
        dropout:     float = 0.1,
        node_level:  bool  = False,
        pooling:     str   = "add",
    ):
        super().__init__()
        self.node_level = node_level
        self.pooling    = pooling

        # Initial node embedding
        self.node_encoder = _mlp(
            [in_channels, hidden_dim, hidden_dim], dropout=dropout
        )

        # Initial edge embedding
        self.edge_encoder = _mlp(
            [edge_in_dim, hidden_dim // 2, hidden_dim // 2], dropout=dropout
        )
        edge_dim = hidden_dim // 2

        # Message passing layers
        self.layers = nn.ModuleList([
            PENNLayer(hidden_dim, edge_dim, hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output head
        if node_level:
            self.head = _mlp([hidden_dim, hidden_dim // 2, out_dim])
        else:
            self.head = _mlp([hidden_dim, hidden_dim // 2, out_dim])

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        num_nodes = x.size(0)

        # Encode nodes and edges
        h = self.node_encoder(x)                          # (N, hidden)

        if edge_attr is not None and edge_attr.shape[0] > 0:
            e = self.edge_encoder(edge_attr)              # (E, edge_dim)
        else:
            # Handle graphs with no edges
            e = torch.zeros(
                edge_index.shape[1], self.layers[0].msg_mlp[0].in_features
                - 2 * h.shape[-1],
                device=x.device,
            )

        # Message passing rounds
        for layer in self.layers:
            h = layer(h, edge_index, e, num_nodes)

        # Readout
        if self.node_level:
            return self.head(h)    # (N, out_dim), equivariant

        # Graph-level invariant readout
        if self.pooling == "add":
            h_graph = global_add_pool(h, batch)
        else:
            h_graph = global_mean_pool(h, batch)

        return self.head(h_graph)   # (B, out_dim), invariant