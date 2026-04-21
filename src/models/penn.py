"""
penn.py
=======
Permutation-Equivariant Neural Network (PENN) for systemic risk approximation.


Implements the message-passing architecture from:
    Gonon et al. (2024). "Computing Systemic Risk Measures with Graph Neural Networks."
    arXiv:2410.07222.


Architecture:
    Node embeddings h_i^{(0)} = MLP_node(x_i)
    Edge embeddings e_ij^{(0)} = MLP_edge(x_i, x_j, L_ij)

    For l = 1..L:
        m_ij = MLP_msg(h_i || h_j || e_ij)
        h_i  = MLP_upd(h_i || sum_{j in N(i)} m_ij)

    Graph readout (invariant):
        z = sum_i h_i^{(L)}
        output = MLP_out(z)

    Node readout (equivariant):
        output_i = MLP_node_out(h_i^{(L)})
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
    Single permutation-equivariant message passing layer.

    For each node i:
        1. m_ij = MLP_msg(h_i || h_j || e_ij)
        2. agg_i = sum_{j in N(i)} m_ij
        3. h_i'  = MLP_upd(h_i || agg_i)

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
        self.msg_mlp = _mlp(
            [node_dim * 2 + edge_dim, hidden_dim, hidden_dim],
            dropout=dropout,
        )
        self.upd_mlp = _mlp(
            [node_dim + hidden_dim, hidden_dim, node_dim],
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        num_nodes:  int,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]

        msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        messages  = self.msg_mlp(msg_input)

        agg = scatter(messages, dst, dim=0, dim_size=num_nodes, reduce="sum")
        upd_input = torch.cat([h, agg], dim=-1)
        h_new     = self.upd_mlp(upd_input)
        h_new     = self.norm(h + h_new)

        return h_new



class PENN(nn.Module):
    """
    Permutation-Equivariant Neural Network for systemic risk prediction.

    Parameters
    ----------
    in_channels  : node feature dimension
    edge_in_dim  : edge feature dimension
    hidden_dim   : hidden dimension throughout
    out_dim      : output dimension
    n_layers     : number of message passing rounds
    dropout      : dropout rate
    node_level   : if True, return node-level outputs (for DebtRank)
    pooling      : "add" or "mean" for graph-level readout
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

        self.node_encoder = _mlp(
            [in_channels, hidden_dim, hidden_dim], dropout=dropout
        )
        self.edge_encoder = _mlp(
            [edge_in_dim, hidden_dim // 2, hidden_dim // 2], dropout=dropout
        )
        edge_dim = hidden_dim // 2

        self.layers = nn.ModuleList([
            PENNLayer(hidden_dim, edge_dim, hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.head = _mlp([hidden_dim, hidden_dim // 2, out_dim])

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        num_nodes = x.size(0)

        h = self.node_encoder(x)

        if edge_attr is not None and edge_attr.shape[0] > 0:
            e = self.edge_encoder(edge_attr)
        else:
            e = torch.zeros(
                edge_index.shape[1], self.layers[0].msg_mlp[0].in_features
                - 2 * h.shape[-1],
                device=x.device,
            )

        for layer in self.layers:
            h = layer(h, edge_index, e, num_nodes)

        if self.node_level:
            return self.head(h)

        if self.pooling == "add":
            h_graph = global_add_pool(h, batch)
        else:
            h_graph = global_mean_pool(h, batch)

        return self.head(h_graph)