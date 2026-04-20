"""
mlp_baseline.py
===============
MLP Baseline: Graph-Agnostic Systemic Risk Predictor
-----------------------------------------------------
Aggregates all node features into a fixed-size global vector
(sum + mean pooling) and passes through a deep MLP.

This model deliberately ignores graph structure and serves as
the baseline that GNN models must outperform to justify the
use of network topology.

Architecture:
    node features (n, F)
     → global sum pooling  → (F,)
     → global mean pooling → (F,)
     → concatenate         → (2F,)
     → MLP(2F → hidden → hidden → out_dim)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool


class MLPBaseline(nn.Module):
    """
    Graph-agnostic MLP baseline for systemic risk prediction.

    Parameters
    ----------
    in_channels : int
        Number of node features F.
    hidden_dim : int
        Hidden layer width.
    out_dim : int
        Output dimension (1 for scalar AS/MBC, n for DebtRank vector).
    n_layers : int
        Number of hidden layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim:  int = 128,
        out_dim:     int = 1,
        n_layers:    int = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim

        # Input: concatenation of sum-pooled and mean-pooled node features
        input_dim = in_channels * 2

        layers = []
        dims   = [input_dim] + [hidden_dim] * n_layers + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Global pooling (ignores edges completely)
        x_sum  = global_add_pool(x, batch)   # (B, F)
        x_mean = global_mean_pool(x, batch)  # (B, F)
        x_glob = torch.cat([x_sum, x_mean], dim=-1)  # (B, 2F)

        return self.mlp(x_glob)   # (B, out_dim)