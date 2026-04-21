"""
gnn_volatility.py
=================
GNN Volatility Predictor
Node-level regression on the dynamic stock correlation graph.
GraphSAGE implemented in pure PyTorch via manual mean-aggregation.
"""


from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data



# ---------------------------------------------------------------------------
# GraphSAGE mean aggregation
# ---------------------------------------------------------------------------


def sage_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Mean aggregation of neighbours.
    x          : (N, F)
    edge_index : (2, E) with [row, col] = [dst, src]
    """
    N, F = x.size()
    row, col = edge_index

    agg = torch.zeros_like(x)
    agg.scatter_add_(0, row.view(-1, 1).expand(-1, F), x[col])

    deg = torch.zeros(N, 1, device=x.device)
    deg.scatter_add_(0, row.view(-1, 1), torch.ones(edge_index.size(1), 1, device=x.device))
    deg = deg.clamp(min=1.0)

    agg = agg / deg
    return agg



class SAGEVolModel(nn.Module):
    """
    GraphSAGE for node-level volatility regression.
    h_i^{(k+1)} = ReLU( W_self h_i^{(k)} + W_neigh mean_{j in N(i)} h_j^{(k)} )
    """
    def __init__(self, in_channels: int, hidden: int = 64,
                 n_layers: int = 3, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.dropout = dropout

        self.W_self  = nn.ModuleList()
        self.W_neigh = nn.ModuleList()
        self.norms   = nn.ModuleList()

        dims = [in_channels] + [hidden] * n_layers
        for i in range(n_layers):
            self.W_self.append(nn.Linear(dims[i], dims[i+1]))
            self.W_neigh.append(nn.Linear(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        for W_s, W_n, norm in zip(self.W_self, self.W_neigh, self.norms):
            neigh = sage_aggregate(x, edge_index)
            x_new = W_s(x) + W_n(neigh)
            x = norm(F.relu(x_new))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.head(x).squeeze(-1)
        return out



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


MODEL_REGISTRY = {
    "sage": SAGEVolModel,
}


def build_model(
    name: str,
    in_channels: int,
    hidden: int = 64,
    n_layers: int = 3,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    return cls(in_channels=in_channels, hidden=hidden,
               n_layers=n_layers, dropout=dropout, **kwargs)



# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    N, F, E = 50, 17, 120
    data = Data(
        x          = torch.randn(N, F),
        edge_index = torch.randint(0, N, (2, E)),
        y          = torch.rand(N),
    )

    model = build_model("sage", in_channels=F, hidden=64)
    out   = model(data)
    params = sum(p.numel() for p in model.parameters())
    print(f"sage | output: {out.shape} | params: {params:,}")