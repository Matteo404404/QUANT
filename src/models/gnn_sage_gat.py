"""
gnn_sage_gat.py
===============
GNN Models: GraphSAGE and GAT implementations.


GraphSAGE implemented via manual sparse mean-aggregation.
GAT implemented without MessagePassing using manual attention computation.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool



# ---------------------------------------------------------------------------
# Manual mean-aggregation SAGE layer
# ---------------------------------------------------------------------------


class PureSAGEConv(nn.Module):
    """
    GraphSAGE conv layer using manual sparse mean-aggregation.
    h_i' = W · CONCAT(h_i, MEAN_{j in N(i)} h_j)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        deg = torch.zeros(n, dtype=torch.float, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1.0)

        agg = torch.zeros_like(x)
        agg.scatter_add_(
            0,
            dst.unsqueeze(-1).expand(-1, x.size(-1)),
            x[src],
        )

        agg = agg / deg.unsqueeze(-1)
        out = torch.cat([x, agg], dim=-1)
        return self.lin(out)



# ---------------------------------------------------------------------------
# GraphSAGE model
# ---------------------------------------------------------------------------


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE for systemic risk prediction.

    Parameters
    ----------
    in_channels  : node feature dimension
    hidden_dim   : width of hidden layers
    out_dim      : output dimension
    n_layers     : number of SAGE conv layers
    dropout      : dropout rate
    pooling      : "mean" or "add"
    node_level   : if True, node-level output
    """

    def __init__(
        self,
        in_channels: int   = 4,
        hidden_dim:  int   = 64,
        out_dim:     int   = 1,
        n_layers:    int   = 3,
        dropout:     float = 0.1,
        pooling:     str   = "mean",
        node_level:  bool  = False,
    ):
        super().__init__()
        self.dropout    = dropout
        self.pooling    = pooling
        self.node_level = node_level

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_channels] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(PureSAGEConv(dims[i], dims[i + 1]))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.node_level:
            return self.head(x)

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)

        return self.head(x)



# ---------------------------------------------------------------------------
# GAT layer
# ---------------------------------------------------------------------------


class GATLayer(nn.Module):
    """
    Single-head GAT layer.
    alpha_ij = softmax( LeakyReLU( a^T [W h_i || W h_j] ) )
    h_i' = sigma( sum_j alpha_ij W h_j )
    """

    def __init__(self, in_channels: int, out_channels: int,
                 n_heads: int = 4, dropout: float = 0.1,
                 concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_ch  = out_channels
        self.concat  = concat
        self.dropout = dropout

        self.W   = nn.Linear(in_channels, out_channels * n_heads, bias=False)
        self.att = nn.Parameter(torch.empty(1, n_heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att.view(1, -1, 1).reshape(1, -1, 2 * out_channels))
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        H, C = self.n_heads, self.out_ch
        src, dst = edge_index[0], edge_index[1]

        Wh = self.W(x).view(n, H, C)

        e_src = (Wh[src] * self.att[:, :, :C]).sum(-1)
        e_dst = (Wh[dst] * self.att[:, :, C:]).sum(-1)
        e     = self.leaky(e_src + e_dst)

        e_max = torch.full((n, H), float('-inf'), device=x.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, H), e, reduce='amax')
        e_exp = torch.exp(e - e_max[dst])
        denom = torch.zeros(n, H, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), e_exp)
        alpha = e_exp / (denom[dst] + 1e-16)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        agg = torch.zeros(n, H, C, device=x.device)
        agg.scatter_add_(
            0,
            dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, C),
            (alpha.unsqueeze(-1) * Wh[src]),
        )

        if self.concat:
            return F.elu(agg.view(n, H * C))
        else:
            return F.elu(agg.mean(dim=1))



class GATModel(nn.Module):
    """Multi-layer GAT for systemic risk prediction."""

    def __init__(
        self,
        in_channels: int   = 4,
        hidden_dim:  int   = 32,
        out_dim:     int   = 1,
        n_layers:    int   = 3,
        n_heads:     int   = 4,
        dropout:     float = 0.1,
        pooling:     str   = "mean",
        node_level:  bool  = False,
    ):
        super().__init__()
        self.dropout    = dropout
        self.pooling    = pooling
        self.node_level = node_level

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATLayer(in_channels, hidden_dim,
                                   n_heads=n_heads, dropout=dropout, concat=True))
        self.norms.append(nn.BatchNorm1d(hidden_dim * n_heads))

        for _ in range(n_layers - 2):
            self.convs.append(GATLayer(hidden_dim * n_heads, hidden_dim,
                                       n_heads=n_heads, dropout=dropout, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim * n_heads))

        if n_layers > 1:
            self.convs.append(GATLayer(hidden_dim * n_heads, hidden_dim,
                                       n_heads=1, dropout=dropout, concat=False))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            final_dim = hidden_dim
        else:
            final_dim = hidden_dim * n_heads

        self.head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, out_dim),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.node_level:
            return self.head(x)

        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)

        return self.head(x)