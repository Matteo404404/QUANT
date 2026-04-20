"""
correlation_graph.py
====================
Dynamic Correlation Graph Builder
-----------------------------------
Builds a per-time_id graph where:
  - Nodes  = stocks
  - Edges  = pairs of stocks with |correlation| > threshold
             computed on rv_full within a rolling window of time_ids
  - Node features = LOB microstructure features from lob_features.py
  - Edge weights  = absolute Pearson correlation of rv_full

This graph is the input to the GNN volatility predictor.

Usage
-----
from src.features.correlation_graph import build_graphs
graphs = build_graphs(features_df, window=50, threshold=0.4)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Node feature columns (exclude ids and target)
# ---------------------------------------------------------------------------

NODE_FEATURE_COLS = [
    "rv_full", "rv_last30", "rv_first30", "rv_ratio",
    "mid_rv", "spread1_mean", "spread1_std",
    "imbalance1_mean", "imbalance1_std",
    "depth_mean", "depth_std",
    "trade_volume", "trade_count",
    "vwap_spread", "price_impact", "trade_rv",
    "n_updates",
]


# ---------------------------------------------------------------------------
# Correlation graph builder
# ---------------------------------------------------------------------------

def build_correlation_matrix(
    pivot:     pd.DataFrame,
    time_ids:  list,
    window:    int,
) -> np.ndarray:
    """
    Compute Pearson correlation matrix of rv_full across stocks
    over a rolling window of time_ids.

    Parameters
    ----------
    pivot    : DataFrame indexed by time_id, columns = stock_ids, values = rv_full
    time_ids : sorted list of all time_ids
    window   : number of past time_ids to use for correlation

    Returns
    -------
    corr : np.ndarray of shape (n_stocks, n_stocks)
    """
    subset = pivot.loc[time_ids]
    corr = subset.corr(method="pearson").values
    # Replace NaN with 0
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def corr_to_edge_index(
    corr:      np.ndarray,
    threshold: float,
    stock_ids: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert correlation matrix to edge_index + edge_attr.
    Only keep edges where |corr| > threshold (excluding self-loops).

    Returns
    -------
    edge_index : torch.LongTensor of shape (2, E)
    edge_attr  : torch.FloatTensor of shape (E, 1)
    """
    n = len(stock_ids)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)

    # vectorized: find all (i, j) where |corr| > threshold
    mask = abs_corr > threshold
    src_arr, dst_arr = np.where(mask)
    weights = abs_corr[src_arr, dst_arr]

    if len(src_arr) == 0:
        # fallback: fully connect if nothing passes threshold
        mask_full = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask_full, False)
        src_arr, dst_arr = np.where(mask_full)
        weights = abs_corr[src_arr, dst_arr]

    edge_index = torch.tensor(np.stack([src_arr, dst_arr]), dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    return edge_index, edge_attr


def build_graphs(
    features_df: pd.DataFrame,
    window:      int   = 50,
    threshold:   float = 0.3,
    min_stocks:  int   = 10,
) -> list[Data]:
    """
    Build one PyG Data graph per time_id snapshot.

    For each time_id t:
      - Uses the past `window` time_ids to compute stock correlations
      - Node features = normalised LOB features at time t for each stock
      - Edge index    = correlation graph at threshold
      - y             = target (realized vol) for each stock at time t

    Parameters
    ----------
    features_df : output of build_feature_matrix()
    window      : rolling window size for correlation
    threshold   : minimum |corr| to include an edge
    min_stocks  : skip time_ids where fewer than this many stocks are available

    Returns
    -------
    list of torch_geometric.data.Data objects
    """
    # Sort
    features_df = features_df.sort_values(["time_id", "stock_id"]).copy()

    # Pivot rv_full: rows=time_id, cols=stock_id
    rv_pivot = features_df.pivot_table(
        index="time_id", columns="stock_id", values="rv_full"
    )

    all_time_ids = sorted(features_df["time_id"].unique())
    all_stock_ids = sorted(features_df["stock_id"].unique())
    n_stocks = len(all_stock_ids)
    stock_idx = {s: i for i, s in enumerate(all_stock_ids)}

    # Fill NaN in pivot with column means
    rv_pivot = rv_pivot.fillna(rv_pivot.mean())

    graphs = []

    for t_pos, tid in enumerate(tqdm(all_time_ids, desc="Building graphs")):
        # Skip if not enough history for correlation window
        if t_pos < window:
            continue

        window_tids = all_time_ids[t_pos - window: t_pos]

        # Rows at this time_id
        t_df = features_df[features_df["time_id"] == tid]

        if len(t_df) < min_stocks:
            continue

        present_stocks = sorted(t_df["stock_id"].unique())
        present_idx    = [stock_idx[s] for s in present_stocks]
        n = len(present_stocks)

        # --- Node features ---
        feat_cols = [c for c in NODE_FEATURE_COLS if c in t_df.columns]
        X = t_df.set_index("stock_id").loc[present_stocks][feat_cols].values.astype(np.float32)

        # Normalise per-graph (z-score)
        mu  = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0,  keepdims=True) + 1e-8
        X   = (X - mu) / std
        X   = np.nan_to_num(X, nan=0.0)

        # --- Targets ---
        y = t_df.set_index("stock_id").loc[present_stocks]["target"].values.astype(np.float32)

        # --- Correlation graph (on present stocks only) ---
        rv_sub = rv_pivot.loc[
            [t for t in window_tids if t in rv_pivot.index],
            [s for s in present_stocks if s in rv_pivot.columns]
        ]

        if rv_sub.shape[0] < 5 or rv_sub.shape[1] < 2:
            continue

        corr = rv_sub.corr(method="pearson").values
        corr = np.nan_to_num(corr, nan=0.0)

        edge_index, edge_attr = corr_to_edge_index(corr, threshold, present_stocks)

        data = Data(
            x          = torch.tensor(X, dtype=torch.float),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor(y, dtype=torch.float),
            time_id    = tid,
            stock_ids  = torch.tensor(present_stocks, dtype=torch.long),
            n_stocks   = n,
        )
        graphs.append(data)

    print(f"\nBuilt {len(graphs)} graphs from {len(all_time_ids)} time_ids")
    print(f"Avg nodes per graph : {np.mean([g.num_nodes for g in graphs]):.1f}")
    print(f"Avg edges per graph : {np.mean([g.num_edges for g in graphs]):.1f}")

    return graphs


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.features.lob_features import PROCESSED_DIR

    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("Run lob_features.py first to generate features.parquet")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    print(f"Loaded features: {df.shape}")

    graphs = build_graphs(df, window=50, threshold=0.3)
    print(graphs[0])