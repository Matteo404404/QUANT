"""
systemic_risk.py
================
Contagion Layer: systemic importance on the volatility correlation graph.


Metrics (robust on dense graphs):
  - Weighted PageRank
  - Weighted strength (total correlation weight)
  - Eigenvector centrality


Output:
  data/processed/systemic_risk.parquet  -- score per (time_id, stock_id)
  data/processed/hub_stocks.csv         -- stock ranking by systemic importance
  results/contagion_matrix.png          -- contagion heatmap (top-20 stocks)
  results/systemic_timeseries.png       -- importance time series (top-5 hubs)
"""


from __future__ import annotations


import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from src.features.lob_features      import PROCESSED_DIR
from src.features.correlation_graph import build_graphs


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)



# ---------------------------------------------------------------------------
# Graph to adjacency matrix
# ---------------------------------------------------------------------------


def graph_to_adj(graph) -> tuple[np.ndarray, list]:
    n_nodes  = graph.x.shape[0]
    edge_idx = graph.edge_index.numpy()
    edge_w   = (
        graph.edge_attr.numpy().squeeze()
        if graph.edge_attr is not None
        else np.ones(edge_idx.shape[1])
    )

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adj[edge_idx[0], edge_idx[1]] = edge_w
    adj = (adj + adj.T) / 2.0

    stock_ids = (
        graph.stock_ids.tolist()
        if hasattr(graph.stock_ids, "tolist")
        else list(range(n_nodes))
    )
    return adj, stock_ids



# ---------------------------------------------------------------------------
# Composite systemic importance
# ---------------------------------------------------------------------------


def compute_systemic_importance(
    adj:        np.ndarray,
    pagerank_alpha: float = 0.85,
    sparsify_pct:   float = 80.0,
    n_iter:         int   = 100,
) -> np.ndarray:
    """
    Composite systemic importance score per node.

    1. Weighted PageRank (alpha=0.85)
    2. Weighted strength (sum of edge weights)
    3. Eigenvector centrality

    The graph is sparsified first, keeping only edges above the
    given percentile to avoid saturation on dense correlation graphs.

    Returns (N,) float32 scores normalised to [0, 1].
    """
    N = adj.shape[0]

    nonzero = adj[adj > 0]
    if len(nonzero) == 0:
        return np.zeros(N, dtype=np.float32)

    threshold  = np.percentile(nonzero, sparsify_pct)
    adj_sparse = adj.copy()
    adj_sparse[adj_sparse < threshold] = 0.0
    np.fill_diagonal(adj_sparse, 0.0)

    G = nx.from_numpy_array(adj_sparse)

    try:
        pr     = nx.pagerank(G, alpha=pagerank_alpha, weight="weight", max_iter=n_iter)
        pr_arr = np.array([pr.get(i, 0.0) for i in range(N)], dtype=np.float64)
    except Exception:
        pr_arr = np.ones(N, dtype=np.float64) / N

    strength = adj_sparse.sum(axis=1).astype(np.float64)

    try:
        ec     = nx.eigenvector_centrality_numpy(G, weight="weight")
        ec_arr = np.array([ec.get(i, 0.0) for i in range(N)], dtype=np.float64)
    except Exception:
        ec_arr = np.ones(N, dtype=np.float64) / N

    def norm01(x: np.ndarray) -> np.ndarray:
        r = x - x.min()
        d = r.max()
        return r / d if d > 1e-10 else np.zeros_like(r)

    score = 0.4 * norm01(pr_arr) + 0.3 * norm01(strength) + 0.3 * norm01(ec_arr)
    return score.astype(np.float32)



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def compute_all_systemic_risk(
    features_df:  pd.DataFrame,
    sample_every: int   = 5,
    n_iter:       int   = 100,
    sparsify_pct: float = 80.0,
    save:         bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute systemic importance for sampled time_ids.

    Parameters
    ----------
    sample_every : compute every N-th time_id
    sparsify_pct : percentile below which edge weights are zeroed
    """
    print("Building graphs for systemic risk analysis...")
    graphs = build_graphs(features_df, window=50, threshold=0.3)
    print(f"Graphs built: {len(graphs)}")

    sampled_graphs = graphs[::sample_every]
    print(
        f"Computing systemic importance on {len(sampled_graphs)} sampled graphs "
        f"(1 every {sample_every})..."
    )

    records = []

    for g in tqdm(sampled_graphs, desc="Systemic risk"):
        adj, stock_ids = graph_to_adj(g)
        tid            = int(g.time_id)

        scores = compute_systemic_importance(
            adj,
            sparsify_pct=sparsify_pct,
            n_iter=n_iter,
        )

        for idx, sid in enumerate(stock_ids):
            records.append({
                "time_id"            : tid,
                "stock_id"           : int(sid),
                "systemic_importance": float(scores[idx]),
                "degree"             : int((adj[idx] > 0).sum()),
                "strength"           : float(adj[idx].sum()),
            })

    result_df = pd.DataFrame(records)
    print(f"Done. Shape: {result_df.shape}")

    hub_df = (
        result_df
        .groupby("stock_id")["systemic_importance"]
        .agg(["mean", "std", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
        .rename(columns={"mean": "avg_score", "std": "std_score", "max": "max_score"})
    )
    hub_df["rank"] = range(1, len(hub_df) + 1)

    print(f"\nTop 15 systemically important stocks:")
    print(hub_df.head(15).to_string(index=False))

    if save:
        sr_path  = PROCESSED_DIR / "systemic_risk.parquet"
        hub_path = PROCESSED_DIR / "hub_stocks.csv"
        result_df.to_parquet(sr_path, index=False)
        hub_df.to_csv(hub_path, index=False)
        print(f"\nSaved to {sr_path}")
        print(f"Saved to {hub_path}")

    return result_df, hub_df



# ---------------------------------------------------------------------------
# Contagion matrix heatmap
# ---------------------------------------------------------------------------


def plot_contagion_matrix(
    graphs:       list,
    hub_df:       pd.DataFrame,
    top_n:        int   = 20,
    sparsify_pct: float = 80.0,
    save_path:    str   = None,
):
    """Heatmap C[i,j] = mean correlation weight between stock i and j, top-N systemic stocks."""
    if save_path is None:
        save_path = str(RESULTS_DIR / "contagion_matrix.png")

    top_stocks = hub_df.head(top_n)["stock_id"].tolist()
    top_set    = set(top_stocks)

    contagion_sum = np.zeros((top_n, top_n), dtype=np.float64)
    count         = 0

    for g in tqdm(graphs[::10], desc="Contagion matrix"):
        adj, stock_ids = graph_to_adj(g)

        present_idx = [i for i, s in enumerate(stock_ids) if s in top_set]
        present_sid = [stock_ids[i] for i in present_idx]
        if len(present_idx) < 2:
            continue

        sub_adj = adj[np.ix_(present_idx, present_idx)]

        nonzero = sub_adj[sub_adj > 0]
        if len(nonzero) > 0:
            thr     = np.percentile(nonzero, sparsify_pct)
            sub_adj = np.where(sub_adj >= thr, sub_adj, 0.0)

        for i, si in enumerate(present_sid):
            for j, sj in enumerate(present_sid):
                if i == j:
                    continue
                gi = top_stocks.index(si)
                gj = top_stocks.index(sj)
                contagion_sum[gi, gj] += sub_adj[i, j]

        count += 1

    if count == 0:
        print("No valid graphs for contagion matrix.")
        return

    contagion_avg = contagion_sum / max(count, 1)

    fig, ax = plt.subplots(figsize=(14, 12))
    labels  = [f"S{s}" for s in top_stocks]

    sns.heatmap(
        contagion_avg,
        ax          = ax,
        xticklabels = labels,
        yticklabels = labels,
        cmap        = "YlOrRd",
        linewidths  = 0.4,
        linecolor   = "#444444",
        annot       = (top_n <= 15),
        fmt         = ".3f" if top_n <= 15 else "",
        cbar_kws    = {"label": "Avg correlation weight (sparsified)"},
    )
    ax.set_title(
        f"Volatility Contagion Matrix -- Top {top_n} Systemic Stocks\n"
        f"(PageRank + Strength + Eigenvector, avg over {count} windows)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Receiver", fontsize=11)
    ax.set_ylabel("Transmitter", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")



# ---------------------------------------------------------------------------
# Time series of systemic importance
# ---------------------------------------------------------------------------


def plot_systemic_timeseries(
    result_df: pd.DataFrame,
    hub_df:    pd.DataFrame,
    top_n:     int = 5,
    save_path: str = None,
):
    if save_path is None:
        save_path = str(RESULTS_DIR / "systemic_timeseries.png")

    top_stocks = hub_df.head(top_n)["stock_id"].tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, top_n))

    for i, sid in enumerate(top_stocks):
        sub = result_df[result_df["stock_id"] == sid].sort_values("time_id")
        if len(sub) < 3:
            continue
        ax.plot(
            sub["time_id"],
            sub["systemic_importance"],
            label     = f"Stock {int(sid)}",
            color     = colors[i],
            linewidth = 1.8,
            alpha     = 0.85,
        )

    ax.set_title(
        f"Systemic Importance Over Time -- Top {top_n} Hub Stocks\n"
        "(Composite: PageRank + Strength + Eigenvector on volatility correlation graph)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Time ID (market session)", fontsize=11)
    ax.set_ylabel("Systemic Importance Score [0-1]", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    features_path = PROCESSED_DIR / "features_with_nn.parquet"
    if not features_path.exists():
        features_path = PROCESSED_DIR / "features.parquet"

    print(f"Loading features: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"Shape: {df.shape}")

    result_df, hub_df = compute_all_systemic_risk(
        df,
        sample_every = 5,
        n_iter       = 100,
        sparsify_pct = 80.0,
        save         = True,
    )

    print("\nBuilding contagion matrix (top 20 stocks)...")
    graphs = build_graphs(df, window=50, threshold=0.3)
    plot_contagion_matrix(graphs, hub_df, top_n=20, sparsify_pct=80.0)

    print("\nPlotting systemic importance time series...")
    plot_systemic_timeseries(result_df, hub_df, top_n=5)

    print("\nSystemic risk analysis complete.")
    print("  data/processed/systemic_risk.parquet")
    print("  data/processed/hub_stocks.csv")
    print("  results/contagion_matrix.png")
    print("  results/systemic_timeseries.png")