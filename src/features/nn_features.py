"""
nn_features.py
==============
Time-ID Nearest Neighbour Features

For each time_id, finds the K most similar market sessions using KNN
on the cross-stock realized volatility profile, then uses their
features as additional predictors. Captures market regime similarity.

Reference: nyanp, Kaggle Optiver 1st place
https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_time_pivot(
    features_df: pd.DataFrame,
    value_col:   str = "rv_full",
) -> pd.DataFrame:
    """Pivot: rows=time_id, cols=stock_id, values=value_col. NaN filled with column mean."""
    pivot = features_df.pivot_table(
        index="time_id", columns="stock_id", values=value_col
    )
    pivot = pivot.fillna(pivot.mean())
    return pivot


def fit_nn_index(
    pivot:    pd.DataFrame,
    n_neighbors: int = 80,
    metric:   str = "canberra",
) -> NearestNeighbors:
    """Fit KNN index on the scaled pivot table."""
    scaler = StandardScaler()
    X = scaler.fit_transform(pivot.values)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1)
    nn.fit(X)
    return nn, scaler


def compute_nn_features(
    features_df:  pd.DataFrame,
    n_neighbors:  int = 10,
    nn_cols:      list = None,
    save:         bool = True,
) -> pd.DataFrame:
    """
    For each (stock_id, time_id), add features from the K nearest time_id neighbours.

    Added columns:
      - nn_rv_mean     : mean rv_full of K nearest time_ids (same stock)
      - nn_rv_std      : std rv_full of K nearest time_ids
      - nn_target_mean : mean target of K nearest time_ids (same stock)
      - nn_target_std  : std target
      - nn_dist_1      : distance to nearest neighbour
      - nn_dist_ratio  : dist_1 / dist_k (neighbourhood stability)
    """
    if nn_cols is None:
        nn_cols = ["rv_full"]

    print(f"Building time-ID nearest neighbour features (K={n_neighbors})...")

    all_time_ids = sorted(features_df["time_id"].unique())
    time_id_to_idx = {t: i for i, t in enumerate(all_time_ids)}

    # pivot for KNN: each row is a time_id, columns = stock x feature
    pivots = {}
    for col in nn_cols:
        pivots[col] = build_time_pivot(features_df, value_col=col)

    base_pivot = pivots["rv_full"]
    nn_model, scaler = fit_nn_index(base_pivot, n_neighbors=n_neighbors)

    X_scaled = scaler.transform(base_pivot.values)
    distances, indices = nn_model.kneighbors(X_scaled)
    # skip self-match at index 0
    distances = distances[:, 1:]   # (T, K)
    indices   = indices[:, 1:]     # (T, K)

    # map pivot index -> time_id
    idx_to_time_id = {i: t for i, t in enumerate(all_time_ids)}

    print("Computing per-stock neighbour statistics...")
    stock_rv     = features_df.pivot_table(index="time_id", columns="stock_id", values="rv_full")
    stock_target = features_df.pivot_table(index="time_id", columns="stock_id", values="target")

    stock_rv     = stock_rv.fillna(stock_rv.mean())
    stock_target = stock_target.fillna(stock_target.mean())

    nn_stats = {}

    for i, tid in enumerate(tqdm(all_time_ids, desc="NN features")):
        neighbor_indices = indices[i]           # (K,)
        neighbor_tids    = [idx_to_time_id[j] for j in neighbor_indices]
        dists            = distances[i]         # (K,)

       
        dist_1     = dists[0]
        dist_k     = dists[-1]
        dist_ratio = dist_1 / (dist_k + 1e-10)

        # per-stock stats from neighbour targets/rv
        if tid in stock_rv.index:
            valid_neighbor_tids = [t for t in neighbor_tids if t in stock_rv.index]

            if valid_neighbor_tids:
                rv_neigh  = stock_rv.loc[valid_neighbor_tids]     # (K, n_stocks)
                tgt_neigh = stock_target.loc[valid_neighbor_tids] # (K, n_stocks)

                nn_stats[tid] = {
                    "nn_rv_mean"    : rv_neigh.mean(axis=0),   # Series index=stock_id
                    "nn_rv_std"     : rv_neigh.std(axis=0),
                    "nn_target_mean": tgt_neigh.mean(axis=0),
                    "nn_target_std" : tgt_neigh.std(axis=0),
                    "nn_dist_1"     : dist_1,
                    "nn_dist_ratio" : dist_ratio,
                }

    # Merge NN features into the dataframe using vectorized pre-allocated arrays
    print("Merging NN features into main dataframe...")

    nn_cols = ["nn_rv_mean", "nn_rv_std", "nn_target_mean",
               "nn_target_std", "nn_dist_1", "nn_dist_ratio"]
    records = {col: np.full(len(features_df), np.nan) for col in nn_cols}

    tid_arr = features_df["time_id"].values
    sid_arr = features_df["stock_id"].values

    for row_i, (tid, sid) in enumerate(tqdm(
        zip(tid_arr, sid_arr), total=len(features_df), desc="Merging"
    )):
        if tid not in nn_stats:
            continue
        s = nn_stats[tid]
        sid = int(sid)
        records["nn_rv_mean"][row_i]     = s["nn_rv_mean"].get(sid, np.nan)
        records["nn_rv_std"][row_i]      = s["nn_rv_std"].get(sid, np.nan)
        records["nn_target_mean"][row_i] = s["nn_target_mean"].get(sid, np.nan)
        records["nn_target_std"][row_i]  = s["nn_target_std"].get(sid, np.nan)
        records["nn_dist_1"][row_i]      = s["nn_dist_1"]
        records["nn_dist_ratio"][row_i]  = s["nn_dist_ratio"]

    result_df = features_df.copy()
    for col, arr in records.items():
        result_df[col] = arr
    result_df = result_df.dropna(subset=["target"])

    print(f"Done. Shape with NN features: {result_df.shape}")

    if save:
        out = PROCESSED_DIR / "features_with_nn.parquet"
        result_df.to_parquet(out, index=False)
        print(f"Saved to {out}")

    return result_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    features_path = PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(features_path)
    print(f"Loaded: {df.shape}")

    result = compute_nn_features(df, n_neighbors=10, save=True)
    print(result[["stock_id", "time_id", "rv_full", "target",
                  "nn_rv_mean", "nn_target_mean", "nn_dist_1"]].head(10))