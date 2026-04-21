"""
nn_features_v2.py
=================
Enhanced time-ID NN features, K=50, dual metric.

Improvements over v1:
  - K=50 instead of K=10 (more regime context)
  - Percentiles (p10, p25, p75) of neighbour targets
  - Weighted mean (weight = 1/distance)
  - Dual metric: Canberra + Euclidean, ensemble of two KNN indices
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_time_pivot(df: pd.DataFrame, value_col: str = "rv_full") -> pd.DataFrame:
    pivot = df.pivot_table(index="time_id", columns="stock_id", values=value_col)
    return pivot.fillna(pivot.mean())


def fit_nn_index(pivot: pd.DataFrame, n_neighbors: int, metric: str):
    scaler = StandardScaler()
    X      = scaler.fit_transform(pivot.values)
    nn     = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1)
    nn.fit(X)
    return nn, scaler, X


def compute_nn_features_v2(
    features_df: pd.DataFrame,
    n_neighbors: int  = 50,
    save:        bool = True,
) -> pd.DataFrame:
    print(f"Building NN features v2 (K={n_neighbors}, dual metric)...")

    all_time_ids   = sorted(features_df["time_id"].unique())
    idx_to_tid     = {i: t for i, t in enumerate(all_time_ids)}
    tid_to_idx     = {t: i for i, t in enumerate(all_time_ids)}

    # Pivot base for KNN
    base_pivot = build_time_pivot(features_df, "rv_full")

    # fit two KNN with different metrics
    nn_canb, scaler_c, X_c = fit_nn_index(base_pivot, n_neighbors, "canberra")
    nn_eucl, scaler_e, X_e = fit_nn_index(base_pivot, n_neighbors, "euclidean")

    dist_c, idx_c = nn_canb.kneighbors(X_c)
    dist_e, idx_e = nn_eucl.kneighbors(X_e)

    # drop self-match at index 0
    dist_c, idx_c = dist_c[:, 1:], idx_c[:, 1:]
    dist_e, idx_e = dist_e[:, 1:], idx_e[:, 1:]

    # pivot target and rv for computing stats
    stock_rv  = build_time_pivot(features_df, "rv_full")
    stock_tgt = build_time_pivot(features_df, "target")

    print("Computing neighbour statistics (dual metric)...")

    nn_stats = {}

    for i, tid in enumerate(tqdm(all_time_ids, desc="NN stats")):
        # Canberra neighbors
        n_tids_c = [idx_to_tid[j] for j in idx_c[i]]
        dists_c  = dist_c[i]

        # Euclidean neighbors
        n_tids_e = [idx_to_tid[j] for j in idx_e[i]]
        dists_e  = dist_e[i]

        # union of both neighbour sets, deduplicated
        all_n_tids = list(dict.fromkeys(n_tids_c + n_tids_e))

        valid_c = [t for t in n_tids_c if t in stock_tgt.index]
        valid_e = [t for t in n_tids_e if t in stock_tgt.index]
        valid_all = [t for t in all_n_tids if t in stock_tgt.index]

        def weighted_stats(tids, dists, col_df):
            """Distance-weighted aggregation of neighbour stats."""
            tids_valid = [t for t in tids if t in col_df.index]
            if not tids_valid:
                return None, None, None, None, None, None
            d_valid = np.array([dists[tids.index(t)] for t in tids_valid], dtype=np.float64)
            w       = 1.0 / (d_valid + 1e-10)
            w      /= w.sum()
            vals    = col_df.loc[tids_valid]  # (K, n_stocks)
            wmean   = (vals.multiply(w, axis=0)).sum(axis=0)
            mean_   = vals.mean(axis=0)
            std_    = vals.std(axis=0)
            p25     = vals.quantile(0.25, axis=0)
            p75     = vals.quantile(0.75, axis=0)
            p10     = vals.quantile(0.10, axis=0)
            return wmean, mean_, std_, p25, p75, p10

        tgt_wmean_c, tgt_mean_c, tgt_std_c, tgt_p25_c, tgt_p75_c, tgt_p10_c = \
            weighted_stats(n_tids_c, dists_c, stock_tgt)
        rv_wmean_c,  rv_mean_c,  rv_std_c,  _,       _,       _       = \
            weighted_stats(n_tids_c, dists_c, stock_rv)

        tgt_wmean_e, tgt_mean_e, tgt_std_e, _, _, _ = \
            weighted_stats(n_tids_e, dists_e, stock_tgt)

        nn_stats[tid] = {
            # Canberra
            "nn_tgt_wmean_c" : tgt_wmean_c,
            "nn_tgt_mean_c"  : tgt_mean_c,
            "nn_tgt_std_c"   : tgt_std_c,
            "nn_tgt_p25_c"   : tgt_p25_c,
            "nn_tgt_p75_c"   : tgt_p75_c,
            "nn_tgt_p10_c"   : tgt_p10_c,
            "nn_rv_wmean_c"  : rv_wmean_c,
            "nn_rv_mean_c"   : rv_mean_c,
            "nn_rv_std_c"    : rv_std_c,
            # Euclidean
            "nn_tgt_wmean_e" : tgt_wmean_e,
            "nn_tgt_mean_e"  : tgt_mean_e,
            "nn_tgt_std_e"   : tgt_std_e,
            # Distanze
            "nn_dist_1_c"    : dists_c[0],
            "nn_dist_k_c"    : dists_c[-1],
            "nn_dist_ratio_c": dists_c[0] / (dists_c[-1] + 1e-10),
            "nn_dist_1_e"    : dists_e[0],
            "nn_dist_ratio_e": dists_e[0] / (dists_e[-1] + 1e-10),
        }

    print("Merging into dataframe...")

    new_cols = [
        "nn_tgt_wmean_c","nn_tgt_mean_c","nn_tgt_std_c",
        "nn_tgt_p25_c","nn_tgt_p75_c","nn_tgt_p10_c",
        "nn_rv_wmean_c","nn_rv_mean_c","nn_rv_std_c",
        "nn_tgt_wmean_e","nn_tgt_mean_e","nn_tgt_std_e",
        "nn_dist_1_c","nn_dist_k_c","nn_dist_ratio_c",
        "nn_dist_1_e","nn_dist_ratio_e",
    ]

    # pre-allocate arrays instead of row-by-row loop
    records = {col: np.full(len(features_df), np.nan) for col in new_cols}

    tid_arr = features_df["time_id"].values
    sid_arr = features_df["stock_id"].values

    for row_i, (tid, sid) in enumerate(tqdm(
        zip(tid_arr, sid_arr), total=len(features_df), desc="Filling"
    )):
        if tid not in nn_stats:
            continue
        s = nn_stats[tid]
        sid = int(sid)

        for col in new_cols:
            val = s.get(col)
            if val is None:
                continue
            if hasattr(val, "get"):         # è una Series
                records[col][row_i] = val.get(sid, np.nan)
            else:                           # è uno scalar (distanze)
                records[col][row_i] = float(val)

    result_df = features_df.copy()
    for col, arr in records.items():
        result_df[col] = arr

    result_df = result_df.dropna(subset=["target"])
    print(f"Done. Shape: {result_df.shape}")

    if save:
        out = PROCESSED_DIR / "features_with_nn_v2.parquet"
        result_df.to_parquet(out, index=False)
        print(f"Saved to {out}")

    return result_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    base = PROCESSED_DIR / "features_with_nn.parquet"
    if not base.exists():
        base = PROCESSED_DIR / "features.parquet"

    df     = pd.read_parquet(base)
    result = compute_nn_features_v2(df, n_neighbors=50, save=True)
    print(result[new_cols[:5]].describe())