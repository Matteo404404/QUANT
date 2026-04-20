"""
cross_stock_features.py
=======================
Cross-stock volatility features.

For each (stock_id, time_id), adds the mean/std volatility of the
most correlated stocks (top-K peers from the correlation graph).
Makes the cross-stock signal explicit as tabular features for LightGBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def compute_cross_stock_features(
    features_df: pd.DataFrame,
    corr_window: int  = 50,
    top_k:       int  = 10,
    save:        bool = True,
) -> pd.DataFrame:
    """
    For each stock, finds top-K most correlated peers (rolling window)
    and adds their rv_full as features.

    Added columns:
      - cs_rv_mean   : mean rv of top-K correlated stocks
      - cs_rv_std    : std rv of top-K correlated stocks
      - cs_rv_max    : max rv of top-K correlated stocks
      - cs_tgt_mean  : mean target of top-K peers
      - cs_corr_mean : mean Pearson correlation with top-K peers
    """
    print(f"Computing cross-stock features (window={corr_window}, top_k={top_k})...")

    all_time_ids = sorted(features_df["time_id"].unique())

    # Pivot rv_full: (time_id, stock_id)
    rv_pivot  = features_df.pivot_table(index="time_id", columns="stock_id", values="rv_full")
    tgt_pivot = features_df.pivot_table(index="time_id", columns="stock_id", values="target")

    rv_pivot  = rv_pivot.fillna(rv_pivot.mean())
    tgt_pivot = tgt_pivot.fillna(tgt_pivot.mean())

    stocks    = rv_pivot.columns.tolist()
    n_stocks  = len(stocks)
    s_to_idx  = {s: i for i, s in enumerate(stocks)}

    new_records = {
        "cs_rv_mean"  : np.full(len(features_df), np.nan),
        "cs_rv_std"   : np.full(len(features_df), np.nan),
        "cs_rv_max"   : np.full(len(features_df), np.nan),
        "cs_tgt_mean" : np.full(len(features_df), np.nan),
        "cs_corr_mean": np.full(len(features_df), np.nan),
    }

    # build (time_id, stock_id) -> row index lookup
    idx_map = {}
    for row_i, (tid, sid) in enumerate(
        zip(features_df["time_id"].values, features_df["stock_id"].values)
    ):
        idx_map[(int(tid), int(sid))] = row_i

    prev_corr_matrix = None

    for t_pos, tid in enumerate(tqdm(all_time_ids, desc="Cross-stock")):

        # rolling window for correlation
        window_start = max(0, t_pos - corr_window)
        window_tids  = all_time_ids[window_start:t_pos + 1]

        if len(window_tids) < 5:
            continue

        window_rv = rv_pivot.loc[
            [t for t in window_tids if t in rv_pivot.index]
        ]

        # recompute correlation every 10 steps (stable enough)
        if t_pos % 10 == 0 or prev_corr_matrix is None:
            corr_matrix       = window_rv.corr().values   # (n_stocks, n_stocks)
            prev_corr_matrix  = corr_matrix
        else:
            corr_matrix = prev_corr_matrix

        # current RV for this time_id
        if tid not in rv_pivot.index:
            continue
        rv_current  = rv_pivot.loc[tid].values    # (n_stocks,)
        tgt_current = tgt_pivot.loc[tid].values if tid in tgt_pivot.index else rv_current

        # for each stock, find top-K most correlated
        for s_idx, sid in enumerate(stocks):
            corr_row = np.abs(corr_matrix[s_idx])
            corr_row[s_idx] = 0.0

            top_k_idx  = np.argsort(corr_row)[-top_k:]
            top_k_rv   = rv_current[top_k_idx]
            top_k_tgt  = tgt_current[top_k_idx]
            top_k_corr = corr_row[top_k_idx]

            row_i = idx_map.get((int(tid), int(sid)))
            if row_i is None:
                continue

            new_records["cs_rv_mean"][row_i]   = float(np.mean(top_k_rv))
            new_records["cs_rv_std"][row_i]    = float(np.std(top_k_rv))
            new_records["cs_rv_max"][row_i]    = float(np.max(top_k_rv))
            new_records["cs_tgt_mean"][row_i]  = float(np.mean(top_k_tgt))
            new_records["cs_corr_mean"][row_i] = float(np.mean(top_k_corr))

    result_df = features_df.copy()
    for col, arr in new_records.items():
        result_df[col] = arr

    result_df = result_df.dropna(subset=["target"])
    print(f"Done. Shape: {result_df.shape}")

    if save:
        out = PROCESSED_DIR / "features_with_cross.parquet"
        result_df.to_parquet(out, index=False)
        print(f"Saved to {out}")

    return result_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    base = PROCESSED_DIR / "features_with_nn_v2.parquet"
    if not base.exists():
        base = PROCESSED_DIR / "features_with_nn.parquet"

    df     = pd.read_parquet(base)
    result = compute_cross_stock_features(df, corr_window=50, top_k=10, save=True)
    print(result[["stock_id","time_id","cs_rv_mean","cs_rv_std","cs_tgt_mean"]].head(10))