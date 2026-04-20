"""
lob_features.py
===============
Market Microstructure Feature Engineering
------------------------------------------
Computes professional-grade LOB features from Optiver
book_train.parquet and trade_train.parquet.

Features per (stock_id, time_id):
----------------------------------
WAP1, WAP2          — Weighted Average Price levels 1 and 2
log_return1/2       — Log returns from WAP
realized_vol_*      — Realized volatility at multiple time horizons
bid_ask_spread1/2   — Spread at levels 1 and 2
order_imbalance1/2  — Bid/ask size imbalance
depth_total         — Total queue depth (bid + ask, all levels)
trade_volume        — Total traded volume per window
trade_count         — Number of trades per window
price_impact        — Avg price impact of trades vs WAP
vwap_spread         — VWAP vs WAP spread (informed trading signal)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
BOOK_TRAIN = DATA_DIR / "book_train.parquet"
TRADE_TRAIN = DATA_DIR / "trade_train.parquet"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# ---------------------------------------------------------------------------
# Core LOB feature functions
# ---------------------------------------------------------------------------

def wap(bid_price: pd.Series, bid_size: pd.Series,
        ask_price: pd.Series, ask_size: pd.Series) -> pd.Series:
    """Weighted Average Price: (bp*as + ap*bs) / (bs + as)"""
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)


def log_return(prices: pd.Series) -> pd.Series:
    """Log returns of a price series."""
    return np.log(prices).diff()


def realized_volatility(returns: pd.Series) -> float:
    """Realized volatility = sqrt(sum of squared log returns)."""
    return np.sqrt(np.sum(returns ** 2))


def order_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """(bid - ask) / (bid + ask) — ranges from -1 to +1."""
    total = bid_size + ask_size
    return (bid_size - ask_size) / total.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Per-window feature extractor (book data)
# ---------------------------------------------------------------------------

def extract_book_features(df: pd.DataFrame) -> dict:
    """
    Extract all LOB features from one (stock_id, time_id) window.

    Parameters
    ----------
    df : DataFrame with columns:
         seconds_in_bucket, bid_price1, ask_price1, bid_size1, ask_size1,
         bid_price2, ask_price2, bid_size2, ask_size2

    Returns
    -------
    dict of scalar features
    """
    df = df.sort_values("seconds_in_bucket").copy()

    # WAP levels 1 and 2
    w1 = wap(df["bid_price1"], df["bid_size1"], df["ask_price1"], df["ask_size1"])
    w2 = wap(df["bid_price2"], df["bid_size2"], df["ask_price2"], df["ask_size2"])

    # Log returns
    lr1 = log_return(w1).dropna()
    lr2 = log_return(w2).dropna()

    # Realized vol at multiple horizons
    rv_full   = realized_volatility(lr1)
    rv_last30 = realized_volatility(lr1.iloc[-30:] if len(lr1) > 30 else lr1)
    rv_first30 = realized_volatility(lr1[:30] if len(lr1) > 30 else lr1)

    # Spreads
    spread1 = (df["ask_price1"] - df["bid_price1"]) / df["bid_price1"]
    spread2 = (df["ask_price2"] - df["bid_price2"]) / df["bid_price2"]

    # Order imbalance
    imb1 = order_imbalance(df["bid_size1"], df["ask_size1"])
    imb2 = order_imbalance(df["bid_size2"], df["ask_size2"])

    # Total depth
    depth = (df["bid_size1"] + df["ask_size1"] +
             df["bid_size2"] + df["ask_size2"])

    # Mid-price volatility
    mid = (df["bid_price1"] + df["ask_price1"]) / 2
    mid_rv = realized_volatility(log_return(mid).dropna())

    return {
        "wap1_mean"         : w1.mean(),
        "wap1_std"          : w1.std(),
        "wap2_mean"         : w2.mean(),
        "rv_full"           : rv_full,
        "rv_last30"         : rv_last30,
        "rv_first30"        : rv_first30,
        "rv_ratio"          : rv_last30 / (rv_first30 + 1e-10),
        "mid_rv"            : mid_rv,
        "spread1_mean"      : spread1.mean(),
        "spread1_std"       : spread1.std(),
        "spread2_mean"      : spread2.mean(),
        "imbalance1_mean"   : imb1.mean(),
        "imbalance1_std"    : imb1.std(),
        "imbalance2_mean"   : imb2.mean(),
        "depth_mean"        : depth.mean(),
        "depth_std"         : depth.std(),
        "n_updates"         : len(df),
    }


# ---------------------------------------------------------------------------
# Per-window feature extractor (trade data)
# ---------------------------------------------------------------------------

def extract_trade_features(df: pd.DataFrame, wap_mean: float) -> dict:
    """
    Extract trade-based features from one (stock_id, time_id) window.

    Parameters
    ----------
    df      : DataFrame with columns: seconds_in_bucket, price, size, order_count
    wap_mean: mean WAP from book features (used for price impact)

    Returns
    -------
    dict of scalar features
    """
    if df is None or len(df) == 0:
        return {
            "trade_volume"   : 0.0,
            "trade_count"    : 0.0,
            "vwap"           : np.nan,
            "vwap_spread"    : np.nan,
            "price_impact"   : np.nan,
            "trade_rv"       : 0.0,
            "order_count"    : 0.0,
        }

    df = df.sort_values("seconds_in_bucket").copy()

    volume      = df["size"].sum()
    vwap        = (df["price"] * df["size"]).sum() / (volume + 1e-10)
    vwap_spread = (vwap - wap_mean) / (wap_mean + 1e-10)
    price_impact = np.abs(df["price"] - wap_mean).mean() / (wap_mean + 1e-10)
    trade_rv    = realized_volatility(log_return(df["price"]).dropna())
    order_count = df["order_count"].sum() if "order_count" in df.columns else len(df)

    return {
        "trade_volume"   : volume,
        "trade_count"    : float(len(df)),
        "vwap"           : vwap,
        "vwap_spread"    : vwap_spread,
        "price_impact"   : price_impact,
        "trade_rv"       : trade_rv,
        "order_count"    : float(order_count),
    }


# ---------------------------------------------------------------------------
# Main builder: processes all stocks and returns feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    stock_ids:  Optional[list] = None,
    max_stocks: Optional[int]  = None,
    save:       bool           = True,
) -> pd.DataFrame:
    """
    Build the full feature matrix for all (stock_id, time_id) pairs.

    Parameters
    ----------
    stock_ids  : list of stock IDs to process (None = all)
    max_stocks : process only first N stocks (useful for testing)
    save       : if True, saves to data/processed/features.parquet

    Returns
    -------
    pd.DataFrame with columns: stock_id, time_id, + all features + target
    """
    # Load targets
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    target_map = train_df.set_index(["stock_id", "time_id"])["target"].to_dict()

    # Discover available stock IDs from book_train
    available = sorted([
        int(p.name.split("=")[1])
        for p in BOOK_TRAIN.iterdir()
        if p.is_dir() and p.name.startswith("stock_id=")
    ])

    if stock_ids is not None:
        available = [s for s in available if s in stock_ids]
    if max_stocks is not None:
        available = available[:max_stocks]

    print(f"Processing {len(available)} stocks...")

    rows = []
    from tqdm import tqdm
    for sid in tqdm(available, desc="Processing stocks"):
        # Load book data for this stock
        book_path  = BOOK_TRAIN / f"stock_id={sid}"
        trade_path = TRADE_TRAIN / f"stock_id={sid}"

        try:
            book_df  = pd.read_parquet(book_path)
            trade_df = pd.read_parquet(trade_path) if trade_path.exists() else None
        except Exception as e:
            print(f"  Stock {sid}: read error — {e}")
            continue

        # Group by time_id and extract features
        book_groups  = book_df.groupby("time_id")
        trade_groups = trade_df.groupby("time_id") if trade_df is not None else {}

        for tid, bgroup in book_groups:
            book_feats  = extract_book_features(bgroup)

            tgroup = None
            if trade_df is not None and tid in trade_groups.groups:
                tgroup = trade_groups.get_group(tid)
            trade_feats = extract_trade_features(tgroup, book_feats["wap1_mean"])

            row = {"stock_id": sid, "time_id": tid}
            row.update(book_feats)
            row.update(trade_feats)
            row["target"] = target_map.get((sid, tid), np.nan)
            rows.append(row)

    features_df = pd.DataFrame(rows)
    features_df = features_df.dropna(subset=["target"])

    print(f"Done. Feature matrix shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / "features.parquet"
        features_df.to_parquet(out, index=False)
        print(f"Saved to {out}")

    return features_df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test on 5 stocks first
    df = build_feature_matrix(max_stocks=5, save=False)
    print(df.head())
    print(df.describe())