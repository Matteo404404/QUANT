"""
baseline.py
===========
LightGBM tabular baseline with target ratio transform.
No graph structure, just LOB + NN features.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_COLS = [
    # LOB microstructure
    "rv_full", "rv_last30", "rv_first30", "rv_ratio", "mid_rv",
    "spread1_mean", "spread1_std", "spread2_mean",
    "imbalance1_mean", "imbalance1_std", "imbalance2_mean",
    "depth_mean", "depth_std", "n_updates",
    "trade_volume", "trade_count", "vwap_spread",
    "price_impact", "trade_rv", "order_count",
    # NN v1 (K=10, Canberra)
    "nn_rv_mean", "nn_rv_std", "nn_target_mean",
    "nn_target_std", "nn_dist_1", "nn_dist_ratio",
    # NN v2 (K=50, dual metric)
    "nn_tgt_wmean_c", "nn_tgt_mean_c", "nn_tgt_std_c",
    "nn_tgt_p25_c", "nn_tgt_p75_c", "nn_tgt_p10_c",
    "nn_rv_wmean_c", "nn_rv_mean_c", "nn_rv_std_c",
    "nn_tgt_wmean_e", "nn_tgt_mean_e", "nn_tgt_std_e",
    "nn_dist_1_c", "nn_dist_k_c", "nn_dist_ratio_c",
    "nn_dist_1_e", "nn_dist_ratio_e",
    # Cross-stock
    "cs_rv_mean", "cs_rv_std", "cs_rv_max",
    "cs_tgt_mean", "cs_corr_mean",
]


LGB_PARAMS = {
    "objective"        : "regression",
    "metric"           : "rmse",
    "learning_rate"    : 0.05,
    "num_leaves"       : 256,
    "min_child_samples": 80,
    "feature_fraction" : 0.8,
    "bagging_fraction" : 0.8,
    "bagging_freq"     : 1,
    "n_estimators"     : 3000,
    "early_stopping_round": 50,
    "verbose"          : -1,
    "n_jobs"           : -1,
}


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(((y_true - y_pred) / (y_true + 1e-9)) ** 2))


def train_lightgbm(
    features_path: str = None,
    n_folds:       int = 5,
    use_nn_feats:  bool = True,
) -> dict:
    """
    Train LightGBM with GroupKFold on time_id (temporal split).

    Returns
    -------
    dict with oof_rmspe, test_rmspe, feature_importances
    """
    if features_path is None:
        # pick best available feature set
        cross_path = PROCESSED_DIR / "features_with_cross.parquet"
        nn_path    = PROCESSED_DIR / "features_with_nn_v2.parquet"
        base_path  = PROCESSED_DIR / "features.parquet"
        features_path = (
            cross_path if cross_path.exists() else
            nn_path    if (use_nn_feats and nn_path.exists()) else
            base_path
        )

    print(f"Loading: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"Shape: {df.shape}")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Features: {len(feat_cols)}")

    # target ratio transform: predict rv_target / rv_current instead of raw rv_target
    df["target_ratio"] = df["target"] / (df["rv_full"] + 1e-9)
    df["target_ratio"] = df["target_ratio"].clip(0.1, 10.0)

    # Temporal train/test split (85/15 su time_id ordinati)
    sorted_tids = sorted(df["time_id"].unique())
    cutoff      = sorted_tids[int(len(sorted_tids) * 0.85)]
    train_df    = df[df["time_id"] <= cutoff].copy()
    test_df     = df[df["time_id"] >  cutoff].copy()

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    X_train = train_df[feat_cols].values
    y_train = train_df["target_ratio"].values
    groups  = train_df["time_id"].values

    X_test  = test_df[feat_cols].values
    y_test  = test_df["target"].values

    # GroupKFold su time_id
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds   = np.zeros(len(train_df))
    test_preds  = np.zeros(len(test_df))
    models      = []

    print(f"\nTraining {n_folds}-fold LightGBM...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            callbacks=[lgb.log_evaluation(200)],
        )

        val_pred_ratio  = model.predict(Xval)
        test_pred_ratio = model.predict(X_test)

        # convert ratio back to absolute volatility
        val_rv_full  = train_df.iloc[val_idx]["rv_full"].values
        test_rv_full = test_df["rv_full"].values

        oof_preds[val_idx] = val_pred_ratio  * (val_rv_full  + 1e-9)
        test_preds         += test_pred_ratio * (test_rv_full + 1e-9) / n_folds

        models.append(model)
        print(f"  Fold {fold+1} done.")

    oof_y_true   = train_df["target"].values
    oof_rmspe    = rmspe(oof_y_true, oof_preds)
    test_rmspe   = rmspe(y_test, test_preds)

    ss_res = np.sum((y_test - test_preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    test_r2 = 1 - ss_res / (ss_tot + 1e-10)

    print(f"\nLightGBM results: OOF RMSPE={oof_rmspe:.5f}, "
          f"Test RMSPE={test_rmspe:.5f}, Test R²={test_r2:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature"   : feat_cols,
        "importance": models[-1].feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\nTop 10 features:\n{importance.head(10).to_string(index=False)}")
    importance.to_csv(PROCESSED_DIR / "lgb_importance.csv", index=False)

    # save predictions for ensemble
    train_df["lgb_pred"] = oof_preds
    test_df["lgb_pred"]  = test_preds
    train_df[["stock_id","time_id","target","lgb_pred"]].to_parquet(
        PROCESSED_DIR / "lgb_oof.parquet", index=False)
    test_df[["stock_id","time_id","target","lgb_pred"]].to_parquet(
        PROCESSED_DIR / "lgb_test.parquet", index=False)
    print("Saved OOF and test predictions.")

    return {
        "oof_rmspe" : oof_rmspe,
        "test_rmspe": test_rmspe,
        "test_r2"   : test_r2,
        "importance": importance,
    }


if __name__ == "__main__":
    train_lightgbm(use_nn_feats=True)