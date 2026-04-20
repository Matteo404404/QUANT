"""
stacking.py
===========
Level-2 stacking: Ridge meta-learner on OOF predictions.

Meta-learner features:
  [lgb_pred, gnn_pred, nn_tgt_wmean_c, nn_tgt_wmean_e, cs_tgt_mean,
   rv_full, nn_dist_ratio_c]

Target: rv_target
Training: Ridge regression with CV on time_id (same split as level 1).
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.lob_features import PROCESSED_DIR
from src.metrics import rmspe, r2_score as r2


def train_meta_learner(use_cross_features: bool = True) -> dict:
    """
    Load OOF and test predictions from all models and train
    a level-2 Ridge meta-learner.
    """
    # load best available feature set
    feat_path = PROCESSED_DIR / "features_with_cross.parquet"
    if not feat_path.exists():
        feat_path = PROCESSED_DIR / "features_with_nn_v2.parquet"
    if not feat_path.exists():
        feat_path = PROCESSED_DIR / "features_with_nn.parquet"

    print(f"Loading features: {feat_path}")
    feat_df = pd.read_parquet(feat_path)

    # same temporal split as baseline
    sorted_tids = sorted(feat_df["time_id"].unique())
    cutoff      = sorted_tids[int(len(sorted_tids) * 0.85)]
    train_feat  = feat_df[feat_df["time_id"] <= cutoff].copy()
    test_feat   = feat_df[feat_df["time_id"] >  cutoff].copy()

    # load LGB predictions
    lgb_oof  = pd.read_parquet(PROCESSED_DIR / "lgb_oof.parquet")
    lgb_test = pd.read_parquet(PROCESSED_DIR / "lgb_test.parquet")

    train_df = train_feat.merge(
        lgb_oof[["stock_id","time_id","lgb_pred"]], on=["stock_id","time_id"], how="left"
    )
    test_df  = test_feat.merge(
        lgb_test[["stock_id","time_id","lgb_pred"]], on=["stock_id","time_id"], how="left"
    )

    # meta-learner feature set
    meta_cols = ["lgb_pred", "rv_full"]

    nn_v2_cols = ["nn_tgt_wmean_c","nn_tgt_wmean_e","nn_dist_ratio_c","nn_dist_ratio_e"]
    meta_cols += [c for c in nn_v2_cols if c in train_df.columns]
    if use_cross_features:
        cs_cols = ["cs_rv_mean","cs_rv_std","cs_tgt_mean","cs_corr_mean"]
        meta_cols += [c for c in cs_cols if c in train_df.columns]

    meta_cols = [c for c in meta_cols if c in train_df.columns]
    print(f"Meta-learner features: {meta_cols}")

    # drop NaN
    train_df = train_df.dropna(subset=meta_cols + ["target"])
    test_df  = test_df.dropna(subset=meta_cols + ["target"])

    X_train = train_df[meta_cols].values
    y_train = train_df["target"].values
    groups  = train_df["time_id"].values

    X_test  = test_df[meta_cols].values
    y_test  = test_df["target"].values

    # scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # RidgeCV with GroupKFold on time_id
    gkf        = GroupKFold(n_splits=5)
    oof_preds  = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    print("Training Ridge meta-learner (5-fold)...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        model = RidgeCV(alphas=alphas, cv=3)
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_preds[val_idx] = model.predict(X_train[val_idx])
        test_preds        += model.predict(X_test) / 5
        print(f"  Fold {fold+1} | alpha={model.alpha_:.4f} | "
              f"val RMSPE={rmspe(y_train[val_idx], oof_preds[val_idx]):.5f}")

    oof_rmspe  = rmspe(y_train, oof_preds)
    test_rmspe = rmspe(y_test,  test_preds)
    test_r2    = r2(y_test, test_preds)

    # compare with LGB-only baseline
    lgb_test_rmspe = rmspe(y_test, test_df["lgb_pred"].values)

    print(f"\nMeta-learner results: OOF RMSPE={oof_rmspe:.5f}, "
          f"Test RMSPE={test_rmspe:.5f}, Test R²={test_r2:.4f}")
    print(f"LGB-only baseline: {lgb_test_rmspe:.5f} | "
          f"Improvement: {(lgb_test_rmspe - test_rmspe):.5f}")

    # save meta predictions
    test_df["meta_pred"] = test_preds
    test_df[["stock_id","time_id","target","lgb_pred","meta_pred"]].to_parquet(
        PROCESSED_DIR / "meta_test.parquet", index=False
    )
    print("Saved to data/processed/meta_test.parquet")

    return {
        "oof_rmspe" : oof_rmspe,
        "test_rmspe": test_rmspe,
        "test_r2"   : test_r2,
        "lgb_rmspe" : lgb_test_rmspe,
    }


if __name__ == "__main__":
    train_meta_learner(use_cross_features=True)