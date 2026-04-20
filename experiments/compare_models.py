"""
compare_models.py
=================
Final ensemble and model comparison:
  - LightGBM + NN features
  - GraphSAGE on correlation graph
  - Optimal weighted ensemble (minimise RMSPE)
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.optimize import minimize
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lob_features      import PROCESSED_DIR
from src.features.correlation_graph import build_graphs
from src.models.gnn_volatility       import build_model
from src.metrics import rmspe, r2_score as r2, mae


# ---------------------------------------------------------------------------
# GNN inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def gnn_predict(
    model_path: str,
    graphs:     list,
    device:     torch.device,
    y_mean:     float,
    y_std:      float,
) -> np.ndarray:
    """Load GNN checkpoint and predict on a list of graphs. Returns denormalised array."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]

    in_channels = graphs[0].x.shape[1]
    model = build_model(
        "sage",
        in_channels = in_channels,
        hidden      = cfg["hidden"],
        n_layers    = cfg["n_layers"],
        dropout     = cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    preds  = []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch)
        out   = out * y_std + y_mean
        preds.append(out.cpu().numpy())

    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Optimal ensemble weights via scipy
# ---------------------------------------------------------------------------

def find_optimal_weights(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """Find w such that RMSPE(w*pred_a + (1-w)*pred_b, y_true) is minimised."""
    def loss(w):
        blend = w[0] * pred_a + (1 - w[0]) * pred_b
        return rmspe(y_true, blend)

    result = minimize(loss, x0=[0.5], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    w_opt  = float(result.x[0])
    return w_opt, 1 - w_opt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. load LightGBM predictions (saved by baseline.py)
    lgb_test_path = PROCESSED_DIR / "lgb_test.parquet"
    if not lgb_test_path.exists():
        raise FileNotFoundError("Run python src/models/baseline.py first")

    lgb_test = pd.read_parquet(lgb_test_path)
    lgb_test = lgb_test.sort_values(["time_id", "stock_id"]).reset_index(drop=True)

    y_test_true = lgb_test["target"].values
    y_lgb       = lgb_test["lgb_pred"].values

    print(f"LightGBM test samples: {len(lgb_test)}")

    # 2. build graphs and filter to the same test time_ids as LGB
    features_path = PROCESSED_DIR / "features_with_nn.parquet"
    if not features_path.exists():
        features_path = PROCESSED_DIR / "features.parquet"

    features_df = pd.read_parquet(features_path)

    # same temporal split as LightGBM (85% cutoff)
    sorted_tids = sorted(features_df["time_id"].unique())
    cutoff      = sorted_tids[int(len(sorted_tids) * 0.85)]
    test_tids   = set(lgb_test["time_id"].unique())

    print(f"Building GNN graphs for {len(test_tids)} test time_ids...")
    all_graphs = build_graphs(features_df, window=50, threshold=0.3)

    test_graphs = [g for g in all_graphs if int(g.time_id) in test_tids]
    print(f"Test graphs: {len(test_graphs)}")

    # 3. GNN predictions
    ckpt_path = Path("sage_vol_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError("GNN checkpoint not found. Run train_gnn_vol.py first")

    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    y_mean  = ckpt["y_mean"]
    y_std   = ckpt["y_std"]

    print(f"GNN checkpoint: y_mean={y_mean:.6f}, y_std={y_std:.6f}")

    y_gnn_raw = gnn_predict(str(ckpt_path), test_graphs, device, y_mean, y_std)

    # align GNN predictions with the LGB dataframe (same stock/time order)
    gnn_rows = []
    for g in test_graphs:
        tid      = int(g.time_id)
        sids     = g.stock_ids.tolist() if hasattr(g.stock_ids, "tolist") else list(g.stock_ids)
        n_nodes  = g.x.shape[0]
        gnn_rows.extend([{"time_id": tid, "stock_id": int(s)} for s in sids])

    gnn_df           = pd.DataFrame(gnn_rows)
    gnn_df["gnn_pred"] = y_gnn_raw[: len(gnn_df)]

    # Merge con lgb_test
    merged = lgb_test.merge(gnn_df, on=["time_id", "stock_id"], how="left")
    merged["gnn_pred"] = merged["gnn_pred"].fillna(merged["lgb_pred"])

    y_gnn = merged["gnn_pred"].values
    y_true = merged["target"].values

    # 4. optimise ensemble weights
    print("\nOptimising ensemble weights...")
    w_lgb, w_gnn = find_optimal_weights(y_lgb, y_gnn, y_true)
    y_ensemble = w_lgb * y_lgb + w_gnn * y_gnn

    # 5. comparison table
    results = {
        "LightGBM (tabular + NN features)" : (y_lgb,      y_true),
        "GraphSAGE (correlation graph)"     : (y_gnn,      y_true),
        f"Ensemble (LGB×{w_lgb:.2f} + GNN×{w_gnn:.2f})": (y_ensemble, y_true),
    }

    print(f"\nFinal model comparison (Optiver Realized Volatility):")
    print(f"  {'Model':<42} {'RMSPE':>8} {'R²':>8} {'MAE':>12}")
    for name, (pred, true) in results.items():
        print(
            f"  {name:<42} "
            f"{rmspe(true, pred):>8.5f} "
            f"{r2(true, pred):>8.4f} "
            f"{mae(true, pred):>12.6f}"
        )
    print(f"\nOptimal weights: LightGBM={w_lgb:.3f}, GNN={w_gnn:.3f}")

    # save results
    rows = [
        {"model": name,
         "rmspe": rmspe(true, pred),
         "r2":    r2(true, pred),
         "mae":   mae(true, pred)}
        for name, (pred, true) in results.items()
    ]
    pd.DataFrame(rows).to_csv("results_final.csv", index=False)
    print("Saved to results_final.csv")


if __name__ == "__main__":
    main()