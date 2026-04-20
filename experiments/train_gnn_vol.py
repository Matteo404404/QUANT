"""
train_gnn_vol.py
================
Train and compare GATv2 / SAGE / GIN on the dynamic stock correlation graph.

Usage
-----
python experiments/train_gnn_vol.py
python experiments/train_gnn_vol.py --model gatv2 --hidden 128 --epochs 100
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lob_features   import PROCESSED_DIR
from src.features.correlation_graph import build_graphs, NODE_FEATURE_COLS
from src.models.gnn_volatility    import build_model
from src.metrics import rmspe_torch as _rmspe_torch, r2_score_torch
from train import EarlyStopping


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    models      = ["sage"],
    hidden      = 64,
    n_layers    = 3,
    dropout     = 0.2,
    lr          = 3e-4,
    weight_decay= 1e-5,
    epochs      = 100,
    patience    = 15,
    batch_size  = 32,
    window      = 50,
    threshold   = 0.3,
    seed        = 42,
    train_frac  = 0.7,
    val_frac    = 0.15,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmspe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _rmspe_torch(pred, target)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return r2_score_torch(pred, target)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        pred = model(batch)
        loss = nn.HuberLoss()(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, y_mean: float, y_std: float):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.append(pred.cpu())
        targets.append(batch.y.cpu())

    preds   = torch.cat(preds)
    targets = torch.cat(targets)

    # Denormalise
    preds_raw   = preds   * y_std + y_mean
    targets_raw = targets * y_std + y_mean

    mse   = nn.MSELoss()(preds_raw, targets_raw).item()
    mae   = nn.L1Loss()(preds_raw, targets_raw).item()
    rmspe_val = rmspe(preds_raw, targets_raw).item()
    r2    = r2_score(preds_raw, targets_raw)

    return {"MSE": mse, "MAE": mae, "RMSPE": rmspe_val, "R2": r2}


# ---------------------------------------------------------------------------
# Single model training run
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    train_loader, val_loader, test_loader,
    in_channels: int,
    cfg: dict,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> dict:

    model = build_model(
        model_name,
        in_channels = in_channels,
        hidden      = cfg["hidden"],
        n_layers    = cfg["n_layers"],
        dropout     = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg["epochs"], eta_min=1e-6
    )

    print(f"\nModel: {model_name.upper()} | Params: {n_params:,} | Device: {device}")

    best_val_rmspe = float("inf")
    best_state     = None
    stopper        = EarlyStopping(patience=cfg["patience"])
    t0             = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimiser, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, device, y_mean, y_std)
            elapsed = time.time() - t0

            print(
                f"  Epoch {epoch:4d} | "
                f"Loss {train_loss:.5f} | "
                f"ValRMSPE {val_metrics['RMSPE']:.5f} | "
                f"R² {val_metrics['R2']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics["RMSPE"] < best_val_rmspe:
                best_val_rmspe = val_metrics["RMSPE"]
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if stopper.step(val_metrics["RMSPE"]):
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, y_mean, y_std)

    print(f"\n  Test RMSPE : {test_metrics['RMSPE']:.5f}")
    print(f"  Test R²    : {test_metrics['R2']:.4f}")
    print(f"  Test MAE   : {test_metrics['MAE']:.6f}")

    # Save checkpoint
    ckpt_path = Path(f"{model_name}_vol_best.pt")
    torch.save({"model_state": best_state, "cfg": cfg,
                "y_mean": y_mean, "y_std": y_std}, ckpt_path)
    print(f"  Saved to {ckpt_path}")

    return {"model": model_name, "params": n_params,
            **{f"test_{k}": v for k, v in test_metrics.items()}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load features ---
    feat_path = PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {feat_path}\n"
            "Run: python src/features/lob_features.py first"
        )

    print(f"Loading features from {feat_path}...")
    features_df = pd.read_parquet(feat_path)
    print(f"  Shape: {features_df.shape} | "
          f"Stocks: {features_df.stock_id.nunique()} | "
          f"TimeIDs: {features_df.time_id.nunique()}")

    # --- Build graphs ---
    print(f"\nBuilding correlation graphs (window={cfg['window']}, "
          f"threshold={cfg['threshold']})...")
    graphs = build_graphs(
        features_df,
        window    = cfg["window"],
        threshold = cfg["threshold"],
    )

    if len(graphs) == 0:
        raise RuntimeError("No graphs built — check window/threshold parameters.")

    # --- Temporal train/val/test split (NO shuffling — time matters) ---
    n      = len(graphs)
    n_train = int(n * cfg["train_frac"])
    n_val   = int(n * cfg["val_frac"])

    train_graphs = graphs[:n_train]
    val_graphs   = graphs[n_train: n_train + n_val]
    test_graphs  = graphs[n_train + n_val:]

    print(f"\nTemporal split: train={len(train_graphs)} | "
          f"val: {len(val_graphs)} | test: {len(test_graphs)}")

    # --- Normalise targets (fit on train only) ---
    all_train_y = torch.cat([g.y for g in train_graphs])
    y_mean = all_train_y.mean().item()
    y_std  = all_train_y.std().item()
    for g in graphs:
        g.y = (g.y - y_mean) / (y_std + 1e-8)

    print(f"Target normalised: mean={y_mean:.6f}, std={y_std:.6f}")

    # --- DataLoaders ---
    train_loader = DataLoader(train_graphs, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=cfg["batch_size"], shuffle=False)

    # --- Feature dimension ---
    in_channels = train_graphs[0].x.shape[1]
    print(f"Node feature dim: {in_channels}")

    # --- Train all models ---
    results = []
    for model_name in cfg["models"]:
        res = train_model(
            model_name,
            train_loader, val_loader, test_loader,
            in_channels, cfg, device, y_mean, y_std,
        )
        results.append(res)

    # --- Final comparison table ---
    print(f"\nFinal comparison (Optiver Realized Volatility, node-level):")
    print(f"  {'Model':<10} {'Params':>10} {'RMSPE':>10} {'R²':>8} {'MAE':>12}")
    for r in sorted(results, key=lambda x: x["test_RMSPE"]):
        print(
            f"  {r['model']:<10} "
            f"{r['params']:>10,} "
            f"{r['test_RMSPE']:>10.5f} "
            f"{r['test_R2']:>8.4f} "
            f"{r['test_MAE']:>12.6f}"
        )

    # Save results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_gnn_vol.csv", index=False)
    print("\nResults saved to results_gnn_vol.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",    nargs="+", default=DEFAULTS["models"])
    parser.add_argument("--hidden",    type=int,   default=DEFAULTS["hidden"])
    parser.add_argument("--n_layers",  type=int,   default=DEFAULTS["n_layers"])
    parser.add_argument("--dropout",   type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--lr",        type=float, default=DEFAULTS["lr"])
    parser.add_argument("--epochs",    type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--patience",  type=int,   default=DEFAULTS["patience"])
    parser.add_argument("--batch_size",type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--window",    type=int,   default=DEFAULTS["window"])
    parser.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--seed",      type=int,   default=DEFAULTS["seed"])
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    main(cfg)