"""
train.py
========
Training Loop for Systemic Risk GNN Models
-------------------------------------------
Includes per-batch target normalisation to fix scale issues.
"""

import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.models.mlp_baseline  import MLPBaseline
from src.models.gnn_sage_gat  import GraphSAGEModel, GATModel
from src.models.penn           import PENN


# ---------------------------------------------------------------------------
# Target extractor
# ---------------------------------------------------------------------------

TARGET_OPTIONS = {
    "as":  lambda d: d.y_as,
    "mbc": lambda d: d.y_mbc,
    "dr":  lambda d: d.y_dr,
}


# ---------------------------------------------------------------------------
# Target statistics (computed once on training set)
# ---------------------------------------------------------------------------

def compute_target_stats(
    data_list,
    target: str,
) -> tuple[float, float]:
    """Compute mean and std of the target across the training set."""
    get_target = TARGET_OPTIONS[target]
    all_vals = []
    for d in data_list:
        all_vals.append(get_target(d).reshape(-1))
    vals  = torch.cat(all_vals)
    mean  = vals.mean().item()
    std   = vals.std().item()
    std   = max(std, 1e-6)   # avoid div by zero
    return mean, std


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimiser,
    target:      str,
    device:      torch.device,
    target_mean: float,
    target_std:  float,
) -> float:
    model.train()
    total_loss = 0.0
    get_target = TARGET_OPTIONS[target]

    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()

        pred = model(batch)
        y    = get_target(batch)

        # Normalise targets to zero mean, unit std
        y_norm = (y - target_mean) / target_std

        if target == "dr":
            pred = pred.squeeze(-1)

        loss = nn.functional.mse_loss(pred.squeeze(-1), y_norm.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    target:      str,
    device:      torch.device,
    target_mean: float = 0.0,
    target_std:  float = 1.0,
) -> dict:
    """Evaluate model. Returns MSE/MAE/R² in ORIGINAL (unnormalised) scale."""
    model.eval()
    get_target = TARGET_OPTIONS[target]
    preds_all, y_all = [], []

    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch)
        y     = get_target(batch)

        if target == "dr":
            pred = pred.squeeze(-1)

        # Denormalise predictions back to original scale
        pred_orig = pred.squeeze(-1) * target_std + target_mean

        preds_all.append(pred_orig.cpu())
        y_all.append(y.squeeze(-1).cpu())

    preds = torch.cat(preds_all)
    y     = torch.cat(y_all)

    mse = nn.functional.mse_loss(preds, y).item()
    mae = nn.functional.l1_loss(preds, y).item()

    ss_res = ((y - preds) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    return {"mse": mse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_data,
    val_data,
    target:       str   = "as",
    epochs:       int   = 200,
    batch_size:   int   = 32,
    lr:           float = 3e-4,
    weight_decay: float = 1e-4,
    patience:     int   = 25,
    device_str:   str   = "auto",
    save_path:    Optional[str] = None,
) -> dict:
    assert target in TARGET_OPTIONS, f"target must be one of {list(TARGET_OPTIONS)}"

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Compute target normalisation stats on training set
    t_mean, t_std = compute_target_stats(train_data, target)

    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )
    stopper = EarlyStopping(patience=patience)

    history = {
        "train_loss": [], "val_mse": [], "val_mae": [], "val_r2": [],
        "lr": [], "best_val_mse": float("inf"), "best_epoch": 0,
        "target_mean": t_mean, "target_std": t_std,
    }
    best_state = None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model.__class__.__name__} | Target: {target.upper()} | "
          f"Params: {n_params:,} | Device: {device}")
    print(f"Target stats: mean={t_mean:.5f}, std={t_std:.5f}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss  = train_epoch(
            model, train_loader, optimiser, target, device, t_mean, t_std
        )
        val_metrics = evaluate(
            model, val_loader, target, device, t_mean, t_std
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])
        history["lr"].append(optimiser.param_groups[0]["lr"])

        if val_metrics["mse"] < history["best_val_mse"]:
            history["best_val_mse"] = val_metrics["mse"]
            history["best_epoch"]   = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:>4d} | "
                f"TrainLoss {train_loss:.5f} | "
                f"ValMSE {val_metrics['mse']:.5f} | "
                f"R² {val_metrics['r2']:.4f} | "
                f"{elapsed:.1f}s"
            )

        if stopper.step(val_metrics["mse"]):
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best epoch: {history['best_epoch']})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - t0
    history["total_time_s"] = total_time
    print(f"\nDone in {total_time:.1f}s. "
          f"Best ValMSE: {history['best_val_mse']:.5f}, "
          f"Best R²: {max(history['val_r2']):.4f}")

    return history