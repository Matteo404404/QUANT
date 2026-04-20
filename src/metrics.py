"""
metrics.py
==========
Shared metric functions used across the project.
Both numpy and torch versions so everything stays consistent.
"""

from __future__ import annotations

import numpy as np


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error (Optiver's official metric)."""
    return float(np.sqrt(np.mean(((y_true - y_pred) / (y_true + 1e-9)) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


try:
    import torch

    def rmspe_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """RMSPE for torch tensors."""
        return torch.sqrt(torch.mean(((target - pred) / (target + 1e-9)) ** 2))

    def r2_score_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
        """R² for torch tensors."""
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return 1.0 - (ss_res / (ss_tot + 1e-10)).item()

except ImportError:
    pass
