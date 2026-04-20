"""
clearing.py
===========
Eisenberg–Noe Clearing Mechanism
---------------------------------
Implements the fixed-point clearing algorithm from:

    Eisenberg, L., & Noe, T. H. (2001).
    "Systemic Risk in Financial Networks."
    Management Science, 47(2), 236–249.

Mathematical Setup
------------------
Consider a financial system of n banks. Define:

    L  ∈ R^{n×n}_+   : nominal liability matrix, L[i,j] = amount bank i owes to bank j
    e  ∈ R^n_+        : vector of external (non-interbank) assets
    p̄  ∈ R^n_+        : vector of total nominal obligations, p̄[i] = Σ_j L[i,j]

The relative liability matrix Π is defined as:
    Π[i,j] = L[i,j] / p̄[i]   if p̄[i] > 0
    Π[i,j] = 0                 otherwise

Clearing Payment Vector
-----------------------
The clearing payment vector p* ∈ [0, p̄] satisfies:

    p*[i] = min( p̄[i],  e[i] + Σ_j Π[j,i] * p*[j] )

Existence & Uniqueness
----------------------
Eisenberg & Noe (2001) Theorem 1 guarantees existence of a greatest
and a least clearing vector. We compute the GREATEST clearing vector
via the fictitious default algorithm (Algorithm 1 in the paper).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ClearingResult:
    """
    Container for Eisenberg–Noe clearing output.

    Attributes
    ----------
    payments : np.ndarray, shape (n,)
        Greatest clearing payment vector p*.
        payments[i] ∈ [0, p_bar[i]].

    equity : np.ndarray, shape (n,)
        Post-clearing equity v*[i] = e[i] + (Π^T @ payments)[i] - p_bar[i].
        Negative equity → bank is in default.

    defaults : np.ndarray of bool, shape (n,)
        defaults[i] = True iff bank i defaults.

    default_loss : float
        Aggregate equity deficit: Σ_{i: defaults[i]} |equity[i]|.
        Primary regression target for the GNN.

    n_defaults : int
        Number of defaulting banks.

    converged : bool
        Whether solver converged (fixed_point only).

    n_iterations : int
        Number of iterations taken.
    """
    payments:      np.ndarray
    equity:        np.ndarray
    defaults:      np.ndarray
    default_loss:  float
    n_defaults:    int
    converged:     bool
    n_iterations:  int

    def __repr__(self) -> str:
        return (
            f"ClearingResult("
            f"n_banks={len(self.payments)}, "
            f"n_defaults={self.n_defaults}, "
            f"default_loss={self.default_loss:.4f}, "
            f"converged={self.converged}, "
            f"iters={self.n_iterations})"
        )


# ---------------------------------------------------------------------------
# Core utility functions
# ---------------------------------------------------------------------------

def compute_relative_liabilities(
    L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute relative liability matrix Π and total obligation vector p̄.

    Parameters
    ----------
    L : np.ndarray, shape (n, n)
        Nominal liability matrix. L[i,j] ≥ 0, diagonal ignored.

    Returns
    -------
    Pi : np.ndarray, shape (n, n)
        Pi[i,j] = L[i,j] / p̄[i]  if p̄[i] > 0 else 0.
    p_bar : np.ndarray, shape (n,)
        p_bar[i] = Σ_j L[i,j].
    """
    L = np.array(L, dtype=np.float64)
    np.fill_diagonal(L, 0.0)

    p_bar = L.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        Pi = np.where(p_bar[:, None] > 0, L / p_bar[:, None], 0.0)

    return Pi, p_bar


def _compute_equity(
    payments: np.ndarray,
    Pi:       np.ndarray,
    p_bar:    np.ndarray,
    e:        np.ndarray,
) -> np.ndarray:
    """
    Post-clearing equity:
        v[i] = e[i] + (Π^T @ payments)[i] - p_bar[i]
    """
    return e + Pi.T @ payments - p_bar


# ---------------------------------------------------------------------------
# Solver 1: Fixed-point iteration
# ---------------------------------------------------------------------------

def clearing_fixed_point(
    L:        np.ndarray,
    e:        np.ndarray,
    max_iter: int   = 1000,
    tol:      float = 1e-10,
) -> ClearingResult:
    """
    Greatest clearing vector via fixed-point iteration.

    Iteration (monotonically decreasing from p̄):
        p^{k+1}[i] = min( p̄[i], max(0, e[i] + (Π^T @ p^k)[i]) )

    Converges to the greatest clearing vector by Tarski's fixed-point
    theorem (lattice monotone map on a complete lattice).

    Parameters
    ----------
    L        : nominal liability matrix, shape (n, n)
    e        : external assets, shape (n,)
    max_iter : iteration cap
    tol      : L∞ convergence threshold

    Returns
    -------
    ClearingResult
    """
    L = np.array(L, dtype=np.float64)
    e = np.array(e, dtype=np.float64)
    n = len(e)

    if L.shape != (n, n):
        raise ValueError(f"L must be ({n},{n}), got {L.shape}")
    if np.any(L < 0):
        raise ValueError("Liability matrix L must be non-negative.")
    if np.any(e < 0):
        warnings.warn("External assets contain negative values.")

    Pi, p_bar = compute_relative_liabilities(L)
    p = p_bar.copy()   # start: everyone pays in full

    converged = False
    delta = np.inf
    for k in range(max_iter):
        receivables = Pi.T @ p
        p_new = np.minimum(p_bar, np.maximum(0.0, e + receivables))
        delta = np.max(np.abs(p_new - p))
        p = p_new
        if delta < tol:
            converged = True
            n_iters = k + 1
            break
    else:
        n_iters = max_iter
        warnings.warn(
            f"Fixed-point did not converge in {max_iter} iters. "
            f"Final Δ = {delta:.2e}."
        )

    equity   = _compute_equity(p, Pi, p_bar, e)
    defaults = equity < -1e-12

    return ClearingResult(
        payments     = p,
        equity       = equity,
        defaults     = defaults,
        default_loss = float(np.sum(np.abs(equity[defaults]))),
        n_defaults   = int(defaults.sum()),
        converged    = converged,
        n_iterations = n_iters,
    )


# ---------------------------------------------------------------------------
# Solver 2: Fictitious Default Algorithm (Eisenberg & Noe 2001, Algorithm 1)
# ---------------------------------------------------------------------------

def clearing_fictitious_default(
    L: np.ndarray,
    e: np.ndarray,
) -> ClearingResult:
    """
    Exact greatest clearing vector via fictitious default algorithm.

    This is the PREFERRED solver. It terminates in at most n+1 rounds
    and is guaranteed to find the unique greatest clearing vector p*.

    Algorithm
    ---------
    1.  Initialise default set D = ∅ (assume all solvent).
    2.  Compute payments: solvent banks pay p̄[i], defaulting banks
        pay all available assets (solve linear system).
    3.  Recompute equity v[i] = e[i] + (Π^T p)[i] - p̄[i].
    4.  Update D ← { i : v[i] < 0 }.
    5.  If D unchanged → stop. Else → go to 2.

    Parameters
    ----------
    L : nominal liability matrix, shape (n, n)
    e : external assets, shape (n,)

    Returns
    -------
    ClearingResult
    """
    L = np.array(L, dtype=np.float64)
    e = np.array(e, dtype=np.float64)
    n = len(e)

    Pi, p_bar = compute_relative_liabilities(L)
    D = np.zeros(n, dtype=bool)

    n_iters = 0
    for _ in range(n + 1):
        n_iters += 1
        p      = _solve_clearing_given_defaults(Pi, p_bar, e, D)
        equity = _compute_equity(p, Pi, p_bar, e)
        D_new  = equity < -1e-12

        if np.array_equal(D_new, D):
            break
        D = D_new
    else:
        warnings.warn("Fictitious default algorithm did not stabilise.")

    equity   = _compute_equity(p, Pi, p_bar, e)
    defaults = equity < -1e-12

    return ClearingResult(
        payments     = p,
        equity       = equity,
        defaults     = defaults,
        default_loss = float(np.sum(np.abs(equity[defaults]))),
        n_defaults   = int(defaults.sum()),
        converged    = True,
        n_iterations = n_iters,
    )


def _solve_clearing_given_defaults(
    Pi:    np.ndarray,
    p_bar: np.ndarray,
    e:     np.ndarray,
    D:     np.ndarray,
) -> np.ndarray:
    """
    Given a fixed default set D, solve for clearing payments.

    Solvent banks (i ∉ D):   p[i] = p̄[i]
    Defaulting banks (i ∈ D): p[i] = e[i] + Σ_j Pi[j,i]*p[j]

    Rearranged as a linear system for p[D]:
        (I - Pi_DD^T) @ p[D] = e[D] + Pi_SD^T @ p̄[S]

    where S = ~D (solvent set).
    """
    n     = len(p_bar)
    p     = p_bar.copy()

    if not np.any(D):
        return p

    idx_D = np.where(D)[0]
    idx_S = np.where(~D)[0]
    Pi_T  = Pi.T

    # RHS: external assets + receivables from solvent banks
    rhs = e[D] + Pi_T[np.ix_(idx_D, idx_S)] @ p_bar[idx_S]

    # Coefficient matrix
    A = np.eye(len(idx_D)) - Pi_T[np.ix_(idx_D, idx_D)]

    try:
        p_D = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        p_D, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)

    p[D] = np.clip(p_D, 0.0, p_bar[D])
    return p


# ---------------------------------------------------------------------------
# Shock utilities
# ---------------------------------------------------------------------------

def apply_shock(
    e:     np.ndarray,
    shock: np.ndarray | float,
    mode:  str = "proportional",
) -> np.ndarray:
    """
    Apply an external asset shock to produce post-shock asset vector.

    Parameters
    ----------
    e     : pre-shock external assets, shape (n,)
    shock : scalar or array of shock magnitudes
    mode  :
        "absolute"     → e_shocked[i] = e[i] - shock[i]
        "proportional" → e_shocked[i] = e[i] * (1 - shock[i])

    Returns
    -------
    e_shocked : np.ndarray, shape (n,)
        Note: NOT clipped to 0 — negative values are valid
        (they represent banks that are already underwater).
    """
    e     = np.array(e, dtype=np.float64)
    shock = np.broadcast_to(
        np.asarray(shock, dtype=np.float64), e.shape
    ).copy()

    if mode == "absolute":
        return e - shock
    elif mode == "proportional":
        return e * (1.0 - shock)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'absolute' or 'proportional'."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_clearing(
    L:          np.ndarray,
    e:          np.ndarray,
    shock:      Optional[np.ndarray | float] = None,
    shock_mode: str   = "proportional",
    method:     str   = "fictitious_default",
    max_iter:   int   = 1000,
    tol:        float = 1e-10,
) -> ClearingResult:
    """
    Compute Eisenberg–Noe clearing for a financial network.

    This is the primary user-facing function.

    Parameters
    ----------
    L          : nominal liability matrix, shape (n, n)
    e          : external assets (pre-shock), shape (n,)
    shock      : optional shock to apply to e before clearing
    shock_mode : "proportional" or "absolute"
    method     : "fictitious_default" (exact, default) or "fixed_point"
    max_iter   : iteration cap (fixed_point only)
    tol        : convergence threshold (fixed_point only)

    Returns
    -------
    ClearingResult

    Examples
    --------
    >>> import numpy as np
    >>> L = np.array([[0, 50, 0],
    ...               [0,  0, 30],
    ...               [20, 0,  0]], dtype=float)
    >>> e = np.array([40.0, 20.0, 10.0])
    >>> result = compute_clearing(L, e)
    >>> result
    ClearingResult(n_banks=3, n_defaults=0, default_loss=0.0000, ...)
    """
    e = np.array(e, dtype=np.float64)

    if shock is not None:
        e = apply_shock(e, shock, mode=shock_mode)

    if method == "fictitious_default":
        return clearing_fictitious_default(L, e)
    elif method == "fixed_point":
        return clearing_fixed_point(L, e, max_iter=max_iter, tol=tol)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Use 'fictitious_default' or 'fixed_point'."
        )