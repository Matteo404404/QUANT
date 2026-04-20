"""
risk_measures.py
================
Systemic Risk Measures for Financial Networks
----------------------------------------------
Implements three canonical systemic risk measures, each grounded
in peer-reviewed literature:

1. Aggregate Shortfall (AS)
   -------------------------
   The simplest measure. Sum of equity deficits across all defaulting
   banks after clearing. Directly available from ClearingResult.
   Used as a baseline regression target.

   Reference:
       Eisenberg & Noe (2001), Management Science 47(2).

2. DebtRank (DR)
   --------------
   A recursive, propagation-based measure of systemic importance.
   For each bank i, DR[i] quantifies the fraction of total system
   equity lost as a result of i's distress, accounting for second-
   and higher-order cascade effects through the network.

   The algorithm runs a discrete-time propagation on the network
   starting from a single shocked node (or a set of nodes), tracking
   how stress diffuses through liability linkages.

   Key property: DebtRank avoids double-counting by marking nodes
   as "inactive" once they have already propagated stress — unlike
   simple cascade models.

   Reference:
       Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G.
       (2012). "DebtRank: Too Central to Fail? Financial Networks, the FED
       and Systemic Risk." Scientific Reports, 2, 541.

3. Minimum Bailout Capital (MBC)
   --------------------------------
   The most sophisticated measure and primary GNN regression target.
   Given a random shock vector ξ ~ F_ξ, MBC is the minimum total
   capital injection m* such that the expected aggregate shortfall
   under optimal allocation stays below an acceptable level α:

       MBC(α) = min{ Σ_i c_i : E[AS(e + c - ξ)] ≤ α }

   where c ∈ R^n_+ is the capital allocation vector.

   This is a stochastic convex optimization problem. We approximate
   it via Monte Carlo sampling of shocks and solve using scipy.optimize.

   Connection to Biagini–Fouque–Frittelli–Meyer-Brandis framework:
   The MBC is a systemic risk measure in the sense of BFFMB (2015),
   satisfying monotonicity, translation invariance, and convexity.
   It maps a random vector of bank values to a scalar risk number.

   Reference:
       Biagini, F., Fouque, J-P., Frittelli, M., & Meyer-Brandis, T.
       (2015). "A Unified Approach to Systemic Risk Measures via
       Acceptance Sets." Mathematical Finance, 29(1), 329–367.

       Gonon, L., Herrera, C., Kruse, T., & Ritter, G. (2024).
       "Computing Systemic Risk Measures with Graph Neural Networks."
       arXiv:2410.07222.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from scipy.optimize import minimize

from src.clearing import compute_clearing, ClearingResult


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SystemicRiskResult:
    """
    Container for all systemic risk measures computed on one network instance.

    Attributes
    ----------
    aggregate_shortfall : float
        Sum of equity deficits across defaulting banks under a single
        deterministic shock. AS = Σ_{i: default} max(0, -equity[i]).

    debtrank : np.ndarray, shape (n,)
        DebtRank score for each bank. DR[i] ∈ [0, 1] is the fraction
        of total system equity lost due to bank i's initial distress.
        Higher = more systemically important.

    debtrank_total : float
        Σ_i DR[i] * (equity_i / total_equity): aggregate system-level
        DebtRank stress index (scalar summary).

    min_bailout_capital : float
        Minimum total capital injection (summed across all banks)
        such that E[AS(e + c* - ξ)] ≤ α, under optimal allocation c*.

    min_bailout_allocation : np.ndarray, shape (n,)
        The optimal per-bank capital allocation c* achieving MBC.

    clearing_result : ClearingResult
        The underlying clearing result for the reference shock.

    n_banks : int
        Number of banks in the system.
    """
    aggregate_shortfall:    float
    debtrank:               np.ndarray
    debtrank_total:         float
    min_bailout_capital:    float
    min_bailout_allocation: np.ndarray
    clearing_result:        ClearingResult
    n_banks:                int

    def __repr__(self) -> str:
        return (
            f"SystemicRiskResult("
            f"n_banks={self.n_banks}, "
            f"AS={self.aggregate_shortfall:.4f}, "
            f"DR_total={self.debtrank_total:.4f}, "
            f"MBC={self.min_bailout_capital:.4f})"
        )


# ---------------------------------------------------------------------------
# Measure 1: Aggregate Shortfall
# ---------------------------------------------------------------------------

def aggregate_shortfall(
    L:          np.ndarray,
    e:          np.ndarray,
    shock:      Optional[np.ndarray | float] = None,
    shock_mode: str = "proportional",
) -> float:
    """
    Compute the Aggregate Shortfall (AS) for a single shock realisation.

    AS = Σ_{i: equity[i] < 0} |equity[i]|
       = Σ_i max(0, p_bar[i] - e[i] - (Π^T p*)[i])

    This is the total capital deficit in the system after clearing.
    Equivalently, it is the minimum aggregate recapitalisation needed
    to make every bank exactly solvent ex-post.

    Parameters
    ----------
    L          : nominal liability matrix, shape (n, n)
    e          : external assets (pre-shock), shape (n,)
    shock      : optional shock applied to e
    shock_mode : "proportional" or "absolute"

    Returns
    -------
    float : aggregate shortfall ≥ 0
    """
    result = compute_clearing(L, e, shock=shock, shock_mode=shock_mode)
    return result.default_loss


def aggregate_shortfall_distribution(
    L:           np.ndarray,
    e:           np.ndarray,
    shocks:      np.ndarray,
    shock_mode:  str = "proportional",
) -> np.ndarray:
    """
    Compute AS for a batch of shock scenarios.

    Parameters
    ----------
    L          : nominal liability matrix, shape (n, n)
    e          : external assets (pre-shock), shape (n,)
    shocks     : shock scenarios, shape (n_scenarios, n) or (n_scenarios,)
    shock_mode : "proportional" or "absolute"

    Returns
    -------
    as_values : np.ndarray, shape (n_scenarios,)
        AS for each shock scenario.
    """
    shocks = np.atleast_2d(shocks)
    if shocks.ndim == 1:
        shocks = shocks[:, None] * np.ones((1, L.shape[0]))

    as_values = np.array([
        aggregate_shortfall(L, e, shock=s, shock_mode=shock_mode)
        for s in shocks
    ])
    return as_values


# ---------------------------------------------------------------------------
# Measure 2: DebtRank
# ---------------------------------------------------------------------------

def debtrank(
    L:            np.ndarray,
    e:            np.ndarray,
    initial_shock: Optional[np.ndarray | float] = None,
    shock_mode:    str = "proportional",
    epsilon:       float = 1e-10,
) -> tuple[np.ndarray, float]:
    """
    Compute DebtRank for each bank (Battiston et al. 2012).

    DebtRank measures the systemic importance of each bank by running
    a stress-propagation process on the interbank network.

    Algorithm (per initial shocked set):
    -------------------------------------
    Each bank i has:
        h_i(t) ∈ [0,1]  : stress level at time t (0=healthy, 1=fully distressed)
        s_i(t) ∈ {U, D, I}: state — Undistressed, Distressed, Inactive

    Propagation rule (from Battiston et al. 2012, Eq. 1):
        h_i(t+1) = min(1,  h_i(t) + Σ_{j: s_j=D} W_{ji} * h_j(t))

    where the impact weight W_{ji} = (L[j,i] * v_j^+) / (Σ_k v_k^+)
    captures: how exposed is j to i, weighted by j's systemic importance.

    After propagation, DR[i] = Σ_j h_j(T) * v_j^+ / Σ_j v_j^+
    where v_j^+ = max(0, equity_j) is the pre-shock positive equity.

    We compute DR for EACH bank i as the initial shocked set {i},
    running n independent propagations.

    Parameters
    ----------
    L             : nominal liability matrix, shape (n, n)
    e             : external assets, shape (n,)
    initial_shock : shock applied before computing pre-shock equity
                    (used to set baseline equity levels)
    shock_mode    : "proportional" or "absolute"
    epsilon       : numerical threshold for state transitions

    Returns
    -------
    dr_scores  : np.ndarray, shape (n,)
        DR[i] = fraction of total system equity lost if bank i is
        initially fully distressed.
    dr_total   : float
        Σ_i DR[i] * equity_weight[i]: system-level aggregate DR index.
    """
    n = L.shape[0]

    # --- Step 1: Compute pre-shock equity (baseline for weighting) ---
    result_base = compute_clearing(L, e)
    v = np.maximum(result_base.equity, 0.0)   # positive equity only
    V_total = v.sum()

    if V_total < 1e-12:
        # Degenerate: system already fully insolvent
        return np.ones(n), 1.0

    # --- Step 2: Impact weight matrix W ---
    # W[j, i] = fraction of j's equity at risk due to j's exposure to i
    # = (L[j,i] / p_bar_j) * (v[j] / V_total)  if p_bar_j > 0
    p_bar = L.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pi = np.where(p_bar[:, None] > 0, L / p_bar[:, None], 0.0)

    # W[j,i]: stress transmitted from j to i when j is distressed
    # W[j,i] = Pi[j,i] * v[j] / V_total
    W = Pi * (v[:, None] / V_total)    # shape (n, n)

    # --- Step 3: Run DebtRank propagation for each initial node ---
    dr_scores = np.zeros(n)

    for seed in range(n):
        h = np.zeros(n)    # stress levels
        s = np.zeros(n, dtype=int)   # 0=Undistressed, 1=Distressed, 2=Inactive

        # Seed node: fully distressed
        h[seed] = 1.0
        s[seed] = 1

        h_prev = h.copy()

        # Propagate until convergence (at most n steps)
        for _ in range(n):
            h_new = h.copy()

            for i in range(n):
                if s[i] == 0:    # Undistressed: can receive stress
                    # Sum stress from all currently Distressed neighbours
                    stress_in = np.sum(
                        W[:, i] * h_prev * (s == 1)
                    )
                    h_new[i] = min(1.0, h[i] + stress_in)

                    if h_new[i] > epsilon:
                        s[i] = 1   # becomes Distressed

            # Transition: previously Distressed → Inactive
            for i in range(n):
                if s[i] == 1 and i != seed:
                    # Check if this node was Distressed last round
                    if h_prev[i] > epsilon and h_new[i] >= h_prev[i] - epsilon:
                        s[i] = 2   # Inactive: already propagated

            delta = np.max(np.abs(h_new - h))
            h = h_new
            h_prev = h_new.copy()

            if delta < epsilon:
                break

        # DR[seed] = fraction of system equity lost = Σ_i h_i * v_i / V_total
        # Subtract initial stress of seed to get IMPACT (not self-stress)
        h_impact = h.copy()
        h_impact[seed] = 0.0   # remove seed's own contribution
        dr_scores[seed] = np.sum(h_impact * v) / V_total

    # Clip to [0,1] for numerical safety
    dr_scores = np.clip(dr_scores, 0.0, 1.0)

    # Aggregate system-level DR index (equity-weighted)
    equity_weights = v / V_total
    dr_total = float(np.sum(dr_scores * equity_weights))

    return dr_scores, dr_total


# ---------------------------------------------------------------------------
# Measure 3: Minimum Bailout Capital (MBC)
# ---------------------------------------------------------------------------

def minimum_bailout_capital(
    L:              np.ndarray,
    e:              np.ndarray,
    shock_sampler:  Callable[[], np.ndarray],
    n_samples:      int   = 500,
    alpha:          float = 0.0,
    method:         str   = "L-BFGS-B",
    tol:            float = 1e-6,
    seed:           Optional[int] = None,
) -> tuple[float, np.ndarray]:
    """
    Compute the Minimum Bailout Capital (MBC) via stochastic optimisation.

    MBC(α) = min_{c ≥ 0}  Σ_i c_i
             subject to:   (1/N) Σ_{k=1}^{N} AS(e + c - ξ^k) ≤ α

    where ξ^k ~ shock_sampler() are i.i.d. shock realisations.

    Interpretation
    --------------
    MBC is the minimum total capital injection (summed across banks) such
    that the AVERAGE aggregate shortfall across shock scenarios is at most α.
    The optimal allocation c* tells us WHERE to inject capital for maximum
    systemic risk reduction — a key regulatory insight.

    With α = 0: we want to eliminate expected shortfall entirely.
    With α > 0: we tolerate some expected loss (more realistic).

    Implementation
    --------------
    We solve the constrained problem by converting to unconstrained via
    a penalty / Lagrangian formulation:

        min_c  Σ_i c_i + λ * max(0, E[AS(e+c-ξ)] - α)^2

    where λ is a large penalty coefficient. We use L-BFGS-B with bounds
    c_i ≥ 0.

    Differentiability note: AS(·) is piecewise linear and not everywhere
    differentiable. We use scipy's gradient-free numerical differentiation
    via L-BFGS-B with finite differences, which works well in practice
    for this scale.

    Parameters
    ----------
    L             : nominal liability matrix, shape (n, n)
    e             : external assets (pre-shock), shape (n,)
    shock_sampler : callable returning shape (n,) shock vector per call
    n_samples     : number of Monte Carlo shock samples
    alpha         : acceptable expected shortfall level (default 0 = full elimination)
    method        : scipy.optimize method (default "L-BFGS-B")
    tol           : optimisation tolerance
    seed          : random seed for reproducibility

    Returns
    -------
    mbc    : float — minimum total capital injection Σ_i c*_i
    c_star : np.ndarray, shape (n,) — optimal per-bank capital allocation
    """
    rng = np.random.default_rng(seed)
    n   = len(e)

    # --- Pre-sample all shocks (fix randomness for stable gradient estimates) ---
    shocks = np.array([shock_sampler() for _ in range(n_samples)])  # (N, n)

    # --- Objective: total capital + penalty for constraint violation ---
    penalty_coeff = 1e4

    def objective(c: np.ndarray) -> float:
        c = np.maximum(c, 0.0)   # enforce non-negativity
        e_recapitalised = e + c

        # Monte Carlo estimate of E[AS(e + c - ξ)]
        as_values = np.array([
            compute_clearing(L, e_recapitalised - s, method="fictitious_default").default_loss
            for s in shocks
        ])
        expected_as = as_values.mean()

        # Penalised objective
        total_capital = c.sum()
        constraint_violation = max(0.0, expected_as - alpha)
        return total_capital + penalty_coeff * constraint_violation ** 2

    # --- Initial guess: zero capital injection ---
    c0     = np.zeros(n)
    bounds = [(0.0, None)] * n    # c_i ≥ 0

    result = minimize(
        objective,
        c0,
        method=method,
        bounds=bounds,
        options={"maxiter": 200, "ftol": tol, "gtol": tol * 0.1},
    )

    c_star = np.maximum(result.x, 0.0)
    mbc    = float(c_star.sum())

    return mbc, c_star


# ---------------------------------------------------------------------------
# Convenience: compute all three measures at once
# ---------------------------------------------------------------------------

def compute_all_risk_measures(
    L:              np.ndarray,
    e:              np.ndarray,
    shock:          Optional[np.ndarray | float] = None,
    shock_mode:     str   = "proportional",
    shock_sampler:  Optional[Callable[[], np.ndarray]] = None,
    n_mbc_samples:  int   = 300,
    alpha:          float = 0.0,
    compute_mbc:    bool  = True,
    seed:           Optional[int] = None,
) -> SystemicRiskResult:
    """
    Compute all three systemic risk measures for a financial network.

    Parameters
    ----------
    L              : nominal liability matrix, shape (n, n)
    e              : external assets (pre-shock), shape (n,)
    shock          : deterministic shock for AS computation
    shock_mode     : shock application mode
    shock_sampler  : callable returning shape (n,) shocks for MBC.
                     If None, defaults to uniform U[0, 0.3] proportional shock.
    n_mbc_samples  : MC samples for MBC estimation
    alpha          : MBC acceptance threshold
    compute_mbc    : if False, skip expensive MBC computation (set to NaN)
    seed           : random seed

    Returns
    -------
    SystemicRiskResult
    """
    n   = L.shape[0]
    rng = np.random.default_rng(seed)

    # --- Aggregate Shortfall ---
    cr = compute_clearing(L, e, shock=shock, shock_mode=shock_mode)
    as_val = cr.default_loss

    # --- DebtRank ---
    dr_scores, dr_total = debtrank(L, e)

    # --- Minimum Bailout Capital ---
    if compute_mbc:
        if shock_sampler is None:
            # Default: uniform proportional shocks in [0, 0.3] per bank
            def shock_sampler() -> np.ndarray:
                return rng.uniform(0.0, 0.3, size=n)

        mbc, c_star = minimum_bailout_capital(
            L, e,
            shock_sampler=shock_sampler,
            n_samples=n_mbc_samples,
            alpha=alpha,
            seed=seed,
        )
    else:
        mbc    = float("nan")
        c_star = np.full(n, float("nan"))

    return SystemicRiskResult(
        aggregate_shortfall    = as_val,
        debtrank               = dr_scores,
        debtrank_total         = dr_total,
        min_bailout_capital    = mbc,
        min_bailout_allocation = c_star,
        clearing_result        = cr,
        n_banks                = n,
    )