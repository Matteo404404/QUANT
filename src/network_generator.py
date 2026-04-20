"""
network_generator.py
====================
Synthetic financial network generators for systemic risk experiments.

We construct three families of directed liability networks, each motivated
by empirical findings in the interbank literature:

1. Erdős–Rényi (ER)
   -----------------
   Baseline random graph G(n, p) with independent edges.
   Captures homogeneous random connectivity.

2. Core–Periphery (CP)
   --------------------
   Small, densely connected core and large, sparsely connected periphery.
   Supported by empirical work on interbank markets:

       Craig, B., & von Peter, G. (2014).
       "Interbank tiering and money center banks."
       Journal of Financial Intermediation, 23(3), 322–347.

3. Scale-Free / Preferential Attachment (BA)
   -----------------------------------------
   Barabási–Albert-type growth with preferential attachment.
   Generates heavy-tailed degree distributions, approximating
   "too-connected-to-fail" hubs.

For each topology, we generate a nominal liability matrix L and an
external asset vector e, calibrated such that:

- Liabilities per bank follow a lognormal distribution (fat-tailed).
- External assets are proportional to outgoing liabilities with
  a configurable leverage factor.

These generators are stochastic but fully controllable via a random seed.

The outputs are designed to feed directly into `compute_all_risk_measures`
from `risk_measures.py` and then into the GNN dataset.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional


NetworkType = Literal["erdos_renyi", "core_periphery", "barabasi_albert"]


@dataclass
class FinancialNetwork:
    """Container for a generated financial network.

    Attributes
    ----------
    L : np.ndarray, shape (n, n)
        Nominal liability matrix. L[i,j] ≥ 0 is amount bank i owes bank j.
        We always ensure diag(L) = 0.

    e : np.ndarray, shape (n,)
        External asset vector. e[i] ≥ 0.

    adjacency : np.ndarray, shape (n, n)
        Binary adjacency matrix (0/1) derived from L: A[i,j] = 1 if L[i,j] > 0.

    network_type : str
        One of {"erdos_renyi", "core_periphery", "barabasi_albert"}.

    metadata : dict
        Additional generation parameters (e.g. p, m, core_size).
    """
    L: np.ndarray
    e: np.ndarray
    adjacency: np.ndarray
    network_type: str
    metadata: dict

    @property
    def n_banks(self) -> int:
        return self.L.shape[0]


# ---------------------------------------------------------------------------
# Helper: sampling liabilities and external assets given adjacency
# ---------------------------------------------------------------------------

def _sample_liabilities_and_assets(
    A: np.ndarray,
    exposure_scale: float = 10.0,
    exposure_lognorm_sigma: float = 0.5,
    leverage_mean: float = 1.5,       # <-- was 0.08, now 1.5
    leverage_sigma: float = 0.3,      # <-- was 0.02, now 0.3
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Given adjacency A, sample nominal liabilities L and external assets e.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Binary adjacency. A[i,j] = 1 indicates potential liability i→j.
    exposure_scale : float
        Sets the typical liability size (in arbitrary units).
    exposure_lognorm_sigma : float
        Volatility of lognormal liability sizes.
    leverage_mean, leverage_sigma : float
        External assets e[i] are sampled as:
            e[i] = (1 / leverage_i) * total_outgoing_liabilities_i
        where leverage_i ~ N(leverage_mean, leverage_sigma).
        Smaller leverage → larger external assets for same liabilities.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        Nominal liabilities.
    e : np.ndarray, shape (n,)
        External assets.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    # Sample liability sizes on each active edge from a lognormal
    log_exposure = rng.normal(
        loc=np.log(exposure_scale) - 0.5 * exposure_lognorm_sigma**2,
        scale=exposure_lognorm_sigma,
        size=(n, n),
    )
    L_raw = np.exp(log_exposure) * A

    # Ensure exact zeros where there is no edge and zero diagonal
    L = L_raw * A
    np.fill_diagonal(L, 0.0)

    # Compute total outgoing nominal liabilities per bank
    total_out = L.sum(axis=1)

    # Sample leverage per bank and compute external assets
    leverage = rng.normal(leverage_mean, leverage_sigma, size=n)
    # Avoid degenerate or negative leverage
    leverage = np.clip(leverage, 0.01, None)

    e = total_out / leverage
    return L, e


# ---------------------------------------------------------------------------
# Generator 1: Erdős–Rényi
# ---------------------------------------------------------------------------

def generate_erdos_renyi_network(
    n: int,
    p: float = 0.1,
    exposure_scale: float = 10.0,
    seed: Optional[int] = None,
) -> FinancialNetwork:
    """Generate an Erdős–Rényi directed liability network.

    Parameters
    ----------
    n : int
        Number of banks.
    p : float
        Edge probability for directed edges i→j (i ≠ j).
    exposure_scale : float
        Typical liability size.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    FinancialNetwork
    """
    rng = np.random.default_rng(seed)

    # Sample adjacency (no self-loops)
    A = rng.random((n, n)) < p
    np.fill_diagonal(A, 0)
    A = A.astype(float)

    L, e = _sample_liabilities_and_assets(A, exposure_scale=exposure_scale, rng=rng)

    return FinancialNetwork(
        L=L,
        e=e,
        adjacency=A,
        network_type="erdos_renyi",
        metadata={"n": n, "p": p, "exposure_scale": exposure_scale, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Generator 2: Core–Periphery
# ---------------------------------------------------------------------------

def generate_core_periphery_network(
    n_core: int,
    n_periphery: int,
    p_cc: float = 0.8,
    p_cp: float = 0.5,
    p_pp: float = 0.1,
    exposure_scale: float = 10.0,
    seed: Optional[int] = None,
) -> FinancialNetwork:
    """Generate a core–periphery directed liability network.

    Banks 0..(n_core-1) are core; the rest are periphery.

    Edge probabilities:
        core → core      : p_cc
        core → periphery : p_cp
        periphery → core : p_cp
        periphery → perp.: p_pp

    Parameters
    ----------
    n_core : int
        Number of core banks.
    n_periphery : int
        Number of periphery banks.
    p_cc, p_cp, p_pp : float
        Edge probabilities between respective tiers.
    exposure_scale : float
        Typical liability size.
    seed : int, optional
        RNG seed.

    Returns
    -------
    FinancialNetwork
    """
    rng = np.random.default_rng(seed)
    n = n_core + n_periphery

    A = np.zeros((n, n), dtype=float)

    core_idx = np.arange(n_core)
    peri_idx = np.arange(n_core, n)

    # Core → core
    if n_core > 1:
        A_cc = rng.random((n_core, n_core)) < p_cc
        np.fill_diagonal(A_cc, 0)
        A[np.ix_(core_idx, core_idx)] = A_cc

    # Core ↔ periphery
    if n_core > 0 and n_periphery > 0:
        A_cp = rng.random((n_core, n_periphery)) < p_cp
        A_pc = rng.random((n_periphery, n_core)) < p_cp
        A[np.ix_(core_idx, peri_idx)] = A_cp
        A[np.ix_(peri_idx, core_idx)] = A_pc

    # Periphery → periphery
    if n_periphery > 1:
        A_pp = rng.random((n_periphery, n_periphery)) < p_pp
        np.fill_diagonal(A_pp, 0)
        A[np.ix_(peri_idx, peri_idx)] = A_pp

    np.fill_diagonal(A, 0)

    L, e = _sample_liabilities_and_assets(A, exposure_scale=exposure_scale, rng=rng)

    return FinancialNetwork(
        L=L,
        e=e,
        adjacency=A,
        network_type="core_periphery",
        metadata={
            "n_core": n_core,
            "n_periphery": n_periphery,
            "p_cc": p_cc,
            "p_cp": p_cp,
            "p_pp": p_pp,
            "exposure_scale": exposure_scale,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Generator 3: Barabási–Albert (preferential attachment)
# ---------------------------------------------------------------------------

def generate_barabasi_albert_network(
    n: int,
    m: int = 2,
    exposure_scale: float = 10.0,
    seed: Optional[int] = None,
) -> FinancialNetwork:
    """Generate a directed Barabási–Albert (BA) preferential attachment network.

    We construct an undirected BA graph for degree distribution via
    preferential attachment, then orient edges randomly to obtain a
    directed liability matrix.

    Parameters
    ----------
    n : int
        Total number of banks (nodes). Must satisfy n > m ≥ 1.
    m : int
        Number of edges each new node attaches to existing nodes.
    exposure_scale : float
        Typical liability size.
    seed : int, optional
        RNG seed.

    Returns
    -------
    FinancialNetwork
    """
    if n <= m:
        raise ValueError("Require n > m >= 1 for BA model.")

    rng = np.random.default_rng(seed)

    # Start with a fully connected seed graph of size m+1
    A_undirected = np.zeros((n, n), dtype=float)

    # Fully connect first (m+1) nodes (no self-loops)
    A_undirected[: m + 1, : m + 1] = 1.0
    np.fill_diagonal(A_undirected, 0.0)

    # Preferential attachment: add nodes one by one
    degree = A_undirected.sum(axis=0)

    for new_node in range(m + 1, n):
        # Attach to m existing nodes, probability ∝ degree
        degree_sum = degree.sum()
        if degree_sum == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = degree[:new_node] / degree_sum

        targets = rng.choice(new_node, size=m, replace=False, p=probs)

        for t in targets:
            A_undirected[new_node, t] = 1.0
            A_undirected[t, new_node] = 1.0

        degree = A_undirected.sum(axis=0)

    # Now orient each undirected edge randomly to obtain a directed adjacency
    A_directed = np.zeros_like(A_undirected)
    for i in range(n):
        for j in range(i + 1, n):
            if A_undirected[i, j] == 1.0:
                # randomly choose direction i→j or j→i
                if rng.random() < 0.5:
                    A_directed[i, j] = 1.0
                else:
                    A_directed[j, i] = 1.0

    L, e = _sample_liabilities_and_assets(A_directed, exposure_scale=exposure_scale, rng=rng)

    return FinancialNetwork(
        L=L,
        e=e,
        adjacency=A_directed,
        network_type="barabasi_albert",
        metadata={"n": n, "m": m, "exposure_scale": exposure_scale, "seed": seed},
    )


# ---------------------------------------------------------------------------
# High-level factory
# ---------------------------------------------------------------------------

def generate_financial_network(
    network_type: NetworkType,
    **kwargs,
) -> FinancialNetwork:
    """Factory to generate a financial network of a given type.

    Parameters
    ----------
    network_type : {"erdos_renyi", "core_periphery", "barabasi_albert"}
        Type of network to generate.
    **kwargs : dict
        Forwarded to the corresponding generator.

    Returns
    -------
    FinancialNetwork
    """
    if network_type == "erdos_renyi":
        return generate_erdos_renyi_network(**kwargs)
    elif network_type == "core_periphery":
        return generate_core_periphery_network(**kwargs)
    elif network_type == "barabasi_albert":
        return generate_barabasi_albert_network(**kwargs)
    else:
        raise ValueError(f"Unknown network_type: {network_type}")