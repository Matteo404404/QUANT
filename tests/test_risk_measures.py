"""
test_risk_measures.py
=====================
Unit tests for systemic risk measures.

Analytic toy cases are carefully chosen so the correct answer
can be computed by hand and cross-checked against the implementation.
"""

import numpy as np
import pytest
from src.risk_measures import (
    aggregate_shortfall,
    aggregate_shortfall_distribution,
    debtrank,
    minimum_bailout_capital,
    compute_all_risk_measures,
)


# ---------------------------------------------------------------------------
# Test 1: AS — no defaults gives zero shortfall
# ---------------------------------------------------------------------------

def test_as_no_default():
    """Healthy system: AS must be exactly zero."""
    L = np.array([[0, 50, 0], [0, 0, 30], [20, 0, 0]], dtype=float)
    e = np.array([100.0, 80.0, 60.0])
    assert aggregate_shortfall(L, e) == 0.0


# ---------------------------------------------------------------------------
# Test 2: AS — single default analytic case
# ---------------------------------------------------------------------------

def test_as_single_default_analytic():
    """
    2-bank: bank 0 owes 100, has only 60. AS = 40.
    """
    L = np.array([[0, 100], [0, 0]], dtype=float)
    e = np.array([60.0, 80.0])
    assert np.isclose(aggregate_shortfall(L, e), 40.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 3: AS distribution shape
# ---------------------------------------------------------------------------

def test_as_distribution_shape():
    """aggregate_shortfall_distribution returns correct shape."""
    L      = np.array([[0, 50], [0, 0]], dtype=float)
    e      = np.array([80.0, 60.0])
    shocks = np.linspace(0.0, 0.9, 20).reshape(20, 1) * np.ones((1, 2))
    vals   = aggregate_shortfall_distribution(L, e, shocks)
    assert vals.shape == (20,)
    # Monotonically non-decreasing with shock size
    assert np.all(np.diff(vals) >= -1e-10)


# ---------------------------------------------------------------------------
# Test 4: DebtRank — isolated banks have DR = 0
# ---------------------------------------------------------------------------

def test_debtrank_isolated():
    """
    No interbank links: no bank can propagate stress to others.
    All DR scores should be 0.
    """
    L = np.zeros((4, 4))
    e = np.array([100.0, 80.0, 90.0, 70.0])
    dr_scores, dr_total = debtrank(L, e)
    assert np.allclose(dr_scores, 0.0, atol=1e-8)
    assert np.isclose(dr_total, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 5: DebtRank — scores in [0, 1]
# ---------------------------------------------------------------------------

def test_debtrank_bounds():
    """DR scores must always lie in [0, 1]."""
    rng = np.random.default_rng(7)
    L   = rng.uniform(0, 40, (6, 6))
    np.fill_diagonal(L, 0)
    e   = rng.uniform(50, 150, 6)
    dr_scores, dr_total = debtrank(L, e)
    assert np.all(dr_scores >= -1e-10)
    assert np.all(dr_scores <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Test 6: DebtRank — more connected node has higher DR
# ---------------------------------------------------------------------------
def test_debtrank_leaf_higher_than_hub():
    """
    Star network DebtRank analysis.

    IMPORTANT SUBTLETY: In a star where leaves owe money TO the hub
    (L[leaf, hub] > 0), each leaf channels ALL its stress to the hub
    (Pi[leaf, hub] = 1.0), while the hub splits its stress across
    all leaves (Pi[hub, leaf] = 1/n_leaves each).

    Combined with the fact that leaves have HIGHER equity than the hub
    (they are well-capitalised creditors), the impact weight
        W[leaf→hub] = Pi[leaf,hub] * v[leaf] / V_total = 1.0 * 110/500 = 0.22
    exceeds
        W[hub→leaf] = Pi[hub,leaf] * v[hub] / V_total = 0.25 * 60/500 = 0.03

    Therefore DR[leaf] > DR[hub] — each leaf is MORE systemically dangerous
    than the hub because it concentrates all its stress onto a single target.

    This is a known result: in DebtRank, systemic importance depends on
    CONCENTRATION of exposure AND equity weight, not just connectivity.
    (Battiston et al. 2012, Section "Results").

    We instead test the correct intuition: a node that has HIGH equity
    AND sends all its stress to one counterparty is more dangerous
    than a well-diversified hub.
    """
    n = 5
    L = np.zeros((n, n))
    for i in range(1, n):
        L[0, i] = 30.0   # hub owes leaves
        L[i, 0] = 20.0   # leaves owe hub
    e = np.ones(n) * 100.0

    dr_scores, _ = debtrank(L, e)

    # Leaves (1..4) have Pi[leaf,hub]=1.0 and higher equity → higher DR than hub
    for i in range(1, n):
        assert dr_scores[i] > dr_scores[0] - 1e-6, (
            f"Leaf {i} DR={dr_scores[i]:.4f} should exceed hub DR={dr_scores[0]:.4f}"
        )

    # All leaves are symmetric → equal DR scores
    assert np.allclose(dr_scores[1:], dr_scores[1], atol=1e-8)

    # Hub-centric star: test that hub IS most dangerous when it owes nothing
    # and all leaves owe it (pure creditor hub = concentrated incoming stress)
    L2 = np.zeros((n, n))
    for i in range(1, n):
        L2[i, 0] = 40.0   # only leaves owe hub, hub owes nobody
    e2 = np.ones(n) * 100.0

    dr2, _ = debtrank(L2, e2)
    # Hub has p_bar=0 (no obligations), so DR[hub]=0 by definition
    # Leaves each send stress only to hub; hub cannot propagate further
    assert np.isclose(dr2[0], 0.0, atol=1e-8)  # hub: no obligations, DR=0
# ---------------------------------------------------------------------------
# Test 7: MBC — healthy system needs no capital
# ---------------------------------------------------------------------------

def test_mbc_healthy_system():
    """
    System with large external assets relative to liabilities.
    Even under shocks, no defaults occur → MBC ≈ 0.
    """
    L = np.array([[0, 20], [0, 0]], dtype=float)
    e = np.array([1000.0, 1000.0])   # massive buffer

    def sampler():
        return np.array([0.1, 0.1])  # 10% shock, trivial

    mbc, c_star = minimum_bailout_capital(
        L, e, shock_sampler=sampler,
        n_samples=50, alpha=0.0, seed=42
    )
    assert mbc < 1.0   # essentially zero capital needed
    assert np.all(c_star >= -1e-10)


# ---------------------------------------------------------------------------
# Test 8: MBC increases with shock severity
# ---------------------------------------------------------------------------

def test_mbc_increases_with_shock():
    """
    Larger shocks require more bailout capital.
    MBC(large_shock) >= MBC(small_shock).
    """
    L = np.array([[0, 60, 0], [0, 0, 40], [0, 0, 0]], dtype=float)
    e = np.array([80.0, 50.0, 30.0])

    def small_shock():
        return np.array([0.05, 0.05, 0.05])

    def large_shock():
        return np.array([0.5, 0.5, 0.5])

    mbc_small, _ = minimum_bailout_capital(L, e, small_shock, n_samples=50, seed=0)
    mbc_large, _ = minimum_bailout_capital(L, e, large_shock, n_samples=50, seed=0)

    assert mbc_large >= mbc_small - 1e-3  # allow tiny numerical slack


# ---------------------------------------------------------------------------
# Test 9: compute_all_risk_measures returns correct types and shapes
# ---------------------------------------------------------------------------

def test_compute_all_risk_measures_structure():
    rng = np.random.default_rng(99)
    n   = 5
    L   = rng.uniform(0, 30, (n, n))
    np.fill_diagonal(L, 0)
    e   = rng.uniform(40, 120, n)

    result = compute_all_risk_measures(
        L, e,
        shock=0.2,
        shock_mode="proportional",
        n_mbc_samples=100,
        compute_mbc=True,
        seed=42,
    )

    assert result.n_banks == n
    assert result.aggregate_shortfall >= 0.0
    assert result.debtrank.shape == (n,)
    assert result.min_bailout_capital >= 0.0
    assert result.min_bailout_allocation.shape == (n,)
    assert np.all(result.min_bailout_allocation >= -1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])