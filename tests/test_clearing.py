"""
test_clearing.py
================
Unit tests for the Eisenberg–Noe clearing module.

Tests are designed around analytically tractable toy cases where
the clearing solution can be computed by hand, ensuring mathematical
correctness before any large-scale experiments.
"""

import numpy as np
import pytest
from src.clearing import (
    compute_clearing,
    compute_relative_liabilities,
    apply_shock,
    ClearingResult,
)


# ---------------------------------------------------------------------------
# Test 1: No defaults (healthy system)
# ---------------------------------------------------------------------------

def test_no_default_small_system():
    """
    3-bank ring network, all banks healthy.
    Bank 0 owes 50 to Bank 1
    Bank 1 owes 30 to Bank 2
    Bank 2 owes 20 to Bank 0
    External assets all large enough to cover obligations.
    Expected: no defaults, all banks pay in full.
    """
    L = np.array([
        [0, 50,  0],
        [0,  0, 30],
        [20, 0,  0],
    ], dtype=float)
    e = np.array([100.0, 80.0, 60.0])

    result = compute_clearing(L, e)

    assert result.n_defaults == 0
    assert np.allclose(result.default_loss, 0.0)
    # All banks pay in full
    p_bar = L.sum(axis=1)
    assert np.allclose(result.payments, p_bar, atol=1e-8)
    # All equity non-negative
    assert np.all(result.equity >= -1e-10)


# ---------------------------------------------------------------------------
# Test 2: Single isolated default
# ---------------------------------------------------------------------------

def test_single_default_no_cascade():
    """
    2-bank system. Bank 0 owes 100 to Bank 1 but has only 60 in assets.
    Bank 1 has no liabilities. No cascade possible.

    Analytic solution:
        p*[0] = e[0] = 60   (pays all it has)
        p*[1] = 0            (no obligations)
        equity[0] = 60 - 100 = -40  (default)
        equity[1] = 80 + 60 - 0 = 140  (solvent, receives partial payment)
    """
    L = np.array([
        [0, 100],
        [0,   0],
    ], dtype=float)
    e = np.array([60.0, 80.0])

    result = compute_clearing(L, e)

    assert result.n_defaults == 1
    assert result.defaults[0] == True
    assert result.defaults[1] == False
    assert np.isclose(result.payments[0], 60.0, atol=1e-8)
    assert np.isclose(result.payments[1], 0.0,  atol=1e-8)
    assert np.isclose(result.equity[0], -40.0,  atol=1e-8)
    assert np.isclose(result.default_loss, 40.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 3: Cascade default
# ---------------------------------------------------------------------------

def test_cascade_default():
    """
    3-bank chain: 0 → 1 → 2
    Bank 0 owes 100 to Bank 1.
    Bank 1 owes 80 to Bank 2.
    Bank 0 has only 40 in external assets → defaults.
    This reduces Bank 1's assets → Bank 1 may also default.

    Analytic solution:
        p*[0] = 40  (all of e[0])
        Bank 1 total assets = e[1] + p*[0] = 20 + 40 = 60 < 80 → defaults
        p*[1] = 60
        Bank 2 equity = e[2] + p*[1] - 0 = 30 + 60 = 90 > 0 → solvent
    """
    L = np.array([
        [0, 100,  0],
        [0,   0, 80],
        [0,   0,  0],
    ], dtype=float)
    e = np.array([40.0, 20.0, 30.0])

    result = compute_clearing(L, e)

    assert result.n_defaults == 2
    assert result.defaults[0] == True
    assert result.defaults[1] == True
    assert result.defaults[2] == False
    assert np.isclose(result.payments[0], 40.0, atol=1e-8)
    assert np.isclose(result.payments[1], 60.0, atol=1e-8)
    # Loss = |equity[0]| + |equity[1]| = 60 + 20 = 80
    assert np.isclose(result.default_loss, 80.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 4: Both solvers agree
# ---------------------------------------------------------------------------

def test_both_solvers_agree():
    """
    On a random network, both fixed_point and fictitious_default
    must return the same clearing vector (up to tolerance).
    """
    rng = np.random.default_rng(42)
    n   = 10
    L   = rng.uniform(0, 50, (n, n))
    np.fill_diagonal(L, 0)
    e   = rng.uniform(20, 100, n)

    r_fp = compute_clearing(L, e, method="fixed_point")
    r_fd = compute_clearing(L, e, method="fictitious_default")

    assert np.allclose(r_fp.payments, r_fd.payments, atol=1e-6), (
        f"Solvers disagree:\n FP:  {r_fp.payments}\n FD:  {r_fd.payments}"
    )
    assert np.allclose(r_fp.equity, r_fd.equity, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Shock application
# ---------------------------------------------------------------------------

def test_proportional_shock():
    """
    A 50% proportional shock to all banks halves their external assets.
    """
    e       = np.array([100.0, 200.0, 150.0])
    shocked = apply_shock(e, shock=0.5, mode="proportional")
    assert np.allclose(shocked, [50.0, 100.0, 75.0])


def test_absolute_shock():
    e       = np.array([100.0, 200.0, 150.0])
    shocked = apply_shock(e, shock=np.array([10.0, 20.0, 30.0]), mode="absolute")
    assert np.allclose(shocked, [90.0, 180.0, 120.0])


# ---------------------------------------------------------------------------
# Test 6: Relative liability matrix properties
# ---------------------------------------------------------------------------

def test_relative_liabilities_row_sum():
    """
    Each row of Pi must sum to 1 (if bank has obligations) or 0.
    """
    L  = np.array([[0, 30, 20], [10, 0, 40], [0, 0, 0]], dtype=float)
    Pi, p_bar = compute_relative_liabilities(L)

    for i in range(3):
        if p_bar[i] > 0:
            assert np.isclose(Pi[i].sum(), 1.0), f"Row {i} of Pi does not sum to 1."
        else:
            assert np.isclose(Pi[i].sum(), 0.0), f"Row {i} of Pi should be zero."


# ---------------------------------------------------------------------------
# Test 7: No liabilities system
# ---------------------------------------------------------------------------

def test_no_interbank_liabilities():
    """
    Banks with no interbank liabilities. Clearing is trivial:
    all payments are zero, equity = external assets.
    """
    L = np.zeros((4, 4))
    e = np.array([50.0, 30.0, 0.0, -10.0])

    result = compute_clearing(L, e)

    assert np.allclose(result.payments, 0.0)
    assert np.allclose(result.equity, e)
    assert result.n_defaults == 1   # only bank 3 (e=-10) defaults


# ---------------------------------------------------------------------------
# Test 8: Full default cascade (all banks)
# ---------------------------------------------------------------------------

def test_full_system_collapse():
    """
    Acyclic chain 0 -> 1 -> 2 -> (external creditor, modelled via p_bar).
    ALL three banks have obligations AND zero external assets.

    Key: every bank must have p_bar[i] > 0, otherwise it has
    no obligations and cannot default by definition.

    Network:
        Bank 0 owes 80 to Bank 1
        Bank 1 owes 60 to Bank 2
        Bank 2 owes 40 to external (modelled as self-loop removed,
        so we give Bank 2 obligations by having it owe Bank 0)
        But to keep it acyclic we use a 4-bank chain instead.

    4-bank acyclic chain: 0->1->2->3
    e = [0, 0, 0, 0]: all external assets wiped.
    Bank 0 pays nothing (e=0, no receivables from anyone).
    Cascade: 1 gets nothing from 0 → defaults, 2 gets nothing → defaults.
    Bank 3 has no obligations (end of chain) → solvent with equity=0.
    Banks 0,1,2 all default. Bank 3 does NOT (no obligations).
    """
    L = np.array([
        [0, 80,  0,  0],
        [0,  0, 60,  0],
        [0,  0,  0, 40],
        [0,  0,  0,  0],   # bank 3: pure creditor, no obligations
    ], dtype=float)
    e = np.zeros(4)

    result = compute_clearing(L, e)

    # Banks 0, 1, 2 default. Bank 3 has p_bar=0, equity=0, solvent.
    assert result.defaults[0] == True
    assert result.defaults[1] == True
    assert result.defaults[2] == True
    assert result.defaults[3] == False   # no obligations → cannot default
    assert result.n_defaults == 3
    assert np.allclose(result.payments[:3], 0.0, atol=1e-8)


def test_cyclic_network_zero_assets_partial_solvency():
    """
    Cyclic network with e=0 demonstrates that interbank payment
    recycling can sustain partial solvency — Pi has eigenvalue 1.
    This is a known property of Eisenberg-Noe on strongly connected
    networks (see E&N 2001, Remark after Theorem 1).
    We assert the algorithm converges and produces a valid result
    (payments in [0, p_bar], equity computed correctly).
    """
    L = np.array([
        [0,  80, 20],
        [50,  0, 30],
        [40, 60,  0],
    ], dtype=float)
    e = np.zeros(3)

    result = compute_clearing(L, e)

    p_bar = L.sum(axis=1)

    # Payments must be in [0, p_bar]
    assert np.all(result.payments >= -1e-10)
    assert np.all(result.payments <= p_bar + 1e-10)
    # At least some defaults (can't all be solvent with e=0)
    assert result.n_defaults >= 1
    # Equity consistency: v = e + Pi^T p - p_bar
    Pi = L / p_bar[:, None]
    equity_check = e + Pi.T @ result.payments - p_bar
    assert np.allclose(result.equity, equity_check, atol=1e-8)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])