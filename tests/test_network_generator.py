"""
test_network_generator.py
=========================
Unit tests for synthetic financial network generators.

We test structural properties (topology correctness) and
financial properties (non-negative liabilities, valid assets).
"""

import numpy as np
import pytest
from src.network_generator import (
    generate_erdos_renyi_network,
    generate_core_periphery_network,
    generate_barabasi_albert_network,
    generate_financial_network,
    FinancialNetwork,
)


# ---------------------------------------------------------------------------
# Shared structural validators
# ---------------------------------------------------------------------------

def assert_valid_financial_network(net: FinancialNetwork):
    """Invariants that must hold for every generated network."""
    n = net.n_banks
    # Shape consistency
    assert net.L.shape == (n, n)
    assert net.e.shape == (n,)
    assert net.adjacency.shape == (n, n)
    # No self-loops
    assert np.allclose(np.diag(net.L), 0.0)
    assert np.allclose(np.diag(net.adjacency), 0.0)
    # Non-negativity
    assert np.all(net.L >= 0.0)
    assert np.all(net.e >= 0.0)
    assert np.all(net.adjacency >= 0.0)
    # Adjacency is binary
    assert set(np.unique(net.adjacency)).issubset({0.0, 1.0})
    # L is consistent with adjacency: L[i,j] > 0 iff adjacency[i,j] = 1
    assert np.all((net.L > 0) == (net.adjacency == 1))


# ---------------------------------------------------------------------------
# Erdős–Rényi tests
# ---------------------------------------------------------------------------

def test_er_structure():
    net = generate_erdos_renyi_network(n=20, p=0.2, seed=0)
    assert net.network_type == "erdos_renyi"
    assert net.n_banks == 20
    assert_valid_financial_network(net)


def test_er_edge_density():
    """Empirical edge density should be close to p."""
    n, p = 50, 0.3
    net = generate_erdos_renyi_network(n=n, p=p, seed=42)
    n_possible = n * (n - 1)
    density = net.adjacency.sum() / n_possible
    # Allow ±0.1 tolerance for finite sample
    assert abs(density - p) < 0.10, f"Density {density:.3f} far from p={p}"


def test_er_reproducible():
    """Same seed must produce identical networks."""
    net1 = generate_erdos_renyi_network(n=15, p=0.2, seed=7)
    net2 = generate_erdos_renyi_network(n=15, p=0.2, seed=7)
    assert np.allclose(net1.L, net2.L)
    assert np.allclose(net1.e, net2.e)


def test_er_different_seeds():
    """Different seeds must produce different networks."""
    net1 = generate_erdos_renyi_network(n=15, p=0.2, seed=1)
    net2 = generate_erdos_renyi_network(n=15, p=0.2, seed=2)
    assert not np.allclose(net1.L, net2.L)


def test_er_no_edges_p0():
    """p=0 → no edges → L is all zeros."""
    net = generate_erdos_renyi_network(n=10, p=0.0, seed=0)
    assert np.allclose(net.L, 0.0)
    assert np.allclose(net.adjacency, 0.0)


# ---------------------------------------------------------------------------
# Core–Periphery tests
# ---------------------------------------------------------------------------

def test_cp_structure():
    net = generate_core_periphery_network(
        n_core=5, n_periphery=15, seed=0
    )
    assert net.network_type == "core_periphery"
    assert net.n_banks == 20
    assert_valid_financial_network(net)


def test_cp_core_denser_than_periphery():
    """
    Core–core block density must exceed periphery–periphery density.
    """
    net = generate_core_periphery_network(
        n_core=5, n_periphery=20,
        p_cc=0.9, p_cp=0.4, p_pp=0.05,
        seed=0,
    )
    nc, np_ = 5, 20
    A = net.adjacency

    # Core–core block (excluding diagonal)
    cc = A[:nc, :nc]
    cc_density = (cc.sum()) / (nc * (nc - 1)) if nc > 1 else 0.0

    # Periphery–periphery block (excluding diagonal)
    pp = A[nc:, nc:]
    pp_density = (pp.sum()) / (np_ * (np_ - 1)) if np_ > 1 else 0.0

    assert cc_density > pp_density, (
        f"Core density {cc_density:.3f} should exceed periphery {pp_density:.3f}"
    )


def test_cp_metadata():
    net = generate_core_periphery_network(n_core=3, n_periphery=7, seed=1)
    assert net.metadata["n_core"] == 3
    assert net.metadata["n_periphery"] == 7


# ---------------------------------------------------------------------------
# Barabási–Albert tests
# ---------------------------------------------------------------------------

def test_ba_structure():
    net = generate_barabasi_albert_network(n=20, m=2, seed=0)
    assert net.network_type == "barabasi_albert"
    assert net.n_banks == 20
    assert_valid_financial_network(net)


def test_ba_degree_distribution_heavy_tail():
    """
    BA networks should produce a heavy-tailed degree distribution.
    The max degree should significantly exceed the mean degree.
    """
    net = generate_barabasi_albert_network(n=100, m=2, seed=42)
    total_degree = net.adjacency.sum(axis=0) + net.adjacency.sum(axis=1)
    assert total_degree.max() > 3 * total_degree.mean(), (
        "BA degree distribution should be heavy-tailed"
    )


def test_ba_invalid_m_raises():
    with pytest.raises(ValueError):
        generate_barabasi_albert_network(n=3, m=5)


def test_ba_reproducible():
    net1 = generate_barabasi_albert_network(n=15, m=2, seed=99)
    net2 = generate_barabasi_albert_network(n=15, m=2, seed=99)
    assert np.allclose(net1.L, net2.L)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

def test_factory_all_types():
    for ntype, kwargs in [
        ("erdos_renyi",    {"n": 10, "p": 0.2, "seed": 0}),
        ("core_periphery", {"n_core": 3, "n_periphery": 7, "seed": 0}),
        ("barabasi_albert",{"n": 10, "m": 2, "seed": 0}),
    ]:
        net = generate_financial_network(ntype, **kwargs)
        assert net.network_type == ntype
        assert_valid_financial_network(net)


def test_factory_invalid_type():
    with pytest.raises(ValueError):
        generate_financial_network("random_walk", n=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])