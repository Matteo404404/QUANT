"""
dataset.py
==========
GNN Dataset Builder for Systemic Risk Prediction
-------------------------------------------------
Bridges network generation + risk measure computation
into a PyTorch Geometric Dataset ready for GNN training.

Each dataset instance is a PyG Data object:

    data.x          : node features,  shape (n, F)
    data.edge_index : COO edge index, shape (2, E)
    data.edge_attr  : edge features,  shape (E, 1)  [liability weight]
    data.y          : regression targets, shape (n_targets,)
    data.n_banks    : scalar int

Node features (F=4):
    [0] e_i                    — external assets (normalised)
    [1] p_bar_i                — total obligations (normalised)
    [2] in_degree_i            — in-degree (normalised)
    [3] out_degree_i           — out-degree (normalised)

Targets (configurable, default: 3):
    [0] aggregate_shortfall    — scalar for whole graph (broadcast to all nodes)
    [1] debtrank_i             — per-node DebtRank score
    [2] min_bailout_capital    — scalar for whole graph (broadcast to all nodes)

All features are normalised per-instance (z-score on node features,
raw values on targets — normalisation of targets done in training loop).

References
----------
Gonon, L., Herrera, C., Kruse, T., & Ritter, G. (2024).
"Computing Systemic Risk Measures with Graph Neural Networks."
arXiv:2410.07222.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, Callable, List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from src.network_generator import (
    FinancialNetwork,
    generate_financial_network,
    NetworkType,
)
from src.risk_measures import compute_all_risk_measures


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def network_to_pyg(
    net: FinancialNetwork,
    risk_result,
    normalise_features: bool = True,
) -> Data:
    """Convert a FinancialNetwork + SystemicRiskResult into a PyG Data object.

    Parameters
    ----------
    net            : FinancialNetwork
    risk_result    : SystemicRiskResult from compute_all_risk_measures
    normalise_features : if True, z-score normalise node features per graph

    Returns
    -------
    torch_geometric.data.Data
    """
    n = net.n_banks
    L = net.L
    e = net.e

    p_bar     = L.sum(axis=1)
    in_deg    = (net.adjacency > 0).sum(axis=0).astype(float)
    out_deg   = (net.adjacency > 0).sum(axis=1).astype(float)

    # --- Node feature matrix (n, 4) ---
    X = np.stack([e, p_bar, in_deg, out_deg], axis=1).astype(np.float32)

    if normalise_features:
        mu  = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X   = (X - mu) / std

    # --- Edge index + edge attributes ---
    rows, cols = np.nonzero(L)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = L[rows, cols].astype(np.float32).reshape(-1, 1)

    # Normalise edge weights per graph
    if edge_attr.shape[0] > 0:
        ea_max = edge_attr.max() + 1e-8
        edge_attr = edge_attr / ea_max

    # --- Target vector ---
    # Scalar targets broadcast to graph-level; per-node targets are vectors
    as_scalar = float(risk_result.aggregate_shortfall)
    mbc_scalar = float(risk_result.min_bailout_capital)
    dr_vector  = risk_result.debtrank.astype(np.float32)   # shape (n,)

    # We store as a dict of tensors for flexibility
    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float),
        # Scalar targets (graph-level)
        y_as       = torch.tensor([as_scalar],  dtype=torch.float),
        y_mbc      = torch.tensor([mbc_scalar], dtype=torch.float),
        # Node-level target
        y_dr       = torch.tensor(dr_vector, dtype=torch.float),
        # Metadata
        n_banks      = torch.tensor([n], dtype=torch.long),
        network_type = net.network_type,
    )

    return data


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class SystemicRiskDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset of synthetic financial networks
    with precomputed systemic risk measure labels.

    Parameters
    ----------
    root : str
        Root directory where the processed dataset is stored.
    n_samples : int
        Number of network instances to generate.
    network_types : list of NetworkType
        Which topologies to include. Instances are drawn uniformly
        from this list.
    n_banks_range : tuple (min, max)
        Range for number of banks per network (sampled uniformly).
    shock : float
        Proportional shock applied to external assets for AS computation.
    n_mbc_samples : int
        Monte Carlo samples for MBC approximation.
    compute_mbc : bool
        If False, skip MBC (faster, useful for large-scale pretraining).
    seed : int, optional
        Master random seed.
    transform, pre_transform : callable, optional
        Standard PyG transforms.
    """

    def __init__(
        self,
        root: str,
        n_samples: int = 1000,
        network_types: Optional[List[NetworkType]] = None,
        n_banks_range: tuple = (10, 30),
        shock: float = 0.2,
        n_mbc_samples: int = 200,
        compute_mbc: bool = True,
        seed: Optional[int] = 42,
        transform=None,
        pre_transform=None,
    ):
        self.n_samples      = n_samples
        self.network_types  = network_types or ["erdos_renyi", "core_periphery", "barabasi_albert"]
        self.n_banks_range  = n_banks_range
        self.shock          = shock
        self.n_mbc_samples  = n_mbc_samples
        self.compute_mbc    = compute_mbc
        self.seed           = seed

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"systemic_risk_N{self.n_samples}_seed{self.seed}.pt"]

    def download(self):
        pass  # data is generated, not downloaded

    def process(self):
        rng = np.random.default_rng(self.seed)
        data_list = []
        failed = 0

        for i in tqdm(range(self.n_samples), desc="Generating networks"):
            try:
                # Sample network type and size
                ntype = rng.choice(self.network_types)
                n     = int(rng.integers(self.n_banks_range[0], self.n_banks_range[1] + 1))
                inst_seed = int(rng.integers(0, 2**31))

                net = self._generate_network(ntype, n, inst_seed)

                # Compute risk measures
                risk = compute_all_risk_measures(
                    L           = net.L,
                    e           = net.e,
                    shock       = self.shock,
                    shock_mode  = "proportional",
                    n_mbc_samples = self.n_mbc_samples,
                    compute_mbc = self.compute_mbc,
                    seed        = inst_seed,
                )

                data = network_to_pyg(net, risk)
                data_list.append(data)

            except Exception as ex:
                warnings.warn(f"Sample {i} failed: {ex}. Skipping.")
                failed += 1

        if failed > 0:
            warnings.warn(f"Total failed samples: {failed}/{self.n_samples}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _generate_network(
        self,
        ntype: str,
        n: int,
        seed: int,
    ) -> FinancialNetwork:
        """Dispatch to the correct generator with sensible default parameters."""
        if ntype == "erdos_renyi":
            return generate_financial_network(
                "erdos_renyi", n=n, p=0.15, seed=seed
            )
        elif ntype == "core_periphery":
            n_core = max(2, n // 4)
            n_peri = n - n_core
            return generate_financial_network(
                "core_periphery",
                n_core=n_core,
                n_periphery=n_peri,
                seed=seed,
            )
        elif ntype == "barabasi_albert":
            return generate_financial_network(
                "barabasi_albert", n=n, m=2, seed=seed
            )
        else:
            raise ValueError(f"Unknown network type: {ntype}")


# ---------------------------------------------------------------------------
# Lightweight in-memory builder (no disk caching, for quick experiments)
# ---------------------------------------------------------------------------

def build_dataset_in_memory(
    n_samples: int = 500,
    network_types: Optional[List[NetworkType]] = None,
    n_banks: int = 20,
    shock: float = 0.2,
    n_mbc_samples: int = 150,
    compute_mbc: bool = True,
    seed: int = 42,
) -> List[Data]:
    """
    Build a list of PyG Data objects without disk caching.
    Useful for fast prototyping and unit tests.

    Parameters
    ----------
    n_samples      : number of instances
    network_types  : list of topology types to sample from
    n_banks        : fixed number of banks per network
    shock          : proportional shock for AS
    n_mbc_samples  : MC samples for MBC
    compute_mbc    : whether to compute MBC (slow)
    seed           : master RNG seed

    Returns
    -------
    list of torch_geometric.data.Data
    """
    rng    = np.random.default_rng(seed)
    ntypes = network_types or ["erdos_renyi", "core_periphery", "barabasi_albert"]
    data_list = []
    failed    = 0

    for i in tqdm(range(n_samples), desc="Building dataset"):
        try:
            ntype     = rng.choice(ntypes)
            inst_seed = int(rng.integers(0, 2**31))

            if ntype == "erdos_renyi":
                net = generate_financial_network(
                    "erdos_renyi", n=n_banks, p=0.15, seed=inst_seed
                )
            elif ntype == "core_periphery":
                n_core = max(2, n_banks // 4)
                net = generate_financial_network(
                    "core_periphery",
                    n_core=n_core,
                    n_periphery=n_banks - n_core,
                    seed=inst_seed,
                )
            else:
                net = generate_financial_network(
                    "barabasi_albert", n=n_banks, m=2, seed=inst_seed
                )

            risk = compute_all_risk_measures(
                L             = net.L,
                e             = net.e,
                shock         = shock,
                shock_mode    = "proportional",
                n_mbc_samples = n_mbc_samples,
                compute_mbc   = compute_mbc,
                seed          = inst_seed,
            )

            data_list.append(network_to_pyg(net, risk))

        except Exception as ex:
            warnings.warn(f"Sample {i} failed: {ex}. Skipping.")
            failed += 1

    if failed > 0:
        warnings.warn(f"Total failed: {failed}/{n_samples}")

    return data_list