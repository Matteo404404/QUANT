import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch_geometric.loader import DataLoader

from src.dataset import build_dataset_in_memory
from src.models.gnn_sage_gat import GraphSAGEModel
from src.models.mlp_baseline import MLPBaseline
from train import train, evaluate


def main():
    # -----------------------------------------------------------
    # 1. Train on MIXED topologies
    # -----------------------------------------------------------
    print("Building mixed training set (ER + CP + BA)...")
    mixed_data = build_dataset_in_memory(
        n_samples      = 900,
        n_banks        = 25,
        shock          = 0.6,
        network_types  = ["erdos_renyi", "core_periphery", "barabasi_albert"],
        compute_mbc    = False,
        seed           = 20,
    )
    train_data = mixed_data[:720]
    val_data   = mixed_data[720:]

    sage = GraphSAGEModel(
        in_channels = 4,
        hidden_dim  = 128,
        out_dim     = 1,
        n_layers    = 5,
        dropout     = 0.1,
        node_level  = False,
        pooling     = "add",
    )

    mlp = MLPBaseline(
        in_channels = 4,
        hidden_dim  = 128,
        out_dim     = 1,
        n_layers    = 3,
        dropout     = 0.1,
    )

    print("\nTraining GraphSAGE on mixed topologies...")
    history_sage = train(
        model       = sage,
        train_data  = train_data,
        val_data    = val_data,
        target      = "as",
        epochs      = 300,
        batch_size  = 32,
        lr          = 3e-4,
        patience    = 30,
        device_str  = "auto",
        save_path   = "sage_as_mixed.pt",
    )

    print("\nTraining MLP baseline on mixed topologies...")
    history_mlp = train(
        model       = mlp,
        train_data  = train_data,
        val_data    = val_data,
        target      = "as",
        epochs      = 300,
        batch_size  = 32,
        lr          = 3e-4,
        patience    = 30,
        device_str  = "auto",
        save_path   = "mlp_as_mixed.pt",
    )

    # -----------------------------------------------------------
    # 2. Held-out test sets (same seeds as topology_generalisation)
    # -----------------------------------------------------------
    print("\nBuilding held-out test sets...")
    test_er = build_dataset_in_memory(
        n_samples=200, n_banks=25, shock=0.6,
        network_types=["erdos_renyi"], compute_mbc=False, seed=13,
    )
    test_cp = build_dataset_in_memory(
        n_samples=200, n_banks=25, shock=0.6,
        network_types=["core_periphery"], compute_mbc=False, seed=11,
    )
    test_ba = build_dataset_in_memory(
        n_samples=200, n_banks=25, shock=0.6,
        network_types=["barabasi_albert"], compute_mbc=False, seed=12,
    )

    # -----------------------------------------------------------
    # 3. Compare SAGE vs MLP on all topologies
    # -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sage.to(device)
    mlp.to(device)

    results = {}
    for name, model, history in [("SAGE", sage, history_sage), ("MLP", mlp, history_mlp)]:
        t_mean = history["target_mean"]
        t_std  = history["target_std"]
        results[name] = {
            "ER": evaluate(model, DataLoader(test_er, batch_size=32), "as", device, t_mean, t_std),
            "CP": evaluate(model, DataLoader(test_cp, batch_size=32), "as", device, t_mean, t_std),
            "BA": evaluate(model, DataLoader(test_ba, batch_size=32), "as", device, t_mean, t_std),
        }

    print("\nMixed training generalisation results:")
    print(f"  {'Model':<8} {'Topology':<6} {'MSE':>10}  {'MAE':>8}  {'R²':>8}")
    for name in ["SAGE", "MLP"]:
        for topo in ["ER", "CP", "BA"]:
            m = results[name][topo]
            print(f"  {name:<8} {topo:<6} {m['mse']:>10.2f}  {m['mae']:>8.3f}  {m['r2']:>8.4f}")
    print("\nKey comparison:")
    print("  SAGE > MLP in R²  = graph structure helps over flat features.")
    print("  R² close across ER/CP/BA = model generalises across topologies.")


if __name__ == "__main__":
    main()