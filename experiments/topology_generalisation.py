import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch_geometric.loader import DataLoader

from src.dataset import build_dataset_in_memory
from src.models.gnn_sage_gat import GraphSAGEModel
from train import train, evaluate, compute_target_stats


def main():
    # -----------------------------------------------------------
    # 1. Train ONLY on Erdos-Renyi networks
    # -----------------------------------------------------------
    print("Building ER training set...")
    er_data = build_dataset_in_memory(
        n_samples      = 600,
        n_banks        = 25,
        shock          = 0.6,
        network_types  = ["erdos_renyi"],
        compute_mbc    = False,
        seed           = 10,
    )
    train_er = er_data[:500]
    val_er   = er_data[500:]

    model = GraphSAGEModel(
        in_channels = 4,
        hidden_dim  = 128,
        out_dim     = 1,
        n_layers    = 5,
        dropout     = 0.1,
        node_level  = False,
        pooling     = "add",
    )

    print("\nTraining on ER only...")
    history = train(
        model       = model,
        train_data  = train_er,
        val_data    = val_er,
        target      = "as",
        epochs      = 300,
        batch_size  = 32,
        lr          = 3e-4,
        patience    = 30,
        device_str  = "auto",
        save_path   = "sage_as_er_only.pt",
    )

    # -----------------------------------------------------------
    # 2. Build held-out test sets (unseen topologies)
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
    # 3. Evaluate generalisation
    # -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t_mean = history["target_mean"]
    t_std  = history["target_std"]

    m_er = evaluate(model, DataLoader(test_er, batch_size=32), "as", device, t_mean, t_std)
    m_cp = evaluate(model, DataLoader(test_cp, batch_size=32), "as", device, t_mean, t_std)
    m_ba = evaluate(model, DataLoader(test_ba, batch_size=32), "as", device, t_mean, t_std)

    print("\nTopology generalisation results:")
    print(f"  In-dist  ER: MSE={m_er['mse']:.4f}  MAE={m_er['mae']:.4f}  R²={m_er['r2']:.4f}")
    print(f"  OOD      CP: MSE={m_cp['mse']:.4f}  MAE={m_cp['mae']:.4f}  R²={m_cp['r2']:.4f}")
    print(f"  OOD      BA: MSE={m_ba['mse']:.4f}  MAE={m_ba['mae']:.4f}  R²={m_ba['r2']:.4f}")
    print("\nInterpretation:")
    print("  R² near 1.0 on all three  = strong cross-topology generalisation.")
    print("  R² drops on CP/BA         = model is topology-specific (interesting finding).")
    print(f"\n  ER  best val R²: {max(history['val_r2']):.4f}")
    print(f"  ER  best epoch : {history['best_epoch']}")


if __name__ == "__main__":
    main()