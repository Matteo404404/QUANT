import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import build_dataset_in_memory
from src.models.penn import PENN
from train import train


def main():
    print("Building dataset...")
    data = build_dataset_in_memory(
        n_samples   = 400,
        n_banks     = 20,
        shock       = 0.6,
        compute_mbc = False,
        seed        = 3,
    )

    train_data = data[:320]
    val_data   = data[320:]

    model = PENN(
        in_channels = 4,
        edge_in_dim = 1,
        hidden_dim  = 64,
        out_dim     = 1,
        n_layers    = 4,
        dropout     = 0.1,
        node_level  = True,
        pooling     = "add",
    )

    history = train(
        model       = model,
        train_data  = train_data,
        val_data    = val_data,
        target      = "dr",
        epochs      = 150,
        batch_size  = 16,
        lr          = 3e-4,
        patience    = 20,
        device_str  = "auto",
        save_path   = "penn_dr_best.pt",
    )

    print("\nBest ValMSE :", history["best_val_mse"])
    print("Best Epoch  :", history["best_epoch"])


if __name__ == "__main__":
    main()