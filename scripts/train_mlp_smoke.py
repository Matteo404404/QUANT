import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import build_dataset_in_memory
from src.models.mlp_baseline import MLPBaseline
from train import train


def main():
    print("Building dataset...")
    data = build_dataset_in_memory(
        n_samples   = 200,
        n_banks     = 20,
        shock       = 0.6,
        compute_mbc = False,
        seed        = 42,
    )

    train_data = data[:160]
    val_data   = data[160:]

    model = MLPBaseline(
        in_channels = 4,
        hidden_dim  = 128,
        out_dim     = 1,
        n_layers    = 3,
        dropout     = 0.1,
    )

    history = train(
        model       = model,
        train_data  = train_data,
        val_data    = val_data,
        target      = "as",
        epochs      = 50,
        batch_size  = 16,
        lr          = 3e-4,
        patience    = 10,
        device_str  = "auto",
        save_path   = "mlp_as_best.pt",
    )

    print("\nBest ValMSE :", history["best_val_mse"])
    print("Best Epoch  :", history["best_epoch"])


if __name__ == "__main__":
    main()