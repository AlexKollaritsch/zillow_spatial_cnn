"""
train_spatial_cnn.py

Trains a global Spatial CNN encoder on Hilbert-ordered, multi-feature city panels.

Outputs:
    data/spatial_cnn_global.pt
"""

import sys
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# Paths & sys.path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CKPT_PATH = DATA_DIR / "spatial_cnn_global.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Imports from project
# ---------------------------------------------------------

from src._01_data import load_processed_data
from src._02_features import create_spatial_cnn_windows
from src.spatial_cnn import SpatialCNNEncoder

# ---------------------------------------------------------
# Reproducibility & device
# ---------------------------------------------------------

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed(SEED)
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

LOOKBACK = 24          # temporal window length
PREDICTION_LENGTH = 12 # horizon (for alignment only)
SPATIAL_CHANNELS = 32
DROPOUT = 0.2

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class SpatialCNNDataset(Dataset):
    """
    Each item:
        X_spatial: (F, C, T)
    """

    def __init__(self, X_spatial: torch.Tensor):
        super().__init__()
        self.X_spatial = X_spatial

    def __len__(self):
        return self.X_spatial.shape[0]

    def __getitem__(self, idx):
        return self.X_spatial[idx]


# ---------------------------------------------------------
# LightningModule wrapper
# ---------------------------------------------------------

class SpatialCNNModule(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        spatial_channels: int = SPATIAL_CHANNELS,
        dropout: float = DROPOUT,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SpatialCNNEncoder(
            num_features=num_features,
            spatial_channels=spatial_channels,
            dropout=dropout,
        )

        # simple reconstruction head: map embedding back to per-city scalar
        self.proj = nn.Linear(spatial_channels, 1)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, X_spatial: torch.Tensor) -> torch.Tensor:
        """
        X_spatial: (B, F, C, T)
        Returns:
            y_hat: (B, C) dummy reconstruction target
        """
        emb = self.encoder(X_spatial)  # (B, C, E)
        y_hat = self.proj(emb).squeeze(-1)  # (B, C)
        return y_hat

    def training_step(self, batch, batch_idx):
        X_spatial = batch  # (B, F, C, T)

        # dummy self-supervised target: mean over time of first feature
        # you can replace this with a better objective later
        with torch.no_grad():
            target = X_spatial[:, 0].mean(dim=-1)  # (B, C)

        y_hat = self(X_spatial)
        loss = nn.functional.mse_loss(y_hat, target)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }


# ---------------------------------------------------------
# Data builder
# ---------------------------------------------------------

def build_spatial_cnn_data():
    """
    Uses load_processed_data() to build:
    - full_features: (T, F*C)
    - spatial CNN windows: X_spatial: (N, F, C, T_s)
    """

    (
        df_pivot_h,
        df_pivot_log,
        full_features,
        train_means,
        train_stds,
        num_features,
        num_cities,
        resid_std
    ) = load_processed_data()


    # create_spatial_cnn_windows expects:
    #   full_features: (T, F*C)
    #   num_cities, num_features, window, horizon
    X_spatial_np, _ = create_spatial_cnn_windows(
        full_features,
        num_cities=num_cities,
        num_features=num_features,
        window=LOOKBACK,
        horizon=PREDICTION_LENGTH,
    )  # (N, F, C, T_s)

    X_spatial = torch.tensor(X_spatial_np, dtype=torch.float32)

    ds = SpatialCNNDataset(X_spatial)

    return {
        "dataset": ds,
        "num_features": num_features,
        "num_cities": num_cities,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = build_spatial_cnn_data()
    ds = data["dataset"]

    train_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    model = SpatialCNNModule(
        num_features=data["num_features"],
        spatial_channels=SPATIAL_CHANNELS,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        precision=32,
    )

    trainer.fit(model, train_loader)

    # Save checkpoint compatible with generate_spatial_embeddings.py
    ckpt = {
        "state_dict": model.state_dict(),
        "hparams": model.hparams,
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"Saved Spatial CNN checkpoint to: {CKPT_PATH}")

    import pickle

    (
        df_pivot_h,
        df_pivot_log,
        full_features,
        train_means,
        train_stds,
        num_features,
        num_cities,
        resid_std
    ) = load_processed_data()

    with open(DATA_DIR / "spatial_cnn_metadata.pkl", "wb") as f:
        pickle.dump({
            "df_pivot_h": df_pivot_h,
            "df_pivot_log": df_pivot_log,
            "full_features": full_features,
            "train_means": train_means,
            "train_stds": train_stds,
            "num_features": num_features,
            "num_cities": num_cities,
            "resid_std": resid_std
        }, f)