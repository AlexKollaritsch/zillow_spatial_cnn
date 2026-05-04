"""
models.py

DeepAR (per-city) with frozen Spatial CNN embeddings using PyTorch Lightning.

- Assumes a separate Spatial CNN has already been trained and its city embeddings
  are saved to disk as: PROJECT_ROOT / "data" / "spatial_embeddings.pt"
- This file trains a single global DeepAR model that:
    * sees one city per sample
    * uses that city's spatial embedding as a static feature
    * can generalize to all cities at inference time (Streamlit)
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------
# Reproducibility & device
# -------------------------------------------------------------------

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

torch.set_float32_matmul_precision("medium")

# -------------------------------------------------------------------
# Paths & imports
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src._01_data import load_deepar_dataset
from src._02_features import (
    scale_per_city,
)

# -------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------

CONTEXT_LENGTH = 12
PREDICTION_LENGTH = 12
BATCH_SIZE = 128
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5

MAX_SAMPLES_PER_CITY = 64  # cap to keep dataset size manageable


# -------------------------------------------------------------------
# DeepAR decoder (per-city, with static spatial embedding)
# -------------------------------------------------------------------

class DeepARDecoder(nn.Module):
    """
    Univariate DeepAR with static spatial embedding per city.

    Inputs:
        y_context: (B, T_c)
        spatial_emb: (B, E)

    Outputs:
        y_pred: (B, T_p)
    """

    def __init__(
        self,
        spatial_emb_dim: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        prediction_length: int = PREDICTION_LENGTH,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length

        # Input: previous target + spatial embedding
        self.input_size = 1 + spatial_emb_dim

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output head: predict next value (Gaussian mean)
        self.proj = nn.Linear(hidden_size, 1)

        # Learnable global log-variance
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, y_context: torch.Tensor, spatial_emb: torch.Tensor) -> torch.Tensor:
        """
        y_context: (B, T_c)
        spatial_emb: (B, E)
        """
        B, T_c = y_context.shape
        E = spatial_emb.shape[1]

        # Prepare context sequence
        y_ctx = y_context.unsqueeze(-1)  # (B, T_c, 1)
        s_rep = spatial_emb.unsqueeze(1).repeat(1, T_c, 1)  # (B, T_c, E)

        gru_input = torch.cat([y_ctx, s_rep], dim=-1)  # (B, T_c, 1+E)

        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=y_context.device)
        _, h = self.gru(gru_input, h0)  # h: (num_layers, B, H)

        # Autoregressive decoding
        h_t = h
        y_t = y_ctx[:, -1:, :]  # (B, 1, 1)

        preds = []

        for _ in range(self.prediction_length):
            s_step = spatial_emb.unsqueeze(1)  # (B, 1, E)
            step_input = torch.cat([y_t, s_step], dim=-1)  # (B, 1, 1+E)

            out_step, h_t = self.gru(step_input, h_t)  # (B, 1, H)
            mean_step = self.proj(out_step)            # (B, 1, 1)

            preds.append(mean_step.squeeze(-1))        # (B, 1)
            y_t = mean_step                            # next input

        preds = torch.cat(preds, dim=1)  # (B, T_p)
        return preds

    def gaussian_nll(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred, y_true: (B, T_p)
        """
        sigma = torch.exp(self.log_sigma) + 1e-6
        var = sigma ** 2

        nll = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * (y_true - y_pred) ** 2 / var
        return nll.mean()


# -------------------------------------------------------------------
# LightningModule
# -------------------------------------------------------------------

class DeepARPerCity(pl.LightningModule):
    """
    Global DeepAR model trained on one city per sample, with frozen spatial embeddings.
    """

    def __init__(
        self,
        spatial_emb_dim: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        prediction_length: int = PREDICTION_LENGTH,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.decoder = DeepARDecoder(
            spatial_emb_dim=spatial_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_length=prediction_length,
        )

        self.prediction_length = prediction_length
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch):
        y_context = batch["y_context"]      # (B, T_c)
        spatial_emb = batch["spatial_emb"]  # (B, E)
        return self.decoder(y_context, spatial_emb)  # (B, T_p)

    def training_step(self, batch, batch_idx):
        y_true = batch["y_future"]  # (B, T_p)
        y_pred = self(batch)        # (B, T_p)

        loss = self.decoder.gaussian_nll(y_pred, y_true)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_true = batch["y_future"]
        y_pred = self(batch)

        loss = self.decoder.gaussian_nll(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_true = batch["y_future"]
        y_pred = self(batch)

        loss = self.decoder.gaussian_nll(y_pred, y_true)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

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
                "monitor": "val_loss",
            },
        }

# -------------------------------------------------------------------
# Dataset (per-city samples)
# -------------------------------------------------------------------

class CityDeepARDataset(Dataset):
    """
    Each item:
        y_context: (T_c,)
        y_future: (T_p,)
        spatial_emb: (E,)
    """

    def __init__(
        self,
        y_context: torch.Tensor,
        y_future: torch.Tensor,
        spatial_emb: torch.Tensor,
    ):
        super().__init__()
        self.y_context = y_context
        self.y_future = y_future
        self.spatial_emb = spatial_emb

    def __len__(self):
        return self.y_context.shape[0]

    def __getitem__(self, idx):
        return {
            "y_context": self.y_context[idx],
            "y_future": self.y_future[idx],
            "spatial_emb": self.spatial_emb[idx],
        }


# -------------------------------------------------------------------
# Data preparation helper
# -------------------------------------------------------------------

def build_deepar_data():
    """
    Builds a per-city DeepAR dataset using:
    - scaled log-ZHVI per city
    - frozen spatial embeddings loaded from disk

    Expects:
        PROJECT_ROOT / "data" / "spatial_embeddings.pt"
        with shape (num_cities, E), aligned with df_pivot columns.
    """

    df_long = load_deepar_dataset()  # (CityState, Date, ZHVI, ...)
    df_long = df_long.sort_values(["Date", "CityState"])

    # Pivot ZHVI: (time, cities)
    df_pivot = df_long.pivot(index="Date", columns="CityState", values="ZHVI").sort_index()

    # ---------------------------------------------------------
    # NEW: Filter pivot to match Spatial CNN city list
    # ---------------------------------------------------------

    # Load the same city list used by the Spatial CNN
    from src._01_data import load_processed_data
    cnn_data = load_processed_data()
    cnn_city_list = cnn_data["df_pivot_log"].columns.tolist()

    # Filter pivot to match CNN cities
    df_pivot = df_pivot[cnn_city_list]

    # Drop cities with all-NaN ZHVI values
    df_pivot = df_pivot.dropna(axis=1, how="all")

    # Ensure no CNN cities were lost
    assert df_pivot.shape[1] == len(cnn_city_list), \
        "Some CNN cities have no ZHVI values — mismatch"


    # Confirm alignment
    assert df_pivot.shape[1] == cnn_data["num_cities"], \
        f"DeepAR pivot has {df_pivot.shape[1]} cities but CNN has {cnn_data['num_cities']}"


    # Log-transform
    df_pivot_log = np.log(df_pivot)

    # Per-city scaling
    df_scaled, means_c, stds_c = scale_per_city(df_pivot_log)  # (T, C)
    arr = df_scaled.to_numpy()  # (T, C)
    T_total, num_cities = arr.shape

    # Load frozen spatial embeddings
    emb_path = PROJECT_ROOT / "data" / "spatial_embeddings.pt"
    spatial_emb_all = torch.load(emb_path)  # (num_cities, E)
    spatial_emb_all.requires_grad_(False)
    if spatial_emb_all.shape[0] != num_cities:
        raise ValueError(
            f"Spatial embeddings first dim {spatial_emb_all.shape[0]} "
            f"does not match num_cities {num_cities}"
        )
    spatial_emb_all = spatial_emb_all.float()

    y_context_list = []
    y_future_list = []
    spatial_list = []

    for city_idx in range(num_cities):
        series = arr[:, city_idx]  # (T,)

        # Build sliding windows for this city
        indices = []
        for t in range(T_total - CONTEXT_LENGTH - PREDICTION_LENGTH + 1):
            indices.append(t)

        # Optional: subsample to limit dataset size
        if len(indices) > MAX_SAMPLES_PER_CITY:
            indices = np.random.choice(
                indices,
                size=MAX_SAMPLES_PER_CITY,
                replace=False,
            )

        for t in indices:
            ctx_start = t
            ctx_end = t + CONTEXT_LENGTH
            fut_end = ctx_end + PREDICTION_LENGTH

            y_ctx = series[ctx_start:ctx_end]   # (T_c,)
            y_fut = series[ctx_end:fut_end]     # (T_p,)

            y_context_list.append(y_ctx)
            y_future_list.append(y_fut)
            spatial_list.append(spatial_emb_all[city_idx].numpy())

    y_context_t = torch.tensor(np.stack(y_context_list), dtype=torch.float32)  # (N, T_c)
    y_future_t = torch.tensor(np.stack(y_future_list), dtype=torch.float32)    # (N, T_p)
    spatial_t = torch.tensor(np.stack(spatial_list), dtype=torch.float32)      # (N, E)

    ds = CityDeepARDataset(y_context_t, y_future_t, spatial_t)

    return {
        "dataset": ds,
        "num_cities": num_cities,
        "spatial_emb_dim": spatial_t.shape[1],
        "means_c": means_c,
        "stds_c": stds_c,
        "city_names": df_pivot.columns.tolist(),
    }


# ---------------------------------------------------------
# Callback to print train + val loss at the end of each epoch
# ---------------------------------------------------------
class PrintLossCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        if train_loss is not None and val_loss is not None:
            print(f"Epoch {trainer.current_epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


# -------------------------------------------------------------------
# Main (example training loop with Lightning Trainer)
# -------------------------------------------------------------------

if __name__ == "__main__":
    data = build_deepar_data()
    ds = data["dataset"]

    # ---------------------------------------------------------
    # Train/Validation split
    # ---------------------------------------------------------
    N = len(ds)
    val_size = int(0.05 * N)
    train_size = N - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = DeepARPerCity(
        spatial_emb_dim=data["spatial_emb_dim"],
        hidden_size=32,
        num_layers=1,
        prediction_length=PREDICTION_LENGTH,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-deepar",
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        precision="16-mixed",
        callbacks=[early_stop, checkpoint, PrintLossCallback()],
    )

    trainer.fit(model, train_loader, val_loader)

    # ---------------------------------------------------------
    # Save trained model + metadata for Streamlit
    # ---------------------------------------------------------
    import pickle

    SAVE_DIR = PROJECT_ROOT / "models"
    SAVE_DIR.mkdir(exist_ok=True)

    # 1. Save model weights
    model_path = SAVE_DIR / "deepar_model.pt"
    best_ckpt_path = checkpoint.best_model_path
    best_state = torch.load(best_ckpt_path)["state_dict"]
    model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)

    print(f"Saved model weights to: {model_path}")

    # 2. Save metadata needed for inference
    metadata = {
        "means_c": data["means_c"],
        "stds_c": data["stds_c"],
        "city_names": data["city_names"],
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
    }

    meta_path = SAVE_DIR / "deepar_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved metadata to: {meta_path}")

    # ---------------------------------------------------------
    # Evaluate DeepAR vs Naive baseline across all horizons
    # ---------------------------------------------------------
    import torch
    import numpy as np

    print("\nRunning model vs naive baseline evaluation...")

    # Reload model in eval mode
    eval_model = DeepARPerCity(
        spatial_emb_dim=data["spatial_emb_dim"],
        hidden_size=32,
        num_layers=1,
        prediction_length=PREDICTION_LENGTH,
    )
    eval_model.load_state_dict(torch.load(model_path))
    eval_model.eval()
    eval_model.to(DEVICE)

    # Extract arrays for evaluation
    y_context = data["dataset"].y_context.to(DEVICE)      # (N, T_c)
    y_future  = data["dataset"].y_future.to(DEVICE)       # (N, T_p)
    spatial   = data["dataset"].spatial_emb.to(DEVICE)    # (N, E)

    y_pred_list = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(y_context), batch_size):
            batch_ctx = y_context[i:i+batch_size]
            batch_sp  = spatial[i:i+batch_size]

            batch_pred = eval_model({
                "y_context": batch_ctx,
                "spatial_emb": batch_sp
            })

            y_pred_list.append(batch_pred.cpu())

    y_pred = torch.cat(y_pred_list, dim=0)


    # Naive baseline: repeat last context value
    y_last = y_context[:, -1].unsqueeze(1)                # (N, 1)
    y_naive = y_last.repeat(1, PREDICTION_LENGTH)         # (N, T_p)

    # Convert to numpy
    true_np  = y_future.cpu().numpy()
    pred_np  = y_pred.cpu().numpy()
    naive_np = y_naive.cpu().numpy()

    # Horizon-wise RMSE
    rmse_model = np.sqrt(((pred_np - true_np) ** 2).mean(axis=0))
    rmse_naive = np.sqrt(((naive_np - true_np) ** 2).mean(axis=0))

    # Aggregate RMSE
    rmse_model_all = rmse_model.mean()
    rmse_naive_all = rmse_naive.mean()

    print("\nHorizon-wise RMSE (Model vs Naive):")
    for h in range(PREDICTION_LENGTH):
        print(f"H{h+1:02d}:  Model={rmse_model[h]:.4f}   Naive={rmse_naive[h]:.4f}")

    print("\nAggregate RMSE over all horizons:")
    print(f"Model: {rmse_model_all:.4f}")
    print(f"Naive: {rmse_naive_all:.4f}")
