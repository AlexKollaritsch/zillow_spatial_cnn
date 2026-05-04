"""
generate_spatial_embeddings.py

Creates spatial_embeddings.pt if it does not already exist.

This script:
- loads your trained Spatial CNN model (spatial_cnn_global.pt)
- loads the SAME processed Zillow data used during CNN training
- builds a single (1, F, C, T) tensor
- runs the Spatial CNN encoder once
- saves the resulting (C, E) embeddings to: data/spatial_embeddings.pt
"""

import sys
from pathlib import Path
import torch
import numpy as np

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EMB_PATH = DATA_DIR / "spatial_embeddings.pt"
MODEL_PATH = DATA_DIR / "spatial_cnn_global.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------
# Imports from your project
# ---------------------------------------------------------

from src._01_data import load_processed_data
from src._02_features import apply_feature_group_scaling
from src.spatial_cnn import SpatialCNNEncoder

# ---------------------------------------------------------
# Step 0 — If embeddings already exist, load and exit
# ---------------------------------------------------------

if EMB_PATH.exists():
    print(f"Spatial embeddings already exist at: {EMB_PATH}")
    spatial_emb = torch.load(EMB_PATH)
    print("Loaded embeddings with shape:", spatial_emb.shape)
    sys.exit(0)

print("No spatial_embeddings.pt found — generating now...")

# ---------------------------------------------------------
# Step 1 — Load processed Zillow data (CNN-ready)
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# Step 2 — Apply the same scaling used during CNN training
# ---------------------------------------------------------

full_scaled = apply_feature_group_scaling(
    full_features,
    train_means,
    train_stds,
    num_cities
)

# ---------------------------------------------------------
# Step 3 — Build the (1, F, C, T) tensor
# ---------------------------------------------------------

arr = full_scaled.to_numpy()  # (T, F*C)
T = arr.shape[0]

# reshape → (T, F, C)
arr = arr.reshape(T, num_features, num_cities)

# transpose → (1, F, C, T)
X = np.transpose(arr, (1, 2, 0))
X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, F, C, T)

print("Spatial CNN input tensor shape:", X.shape)

# ---------------------------------------------------------
# Step 4 — Load trained Spatial CNN model
# ---------------------------------------------------------

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
state_dict = checkpoint["state_dict"]

# Infer spatial embedding dimension from model
spatial_channels = state_dict["encoder.spatial_pw2.weight"].shape[0]

encoder = SpatialCNNEncoder(
    num_features=num_features,
    spatial_channels=spatial_channels,
    dropout=0.0,
)

# Load only encoder weights
encoder.load_state_dict({
    k.replace("encoder.", ""): v
    for k, v in state_dict.items()
    if k.startswith("encoder.")
})

encoder.eval()

# ---------------------------------------------------------
# Step 5 — Run encoder to get (1, C, E) → save (C, E)
# ---------------------------------------------------------

with torch.no_grad():
    spatial_emb = encoder(X)  # (1, C, E)
    spatial_emb = spatial_emb.squeeze(0)  # (C, E)

torch.save(spatial_emb, EMB_PATH)

print(f"Saved spatial embeddings to: {EMB_PATH}")
print("Embedding matrix shape:", spatial_emb.shape)