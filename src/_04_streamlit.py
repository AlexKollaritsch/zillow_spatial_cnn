import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src._02_features import apply_feature_group_scaling
from src.train_spatial_cnn import LOOKBACK
from src.spatial_cnn import SpatialCNNEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Model wrapper (matches training architecture)
# ---------------------------------------------------------

class SpatialCNNWrapper(torch.nn.Module):
    def __init__(self, num_features, spatial_channels=32, dropout=0.2):
        super().__init__()
        self.encoder = SpatialCNNEncoder(
            num_features=num_features,
            spatial_channels=spatial_channels,
            dropout=dropout
        )
        self.proj = torch.nn.Linear(spatial_channels, 1)

    def forward(self, X):
        emb = self.encoder(X)          # (B, C, E)
        out = self.proj(emb).squeeze(-1)  # (B, C)
        return out

# ---------------------------------------------------------
# Cached metadata loader
# ---------------------------------------------------------

@st.cache_resource
def get_metadata():
    with open(DATA_DIR / "spatial_cnn_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return (
        metadata["df_pivot_h"],
        metadata["df_pivot_log"],
        metadata["full_features"],
        metadata["train_means"],
        metadata["train_stds"],
        int(metadata["num_features"]),
        int(metadata["num_cities"]),
        metadata["resid_std"],
    )

# ---------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------

@st.cache_resource
def load_model(num_features):
    model = SpatialCNNWrapper(
        num_features=num_features,
        spatial_channels=32,
        dropout=0.2
    ).to(DEVICE)

    state = torch.load(
        DATA_DIR / "spatial_cnn_global.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(state["state_dict"])
    model.eval()
    return model

# ---------------------------------------------------------
# Cached latest window builder
# ---------------------------------------------------------

@st.cache_resource
def get_latest_window(full_features, train_means, train_stds, num_cities, num_features):
    scaled = apply_feature_group_scaling(
        full_features, train_means, train_stds, num_cities
    )

    arr = scaled.to_numpy()
    arr = arr[-LOOKBACK:]
    arr = arr.reshape(LOOKBACK, num_features, num_cities)
    arr = np.transpose(arr, (1, 2, 0))

    X = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return X.to(DEVICE)

# ---------------------------------------------------------
# Cached city list
# ---------------------------------------------------------

@st.cache_data
def get_city_list(df_pivot_h):
    return list(df_pivot_h.columns)

# ---------------------------------------------------------
# Load metadata + model (cached)
# ---------------------------------------------------------

(
    df_pivot_h,
    df_pivot_log,
    full_features,
    train_means,
    train_stds,
    num_features,
    num_cities,
    resid_std,
) = get_metadata()

model = load_model(num_features)
city_list = get_city_list(df_pivot_h)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("🏠 Zillow ZHVI Forecasting — Spatial CNN")

city = st.selectbox("Select a city:", city_list)
horizon = st.slider("Forecast horizon (months ahead):", 1, 12, 6)

if st.button("Predict"):

    # Build cached input window
    X = get_latest_window(full_features, train_means, train_stds, num_cities, num_features)

    with torch.no_grad():
        preds_scaled = model(X).squeeze(0)  # (num_cities,)

    city_idx = city_list.index(city)
    pred_scaled = preds_scaled[city_idx].item()

    # Last observed log(ZHVI)
    last_log = df_pivot_log[city].iloc[-1]
    last_raw = float(np.exp(last_log))

    # Multi-horizon recursive forecast
    preds_raw = []
    curr_log = last_log

    for _ in range(horizon):
        delta = pred_scaled * resid_std.iloc[city_idx]
        curr_log = curr_log + delta
        preds_raw.append(np.exp(curr_log))

    # -----------------------------------------------------
    # Display metrics
    # -----------------------------------------------------

    st.subheader(f"{city} — {horizon}-Month Forecast")

    final_pred = preds_raw[-1]
    pct_change = (final_pred / last_raw - 1) * 100

    st.metric(
        label="Forecasted ZHVI",
        value=f"${final_pred:,.0f}",
        delta=f"{pct_change:.2f}%"
    )

    st.write(f"Last observed ZHVI: ${last_raw:,.0f}")
    st.write(f"Model residual (scaled): {pred_scaled:.4f}")

    # -----------------------------------------------------
    # Plot 1 — Historical ZHVI
    # -----------------------------------------------------

    import matplotlib.pyplot as plt

    st.subheader("Historical ZHVI")

    hist_raw = df_pivot_h[city]

    import plotly.express as px

    fig1 = px.line(
        x=hist_raw.index,
        y=hist_raw.values,
        title=f"{city} — Historical ZHVI",
        labels={"x": "Date", "y": "ZHVI ($)"}
    )

    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------------------------------
    # Plot 2 — Multi-Horizon Forecast Curve
    # -----------------------------------------------------

    st.subheader("Forecast Curve")

    future_dates = [
        hist_raw.index[-1] + pd.DateOffset(months=h)
        for h in range(1, horizon + 1)
    ]

    fig2 = px.line(
        x=future_dates,
        y=preds_raw,
        markers=True,
        title=f"{city} — {horizon}-Month Forecast Curve",
        labels={"x": "Date", "y": "ZHVI ($)"}
    )
    fig2.add_hline(y=last_raw, line_dash="dash", line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)
