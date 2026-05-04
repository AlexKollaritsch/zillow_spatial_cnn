"""
features.py
Feature engineering utilities for:
- supervised windows (multi-feature CNN)
- spatial CNN windows
- Hilbert ordering
- coordinate filtering
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# 1. Filter cities missing coordinates (spatial models only)
# ============================================================

def filter_cities_with_coordinates(df_pivot, coords):
    mask = coords["Latitude"].notna() & coords["Longitude"].notna()
    keep_cols = coords.index[mask]

    df_pivot_f = df_pivot.loc[:, keep_cols]
    coords_f = coords.loc[keep_cols]

    return df_pivot_f, coords_f


# ============================================================
# 2. Hilbert ordering
# ============================================================

def hilbert_order_cities(coords, p=10):
    from hilbertcurve.hilbertcurve import HilbertCurve

    coords_norm = (coords - coords.min()) / (coords.max() - coords.min())
    pts = (coords_norm.to_numpy() * (2**p - 1)).astype(int)

    hilbert = HilbertCurve(p, 2)
    hilbert_indices = [hilbert.distance_from_point(list(pt)) for pt in pts]

    return np.argsort(hilbert_indices)


def apply_hilbert_order(df_pivot, coords, hilbert_order):
    df_pivot_h = df_pivot.iloc[:, hilbert_order]
    coords_h = coords.iloc[hilbert_order]
    return df_pivot_h, coords_h


# ============================================================
# 3. Multi-feature spatial CNN window creation
# ============================================================

def create_spatial_cnn_windows(df, num_cities, num_features, window=24, horizon=1):
    """
    df shape: (time, num_features * num_cities)

    Returns:
        X: (samples, num_features, num_cities, window)
        y: (samples, num_cities)
    """

    arr = df.to_numpy()  # (T, num_features * num_cities)
    T, total_cols = arr.shape

    # Safety check
    if total_cols != num_cities * num_features:
        raise ValueError(
            f"Feature matrix shape mismatch: expected {num_cities*num_features} columns, got {total_cols}"
        )

    X_list = []
    y_list = []

    for t in range(T - window - horizon + 1):
        window_slice = arr[t:t+window]  # (window, total_cols)

        # reshape → (window, num_features, num_cities)
        window_slice = window_slice.reshape(window, num_features, num_cities)

        # transpose → (num_features, num_cities, window)
        window_slice = np.transpose(window_slice, (1, 2, 0))

        X_list.append(window_slice)

        # target = raw ZHVI only (feature 0)
        y_list.append(arr[t + window + horizon - 1, :num_cities])

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)

    return X, y

def create_multi_horizon_targets(df, num_cities, window=24, max_horizon=12):
    """
    df shape: (time, num_cities)

    Returns:
        X_last: (samples, num_cities)
        y: (samples, num_cities, max_horizon)
    """

    arr = df.to_numpy()
    T = arr.shape[0]

    X_last_list = []
    y_list = []

    for t in range(T - window - max_horizon + 1):
        last_obs = arr[t + window - 1]  # (num_cities)

        horizons = []
        for h in range(1, max_horizon + 1):
            future = arr[t + window + h - 1]
            horizons.append(future - last_obs)  # residual

        y_list.append(np.stack(horizons, axis=-1))  # (num_cities, H)
        X_last_list.append(last_obs)

    X_last = torch.tensor(np.stack(X_last_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)

    return X_last, y

def create_deepar_sequences(
    target_pivot: pd.DataFrame,
    feature_matrix: pd.DataFrame | None = None,
    context_length: int = 24,
):
    """
    target_pivot: (time, num_cities) – ZHVI (scaled)
    feature_matrix: (time, num_features * num_cities) or (time, global_features)
        If None, we use only the target as input.

    Returns:
        sequences: dict[city] -> {
            "y": (T,),
            "x": (T, F) or None
        }
    """
    cities = target_pivot.columns
    sequences = {}

    for i, city in enumerate(cities):
        y = target_pivot[city].to_numpy()  # (T,)

        if feature_matrix is not None:
            # city-major: features for this city are every num_cities step
            # or you can pre-slice per city before calling this
            x = feature_matrix.filter(like=f"{city}_").to_numpy()
        else:
            x = None

        sequences[city] = {"y": y, "x": x}

    return sequences

# ============================================================
# 4. PyTorch Dataset
# ============================================================

class SpatialCnnDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class DeepARDataset(Dataset):
    def __init__(self, sequences, context_length: int, prediction_length: int):
        self.sequences = list(sequences.values())
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        y = seq["y"]
        x = seq["x"]

        T = len(y)
        # you can later sample random cut points; for now use full history
        context = y[:T - self.prediction_length]
        future = y[T - self.prediction_length:]

        if x is not None:
            x_context = x[:T - self.prediction_length]
            x_future = x[T - self.prediction_length:]
        else:
            x_context = None
            x_future = None

        return {
            "y_context": torch.tensor(context, dtype=torch.float32),
            "y_future": torch.tensor(future, dtype=torch.float32),
            "x_context": None if x_context is None else torch.tensor(x_context, dtype=torch.float32),
            "x_future": None if x_future is None else torch.tensor(x_future, dtype=torch.float32),
        }

# ============================================================
# 5. Feature engineering
# ============================================================

def build_seasonality_features(dates: pd.DatetimeIndex, city_names) -> pd.DataFrame:
    month = dates.month
    quarter = dates.quarter

    base = pd.DataFrame(index=dates)
    base["month_sin"] = np.sin(2 * np.pi * month / 12)
    base["month_cos"] = np.cos(2 * np.pi * month / 12)
    base["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    base["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)

    # Broadcast with city-first naming
    expanded = pd.concat(
        [
            base.rename(columns=lambda col: f"{city}_{col}")
            for city in city_names
        ],
        axis=1
    )

    return expanded


def build_rolling_features(df_pivot: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Rolling mean only (no rolling volatility).
    """
    roll_mean = df_pivot.rolling(window=window, min_periods=1).mean()
    roll_mean.columns = [f"{c}_roll_mean" for c in df_pivot.columns]
    return roll_mean


def build_momentum_features(df_pivot: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    momentum = df_pivot.diff(periods=periods).fillna(0.0)

    momentum.columns = [f"{c}_momentum" for c in df_pivot.columns]

    return momentum


def build_full_feature_matrix(df_pivot: pd.DataFrame) -> pd.DataFrame:
    cities = df_pivot.columns

    rolling = build_rolling_features(df_pivot)
    momentum = build_momentum_features(df_pivot)
    seasonality = build_seasonality_features(df_pivot.index, cities)

    blocks = []
    for city in cities:
        cols = []

        # raw
        cols.append(df_pivot[[city]])

        # rolling
        cols.append(rolling[[f"{city}_roll_mean"]])

        # momentum
        cols.append(momentum[[f"{city}_momentum"]])

        # seasonality (4 features)
        cols.append(
            seasonality[
                [
                    f"{city}_month_sin",
                    f"{city}_month_cos",
                    f"{city}_quarter_sin",
                    f"{city}_quarter_cos",
                ]
            ]
        )

        blocks.append(pd.concat(cols, axis=1))

    full = pd.concat(blocks, axis=1)
    return full

def scale_feature_groups(df, num_cities):
    """
    df shape: (time, num_features * num_cities)
    """

    total_cols = df.shape[1]
    num_features = total_cols // num_cities

    scaled = df.copy()
    means = {}
    stds = {}

    for f in range(num_features):
        start = f * num_cities
        end = (f + 1) * num_cities

        group = df.iloc[:, start:end]
        mean = group.mean()
        std = group.std().replace(0, 1)

        scaled.iloc[:, start:end] = (group - mean) / std

        means[f] = mean
        stds[f] = std

    return scaled, means, stds

def scale_per_city(df_pivot: pd.DataFrame):
    """
    df_pivot: (time, num_cities) – e.g., ZHVI pivot
    Returns:
        scaled: same shape
        means: per-city mean (Series)
        stds: per-city std (Series)
    """
    means = df_pivot.mean(axis=0)
    stds = df_pivot.std(axis=0).replace(0, 1)

    scaled = (df_pivot - means) / stds
    return scaled, means, stds


def apply_per_city_scaling(df_pivot: pd.DataFrame, means, stds):
    return (df_pivot - means) / stds


def apply_feature_group_scaling(df, means, stds, num_cities):
    total_cols = df.shape[1]
    num_features = total_cols // num_cities

    scaled = df.copy()

    for f in range(num_features):
        start = f * num_cities
        end = (f + 1) * num_cities

        group = df.iloc[:, start:end]
        scaled.iloc[:, start:end] = (group - means[f]) / stds[f]

    return scaled