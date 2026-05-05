# Zillow ZHVI Forecasting with Spatial CNN
End‑to‑End Machine Learning Pipeline for City‑Level Home Price Forecasting

This project builds a complete, production‑style machine learning system for forecasting Zillow Home Value Index (ZHVI) at the city level across the United States. It combines large‑scale data engineering, spatial modeling, deep learning, and interactive visualization into a single, cohesive pipeline.

The goal of this project is to demonstrate real‑world ML engineering skills — not just model training, but the entire lifecycle: data ingestion, feature engineering, model architecture design, training, evaluation, and deployment through a Streamlit app.

## Public Data Sources (Automatically Downloaded or Provided by User)

This project uses publicly available macroeconomic and geographic datasets:

- Federal Funds Rate (FRED)
- 30-Year Mortgage Rate (FRED)
- CPI All Items (FRED)
- Core CPI (FRED)
- US Gazetteer (Census Bureau)

These files are not included in the repository or release because they are publicly accessible and can be downloaded directly from their official sources.

## Download Required Files (Large Artifacts)
GitHub does not allow large files (>100MB) to be stored directly in the repository.
To run the full pipeline or the Streamlit app locally, download the following files and place them in the data/ directory:

## Model + Metadata
### Spatial CNN Model Checkpoint (spatial_cnn_global.pt)  
Download: https://github.com/AlexKollaritsch/zillow_spatial_cnn/releases/download/v1.0/spatial_cnn_global.pt

### Spatial CNN Metadata (spatial_cnn_metadata.pkl)  
Download: https://github.com/AlexKollaritsch/zillow_spatial_cnn/releases/download/v1.0/spatial_cnn_metadata.pkl

### Spatial Embeddings (spatial_embeddings.pt)  
Download: https://github.com/AlexKollaritsch/zillow_spatial_cnn/releases/download/v1.0/spatial_embeddings.pt

### Large Zillow Dataset
City‑level ZHVI CSV (City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv)  
Download: https://github.com/AlexKollaritsch/zillow_spatial_cnn/releases/download/v1.0/City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv

After downloading, place all files into:
zillow_project/data/

This keeps the repository lightweight while still allowing full reproducibility.

## Key Features
### Full end‑to‑end ML pipeline
Raw Zillow data → cleaned, merged, pivoted, and transformed

Feature engineering across 124,000+ features

Spatial coordinate integration using US Gazetteer

Hilbert‑curve ordering for spatial locality

Train/validation splits with leakage‑safe design

Model training, checkpointing, and metadata management

## Custom Spatial CNN architecture
Depthwise + pointwise convolutions

Spatial channels = 32

Hilbert‑ordered adjacency

Residual forecasting head

PyTorch + PyTorch Lightning

## Efficient training pipeline
GPU‑accelerated training

Self‑supervised residual prediction

Clean, reproducible training scripts

Automatic checkpoint saving

Metadata packaging for inference

## Interactive Streamlit app
Fast, cached inference

City‑level selection

Multi‑month forecast horizon

Historical ZHVI visualization

Forecast curve visualization

Plotly‑based interactive charts

## Production‑style engineering
Modular code structure

Cached data loaders

Deterministic preprocessing

Device‑agnostic inference

Clear separation of training vs. inference logic

## Model Overview
### SpatialCNNEncoder
A custom convolutional architecture designed to capture spatial relationships between cities. Instead of using geographic distance directly, cities are mapped onto a Hilbert curve, which preserves locality and allows 1D convolutions to approximate spatial adjacency.

### Residual Forecasting
The model predicts the next‑month residual change in log‑ZHVI for each city. This is a stable, self‑supervised target that avoids scale issues and allows the model to learn directional momentum.

### Recursive Multi‑Horizon Forecasting
For multi‑month forecasts, the model’s predicted residual is applied recursively. This produces a smooth, interpretable trend suitable for short‑term directional forecasting.

## Project Structure
Code
zillow_project/
│
├── data/                         # Checkpoints, metadata, embeddings
├── notebooks/                    # Exploratory analysis
├── src/
│   ├── _01_data.py               # Data loading + pivoting
│   ├── _02_features.py           # Feature engineering + scaling
│   ├── spatial_cnn.py            # Model architecture
│   ├── train_spatial_cnn.py      # Training loop
│   ├── generate_spatial_embeddings.py
│   ├── _04_streamlit.py          # Streamlit app
│   └── utils/                    # Helpers
│
└── README.md
## Streamlit App
The app provides:

## City selection
Choose from 17,771 cities with valid coordinates.

## Forecast horizon
1–12 months ahead.

## Visualizations
Historical ZHVI (interactive Plotly chart)

Forecast curve with last observed value

Forecasted ZHVI and percent change

## Performance
All heavy computations cached

Model inference is instant

Plotly charts render quickly

## Data Sources
Zillow Home Value Index (ZHVI) — city‑level housing price data

US Gazetteer — latitude/longitude for spatial modeling

Federal Reserve Bank of St. Louis — Fed funds, mortgage rates, CPI, Core CPI

## Results
The model produces:

Smooth, stable short‑term forecasts

City‑level directional signals

Spatially informed predictions

Fast inference suitable for interactive use

While the recursive residual method produces linear multi‑month trends, it is highly effective for short‑term directional forecasting and demonstrates the core spatial modeling capability.

## Lessons Learned
Designing and training a custom neural architecture

Encoding spatial structure using Hilbert curves

Building a large‑scale feature pipeline

Managing metadata for reproducible inference

Deploying a deep learning model in Streamlit

Debugging complex GPU, PyTorch, and Lightning issues

Structuring a real ML project like a production system

## Future Improvements
Multi‑horizon prediction head (predict 12 months directly)

Attention‑based temporal modeling

Seasonal decomposition

Uncertainty intervals

Similar‑city recommendations using embeddings

Model comparison (TFT, N‑BEATS, DeepAR)