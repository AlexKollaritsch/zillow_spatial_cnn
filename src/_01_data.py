import pandas as pd
from pathlib import Path
import zipfile
import numpy as np
import sys

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._02_features import (
    filter_cities_with_coordinates,
    hilbert_order_cities,
    apply_hilbert_order,
    build_full_feature_matrix,
    scale_feature_groups,
)

DATA_DIR = PROJECT_ROOT / "data"

ZILLOW_FILE = DATA_DIR / "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
GAZ_ZIP_FILE = DATA_DIR / "2025_Gaz_place_national.zip"

FEDFUNDS_FILE = DATA_DIR / "FEDFUNDS.csv"
MORTGAGE_FILE = DATA_DIR / "MORTGAGE30US.csv"
CPIAUCSI_FILE = DATA_DIR / "CPIAUCSL.csv"
CPILFESL_FILE = DATA_DIR / "CPILFESL.csv"

# ============================================================
# Zillow
# ============================================================

def load_zillow_data():
    df = pd.read_csv(ZILLOW_FILE)

    # Keep only city-level rows
    df = df[df["RegionType"] == "city"].copy()

    def clean_zillow_city(name: str) -> str:
        if not isinstance(name, str):
            return ""
        name = name.replace("St.", "Saint")
        name = name.replace("St ", "Saint ")
        name = name.replace("Ft.", "Fort")
        name = name.replace("Ft ", "Fort ")
        return name.title()

    # Create unique city identifier
    df["CITY_CLEAN"] = df["RegionName"].apply(clean_zillow_city)
    df["CityState"] = df["CITY_CLEAN"] + ", " + df["StateName"]


    # Keep static features
    df["State"] = df["StateName"]
    df["County"] = df["CountyName"]

    return df


def collapse_duplicate_cities(df):
    # Zillow is sorted by SizeRank, so first = most relevant
    df = df.sort_values(["CityState", "SizeRank"], ascending=[True, True])

    # Identify ZHVI date columns
    date_cols = [c for c in df.columns if c[:4].isdigit()]

    # Compute averaged ZHVI
    df_vals = df.groupby("CityState")[date_cols].mean().reset_index()

    # Keep only the FIRST metadata row for each CityState
    df_meta = df.drop_duplicates(subset=["CityState"], keep="first")

    # Merge metadata with averaged ZHVI
    df_final = df_meta.drop(columns=date_cols).merge(df_vals, on="CityState", how="left")

    # ⭐ THIS IS THE CRITICAL FIX ⭐
    # Ensure df_final contains ONLY one row per CityState
    df_final = df_final.drop_duplicates(subset=["CityState"], keep="first")

    return df_final

# ============================================================
# Gazetteer (coords)
# ============================================================

def clean_gazetteer_city(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = name.replace("St.", "Saint")
    name = name.replace("St ", "Saint ")
    name = name.replace("Ft.", "Fort")
    name = name.replace("Ft ", "Fort ")

    # remove descriptors
    for token in [" city", " town", " village", " municipality", " borough", " cdp"]:
        if name.endswith(token):
            name = name.replace(token, "")

    # remove parentheses like "Springfield (CDP)"
    if "(" in name:
        name = name.split("(")[0].strip()

    return name.title()

def load_gazetteer(zip_path: Path = GAZ_ZIP_FILE) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        txt_file = [f for f in z.namelist() if f.endswith(".txt")][0]

        with z.open(txt_file) as f:
            df = pd.read_csv(f, sep="|", dtype=str)

    df["INTPTLAT"] = pd.to_numeric(df["INTPTLAT"], errors="coerce")
    df["INTPTLONG"] = pd.to_numeric(df["INTPTLONG"], errors="coerce")

    # Create cleaned city name
    df["CITY_CLEAN"] = df["NAME"].apply(clean_gazetteer_city)
    df["CityState"] = df["CITY_CLEAN"] + ", " + df["USPS"]

    return df

def merge_coords(df, gaz):
    df = df.merge(
        gaz[["CityState", "INTPTLAT", "INTPTLONG"]],
        on="CityState",
        how="left"
    )

    df = df.rename(columns={
        "INTPTLAT": "Latitude",
        "INTPTLONG": "Longitude"
    })

    return df

# ============================================================
# Manual coordinate overrides
# ============================================================

MANUAL_COORDS = {
    "Anchorage, AK": (61.2181, -149.9003),
    "Honolulu, HI": (21.3069, -157.8583),
    "Indianapolis, IN": (39.7684, -86.1581),
    "Nashville, TN": (36.1627, -86.7816),
    "Lexington, KY": (38.0406, -84.5037),
    "Saint Louis, MO": (38.6270, -90.1994),
    "Saint Paul, MN": (44.9537, -93.0900),
    "Saint Petersburg, FL": (27.7676, -82.6403),
    "Boise, ID": (43.6150, -116.2023),
    "Port Saint Lucie, FL": (27.2730, -80.3582),
    "Cypress, TX": (29.9187, -95.5603),
    "Henrico, VA": (37.5052, -77.3324),
    "Augusta, GA": (33.4735, -82.0105),
    "Macon, GA": (32.8407, -83.6324),
    "Saint Augustine, FL": (29.9012, -81.3124),
    "Athens, GA": (33.9519, -83.3576),
    "Ventura, CA": (34.2746, -119.2290),
    "Saint Charles, MO": (38.7881, -90.4975),
    "Edison, NJ": (40.5187, -74.4121),
    "Clinton Township, MI": (42.5803, -82.9185),
    "Lees Summit, MO": (38.9108, -94.3822),
    "Canton, MI": (42.3214, -83.4822),
    "Saint George, UT": (37.0965, -113.5684),
    "Macomb, MI": (42.5934, -82.9185),
    "Stafford, VA": (38.5831, -77.2644),
    "Saint Cloud, FL": (28.2484, -81.2812),
    "O Fallon, MO": (38.8106, -90.6998),
    "Chesterfield, VA": (37.3673, -77.6078),
    "Shelby Township, MI": (42.6061, -82.8648),
    "Saint Joseph, MO": (39.7675, -94.8467),
    "Land O Lakes, FL": (28.1492, -82.3878),
    "Brick, NJ": (40.0995, -74.0987),
    "Deland, FL": (29.0288, -81.3031),
    "Saint Cloud, MN": (45.5579, -94.1631),
    "Waterford, MI": (42.6553, -83.4148),
    "West Bloomfield, MI": (42.5834, -83.3773),
    "Carson City, NV": (39.1638, -119.7674),
    "Fountainbleau, FL": (25.9583, -80.1334),
    "West Chester, OH": (39.4020, -84.4089),
    "North Bergen Township, NJ": (40.7870, -74.0141),
    "Hamden, CT": (41.7773, -72.9059),
    "Piscataway, NJ": (40.5512, -74.4637),
    "Saint Peters, MO": (38.7879, -90.4975),
    "Irvington, NJ": (40.7378, -74.2298),
    "Fairfield, CT": (41.1408, -73.2610),
    "Jackson, NJ": (40.0968, -74.3284),
    "Desoto, TX": (32.6638, -96.8570),
    "Weymouth, MA": (42.2188, -70.9410),
    "Wayne, NJ": (40.9198, -74.2581),
    "Mililani, HI": (21.4324, -158.0031),
    "Lagrange, GA": (33.0361, -85.0468),
    "Saint Clair Shores, MI": (42.4977, -82.8964),
    "Bloomfield, NJ": (40.8081, -74.1938),
    "Fuquay Varina, NC": (35.4898, -78.7878),
    "Stratford, CT": (41.1918, -73.1952),
    "Saint Louis Park, MN": (44.9591, -93.3700),
    "Redford, MI": (42.2059, -83.1447),
    "East Brunswick, NJ": (40.4259, -74.3633),
    "West Orange, NJ": (40.7695, -74.2390),
    "Kailua, HI": (21.4028, -157.7394),
    "Cumberland, RI": (41.9228, -71.4370),
    "Coventry, RI": (41.7501, -71.4370),
    "Butte, MT": (46.0038, -112.5348),
    "Dartmouth, MA": (41.5840, -70.9975),
    "Fort Worth, TX": (32.7555, -97.3308),
}

def apply_manual_coordinates(df):
    for citystate, (lat, lon) in MANUAL_COORDS.items():
        mask = df["CityState"] == citystate
        df.loc[mask, "Latitude"] = lat
        df.loc[mask, "Longitude"] = lon
    return df


# ============================================================
# FRED macro data
# ============================================================

def load_fred_series(path, name):
    df = pd.read_csv(path)

    # Identify the date column
    if "observation_date" in df.columns:
        date_col = "observation_date"
    elif "DATE" in df.columns:
        date_col = "DATE"
    else:
        raise ValueError(f"No date column found in {path}")

    # Identify the value column (first non-date column)
    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) == 0:
        raise ValueError(f"No value column found in {path}")

    value_col = value_cols[0]

    # Rename to standard names
    df = df.rename(columns={
        date_col: "Date",
        value_col: name
    })

    # Convert to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=["Date"])

    # Set index
    df = df.set_index("Date").sort_index()

    return df

def load_macro_data():
    fed = load_fred_series(FEDFUNDS_FILE, "fed_funds")
    mort = load_fred_series(MORTGAGE_FILE, "mortgage_rate")
    cpi = load_fred_series(CPIAUCSI_FILE, "cpi_all")
    core = load_fred_series(CPILFESL_FILE, "cpi_core")

    macro = fed.join(mort, how="outer")
    macro = macro.join(cpi, how="outer")
    macro = macro.join(core, how="outer")

    return macro.sort_index().ffill()


# ============================================================
# Transform Zillow → long format
# ============================================================

def to_long(df):
    date_cols = [c for c in df.columns if c[:4].isdigit()]

    df_long = df.melt(
        id_vars=["CityState", "State", "County", "Latitude", "Longitude"],
        value_vars=date_cols,
        var_name="Date",
        value_name="ZHVI"
    )

    df_long["Date"] = pd.to_datetime(df_long["Date"])
    df_long["ZHVI"] = pd.to_numeric(df_long["ZHVI"], errors="coerce")

    df_long = df_long.sort_values(["CityState", "Date"])

    return df_long


# ============================================================
# Merge macro into long format
# ============================================================

def merge_macro(df_long, macro):
    # macro already has a Date column
    df = df_long.merge(macro, on="Date", how="left")
    df = df.sort_values(["CityState", "Date"])
    return df

# ============================================================
# Final pipeline (DeepAR-ready)
# ============================================================

def load_deepar_dataset():
    # Zillow
    df = load_zillow_data()
    # df = collapse_duplicate_cities(df)

    # Coordinates
    gaz = load_gazetteer()
    df = merge_coords(df, gaz)
    df = apply_manual_coordinates(df)
    df = collapse_duplicate_cities(df)

    # Long format
    df_long = to_long(df)

    # Convert Zillow end-of-month dates → month-start
    df_long["Date"] = df_long["Date"].values.astype("datetime64[M]")

    # Fill missing ZHVI
    df_long["ZHVI"] = df_long.groupby("CityState")["ZHVI"].ffill().bfill()

    # Macro
    macro = load_macro_data()

    # Align macro to monthly Zillow dates
    date_min = df_long["Date"].min()
    date_max = df_long["Date"].max()
    monthly_index = pd.date_range(start=date_min, end=date_max, freq="MS")

    # Reindex first (this drops the index name)
    macro = macro.reindex(monthly_index).ffill()

    # Restore index name AFTER reindexing
    macro.index.name = "Date"

    # Convert index → column
    macro = macro.reset_index()

    # Merge
    df_final = merge_macro(df_long, macro)

    return df_final

# ============================================================
# CNN-ready processed dataset (restores old load_processed_data)
# ============================================================

def load_processed_data():
    """
    Returns the exact fields needed for Spatial CNN training and
    for generating spatial embeddings.
    """

    print("DEBUG: load_processed_data() called from:", __file__)

    # Load long-format DeepAR dataset
    df_long = load_deepar_dataset()

    # Pivot to (time, cities)
    df_pivot = df_long.pivot(index="Date", columns="CityState", values="ZHVI").sort_index()

    # Log-transform
    df_pivot_log = np.log(df_pivot)

    # Coordinates (needed for Hilbert ordering)
    coords = df_long.drop_duplicates("CityState")[["CityState", "Latitude", "Longitude"]]
    coords = coords.set_index("CityState").sort_index()

    # Filter cities missing coords
    df_pivot_f, coords_f = filter_cities_with_coordinates(df_pivot_log, coords)

    # Hilbert ordering
    order = hilbert_order_cities(coords_f)
    df_pivot_h, coords_h = apply_hilbert_order(df_pivot_f, coords_f, order)

    # Build full feature matrix
    full_features = build_full_feature_matrix(df_pivot_h)

    num_cities = df_pivot_h.shape[1]
    total_cols = full_features.shape[1]
    num_features = total_cols // num_cities

    # Scale feature groups
    full_scaled, means, stds = scale_feature_groups(full_features, num_cities)

    # Compute residual std per city (needed for residual model)
    residuals = df_pivot_log.diff().iloc[1:]
    resid_std = residuals.std()

    return (
        df_pivot_h,
        df_pivot_log,
        full_features,
        means,
        stds,
        num_features,
        num_cities,
        resid_std
    )

# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":
    df = load_deepar_dataset()
    d = load_processed_data()
    print(df.head())
    print(df[['Latitude', 'Longitude']].head())
    print(df.shape)
    print(df.columns)
    print('NaN values: ', df.isna().sum())
    print(df[['CityState', 'Latitude', 'Longitude']])
    print("Percent of cities with coordinates: ", df['Latitude'].notna().mean())
    print("Coordinates for Specific City (e.g., 'Anchorage, AK'): ", df[df['CityState'] == 'Fort Worth, TX'][['Latitude', 'Longitude']].head(1))
    print("CNN cities:", d["num_cities"])
    print("DeepAR cities:", df["CityState"].nunique())