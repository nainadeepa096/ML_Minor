"""Loading, cleaning and feature engineering for used-car listings.

Supports two common CarDekho layouts from Kaggle:
  1. "Vehicle dataset from CarDekho" (Car_Name, Year, Selling_Price, Present_Price,
     Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner)
  2. "CAR DETAILS FROM CAR DEKHO" (name, year, selling_price [INR], km_driven,
     fuel, seller_type, transmission, owner)
"""
from pathlib import Path

import pandas as pd

from src import config

ALIASES = {"name": "car_name", "km_driven": "kms_driven", "fuel": "fuel_type"}
OWNER_MAP = {
    "first owner": 0, "second owner": 1, "third owner": 2,
    "fourth & above owner": 3, "test drive car": 0,
}
REQUIRED = ["car_name", "year", "selling_price", "kms_driven",
            "fuel_type", "seller_type", "transmission", "owner"]


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate_sample_data` or place "
            "the CarDekho CSV there (see README).")
    return pd.read_csv(path)


def clean(df: pd.DataFrame, reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    """Standardise columns, fix types, drop bad rows and add engineered features."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=ALIASES)

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if not pd.api.types.is_numeric_dtype(df["owner"]):
        df["owner"] = df["owner"].astype(str).str.strip().str.lower().map(OWNER_MAP)

    for col in ("fuel_type", "seller_type", "transmission"):
        df[col] = df[col].astype(str).str.strip().str.title()

    numeric_cols = ["year", "selling_price", "kms_driven", "owner"]
    if "present_price" in df.columns:
        numeric_cols.append("present_price")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates().dropna(subset=REQUIRED)

    # Prices given in rupees -> convert to lakhs
    if df["selling_price"].median() > 1000:
        df["selling_price"] = df["selling_price"] / 1e5
        if "present_price" in df.columns:
            df["present_price"] = df["present_price"] / 1e5

    df = df[(df["selling_price"] > 0)
            & (df["kms_driven"].between(0, 1_000_000))
            & (df["year"].between(1980, reference_year))]

    df["owner"] = df["owner"].clip(0, 3).astype(int)
    df["brand"] = df["car_name"].astype(str).str.strip().str.split().str[0].str.title()
    df["car_age"] = reference_year - df["year"]
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame):
    """Return (numeric, categorical) feature lists that exist in the data."""
    num = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    return num, cat
