"""Generate a realistic *synthetic* used-car dataset (CarDekho schema).

Prices follow a depreciation model with noise. It is meant for demos/tests so the
project runs out of the box. Replace data/car_data.csv with the real CarDekho
dataset for genuine results.

Usage:
    python -m src.generate_sample_data [--rows 1500]
"""
import argparse

import numpy as np
import pandas as pd

from src import config

# model name -> approximate new price (lakhs INR)
MODELS = {
    "Maruti Alto": 3.8, "Maruti Swift": 6.0, "Maruti Baleno": 7.5,
    "Maruti Dzire": 7.0, "Hyundai i10": 5.0, "Hyundai i20": 8.0,
    "Hyundai Verna": 11.0, "Hyundai Creta": 15.0, "Honda Amaze": 8.0,
    "Honda City": 12.0, "Toyota Innova": 22.0, "Toyota Fortuner": 38.0,
    "Ford EcoSport": 9.5, "Mahindra XUV500": 17.0, "Mahindra Scorpio": 14.0,
    "Tata Nexon": 10.0, "Tata Tiago": 5.5, "Renault Kwid": 4.5,
    "Volkswagen Polo": 8.0, "Skoda Octavia": 25.0,
}


def generate(n: int = 1500, seed: int = config.RANDOM_STATE,
             reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = rng.choice(list(MODELS), n)
    base = np.array([MODELS[m] for m in names])

    year = reference_year - rng.integers(1, 17, n)
    age = reference_year - year
    fuel = rng.choice(["Petrol", "Diesel", "CNG"], n, p=[0.55, 0.40, 0.05])
    seller = rng.choice(["Dealer", "Individual"], n, p=[0.35, 0.65])
    trans = rng.choice(["Manual", "Automatic"], n, p=[0.8, 0.2])
    owner = rng.choice([0, 1, 2, 3], n, p=[0.72, 0.2, 0.06, 0.02])
    kms = np.clip(age * rng.normal(11000, 4000, n), 1000, 300000).round(-2)

    present = base * rng.lognormal(0, 0.08, n)
    present *= np.where(fuel == "Diesel", 1.12, 1.0)
    present *= np.where(trans == "Automatic", 1.15, 1.0)

    price = present * np.exp(-0.13 * age) * (0.94 ** owner)
    price *= np.exp(-kms / 450000)
    price *= np.where(seller == "Dealer", 1.08, 1.0)
    price *= np.where(fuel == "CNG", 0.92, 1.0)
    price *= rng.lognormal(0, 0.09, n)

    return pd.DataFrame({
        "Car_Name": names, "Year": year,
        "Selling_Price": price.round(2).clip(min=0.2),
        "Present_Price": present.round(2),
        "Kms_Driven": kms.astype(int),
        "Fuel_Type": fuel, "Seller_Type": seller,
        "Transmission": trans, "Owner": owner,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1500)
    args = parser.parse_args()
    config.DATA_PATH.parent.mkdir(exist_ok=True)
    df = generate(args.rows)
    df.to_csv(config.DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {config.DATA_PATH}")


if __name__ == "__main__":
    main()
