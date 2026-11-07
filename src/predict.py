"""Predict the resale price of a used car.

Usage:
    python -m src.predict --brand Maruti --year 2018 --kms 45000 --fuel Petrol \
        --seller Individual --transmission Manual --owner 0 --present-price 6.5
"""
import argparse

import joblib
import pandas as pd

from src import config


def load_bundle(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("No trained model found. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_price(bundle, *, brand, year, kms_driven, fuel_type, seller_type,
                  transmission, owner=0, present_price=None) -> float:
    """Return the estimated selling price in lakhs of INR."""
    row = {
        "brand": brand.strip().title(),
        "car_age": bundle["reference_year"] - int(year),
        "kms_driven": kms_driven,
        "owner": owner,
        "fuel_type": fuel_type.title(),
        "seller_type": seller_type.title(),
        "transmission": transmission.title(),
        "present_price": present_price,
    }
    X = pd.DataFrame([row])[bundle["features"]]
    return float(bundle["model"].predict(X)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brand", required=True, help="First word of the car name, e.g. Maruti")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--kms", type=int, required=True)
    p.add_argument("--fuel", required=True, choices=["Petrol", "Diesel", "CNG"])
    p.add_argument("--seller", required=True, choices=["Dealer", "Individual"])
    p.add_argument("--transmission", required=True, choices=["Manual", "Automatic"])
    p.add_argument("--owner", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--present-price", type=float, default=None,
                   help="Current showroom price in lakhs (if the model uses it)")
    a = p.parse_args()

    bundle = load_bundle()
    price = predict_price(
        bundle, brand=a.brand, year=a.year, kms_driven=a.kms, fuel_type=a.fuel,
        seller_type=a.seller, transmission=a.transmission, owner=a.owner,
        present_price=a.present_price)
    print(f"Estimated resale price: ₹{price:.2f} lakhs (≈ ₹{price * 1e5:,.0f})")


if __name__ == "__main__":
    main()
