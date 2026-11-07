"""Central configuration: paths, constants and feature lists."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "car_data.csv"
MODEL_PATH = ROOT / "models" / "car_price_model.joblib"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
REFERENCE_YEAR = date.today().year  # used to compute car age

TARGET = "selling_price"  # in lakhs of INR
NUMERIC_FEATURES = ["car_age", "kms_driven", "owner", "present_price"]
CATEGORICAL_FEATURES = ["brand", "fuel_type", "seller_type", "transmission"]
