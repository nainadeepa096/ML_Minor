import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config, data
from src.generate_sample_data import generate
from src.model import build_pipeline, candidates

REF_YEAR = 2026


@pytest.fixture(scope="module")
def clean_df():
    return data.clean(generate(600, seed=1, reference_year=REF_YEAR), REF_YEAR)


def test_clean_adds_features(clean_df):
    assert {"car_age", "brand"} <= set(clean_df.columns)
    assert (clean_df["car_age"] >= 0).all()
    assert clean_df["selling_price"].gt(0).all()


def test_clean_removes_bad_rows():
    raw = generate(50, seed=2, reference_year=REF_YEAR)
    bad = raw.iloc[:1].copy()
    bad["Selling_Price"] = -5
    out = data.clean(pd.concat([raw, bad]), REF_YEAR)
    assert (out["selling_price"] > 0).all()


def test_clean_requires_columns():
    with pytest.raises(ValueError):
        data.clean(pd.DataFrame({"foo": [1]}), REF_YEAR)


def test_alternate_schema_in_rupees():
    raw = pd.DataFrame({
        "name": ["Maruti 800 AC"], "year": [2012], "selling_price": [250000],
        "km_driven": [50000], "fuel": ["Petrol"], "seller_type": ["Individual"],
        "transmission": ["Manual"], "owner": ["First Owner"]})
    out = data.clean(raw, REF_YEAR)
    assert out.loc[0, "selling_price"] == pytest.approx(2.5)
    assert out.loc[0, "brand"] == "Maruti"


def test_model_learns_signal(clean_df):
    num, cat = data.feature_columns(clean_df)
    X, y = clean_df[num + cat], clean_df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    est, _ = candidates()["Ridge"]
    model = build_pipeline(est, num, cat).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    assert (pred > 0).all()
    assert model.score(X_te, y_te) > 0.8


def test_older_cars_cost_less(clean_df):
    num, cat = data.feature_columns(clean_df)
    est, _ = candidates()["Ridge"]
    model = build_pipeline(est, num, cat).fit(clean_df[num + cat], clean_df[config.TARGET])
    base = clean_df[num + cat].iloc[[0]].copy()
    young, old = base.copy(), base.copy()
    young["car_age"], old["car_age"] = 2, 12
    assert model.predict(young)[0] > model.predict(old)[0]
