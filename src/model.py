"""Model pipelines and hyper-parameter grids."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_pipeline(estimator, numeric, categorical):
    """Preprocessing + estimator, with log-transformed target.

    Car prices are right-skewed, so we model log(1 + price) and convert back.
    """
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    pipe = Pipeline([("prep", pre), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def candidates():
    """name -> (estimator, param grid). Grid keys are relative to the estimator."""
    return {
        "Ridge": (Ridge(), {"alpha": [0.1, 1, 10]}),
        "RandomForest": (
            RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 12],
             "min_samples_leaf": [1, 2]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=config.RANDOM_STATE),
            {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1],
             "max_depth": [2, 3]},
        ),
    }
