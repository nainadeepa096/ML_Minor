"""Train, tune, compare and save the best used-car price model.

Usage:
    python -m src.train [--data data/car_data.csv]
"""
import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src import config, data
from src.model import build_pipeline, candidates


def evaluate(y_true, y_pred):
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae_lakhs": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse_lakhs": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def save_plots(model, X_test, y_test, pred, name):
    config.REPORTS_DIR.mkdir(exist_ok=True)

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, pred, alpha=0.5, s=14, color="#2a6f97")
    lim = [0, max(y_test.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="Actual price (lakhs)", ylabel="Predicted price (lakhs)",
           title=f"Actual vs Predicted - {name}")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_test - pred, bins=40, color="#2a6f97", edgecolor="white")
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set(xlabel="Residual (lakhs)", ylabel="Count", title="Residual distribution")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "residuals.png", dpi=150)
    plt.close(fig)

    # Permutation importance on raw input columns
    imp = permutation_importance(model, X_test, y_test, n_repeats=10,
                                 random_state=config.RANDOM_STATE, scoring="r2")
    s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    s.plot.barh(ax=ax, color="#2a6f97")
    ax.set(xlabel="Drop in R² when shuffled", title="Feature importance (permutation)")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(config.DATA_PATH))
    args = parser.parse_args()

    df = data.clean(data.load_raw(args.data))
    numeric, categorical = data.feature_columns(df)
    features = numeric + categorical
    print(f"Loaded {len(df)} clean rows | features: {features}")

    X, y = df[features], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    rows, fitted = [], {}
    for name, (est, grid) in candidates().items():
        pipe = build_pipeline(est, numeric, categorical)
        grid = {f"regressor__model__{k}": v for k, v in grid.items()}
        search = GridSearchCV(pipe, grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        m = evaluate(y_test, pred)
        rows.append({"model": name, "cv_mae_lakhs": round(-search.best_score_, 4), **m,
                     "best_params": {k.split("__", 2)[-1]: v
                                     for k, v in search.best_params_.items()}})
        fitted[name] = search.best_estimator_
        print(f"{name:<17} CV MAE={-search.best_score_:.3f}  "
              f"Test R2={m['r2']:.3f}  MAE={m['mae_lakhs']:.3f} lakhs")

    # Select by cross-validation score (not test score) to avoid test-set leakage
    results = pd.DataFrame(rows).sort_values("cv_mae_lakhs")
    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    config.REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"\nBest model (by CV MAE): {best_name}")

    save_plots(best, X_test, y_test, best.predict(X_test), best_name)

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": best, "features": features, "model_name": best_name,
        "reference_year": config.REFERENCE_YEAR,
        "brands": sorted(df["brand"].unique()),
        "metrics": results.iloc[0][["r2", "mae_lakhs", "rmse_lakhs"]].to_dict(),
    }, config.MODEL_PATH)
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
