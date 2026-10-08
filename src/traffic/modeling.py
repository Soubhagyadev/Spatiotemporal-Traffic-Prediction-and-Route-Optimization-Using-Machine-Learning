"""Model fitting and evaluation for hourly traffic-speed prediction.

Primary model: multiple linear regression (interpretable, course scope).
Comparison model: random forest (non-linear reference), trained on a
subsample so that the comparison stays cheap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
RF_SAMPLE_SIZE = 200_000
RF_PARAMS = {
    "n_estimators": 60,
    "max_depth": 24,
    "min_samples_leaf": 5,
    "max_features": 0.5,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def evaluate(y_true, y_pred) -> dict:
    """MAE, RMSE, R² and median absolute error."""
    truth = np.asarray(y_true, dtype="float64")
    prediction = np.asarray(y_pred, dtype="float64")
    residual = truth - prediction
    return {
        "rows": int(len(truth)),
        "mae": round(float(mean_absolute_error(truth, prediction)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(truth, prediction))), 4),
        "r2": round(float(r2_score(truth, prediction)), 4),
        "median_abs_error": round(float(np.median(np.abs(residual))), 4),
        "bias": round(float(residual.mean()), 4),
    }


def fit_linear_regression(x_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(x_train.to_numpy(dtype="float64"), y_train.to_numpy(dtype="float64"))
    return model


def fit_random_forest(
    x_train: pd.DataFrame, y_train: pd.Series, sample_size: int = RF_SAMPLE_SIZE
) -> tuple[RandomForestRegressor, int]:
    """Fit the comparison model on a random subsample of the training rows."""
    if len(x_train) > sample_size:
        generator = np.random.default_rng(RANDOM_STATE)
        rows = generator.choice(len(x_train), size=sample_size, replace=False)
        x_sample = x_train.iloc[rows]
        y_sample = y_train.iloc[rows]
    else:
        x_sample, y_sample = x_train, y_train
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(x_sample.to_numpy(dtype="float32"), y_sample.to_numpy(dtype="float64"))
    return model, len(x_sample)


def predict(model, features: pd.DataFrame) -> np.ndarray:
    """Predict, tolerating the float32 input used by the forest."""
    values = features.to_numpy(dtype="float32" if isinstance(model, RandomForestRegressor) else "float64")
    return np.asarray(model.predict(values), dtype="float64")


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reference predictors the models must beat."""
    predictions = {
        "persistence_lag_1h": frame["speed_lag_1h"].to_numpy(dtype="float64"),
        "link_hour_profile": frame["link_hour_speed_profile_enc"].to_numpy(dtype="float64"),
    }
    return predictions


def evaluate_splits(
    models: dict, splits: dict[str, pd.DataFrame], features: list[str], target: str
) -> dict:
    """Evaluate baselines and models on every split."""
    results: dict[str, dict] = {"baselines": {}, "models": {}}
    for split_name, frame in splits.items():
        truth = frame[target]
        for name, prediction in baseline_predictions(frame).items():
            results["baselines"].setdefault(name, {})[split_name] = evaluate(truth, prediction)
        for name, model in models.items():
            results["models"].setdefault(name, {})[split_name] = evaluate(
                truth, predict(model, frame[features])
            )
    return results


def coefficient_table(model: LinearRegression, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Raw and standardised linear coefficients, sorted by absolute effect."""
    coefficients = np.asarray(model.coef_, dtype="float64")
    x_std = x_train.to_numpy(dtype="float64").std(axis=0)
    y_std = float(y_train.to_numpy(dtype="float64").std())
    table = pd.DataFrame(
        {
            "feature": list(x_train.columns),
            "coefficient": coefficients,
            "std_coefficient": coefficients * x_std / (y_std if y_std else 1.0),
            "feature_std": x_std,
        }
    )
    return table.reindex(table["std_coefficient"].abs().sort_values(ascending=False).index)


def variance_inflation_factors(
    features: pd.DataFrame, sample_size: int = 50_000
) -> dict[str, float]:
    """Variance inflation factor per feature (multicollinearity diagnostics).

    A feature with no variance carries no information and cannot be regressed
    on the others, so it is reported as ``inf``.
    """
    if len(features) > sample_size:
        generator = np.random.default_rng(RANDOM_STATE)
        features = features.iloc[generator.choice(len(features), sample_size, replace=False)]
    values = features.to_numpy(dtype="float64")
    scores: dict[str, float] = {}
    for index, column in enumerate(features.columns):
        if float(values[:, index].std()) == 0.0:
            scores[column] = float("inf")
            continue
        others = np.delete(values, index, axis=1)
        model = LinearRegression().fit(others, values[:, index])
        r_squared = float(model.score(others, values[:, index]))
        scores[column] = round(float(1.0 / max(1e-9, 1.0 - r_squared)), 3)
    return scores


def feature_importance(model, columns: list[str], top: int = 20) -> list[dict]:
    """Importance list for tree models (impurity importance)."""
    if not hasattr(model, "feature_importances_"):
        return []
    importance = np.asarray(model.feature_importances_, dtype="float64")
    order = np.argsort(importance)[::-1][:top]
    return [
        {"feature": columns[i], "importance": round(float(importance[i]), 5)} for i in order
    ]


def residual_diagnostics(frame: pd.DataFrame, prediction: np.ndarray, target: str) -> dict:
    """Where the model is wrong: per hour, per weekday and worst segments."""
    residuals = frame[target].to_numpy(dtype="float64") - prediction
    diagnostics = frame[["link_id", "hour_ts"]].copy()
    diagnostics["residual"] = residuals
    by_hour = diagnostics.groupby(diagnostics["hour_ts"].dt.hour)["residual"].mean().round(3)
    by_weekday = diagnostics.groupby(diagnostics["hour_ts"].dt.dayofweek)["residual"].mean().round(3)
    by_link = (
        diagnostics.assign(abs_residual=np.abs(residuals))
        .groupby("link_id")["abs_residual"]
        .agg(["mean", "size"])
        .sort_values("mean", ascending=False)
    )
    return {
        "mean_residual_by_hour": {int(k): float(v) for k, v in by_hour.items()},
        "mean_residual_by_weekday": {int(k): float(v) for k, v in by_weekday.items()},
        "worst_links": [
            {"link_id": int(link), "mae": round(float(row["mean"]), 3), "rows": int(row["size"])}
            for link, row in by_link.head(10).iterrows()
        ],
    }
