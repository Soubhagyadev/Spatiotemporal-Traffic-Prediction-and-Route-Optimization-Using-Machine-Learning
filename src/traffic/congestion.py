"""Congestion categories derived from dataset speed thresholds.

The plan asks for predicted speed to be mapped onto Low/Medium/High
congestion categories using thresholds derived from the data.  Thresholds are
always derived from the *training period* so that the categorisation cannot
leak information from validation or test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered from most to least congested.  Low speed means high congestion.
CATEGORY_ORDER = ["High", "Medium", "Low"]


def derive_thresholds(
    train_speed: pd.Series, low_quantile: float = 1 / 3, high_quantile: float = 2 / 3
) -> dict:
    """Speed cut-offs separating high / medium / low congestion.

    ``high_congestion_below`` is the speed under which traffic counts as
    highly congested; ``low_congestion_above`` is the free-flowing threshold.
    """
    speed = pd.Series(train_speed).dropna()
    below = float(speed.quantile(low_quantile))
    above = float(speed.quantile(high_quantile))
    return {
        "high_congestion_below_mph": round(below, 4),
        "low_congestion_above_mph": round(above, 4),
        "derived_from": "training period hourly mean speed",
        "train_speed_median_mph": round(float(speed.median()), 4),
        "train_rows": int(len(speed)),
    }


def categorize(speed, thresholds: dict) -> np.ndarray:
    """Map speeds to congestion labels (low speed -> high congestion)."""
    values = np.asarray(speed, dtype="float64")
    below = thresholds["high_congestion_below_mph"]
    above = thresholds["low_congestion_above_mph"]
    labels = np.where(values >= above, "Low", np.where(values >= below, "Medium", "High"))
    labels = np.where(np.isnan(values), "Unknown", labels)
    return labels.astype(object)


def free_flow_ratio(speed: pd.Series, free_flow_speed: pd.Series) -> pd.Series:
    """Speed as a share of each segment's own free-flow speed (p95)."""
    return (speed / free_flow_speed.replace(0.0, np.nan)).clip(upper=1.5)


def category_report(actual_speed, predicted_speed, thresholds: dict) -> dict:
    """Agreement between categories of observed and predicted speed."""
    actual = categorize(actual_speed, thresholds)
    predicted = categorize(predicted_speed, thresholds)
    total = len(actual)
    if total == 0:
        return {"rows": 0}
    matrix = pd.crosstab(
        pd.Series(actual, name="actual"),
        pd.Series(predicted, name="predicted"),
    ).reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)
    report = {
        "rows": int(total),
        "accuracy": round(float((actual == predicted).mean()), 4),
        "confusion_matrix": matrix.to_dict(),
        "per_category": {},
    }
    for category in CATEGORY_ORDER:
        mask = actual == category
        if mask.sum() == 0:
            continue
        report["per_category"][category] = {
            "support": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "recall": round(float((predicted[mask] == category).mean()), 4),
            "precision": round(
                float((actual[predicted == category] == category).mean())
                if (predicted == category).sum()
                else 0.0,
                4,
            ),
        }
    return report


def category_distribution(speed, thresholds: dict) -> dict:
    """Share of observations in each congestion category."""
    labels = categorize(speed, thresholds)
    counts = pd.Series(labels).value_counts(normalize=True)
    return {category: round(float(counts.get(category, 0.0)), 4) for category in CATEGORY_ORDER}
