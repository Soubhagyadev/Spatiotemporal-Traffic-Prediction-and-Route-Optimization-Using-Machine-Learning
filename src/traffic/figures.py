"""Diagnostic figures for the data-quality and model reports."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-traffic-config")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

FIGURE_DPI = 130
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def plot_speed_over_time(
    hourly: pd.DataFrame, split_bounds: dict, path: Path | None = None
) -> Path:
    """Daily mean speed across the whole observation period."""
    path = path or config.FIGURE_DIR / "speed_over_time.png"
    daily = hourly.set_index("hour_ts")["speed_mean"].resample("D").mean()
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(daily.index, daily.to_numpy(), color=BLUE, linewidth=1.1)
    axis.plot(
        daily.rolling(7, min_periods=3).mean().index,
        daily.rolling(7, min_periods=3).mean().to_numpy(),
        color=ORANGE,
        linewidth=1.8,
        label="7-day mean",
    )
    for name, (start, end, colour) in split_bounds.items():
        axis.axvspan(start, end, color=colour, alpha=0.08)
        axis.annotate(
            name,
            xy=(start + (end - start) / 2, axis.get_ylim()[1]),
            ha="center",
            va="top",
            fontsize=8,
            color=GREY,
        )
    axis.set_title("Mean traffic speed over time (all road segments)")
    axis.set_ylabel("speed (mph)")
    axis.legend(loc="lower right", fontsize=8)
    return _save(figure, path)


def plot_hourly_pattern(hourly: pd.DataFrame, path: Path | None = None) -> Path:
    """Average speed by hour of day, weekday versus weekend."""
    path = path or config.FIGURE_DIR / "hourly_pattern.png"
    frame = hourly.assign(
        hour=hourly["hour_ts"].dt.hour, weekend=hourly["hour_ts"].dt.dayofweek >= 5
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    for weekend, label, colour in ((False, "weekday", BLUE), (True, "weekend", ORANGE)):
        profile = frame[frame["weekend"] == weekend].groupby("hour")["speed_mean"].mean()
        axis.plot(profile.index, profile.to_numpy(), marker="o", markersize=3, label=label, color=colour)
    axis.set_title("Average speed by hour of day")
    axis.set_xlabel("hour of day")
    axis.set_ylabel("speed (mph)")
    axis.set_xticks(range(0, 24, 2))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_dow_hour_heatmap(hourly: pd.DataFrame, path: Path | None = None) -> Path:
    """Mean speed by day of week and hour (diurnal congestion structure)."""
    path = path or config.FIGURE_DIR / "weekday_hour_heatmap.png"
    frame = hourly.assign(
        hour=hourly["hour_ts"].dt.hour, day=hourly["hour_ts"].dt.dayofweek
    )
    grid = frame.pivot_table(index="day", columns="hour", values="speed_mean", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(10, 3.6))
    image = axis.imshow(grid.to_numpy(), aspect="auto", cmap="RdYlGn", origin="lower")
    axis.set_yticks(range(len(grid.index)))
    axis.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axis.set_xticks(range(0, 24, 2))
    axis.set_xticklabels(range(0, 24, 2))
    axis.set_xlabel("hour of day")
    axis.set_title("Mean speed (mph) by day of week and hour")
    figure.colorbar(image, ax=axis, label="mph")
    return _save(figure, path)


def plot_congestion_by_segment(
    hourly: pd.DataFrame, metadata: pd.DataFrame, thresholds: dict, path: Path | None = None
) -> Path:
    """Segments spending the largest share of hours in high congestion."""
    from .congestion import categorize

    path = path or config.FIGURE_DIR / "congestion_by_segment.png"
    frame = hourly.assign(category=categorize(hourly["speed_mean"], thresholds))
    share = (
        frame.assign(high=(frame["category"] == "High").astype(float))
        .groupby("link_id")["high"]
        .mean()
        .sort_values(ascending=False)
    )
    top = share.head(15)
    names = metadata.set_index("link_id")["link_name"].reindex(top.index).fillna("")
    labels = [
        f"{str(name)[:34]} ({link})" for link, name in zip(top.index, names.to_numpy())
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.barh(labels[::-1], (top.to_numpy() * 100)[::-1], color=ORANGE)
    axis.set_xlabel("share of observed hours classified as high congestion (%)")
    axis.set_title("Most congested road segments (training thresholds)")
    axis.tick_params(labelsize=7)
    return _save(figure, path)


def plot_predicted_vs_actual(
    y_true, y_pred, metrics: dict, path: Path | None = None
) -> Path:
    """Density of predicted versus observed speed on the test period."""
    path = path or config.FIGURE_DIR / "predicted_vs_actual.png"
    truth = np.asarray(y_true, dtype="float64")
    prediction = np.asarray(y_pred, dtype="float64")
    figure, axis = plt.subplots(figsize=(5.6, 5.2))
    image = axis.hexbin(truth, prediction, gridsize=60, cmap="viridis", bins="log", mincnt=1)
    limits = [0, min(90, max(truth.max(), prediction.max()))]
    axis.plot(limits, limits, color="white", linewidth=1.2, linestyle="--")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("observed speed (mph)")
    axis.set_ylabel("predicted speed (mph)")
    axis.set_title(
        f"Test period: MAE {metrics['mae']:.2f} mph, RMSE {metrics['rmse']:.2f} mph, "
        f"R² {metrics['r2']:.3f}",
        fontsize=9,
    )
    figure.colorbar(image, ax=axis, label="link-hours (log)")
    return _save(figure, path)


def plot_model_comparison(results: dict, metric: str = "mae", path: Path | None = None) -> Path:
    """Baseline and model scores side by side for each split."""
    path = path or config.FIGURE_DIR / f"model_comparison_{metric}.png"
    rows = []
    for group in ("baselines", "models"):
        for name, splits in results[group].items():
            for split, scores in splits.items():
                rows.append({"name": name, "split": split, metric: scores[metric]})
    frame = pd.DataFrame(rows)
    splits = [s for s in ("train", "val", "test") if s in set(frame["split"])]
    names = list(dict.fromkeys(frame["name"]))
    width = 0.8 / len(splits)
    figure, axis = plt.subplots(figsize=(9, 4.2))
    for index, split in enumerate(splits):
        values = [
            float(frame[(frame["name"] == name) & (frame["split"] == split)][metric].mean())
            for name in names
        ]
        axis.bar(np.arange(len(names)) + index * width, values, width, label=split)
    axis.set_xticks(np.arange(len(names)) + width * (len(splits) - 1) / 2)
    axis.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    axis.set_ylabel(f"{metric.upper()} (mph)")
    axis.set_title(f"{metric.upper()} by predictor and split")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_residuals(y_true, y_pred, hours, path: Path | None = None) -> Path:
    """Residual distribution and mean residual by hour of day."""
    path = path or config.FIGURE_DIR / "residuals.png"
    truth = np.asarray(y_true, dtype="float64")
    residual = truth - np.asarray(y_pred, dtype="float64")
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(np.clip(residual, -30, 30), bins=60, color=BLUE)
    axes[0].set_title(f"Residuals (mean {residual.mean():.2f} mph)")
    axes[0].set_xlabel("observed - predicted (mph)")
    by_hour = pd.Series(residual).groupby(np.asarray(hours)).mean()
    axes[1].bar(by_hour.index, by_hour.to_numpy(), color=ORANGE)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean residual by hour of day")
    axes[1].set_xlabel("hour of day")
    axes[1].set_xticks(range(0, 24, 2))
    return _save(figure, path)


def plot_feature_effects(
    coefficients: pd.DataFrame, importances: list[dict], path: Path | None = None
) -> Path:
    """Linear-model coefficients and random-forest importances."""
    path = path or config.FIGURE_DIR / "feature_effects.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    top = coefficients.head(15).iloc[::-1]
    axes[0].barh(top["feature"], top["std_coefficient"], color=BLUE)
    axes[0].set_title("Linear regression: standardised coefficients")
    axes[0].tick_params(labelsize=7)
    if importances:
        frame = pd.DataFrame(importances).head(15).iloc[::-1]
        axes[1].barh(frame["feature"], frame["importance"], color=ORANGE)
        axes[1].set_title("Random forest: impurity importance")
        axes[1].tick_params(labelsize=7)
    else:
        axes[1].axis("off")
    return _save(figure, path)


def plot_congestion_confusion(report: dict, path: Path | None = None) -> Path:
    """Confusion matrix between observed and predicted congestion categories."""
    from .congestion import CATEGORY_ORDER

    path = path or config.FIGURE_DIR / "congestion_confusion.png"
    matrix = pd.DataFrame(report["confusion_matrix"]).reindex(
        index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0
    )
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    axis.set_xticks(range(len(CATEGORY_ORDER)))
    axis.set_xticklabels(CATEGORY_ORDER)
    axis.set_yticks(range(len(CATEGORY_ORDER)))
    axis.set_yticklabels(CATEGORY_ORDER)
    axis.set_xlabel("predicted congestion")
    axis.set_ylabel("observed congestion")
    for row in range(len(CATEGORY_ORDER)):
        for column in range(len(CATEGORY_ORDER)):
            axis.text(
                column,
                row,
                f"{int(matrix.iloc[row, column]):,}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if matrix.iloc[row, column] < matrix.to_numpy().max() * 0.6 else "white",
            )
    axis.set_title(f"Category agreement: {report['accuracy']:.1%}", fontsize=9)
    figure.colorbar(image, ax=axis, label="link-hours")
    return _save(figure, path)
