"""Plotly figures describing how well the model predicts.

These read the stored metrics and scored test rows served by the API.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .figures import COLORWAY, styled_layout


def predicted_vs_actual(points: list[dict], column: str = "predicted_speed") -> go.Figure:
    """Observed versus predicted speed on the held-out test period."""
    figure = go.Figure()
    frame = pd.DataFrame(points)
    if frame.empty or not {"observed_speed", column}.issubset(frame.columns):
        return styled_layout(figure, "Predicted vs actual (no data)")
    frame = frame.dropna(subset=[column])
    sample = frame.sample(min(len(frame), 6000), random_state=0)
    figure.add_trace(
        go.Scattergl(
            x=sample["observed_speed"],
            y=sample[column],
            mode="markers",
            marker=dict(
                size=5,
                color=sample[column] - sample["observed_speed"],
                colorscale="RdBu",
                cmid=0,
                colorbar=dict(title="error (mph)"),
                opacity=0.7,
            ),
            hovertemplate="observed %{x:.1f} mph<br>predicted %{y:.1f} mph<extra></extra>",
        )
    )
    limits = [0, float(max(frame["observed_speed"].max(), frame[column].max())) * 1.03]
    figure.add_trace(
        go.Scatter(
            x=limits,
            y=limits,
            mode="lines",
            name="perfect prediction",
            line=dict(color="#2d3748", dash="dash", width=1.2),
        )
    )
    return styled_layout(
        figure,
        "Predicted vs observed speed (test period)",
        xaxis_title="observed speed (mph)",
        yaxis_title="predicted speed (mph)",
        xaxis=dict(range=limits),
        yaxis=dict(range=limits),
        height=430,
    )


def metrics_bar(results: dict) -> go.Figure:
    """MAE by predictor and split."""
    figure = go.Figure()
    rows = []
    for group in ("baselines", "models"):
        for name, splits in (results or {}).get(group, {}).items():
            for split, scores in splits.items():
                rows.append({"name": name, "split": split, "mae": scores["mae"]})
    if not rows:
        return styled_layout(figure, "Model performance (no data)")
    frame = pd.DataFrame(rows)
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        if subset.empty:
            continue
        figure.add_trace(
            go.Bar(x=subset["name"], y=subset["mae"], name=split)
        )
    return styled_layout(
        figure,
        "Mean absolute error by predictor",
        yaxis_title="MAE (mph)",
        barmode="group",
        height=380,
        xaxis=dict(tickangle=-20, tickfont=dict(size=9)),
    )


def feature_effects(coefficients: list[dict], top: int = 12) -> go.Figure:
    """Standardised linear coefficients, largest effect first."""
    figure = go.Figure()
    if not coefficients:
        return styled_layout(figure, "Feature effects (no data)")
    frame = pd.DataFrame(coefficients).head(top).iloc[::-1]
    colours = [COLORWAY[0] if value >= 0 else COLORWAY[1] for value in frame["std_coefficient"]]
    figure.add_trace(
        go.Bar(
            x=frame["std_coefficient"],
            y=frame["feature"],
            orientation="h",
            marker=dict(color=colours),
            hovertemplate="%{y}: %{x:.3f} standardised<extra></extra>",
        )
    )
    return styled_layout(
        figure,
        "What drives the linear prediction",
        xaxis_title="standardised coefficient",
        height=380,
        margin=dict(l=180, r=20, t=45, b=40),
    )
