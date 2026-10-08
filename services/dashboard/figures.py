"""Plotly figures describing how the traffic behaves.

Each function takes the JSON payload returned by the Flask API and returns a
ready-to-render figure.  Model-diagnostic figures live in
:mod:`services.dashboard.model_figures`; both share the style helpers here.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

SPEED_SCALE = "RdYlGn"
COLORWAY = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c"]
BASE_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=55, r=20, t=45, b=40),
    colorway=COLORWAY,
    hovermode="closest",
)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def styled_layout(figure: go.Figure, title: str, **overrides) -> go.Figure:
    figure.update_layout(**BASE_LAYOUT, title=title)
    figure.update_layout(**overrides)
    return figure


def speed_over_time(points: list[dict], granularity: str = "day", scope: str = "") -> go.Figure:
    """Mean speed per day/hour, with a 7-point rolling mean for daily data."""
    figure = go.Figure()
    if not points:
        return styled_layout(figure, "Traffic speed over time (no data)")
    frame = pd.DataFrame(points)
    figure.add_trace(
        go.Scatter(
            x=frame["bucket"],
            y=frame["mean_speed_mph"],
            mode="lines",
            name="mean speed",
            line=dict(width=1.4),
        )
    )
    if granularity == "day" and len(frame) > 7:
        figure.add_trace(
            go.Scatter(
                x=frame["bucket"],
                y=frame["mean_speed_mph"].rolling(7, min_periods=3).mean(),
                mode="lines",
                name="7-day mean",
                line=dict(width=2.6),
            )
        )
    return styled_layout(
        figure,
        f"Traffic speed over time{' — ' + scope if scope else ''}",
        yaxis_title="speed (mph)",
        xaxis_title=granularity,
    )


def hourly_pattern(points: list[dict]) -> go.Figure:
    """Average speed by hour of day, weekday versus weekend."""
    figure = go.Figure()
    if not points:
        return styled_layout(figure, "Hourly pattern (no data)")
    frame = pd.DataFrame(points).sort_values("hour")
    for day_type, colour in (("weekday", COLORWAY[0]), ("weekend", COLORWAY[1])):
        subset = frame[frame["day_type"] == day_type]
        figure.add_trace(
            go.Scatter(
                x=subset["hour"],
                y=subset["mean_speed_mph"],
                name=day_type,
                mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(size=5),
            )
        )
    return styled_layout(
        figure,
        "Average speed by hour of day",
        xaxis_title="hour of day",
        yaxis_title="speed (mph)",
        xaxis=dict(dtick=2),
    )


def weekday_hour_heatmap(points: list[dict]) -> go.Figure:
    """Mean speed per (day of week, hour) cell."""
    figure = go.Figure()
    if not points:
        return styled_layout(figure, "Weekday × hour pattern (no data)")
    grid = pd.DataFrame(points).pivot_table(
        index="weekday", columns="hour", values="mean_speed_mph", aggfunc="mean"
    )
    figure.add_trace(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[int(column) for column in grid.columns],
            y=[WEEKDAYS[int(index)] for index in grid.index],
            colorscale=SPEED_SCALE,
            colorbar=dict(title="mph"),
        )
    )
    return styled_layout(
        figure,
        "Mean speed by day of week and hour",
        xaxis_title="hour of day",
        yaxis=dict(autorange="reversed"),
    )


def congestion_bar(segments: list[dict]) -> go.Figure:
    """Share of hours spent in high congestion, most congested first."""
    figure = go.Figure()
    if not segments:
        return styled_layout(figure, "Congestion by segment (no data)")
    frame = pd.DataFrame(segments).iloc[::-1]
    labels = [
        f"{str(name)[:34]} ({link})"
        for name, link in zip(frame["link_name"], frame["link_id"])
    ]
    figure.add_trace(
        go.Bar(
            x=frame["high_congestion_share_pct"],
            y=labels,
            orientation="h",
            marker=dict(color=COLORWAY[1]),
            customdata=frame[["mean_speed_mph", "borough", "link_hours"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>high congestion: %{x:.1f}% of hours"
                "<br>mean speed: %{customdata[0]:.1f} mph"
                "<br>borough: %{customdata[1]}<br>link-hours: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    return styled_layout(
        figure,
        "Most congested road segments",
        xaxis_title="share of hours in high congestion (%)",
        height=430,
        margin=dict(l=250, r=20, t=45, b=40),
    )


def borough_bar(boroughs: list[dict]) -> go.Figure:
    """High versus low congestion share per borough."""
    figure = go.Figure()
    if not boroughs:
        return styled_layout(figure, "Congestion by borough (no data)")
    frame = pd.DataFrame(boroughs)
    figure.add_trace(
        go.Bar(
            x=frame["borough"],
            y=frame["high_congestion_share_pct"],
            name="high congestion",
            marker=dict(color=COLORWAY[1]),
        )
    )
    figure.add_trace(
        go.Bar(
            x=frame["borough"],
            y=frame["low_congestion_share_pct"],
            name="free flowing",
            marker=dict(color=COLORWAY[2]),
        )
    )
    return styled_layout(
        figure,
        "Congestion mix by borough",
        yaxis_title="share of link-hours (%)",
        barmode="group",
        height=430,
    )


def segment_map(links: list[dict], metric: str = "mean_speed_mph") -> go.Figure:
    """Segment start points on an OpenStreetMap base layer."""
    figure = go.Figure()
    frame = pd.DataFrame(links)
    required = {"start_lat", "start_lon", metric}
    if frame.empty or not required.issubset(frame.columns):
        return styled_layout(figure, "Segment locations (no coordinates)")
    frame = frame.dropna(subset=["start_lat", "start_lon"])
    figure.add_trace(
        go.Scattermap(
            lat=frame["start_lat"],
            lon=frame["start_lon"],
            mode="markers",
            marker=dict(
                size=9,
                color=frame[metric],
                colorscale=SPEED_SCALE,
                colorbar=dict(title="mph"),
                showscale=True,
            ),
            text=frame["link_name"],
            customdata=frame[["link_id", "borough", "hours_observed"]].to_numpy(),
            hovertemplate=(
                "%{text}<br>link %{customdata[0]} · %{customdata[1]}"
                "<br>mean speed: %{marker.color:.1f} mph"
                "<br>hours observed: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **BASE_LAYOUT,
        title="Road segments (colour = mean speed)",
        map=dict(style="open-street-map", center=dict(lat=40.72, lon=-73.95), zoom=9.2),
        height=470,
    )
    return figure


def segment_detail(history: list[dict], forecast: dict | None, name: str = "") -> go.Figure:
    """Observed hourly speed for one segment plus the forecast horizon."""
    figure = go.Figure()
    if not history:
        return styled_layout(figure, "Segment detail (no data)")
    frame = pd.DataFrame(history)
    figure.add_trace(
        go.Scatter(
            x=frame["hour_ts"],
            y=frame["speed_mean"],
            mode="lines",
            name="observed",
            line=dict(width=1.3, color=COLORWAY[0]),
        )
    )
    predictions = (forecast or {}).get("predictions", [])
    if predictions:
        future = pd.DataFrame(predictions)
        figure.add_trace(
            go.Scatter(
                x=future["hour_ts"],
                y=future["predicted_speed_mph"],
                mode="lines+markers",
                name="forecast",
                line=dict(width=2.4, color=COLORWAY[1], dash="dot"),
                marker=dict(size=7),
            )
        )
    figure.add_hline(
        y=(forecast or {}).get("congestion_reference", 31.53),
        line=dict(color="#718096", dash="dash", width=1),
        annotation_text="high-congestion threshold",
        annotation_position="bottom right",
    )
    return styled_layout(
        figure,
        f"Segment detail{': ' + name if name else ''}",
        xaxis_title="hour",
        yaxis_title="speed (mph)",
        height=380,
    )


