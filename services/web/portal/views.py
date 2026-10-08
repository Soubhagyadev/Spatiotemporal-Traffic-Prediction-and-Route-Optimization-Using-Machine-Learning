"""Views for the traffic portal.

Every page reads its data from the Flask API through ``portal.api_client`` and
degrades to empty tables plus a warning banner when the API is unavailable.
"""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
from django.conf import settings
from django.http import Http404
from django.shortcuts import render
from plotly.offline import plot

from . import api_client
from .forms import PredictionForm

CHART_HOURS = 720
SPLITS = ["train", "val", "test"]
CATEGORIES = ["High", "Medium", "Low"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _page(client, active: str, title: str, **extra) -> dict:
    """Banner flags plus page identity; call after the view's API calls."""
    error = client.last_error
    available = client.is_available()
    return {
        "dashboard_url": settings.TRAFFIC_DASHBOARD_URL,
        "api_available": available,
        "api_error": error or client.last_error,
        "active": active,
        "title": title,
        **extra,
    }


def _next_hour(timestamp) -> str | None:
    """``timestamp`` plus one hour, formatted the way the API expects."""
    if not timestamp:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(timestamp).replace(" ", "T", 1))
    except ValueError:
        return None
    return (parsed + dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")


def _speed_chart(points: list[dict], link: dict) -> str:
    """Hourly speed history (last 30 days) as a Plotly div."""
    window = points[-CHART_HOURS:]
    figure = go.Figure(
        go.Scatter(
            x=[point.get("hour_ts") for point in window],
            y=[point.get("speed_mean") for point in window],
            mode="lines",
            name="Hourly mean speed",
            line={"color": "#2563eb", "width": 1.4},
        )
    )
    if link.get("mean_speed_mph") is not None:
        figure.add_hline(
            y=link["mean_speed_mph"],
            line_dash="dot",
            line_color="#64748b",
            annotation_text="segment mean",
            annotation_position="top left",
        )
    figure.update_layout(
        height=360,
        margin={"l": 55, "r": 20, "t": 25, "b": 45},
        template="plotly_white",
        xaxis_title="Hour",
        yaxis_title="Speed (mph)",
        hovermode="x unified",
        showlegend=False,
    )
    return plot(figure, include_plotlyjs="cdn", output_type="div")


def _weekday(day: int) -> str:
    return WEEKDAYS[day] if 0 <= day < len(WEEKDAYS) else str(day)


def _congestion_share(row: dict) -> float:
    value = row.get("high_congestion_share_pct")
    return -1.0 if value is None else value


SORT_KEYS = {
    "name": lambda row: (row.get("link_name") or "").lower(),
    "speed": lambda row: row.get("mean_speed_mph") or 0.0,
    "congestion": _congestion_share,
}


def overview(request):
    """Landing page: dataset window, KPI cards, borough and segment tables."""
    client = api_client.get_client()
    meta = client.meta()
    models = (client.model_metrics().get("results") or {}).get("models") or {}
    headline = [
        {"name": name, **(splits.get("test") or {})} for name, splits in sorted(models.items())
    ]
    return render(request, "portal/overview.html", _page(
        client, "overview", "Overview",
        dataset=meta.get("dataset") or {},
        model_info=meta.get("model") or {},
        thresholds=meta.get("congestion_thresholds") or {},
        summary=client.congestion_summary(),
        boroughs=client.congestion_boroughs(),
        segments=client.congestion_segments(limit=10),
        headline=headline,
    ))


def segments(request):
    """Searchable, sortable table of every segment known to the API."""
    client = api_client.get_client()
    borough = (request.GET.get("borough") or "").strip()
    query = (request.GET.get("q") or "").strip().lower()
    sort_key = request.GET.get("sort") or "congestion"
    if sort_key not in SORT_KEYS:
        sort_key = "congestion"

    congestion = {
        row.get("link_id"): row
        for row in client.congestion_segments(limit=200, borough=borough or None)
    }
    rows = []
    for link in client.links(borough or None):
        stats = congestion.get(link.get("link_id")) or {}
        row = {
            **link,
            "high_congestion_share_pct": stats.get("high_congestion_share_pct"),
        }
        haystack = f"{row.get('link_name') or ''} {row.get('link_id')}".lower()
        if query and query not in haystack:
            continue
        rows.append(row)
    rows.sort(key=SORT_KEYS[sort_key], reverse=sort_key in {"speed", "congestion"})
    return render(request, "portal/segments.html", _page(
        client, "segments", "Segments",
        rows=rows,
        boroughs=client.meta().get("boroughs") or [],
        selected_borough=borough,
        query=request.GET.get("q") or "",
        sort_key=sort_key,
    ))


def segment_detail(request, link_id: int):
    """One segment: metadata, recent hours, congestion summary, chart, forecast."""
    client = api_client.get_client()
    link = client.link(link_id)
    if not link and client.is_available():
        raise Http404(f"Segment {link_id} is unknown to the traffic API.")
    history = client.link_history(link_id, "hour")
    start_hour = _next_hour(link.get("last_hour_ts"))
    return render(request, "portal/segment_detail.html", _page(
        client, "segments", f"Segment {link_id}",
        link=link,
        recent_hours=list(reversed((link.get("recent_hours") or [])[-24:])),
        summary=client.congestion_summary(link_id=link_id),
        chart=_speed_chart(history, link),
        chart_hours=min(len(history), CHART_HOURS),
        history_hours=len(history),
        forecast=api_client.horizon(link_id, start_hour, steps=6) if start_hour else [],
        forecast_start=start_hour,
    ))


def predict_view(request):
    """Single-step prediction plus an optional multi-step forecast."""
    client = api_client.get_client()
    form = PredictionForm(
        request.POST or None, link_choices=api_client.link_choices(client.links())
    )
    if request.method == "GET":
        dataset_end = (client.meta().get("dataset") or {}).get("end")
        form.initial.setdefault("hour_ts", _next_hour(dataset_end) or "")

    prediction: dict = {}
    steps: list[dict] = []
    failure = None
    if request.method == "POST" and form.is_valid():
        link_id = int(form.cleaned_data["link_id"])
        hour_ts = form.cleaned_data["hour_ts"]
        model = form.cleaned_data["model"]
        prediction = api_client.predict(link_id, hour_ts, model)
        if not prediction:
            failure = client.last_error or "The prediction API returned no data."
        elif form.cleaned_data["steps"] > 1:
            steps = api_client.horizon(link_id, hour_ts, form.cleaned_data["steps"], model)
            if not steps:
                failure = client.last_error or "The horizon API returned no data."
    return render(request, "portal/predict.html", _page(
        client, "predict", "Predict",
        form=form,
        prediction=prediction,
        steps=steps,
        failure=failure,
    ))


def model_performance(request):
    """Metrics, category report, coefficients, importances and residuals."""
    client = api_client.get_client()
    metrics = client.model_metrics()
    metric_rows = [
        {"group": group, "name": name, "split": split, **(splits.get(split) or {})}
        for group, entries in (metrics.get("results") or {}).items()
        for name, splits in sorted((entries or {}).items())
        for split in SPLITS
    ]
    coefficients = sorted(
        metrics.get("coefficients") or [],
        key=lambda row: abs(row.get("std_coefficient") or 0.0),
        reverse=True,
    )[:12]
    importances = sorted(
        metrics.get("random_forest_importances") or [],
        key=lambda row: row.get("importance") or 0.0,
        reverse=True,
    )[:12]
    report = metrics.get("congestion_category_report") or {}
    confusion = report.get("confusion_matrix") or {}
    confusion_rows = [
        {
            "actual": actual,
            "cells": [(confusion.get(actual) or {}).get(predicted, 0) for predicted in CATEGORIES],
        }
        for actual in CATEGORIES
    ]
    category_rows = [
        {"name": name, **((report.get("per_category") or {}).get(name) or {})}
        for name in CATEGORIES
    ]
    residuals = metrics.get("residual_diagnostics") or {}
    residual_hours = [
        {"label": f"{int(hour):02d}:00", "value": value}
        for hour, value in sorted(
            (residuals.get("mean_residual_by_hour") or {}).items(), key=lambda item: int(item[0])
        )
    ]
    residual_weekdays = [
        {"label": _weekday(int(day)), "value": value}
        for day, value in sorted(
            (residuals.get("mean_residual_by_weekday") or {}).items(),
            key=lambda item: int(item[0]),
        )
    ]
    return render(request, "portal/model_performance.html", _page(
        client, "model", "Model performance",
        metrics=metrics,
        metric_rows=metric_rows,
        report=report,
        confusion_rows=confusion_rows,
        category_rows=category_rows,
        categories=CATEGORIES,
        coefficients=coefficients,
        importances=importances,
        worst_links=client.model_errors(limit=12),
        residual_hours=residual_hours,
        residual_weekdays=residual_weekdays,
    ))


def about(request):
    """Static description of the project plus the live dataset window."""
    client = api_client.get_client()
    meta = client.meta()
    return render(request, "portal/about.html", _page(
        client, "about", "About",
        dataset=meta.get("dataset") or {},
        model_info=meta.get("model") or {},
        thresholds=meta.get("congestion_thresholds") or {},
    ))
