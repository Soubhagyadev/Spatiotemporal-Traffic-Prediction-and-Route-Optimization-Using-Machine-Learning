"""Dash callbacks: connect the controls to the API and the figures."""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, callback, html

from . import figures, layouts, model_figures

ALL = "all"


def _scope(borough: str | None, link_id) -> tuple[str | None, int | None]:
    return (
        None if borough in (None, ALL) else borough,
        None if link_id in (None, ALL) else int(link_id),
    )


def register_callbacks(app, client) -> None:
    """Attach every callback; ``client`` is a :class:`TrafficApiClient`."""

    @callback(
        Output("ctl-link", "options"),
        Output("ctl-link", "value"),
        Input("ctl-borough", "value"),
    )
    def _segment_options(borough):
        links = client.links(None if borough in (None, ALL) else borough)
        options = [{"label": "All segments", "value": ALL}]
        options += [
            {
                "label": f"{str(row['link_name'])[:38]} ({row['link_id']})",
                "value": row["link_id"],
            }
            for row in links
        ]
        return options, ALL

    # ------------------------------------------------------------------
    # scope-driven views
    # ------------------------------------------------------------------
    @callback(
        Output("kpi-speed", "children"),
        Output("kpi-linkhours", "children"),
        Output("kpi-segments", "children"),
        Output("kpi-congestion", "children"),
        Output("chart-over-time", "figure"),
        Output("chart-hourly", "figure"),
        Output("chart-heatmap", "figure"),
        Output("api-status", "children"),
        Input("ctl-borough", "value"),
        Input("ctl-link", "value"),
        Input("ctl-granularity", "value"),
    )
    def _scope_views(borough, link_id, granularity):
        borough, link_id = _scope(borough, link_id)
        status = ""
        if client.last_error:
            status = f"API problem: {client.last_error}"

        summary = client.congestion_summary(borough=borough, link_id=link_id)
        over_time = client.over_time(borough=borough, granularity=granularity, link_id=link_id)
        scope_label = borough or (f"segment {link_id}" if link_id else "all segments")

        mean_speed = summary.get("mean_speed_mph")
        return (
            f"{mean_speed:.1f} mph" if mean_speed else "—",
            f"{summary.get('link_hours', 0):,}",
            f"{summary.get('segments', 0):,}",
            f"{summary.get('high_share_pct', 0):.1f}%",
            figures.speed_over_time(over_time, granularity, scope_label),
            figures.hourly_pattern(client.hourly_pattern(borough=borough)),
            figures.weekday_hour_heatmap(client.weekday_hour(borough=borough)),
            status,
        )

    @callback(
        Output("chart-congestion", "figure"),
        Output("chart-borough", "figure"),
        Output("chart-map", "figure"),
        Output("chart-segment", "figure"),
        Output("segment-heading", "children"),
        Input("ctl-borough", "value"),
        Input("ctl-link", "value"),
    )
    def _congestion_views(borough, link_id):
        borough, link_id = _scope(borough, link_id)
        links = client.links(borough)
        detail = client.link(link_id) if link_id else None
        forecast = None
        history: list[dict] = []
        if link_id and detail:
            history = client.link_history(link_id, granularity="hour")
            last = detail.get("last_hour_ts")
            if last:
                start = (pd.Timestamp(last) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
                forecast = client.predict_horizon(link_id, start, steps=6)
                forecast["congestion_reference"] = client.meta().get(
                    "congestion_thresholds", {}
                ).get("high_congestion_below_mph")
        elif link_id:
            history = client.link_history(link_id, granularity="hour")
        heading = "Selected segment"
        if detail:
            heading = (
                f"Selected segment: {detail.get('link_name')} ({link_id}) · "
                f"{detail.get('borough')} · {detail.get('implied_length_miles', 0):.2f} mi"
            )
        return (
            figures.congestion_bar(client.congestion_segments(limit=15, borough=borough)),
            figures.borough_bar(client.congestion_boroughs()),
            figures.segment_map(links),
            figures.segment_detail(
                history[-24 * 14 :] if history else [], forecast, detail.get("link_name", "") if detail else ""
            ),
            heading,
        )

    # ------------------------------------------------------------------
    # model views
    # ------------------------------------------------------------------
    @callback(
        Output("kpi-mae", "children"),
        Output("kpi-r2", "children"),
        Output("chart-predicted", "figure"),
        Output("chart-metrics", "figure"),
        Output("chart-features", "figure"),
        Output("metrics-table", "children"),
        Output("table-errors", "children"),
        Output("table-boroughs", "children"),
        Input("ctl-model", "value"),
        Input("ctl-link", "value"),
    )
    def _model_views(model, link_id):
        link_id = None if link_id in (None, ALL) else int(link_id)
        metrics = client.model_metrics()
        results = metrics.get("results", {})
        column = "predicted_speed_rf" if model == "random_forest" else "predicted_speed"
        scores = results.get("models", {}).get(model, {}).get("test", {})

        rows = []
        for group in ("baselines", "models"):
            for name, splits in results.get(group, {}).items():
                test = splits.get("test", {})
                rows.append(
                    [
                        name.replace("_", " "),
                        "baseline" if group == "baselines" else "model",
                        f"{test.get('mae', float('nan')):.2f}",
                        f"{test.get('rmse', float('nan')):.2f}",
                        f"{test.get('r2', float('nan')):.3f}",
                    ]
                )
        table = layouts.data_table(
            "metrics-inner", ["Predictor", "Kind", "MAE", "RMSE", "R²"]
        )
        table.children.append(html.Tbody(layouts.table_rows(rows)))

        errors = [
            [
                f"{str(row['link_name'])[:30]} ({row['link_id']})",
                row["borough"],
                f"{row['mae_mph']:.2f}",
                f"{row['link_hours']:,}",
            ]
            for row in client.model_errors(limit=8)
        ]
        error_body = html.Tbody(layouts.table_rows(errors))
        errors_table = layouts.data_table(
            "errors-inner", ["Segment", "Borough", "MAE (mph)", "Link-hours"]
        )
        errors_table.children.append(error_body)

        boroughs = [
            [
                row["borough"],
                f"{row['segments']:,}",
                f"{row['mean_speed_mph']:.1f} mph",
                f"{row['high_congestion_share_pct']:.1f}%",
            ]
            for row in client.congestion_boroughs()
        ]
        borough_table = layouts.data_table(
            "boroughs-inner", ["Borough", "Segments", "Mean speed", "High cong."]
        )
        borough_table.children.append(html.Tbody(layouts.table_rows(boroughs)))

        return (
            f"{scores.get('mae', float('nan')):.2f} mph",
            f"{scores.get('r2', float('nan')):.3f}",
            model_figures.predicted_vs_actual(client.predictions(link_id=link_id), column),
            model_figures.metrics_bar(results),
            model_figures.feature_effects(metrics.get("coefficients", [])),
            table,
            errors_table,
            borough_table,
        )
