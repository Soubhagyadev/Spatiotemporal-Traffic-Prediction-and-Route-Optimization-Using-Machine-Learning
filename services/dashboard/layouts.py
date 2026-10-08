"""Page structure for the Dash dashboard."""

from __future__ import annotations

from dash import dcc, html

CARD_STYLE = "kpi-card"
SECTION = "section"


def kpi_card(identifier: str, title: str) -> html.Div:
    """A single headline number with a caption."""
    return html.Div(
        [html.Div(title, className="kpi-title"), html.Div("—", id=identifier, className="kpi-value")],
        className=CARD_STYLE,
    )


def data_table(identifier: str, headers: list[str]) -> html.Table:
    """Empty table shell that callbacks fill with rows."""
    return html.Table(
        [html.Thead(html.Tr([html.Th(header) for header in headers]))],
        id=identifier,
        className="data-table",
    )


def table_rows(rows: list[list]) -> list:
    return [html.Tr([html.Td(cell) for cell in row]) for row in rows]


def controls(meta: dict) -> html.Div:
    """Scope controls shared by every chart."""
    boroughs = meta.get("boroughs", []) or []
    thresholds = meta.get("congestion_thresholds", {})
    low = thresholds.get("high_congestion_below_mph")
    high = thresholds.get("low_congestion_above_mph")
    caption = (
        f"Congestion thresholds from the training period: "
        f"high < {low:.1f} mph ≤ medium < {high:.1f} mph ≤ low"
        if low and high
        else "Congestion thresholds unavailable"
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Borough", className="control-label"),
                    dcc.Dropdown(
                        id="ctl-borough",
                        options=[{"label": "All boroughs", "value": "all"}]
                        + [{"label": name, "value": name} for name in boroughs],
                        value="all",
                        clearable=False,
                    ),
                ],
                className="control",
            ),
            html.Div(
                [
                    html.Label("Road segment", className="control-label"),
                    dcc.Dropdown(id="ctl-link", options=[], value="all", clearable=False),
                ],
                className="control control-wide",
            ),
            html.Div(
                [
                    html.Label("Time resolution", className="control-label"),
                    dcc.RadioItems(
                        id="ctl-granularity",
                        options=[
                            {"label": " Daily", "value": "day"},
                            {"label": " Hourly", "value": "hour"},
                        ],
                        value="day",
                        className="radio-inline",
                    ),
                ],
                className="control",
            ),
            html.Div(
                [
                    html.Label("Prediction model", className="control-label"),
                    dcc.RadioItems(
                        id="ctl-model",
                        options=[
                            {"label": " Linear regression", "value": "linear_regression"},
                            {"label": " Random forest", "value": "random_forest"},
                        ],
                        value="linear_regression",
                        className="radio-inline",
                    ),
                ],
                className="control",
            ),
            html.Div(caption, className="control-caption"),
        ],
        className="controls",
    )


def page_layout(meta: dict, web_url: str = "http://127.0.0.1:8000") -> html.Div:
    """Full dashboard layout.

    ``web_url`` is the Django portal, linked from the top-right corner so the
    two frontends can reach each other.
    """
    dataset = meta.get("dataset", {})
    model = meta.get("model", {})
    window = f"{dataset.get('start', '?')} → {dataset.get('end', '?')}"
    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("NYC Traffic Analytics"),
                            html.P(
                                f"Spatiotemporal traffic prediction dashboard · {window} · "
                                f"{dataset.get('links', '?')} road segments · "
                                f"{dataset.get('link_hours', 0):,} link-hours · "
                                f"model: {', '.join(model.get('models', []))}",
                                className="subtitle",
                            ),
                        ],
                        className="header-text",
                    ),
                    html.A("Web interface →", href=web_url, className="header-link"),
                ],
                className="header",
            ),
            html.Div(id="api-status", className="status"),
            controls(meta),
            html.Div(
                [
                    kpi_card("kpi-speed", "Mean speed in scope"),
                    kpi_card("kpi-linkhours", "Link-hours in scope"),
                    kpi_card("kpi-segments", "Segments in scope"),
                    kpi_card("kpi-congestion", "Hours highly congested"),
                    kpi_card("kpi-mae", f"Model MAE ({model.get('target', 'speed')})"),
                    kpi_card("kpi-r2", "Model R² (test)"),
                ],
                className="kpi-row",
            ),
            html.Section(
                [dcc.Graph(id="chart-over-time", config={"displaylogo": False})],
                className=SECTION,
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="chart-hourly", config={"displaylogo": False}),
                        className="panel",
                    ),
                    html.Div(
                        dcc.Graph(id="chart-heatmap", config={"displaylogo": False}),
                        className="panel",
                    ),
                ],
                className="grid-2",
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="chart-congestion", config={"displaylogo": False}),
                        className="panel",
                    ),
                    html.Div(
                        dcc.Graph(id="chart-borough", config={"displaylogo": False}),
                        className="panel",
                    ),
                ],
                className="grid-2",
            ),
            html.Section(
                [
                    html.H2("Where the segments are"),
                    dcc.Graph(id="chart-map", config={"displaylogo": False}),
                ],
                className=SECTION,
            ),
            html.Section(
                [
                    html.H2(id="segment-heading", children="Selected segment"),
                    dcc.Graph(id="chart-segment", config={"displaylogo": False}),
                    html.P(
                        "The dotted continuation is the model rolled forward from the last "
                        "observed hour (forecast mode: each prediction becomes the input "
                        "for the next step).",
                        className="note",
                    ),
                ],
                className=SECTION,
            ),
            html.Section(
                [
                    html.H2("Model performance on the held-out test period"),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(id="chart-predicted", config={"displaylogo": False}),
                                className="panel",
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="chart-metrics", config={"displaylogo": False}
                                    ),
                                    html.Div(id="metrics-table"),
                                ],
                                className="panel",
                            ),
                        ],
                        className="grid-2",
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(id="chart-features", config={"displaylogo": False}),
                                className="panel",
                            ),
                            html.Div(
                                [
                                    html.H3("Largest segment-level errors"),
                                    data_table(
                                        "table-errors",
                                        ["Segment", "Borough", "MAE (mph)", "Link-hours"],
                                    ),
                                    html.H3("Borough summary"),
                                    data_table(
                                        "table-boroughs",
                                        ["Borough", "Segments", "Mean speed", "High cong."],
                                    ),
                                ],
                                className="panel",
                            ),
                        ],
                        className="grid-2",
                    ),
                ],
                className=SECTION,
            ),
            html.Footer(
                "Data: NYC DOT Traffic Speeds (i4gi-tjb9) · Part 1 pipeline: cleaning → "
                "hourly aggregation → features → model · This dashboard reads the Flask API only.",
                className="footer",
            ),
        ],
        className="page",
    )
