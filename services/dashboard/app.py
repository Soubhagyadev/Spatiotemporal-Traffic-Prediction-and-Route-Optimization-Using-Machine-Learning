"""Plotly Dash traffic analytics dashboard.

Usage
-----
    python scripts/run_dashboard.py [--port 8050]

The dashboard is a pure HTTP client of the Flask API: it never opens the
database or the model, so the analytics UI and the ML pipeline stay decoupled.
"""

from __future__ import annotations

import os

from dash import Dash

from services.common import DEFAULT_BASE_URL, TrafficApiClient

from . import callbacks, layouts

DEFAULT_WEB_URL = "http://127.0.0.1:8000"  # Django portal


def create_dashboard(api_base_url: str | None = None, web_url: str | None = None) -> Dash:
    """Build the Dash application, bound to a running traffic API."""
    client = TrafficApiClient(api_base_url or os.environ.get("TRAFFIC_API_URL", DEFAULT_BASE_URL))
    portal = web_url or os.environ.get("TRAFFIC_WEB_URL", DEFAULT_WEB_URL)
    meta = client.meta()
    app = Dash(__name__, title="NYC Traffic Analytics", suppress_callback_exceptions=True)
    app.layout = layouts.page_layout(meta, web_url=portal)
    callbacks.register_callbacks(app, client)
    app.traffic_client = client
    return app


def run(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    app = create_dashboard()
    app.run(host=host, port=port, debug=debug)
