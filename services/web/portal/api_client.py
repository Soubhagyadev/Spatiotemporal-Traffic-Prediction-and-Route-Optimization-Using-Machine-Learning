"""API glue for the portal: builds the shared client and adds POST helpers.

``TrafficApiClient`` already covers the GET endpoints; only the single-step
prediction POST has no wrapper there, so it is added here.
"""

from __future__ import annotations

from django.conf import settings

from services.common import TrafficApiClient

_CLIENTS: dict[str, TrafficApiClient] = {}


def get_client(base_url: str | None = None) -> TrafficApiClient:
    """Return a process-wide cached client for the configured API base URL."""
    url = (base_url or settings.TRAFFIC_API_URL).rstrip("/")
    client = _CLIENTS.get(url)
    if client is None:
        client = _CLIENTS[url] = TrafficApiClient(url)
    return client


def predict(
    link_id: int, hour_ts: str, model: str = "linear_regression"
) -> dict:
    """Single-step prediction; ``{}`` when the API is unavailable."""
    payload = get_client()._request(
        "/api/predict",
        method="post",
        json={"link_id": int(link_id), "hour_ts": hour_ts, "model": model},
    )
    return payload or {}


def horizon(
    link_id: int, start_hour_ts: str, steps: int = 6, model: str = "linear_regression"
) -> list[dict]:
    """Multi-step forecast rows; ``[]`` when the API is unavailable."""
    payload = get_client().predict_horizon(int(link_id), start_hour_ts, steps, model)
    return payload.get("predictions", [])


def link_choices(links: list[dict]) -> list[tuple[str, str]]:
    """Dropdown choices ``(link_id, label)`` ordered by borough then name."""
    ordered = sorted(
        links, key=lambda link: ((link.get("borough") or ""), (link.get("link_name") or ""))
    )
    return [
        (
            str(link["link_id"]),
            f"{link['link_id']} · {link.get('link_name') or 'unnamed'} "
            f"({link.get('borough') or 'unknown'})",
        )
        for link in ordered
        if link.get("link_id") is not None
    ]
