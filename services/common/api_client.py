"""HTTP client for the Flask API used by the Dash dashboard.

The dashboard never reads the database or the model directly: every number it
shows comes through the API, which keeps the presentation layer independent of
the ML pipeline.  A tiny TTL cache avoids repeating identical requests during
a callback burst.
"""

from __future__ import annotations

import time

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:5001"
CACHE_TTL_SECONDS = 30.0


class TrafficApiClient:
    """Thin, cached wrapper around the traffic API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self._cache: dict[str, tuple[float, object]] = {}
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    def _request(self, path: str, params: dict | None = None, method: str = "get", **kwargs):
        key = f"{method}:{path}:{sorted((params or {}).items())}:{kwargs.get('json')}"
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]
        try:
            response = self.session.request(
                method, f"{self.base_url}{path}", params=params, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            payload = response.json()
            self._cache[key] = (time.time(), payload)
            self.last_error = None
            return payload
        except requests.RequestException as exc:
            self.last_error = f"{path}: {exc}"
            return {}
        except ValueError as exc:  # non-JSON response
            self.last_error = f"{path}: invalid JSON ({exc})"
            return {}

    def is_available(self) -> bool:
        return bool(self._request("/health"))

    # ------------------------------------------------------------------
    def health(self) -> dict:
        return self._request("/health")

    def meta(self) -> dict:
        return self._request("/api/meta")

    def links(self, borough: str | None = None) -> list[dict]:
        params = {"borough": borough} if borough else None
        return self._request("/api/links", params).get("links", [])

    def link(self, link_id: int) -> dict:
        return self._request(f"/api/links/{int(link_id)}")

    def link_history(self, link_id: int, granularity: str = "hour") -> list[dict]:
        payload = self._request(
            f"/api/links/{int(link_id)}/history", {"granularity": granularity}
        )
        return payload.get("points", [])

    def over_time(
        self, borough: str | None = None, granularity: str = "day", link_id: int | None = None
    ) -> list[dict]:
        """Mean speed per bucket; the same shape for a scope or a single segment."""
        params: dict = {"granularity": granularity}
        if borough:
            params["borough"] = borough
        if link_id:
            params["link_id"] = int(link_id)
        return self._request("/api/traffic/over_time", params).get("points", [])

    def hourly_pattern(self, borough: str | None = None) -> list[dict]:
        params = {"borough": borough} if borough else None
        return self._request("/api/patterns/hourly", params).get("points", [])

    def weekday_hour(self, borough: str | None = None) -> list[dict]:
        params = {"borough": borough} if borough else None
        return self._request("/api/patterns/weekday_hour", params).get("points", [])

    def congestion_segments(self, limit: int = 15, borough: str | None = None) -> list[dict]:
        params: dict = {"limit": limit}
        if borough:
            params["borough"] = borough
        return self._request("/api/congestion/segments", params).get("segments", [])

    def congestion_boroughs(self) -> list[dict]:
        return self._request("/api/congestion/boroughs").get("boroughs", [])

    def congestion_summary(self, borough: str | None = None, link_id: int | None = None) -> dict:
        params: dict = {}
        if borough:
            params["borough"] = borough
        if link_id:
            params["link_id"] = link_id
        return self._request("/api/congestion/summary", params).get("summary", {})

    # ------------------------------------------------------------------
    def model_metrics(self) -> dict:
        return self._request("/api/model/metrics")

    def predictions(self, limit: int = 4000, link_id: int | None = None) -> list[dict]:
        params: dict = {"limit": limit}
        if link_id:
            params["link_id"] = link_id
        return self._request("/api/model/predictions", params).get("points", [])

    def model_errors(self, limit: int = 10) -> list[dict]:
        return self._request("/api/model/errors", {"limit": limit}).get("segments", [])

    # ------------------------------------------------------------------
    def predict_horizon(
        self, link_id: int, start_hour_ts: str, steps: int = 6, model: str = "linear_regression"
    ) -> dict:
        return self._request(
            "/api/predict/horizon",
            method="post",
            json={
                "link_id": int(link_id),
                "start_hour_ts": start_hour_ts,
                "steps": steps,
                "model": model,
            },
        )
