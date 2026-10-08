"""Shared HTTP client for the traffic API (used by the dashboard and Django)."""

from .api_client import DEFAULT_BASE_URL, TrafficApiClient

__all__ = ["DEFAULT_BASE_URL", "TrafficApiClient"]
