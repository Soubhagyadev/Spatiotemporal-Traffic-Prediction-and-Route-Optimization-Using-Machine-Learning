"""Read-side application service for the traffic API.

Framework-independent: it wraps the SQLite store and the analytics queries and
returns plain dictionaries, so the Flask routes stay thin and the same data can
be reused by tests or other consumers.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from traffic import analytics, config, store

GRANULARITIES = ("hour", "day")


class TrafficDataService:
    """Queries over the processed dataset."""

    def __init__(self, db_path: Path = config.DB_PATH, metrics_path: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.metrics_path = metrics_path or (config.MODEL_DIR / "metrics.json")

    def connect(self) -> sqlite3.Connection:
        return store.connect(self.db_path)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def health(self) -> dict:
        connection = self.connect()
        try:
            bounds = store.dataset_bounds(connection)
        finally:
            connection.close()
        return {
            "status": "ok",
            "database": self.db_path.name,
            "database_exists": self.db_path.exists(),
            "dataset": bounds,
        }

    def meta(self) -> dict:
        connection = self.connect()
        try:
            return {
                "dataset": store.dataset_bounds(connection),
                "boroughs": store.boroughs(connection),
                "congestion_thresholds": self.model_metrics()["congestion_thresholds"],
            }
        finally:
            connection.close()

    def model_metrics(self) -> dict:
        if not self.metrics_path.exists():
            return {"available": False}
        metrics = json.loads(self.metrics_path.read_text())
        metrics["available"] = True
        return metrics

    # ------------------------------------------------------------------
    # segments
    # ------------------------------------------------------------------
    def links(self, borough: str | None = None) -> list[dict]:
        connection = self.connect()
        try:
            return store.list_links(connection, borough=borough)
        finally:
            connection.close()

    def link(self, link_id: int) -> dict | None:
        connection = self.connect()
        try:
            detail = store.get_link(connection, link_id)
            if detail is None:
                return None
            detail["recent_hours"] = store.link_history(connection, link_id, limit=168)
            return detail
        finally:
            connection.close()

    def link_history(
        self,
        link_id: int,
        start: str | None = None,
        end: str | None = None,
        granularity: str = "hour",
    ) -> list[dict]:
        connection = self.connect()
        try:
            if granularity == "hour":
                return store.link_history(connection, link_id, start=start, end=end)
            return analytics.speed_over_time(
                connection, start=start, end=end, granularity="day", link_id=link_id
            )
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # aggregate analytics
    # ------------------------------------------------------------------
    def over_time(
        self,
        start: str | None = None,
        end: str | None = None,
        granularity: str = "day",
        borough: str | None = None,
        link_id: int | None = None,
    ) -> list[dict]:
        connection = self.connect()
        try:
            return analytics.speed_over_time(
                connection,
                start=start,
                end=end,
                granularity=granularity if granularity in GRANULARITIES else "day",
                borough=borough,
                link_id=link_id,
            )
        finally:
            connection.close()

    def hourly_pattern(
        self, borough: str | None = None, link_id: int | None = None
    ) -> list[dict]:
        connection = self.connect()
        try:
            return analytics.hourly_profile(connection, link_id=link_id, borough=borough)
        finally:
            connection.close()

    def weekday_hour(self, borough: str | None = None) -> list[dict]:
        connection = self.connect()
        try:
            return analytics.weekday_hour_grid(connection, borough=borough)
        finally:
            connection.close()

    def congestion_segments(
        self, limit: int = 20, borough: str | None = None
    ) -> list[dict]:
        thresholds = self.meta()["congestion_thresholds"]
        connection = self.connect()
        try:
            return analytics.congestion_ranking(
                connection, thresholds, borough=borough, limit=limit
            )
        finally:
            connection.close()

    def congestion_summary(
        self, borough: str | None = None, link_id: int | None = None
    ) -> dict:
        thresholds = self.meta()["congestion_thresholds"]
        connection = self.connect()
        try:
            return analytics.congestion_summary(
                connection, thresholds, borough=borough, link_id=link_id
            )
        finally:
            connection.close()

    def congestion_boroughs(self) -> list[dict]:
        thresholds = self.meta()["congestion_thresholds"]
        connection = self.connect()
        try:
            return analytics.borough_summary(connection, thresholds)
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # stored model predictions
    # ------------------------------------------------------------------
    def prediction_sample(self, limit: int = 4000, link_id: int | None = None) -> list[dict]:
        connection = self.connect()
        try:
            return analytics.prediction_sample(connection, limit=limit, link_id=link_id)
        finally:
            connection.close()

    def prediction_errors(self, limit: int = 15) -> list[dict]:
        connection = self.connect()
        try:
            return analytics.prediction_error_by_segment(connection, limit=limit)
        finally:
            connection.close()
