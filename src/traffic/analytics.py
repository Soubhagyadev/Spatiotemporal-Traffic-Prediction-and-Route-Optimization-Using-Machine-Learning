"""Aggregate analytics queries behind the dashboard and the web interface.

These are read-only roll-ups over the hourly table.  Keeping them out of the
services means the Flask API, the Dash dashboard and Django answer "traffic
over time", "hourly pattern" and "congestion ranking" questions with exactly
the same SQL.
"""

from __future__ import annotations

import sqlite3

from . import congestion, store


def _range(column: str, start: str | None, end: str | None, params: list) -> str:
    clause = ""
    if start:
        clause += f" AND {column} >= ?"
        params.append(start)
    if end:
        clause += f" AND {column} <= ?"
        params.append(end)
    return clause


def speed_over_time(
    connection: sqlite3.Connection,
    start: str | None = None,
    end: str | None = None,
    granularity: str = "day",
    link_id: int | None = None,
    borough: str | None = None,
) -> list[dict]:
    """Mean speed per day (or hour) across the selected segments."""
    group = "substr(h.hour_ts, 1, 10)" if granularity == "day" else "h.hour_ts"
    params: list = []
    query = f"""
        SELECT {group} AS bucket,
               ROUND(AVG(h.speed_mean), 3) AS mean_speed_mph,
               ROUND(AVG(h.travel_time_mean), 1) AS mean_travel_time_s,
               COUNT(*) AS link_hours,
               COUNT(DISTINCT h.link_id) AS segments
        FROM hourly_traffic h
        JOIN link_metadata m ON m.link_id = h.link_id
        WHERE 1 = 1
    """
    query += _range("h.hour_ts", start, end, params)
    if link_id is not None:
        query += " AND h.link_id = ?"
        params.append(link_id)
    if borough:
        query += " AND m.borough = ?"
        params.append(borough)
    query += f" GROUP BY bucket ORDER BY bucket"
    return store.rows(connection.execute(query, params))


def hourly_profile(
    connection: sqlite3.Connection,
    link_id: int | None = None,
    borough: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> list[dict]:
    """Average speed and volume by hour of day, weekday versus weekend."""
    params: list = []
    query = """
        SELECT CAST(substr(h.hour_ts, 12, 2) AS INTEGER) AS hour,
               CASE WHEN CAST(strftime('%w', h.hour_ts) AS INTEGER) IN (0, 6)
                    THEN 'weekend' ELSE 'weekday' END AS day_type,
               ROUND(AVG(h.speed_mean), 3) AS mean_speed_mph,
               ROUND(AVG(h.obs_count), 1) AS mean_observations,
               COUNT(*) AS link_hours
        FROM hourly_traffic h
        JOIN link_metadata m ON m.link_id = h.link_id
        WHERE 1 = 1
    """
    query += _range("h.hour_ts", start, end, params)
    if link_id is not None:
        query += " AND h.link_id = ?"
        params.append(link_id)
    if borough:
        query += " AND m.borough = ?"
        params.append(borough)
    query += " GROUP BY hour, day_type ORDER BY day_type, hour"
    return store.rows(connection.execute(query, params))


def weekday_hour_grid(
    connection: sqlite3.Connection,
    borough: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> list[dict]:
    """Mean speed per (day of week, hour) cell for the heatmap."""
    params: list = []
    query = """
        SELECT CAST(strftime('%w', h.hour_ts) AS INTEGER) AS weekday,
               CAST(substr(h.hour_ts, 12, 2) AS INTEGER) AS hour,
               ROUND(AVG(h.speed_mean), 3) AS mean_speed_mph,
               COUNT(*) AS link_hours
        FROM hourly_traffic h
        JOIN link_metadata m ON m.link_id = h.link_id
        WHERE 1 = 1
    """
    query += _range("h.hour_ts", start, end, params)
    if borough:
        query += " AND m.borough = ?"
        params.append(borough)
    query += " GROUP BY weekday, hour ORDER BY weekday, hour"
    return store.rows(connection.execute(query, params))


def congestion_ranking(
    connection: sqlite3.Connection,
    thresholds: dict,
    borough: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Segments ordered by the share of hours spent in high congestion."""
    below = thresholds["high_congestion_below_mph"]
    params: list = [below]
    query = """
        SELECT h.link_id, m.borough, m.link_name,
               ROUND(AVG(h.speed_mean), 3) AS mean_speed_mph,
               ROUND(100.0 * SUM(CASE WHEN h.speed_mean < ? THEN 1 ELSE 0 END)
                     / COUNT(*), 2) AS high_congestion_share_pct,
               COUNT(*) AS link_hours
        FROM hourly_traffic h
        JOIN link_metadata m ON m.link_id = h.link_id
        WHERE 1 = 1
    """
    query += _range("h.hour_ts", start, end, params)
    if borough:
        query += " AND m.borough = ?"
        params.append(borough)
    query += " GROUP BY h.link_id ORDER BY high_congestion_share_pct DESC, mean_speed_mph LIMIT ?"
    params.append(limit)
    return store.rows(connection.execute(query, params))


def congestion_summary(
    connection: sqlite3.Connection,
    thresholds: dict,
    borough: str | None = None,
    link_id: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Overall congestion mix for the current scope (KPI cards)."""
    below = thresholds["high_congestion_below_mph"]
    above = thresholds["low_congestion_above_mph"]
    query = """
        SELECT COUNT(*) AS link_hours,
               ROUND(AVG(h.speed_mean), 3) AS mean_speed_mph,
               SUM(CASE WHEN h.speed_mean < ? THEN 1 ELSE 0 END) AS high,
               SUM(CASE WHEN h.speed_mean >= ? AND h.speed_mean < ? THEN 1 ELSE 0 END) AS medium,
               SUM(CASE WHEN h.speed_mean >= ? THEN 1 ELSE 0 END) AS low,
               COUNT(DISTINCT h.link_id) AS segments
        FROM hourly_traffic h
        JOIN link_metadata m ON m.link_id = h.link_id
        WHERE 1 = 1
    """
    params: list = [below, below, above, above]
    query += _range("h.hour_ts", start, end, params)
    if borough:
        query += " AND m.borough = ?"
        params.append(borough)
    if link_id is not None:
        query += " AND h.link_id = ?"
        params.append(link_id)
    row = dict(connection.execute(query, params).fetchone())
    total = max(1, int(row.get("link_hours") or 0))
    for label, key in (("high", "high"), ("medium", "medium"), ("low", "low")):
        row[f"{key}_share_pct"] = round(100.0 * (row.get(key) or 0) / total, 2)
    return row


def borough_summary(connection: sqlite3.Connection, thresholds: dict) -> list[dict]:
    """Per-borough speed, congestion mix and segment counts."""
    below = thresholds["high_congestion_below_mph"]
    above = thresholds["low_congestion_above_mph"]
    return store.rows(
        connection.execute(
            """
            SELECT m.borough,
                   COUNT(DISTINCT m.link_id) AS segments,
                   ROUND(AVG(h.speed_mean), 3) AS mean_speed_mph,
                   ROUND(100.0 * SUM(CASE WHEN h.speed_mean < ? THEN 1 ELSE 0 END)
                         / COUNT(*), 2) AS high_congestion_share_pct,
                   ROUND(100.0 * SUM(CASE WHEN h.speed_mean >= ? THEN 1 ELSE 0 END)
                         / COUNT(*), 2) AS low_congestion_share_pct
            FROM hourly_traffic h
            JOIN link_metadata m ON m.link_id = h.link_id
            WHERE m.borough IS NOT NULL
            GROUP BY m.borough ORDER BY high_congestion_share_pct DESC
            """,
            (below, above),
        )
    )


def prediction_sample(
    connection: sqlite3.Connection, limit: int = 4000, link_id: int | None = None
) -> list[dict]:
    """Observed versus predicted speeds for the scatter plot."""
    query = """
        SELECT link_id, hour_ts, observed_speed, predicted_speed, predicted_speed_rf,
               congestion_actual, congestion_predicted
        FROM test_predictions
    """
    params: list = []
    if link_id is not None:
        query += " WHERE link_id = ?"
        params.append(link_id)
    query += " ORDER BY hour_ts"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    return store.rows(connection.execute(query, params))


def prediction_error_by_segment(connection: sqlite3.Connection, limit: int = 15) -> list[dict]:
    """Segments with the largest mean absolute prediction error."""
    return store.rows(
        connection.execute(
            """
            SELECT p.link_id, m.borough, m.link_name,
                   ROUND(AVG(ABS(p.observed_speed - p.predicted_speed)), 3) AS mae_mph,
                   COUNT(*) AS link_hours
            FROM test_predictions p
            JOIN link_metadata m ON m.link_id = p.link_id
            GROUP BY p.link_id ORDER BY mae_mph DESC LIMIT ?
            """,
            (limit,),
        )
    )


def congestion_category_mix(connection: sqlite3.Connection) -> list[dict]:
    """Observed congestion mix per borough, for the dashboard summary cards."""
    return store.rows(
        connection.execute(
            """
            SELECT m.borough, p.congestion_actual AS category, COUNT(*) AS link_hours
            FROM test_predictions p
            JOIN link_metadata m ON m.link_id = p.link_id
            GROUP BY m.borough, p.congestion_actual
            ORDER BY m.borough, category
            """
        )
    )


def category_labels() -> list[str]:
    return list(congestion.CATEGORY_ORDER)