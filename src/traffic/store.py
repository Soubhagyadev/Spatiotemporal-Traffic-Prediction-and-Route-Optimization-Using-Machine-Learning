"""SQLite storage layer shared by the API, the dashboard and the web interface.

The database is built once from the Part 1 artifacts
(``hourly_traffic.csv`` and ``link_metadata.csv``) and is then the read-only
source of truth for every Part 2 service.  Keeping the queries here means the
Flask API, the Dash dashboard and Django all read identical numbers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config

HOURLY_COLUMNS = [
    "link_id",
    "hour_idx",
    "hour_ts",
    "obs_count",
    "speed_mean",
    "speed_median",
    "speed_std",
    "speed_min",
    "speed_max",
    "travel_time_mean",
    "travel_time_median",
    "travel_time_std",
    "travel_time_min",
    "travel_time_max",
    "implied_length_miles",
]

LINK_COLUMNS = [
    "link_id",
    "borough",
    "link_name",
    "owner",
    "geometry_source",
    "geometry_complete",
    "n_polyline_points",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "geom_length_miles",
    "link_points_truncated",
    "records_in",
    "duplicate_records",
    "outlier_records",
    "records_used",
    "hours_observed",
    "implied_length_miles",
    "first_hour_ts",
    "last_hour_ts",
]

SCHEMA = """
CREATE TABLE hourly_traffic (
    link_id               INTEGER NOT NULL,
    hour_idx              INTEGER NOT NULL,
    hour_ts               TEXT    NOT NULL,
    obs_count             INTEGER NOT NULL,
    speed_mean            REAL,
    speed_median          REAL,
    speed_std             REAL,
    speed_min             REAL,
    speed_max             REAL,
    travel_time_mean      REAL,
    travel_time_median    REAL,
    travel_time_std       REAL,
    travel_time_min       REAL,
    travel_time_max       REAL,
    implied_length_miles  REAL,
    PRIMARY KEY (link_id, hour_ts)
) WITHOUT ROWID;

CREATE TABLE link_metadata (
    link_id                INTEGER PRIMARY KEY,
    borough                TEXT,
    link_name              TEXT,
    owner                  TEXT,
    geometry_source        TEXT,
    geometry_complete      INTEGER,
    n_polyline_points      INTEGER,
    start_lat              REAL,
    start_lon              REAL,
    end_lat                REAL,
    end_lon                REAL,
    geom_length_miles      REAL,
    link_points_truncated  INTEGER,
    records_in             REAL,
    duplicate_records      REAL,
    outlier_records        REAL,
    records_used           REAL,
    hours_observed         REAL,
    implied_length_miles   REAL,
    first_hour_ts          TEXT,
    last_hour_ts           TEXT
);

CREATE TABLE test_predictions (
    link_id            INTEGER NOT NULL,
    hour_ts            TEXT    NOT NULL,
    observed_speed     REAL,
    predicted_speed    REAL,
    predicted_speed_rf REAL,
    congestion_actual  TEXT,
    congestion_predicted TEXT,
    PRIMARY KEY (link_id, hour_ts)
) WITHOUT ROWID;

CREATE INDEX idx_hourly_hour_ts ON hourly_traffic (hour_ts);
CREATE INDEX idx_hourly_link ON hourly_traffic (link_id, hour_ts);
"""


def connect(path: Path = config.DB_PATH, read_only: bool = True) -> sqlite3.Connection:
    """Open the database with dictionary-style rows.

    Services open it read-only; only the build scripts request write access.
    """
    if read_only:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _check_columns(frame: pd.DataFrame, expected: list[str], table: str) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{table} is missing expected columns: {missing}")


def build_database(
    hourly_path: Path = config.HOURLY_DATASET_PATH,
    metadata_path: Path = config.LINK_METADATA_PATH,
    db_path: Path = config.DB_PATH,
) -> dict:
    """Create the SQLite database from the processed Part 1 artifacts."""
    hourly = pd.read_csv(hourly_path)
    metadata = pd.read_csv(metadata_path)
    _check_columns(hourly, HOURLY_COLUMNS, "hourly_traffic")
    _check_columns(metadata, LINK_COLUMNS, "link_metadata")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(SCHEMA)
        hourly[HOURLY_COLUMNS].to_sql(
            "hourly_traffic", connection, if_exists="append", index=False
        )
        metadata[LINK_COLUMNS].to_sql("link_metadata", connection, if_exists="append", index=False)
        connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()
    return {
        "database": str(db_path),
        "hourly_rows": int(len(hourly)),
        "links": int(len(metadata)),
        "size_mb": round(db_path.stat().st_size / 1e6, 1),
    }


def rows(cursor: sqlite3.Cursor) -> list[dict]:
    return [dict(row) for row in cursor.fetchall()]


def dataset_bounds(connection: sqlite3.Connection) -> dict:
    """Observation window and dataset size, used for range validation."""
    row = connection.execute(
        """
        SELECT MIN(hour_ts) AS start, MAX(hour_ts) AS end,
               COUNT(*) AS link_hours, COUNT(DISTINCT link_id) AS links,
               COUNT(DISTINCT hour_ts) AS hours,
               SUM(obs_count) AS observations
        FROM hourly_traffic
        """
    ).fetchone()
    return dict(row)


def list_links(connection: sqlite3.Connection, borough: str | None = None) -> list[dict]:
    query = """
        SELECT m.link_id, m.borough, m.link_name, m.owner, m.start_lat, m.start_lon,
               m.end_lat, m.end_lon, m.geom_length_miles, m.implied_length_miles,
               m.hours_observed, m.first_hour_ts, m.last_hour_ts, m.geometry_complete,
               ROUND(AVG(h.speed_mean), 3) AS mean_speed_mph
        FROM link_metadata m
        LEFT JOIN hourly_traffic h ON h.link_id = m.link_id
    """
    parameters: tuple = ()
    if borough:
        query += " WHERE m.borough = ?"
        parameters = (borough,)
    query += " GROUP BY m.link_id ORDER BY m.borough, m.link_name"
    return rows(connection.execute(query, parameters))


def get_link(connection: sqlite3.Connection, link_id: int) -> dict | None:
    row = connection.execute(
        """
        SELECT m.*, (SELECT ROUND(AVG(speed_mean), 3) FROM hourly_traffic h
                     WHERE h.link_id = m.link_id) AS mean_speed_mph
        FROM link_metadata m WHERE m.link_id = ?
        """,
        (link_id,),
    ).fetchone()
    return None if row is None else dict(row)


def link_history(
    connection: sqlite3.Connection,
    link_id: int,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Hourly series for one segment, oldest first."""
    query = """
        SELECT hour_ts, obs_count, speed_mean, speed_median, travel_time_mean,
               travel_time_median, implied_length_miles
        FROM hourly_traffic WHERE link_id = ?
    """
    parameters: list = [link_id]
    if start:
        query += " AND hour_ts >= ?"
        parameters.append(start)
    if end:
        query += " AND hour_ts <= ?"
        parameters.append(end)
    query += " ORDER BY hour_ts"
    if limit:
        query += " LIMIT ?"
        parameters.append(limit)
    return rows(connection.execute(query, parameters))


def recent_observations(
    connection: sqlite3.Connection,
    link_id: int,
    before: str,
    limit: int = 24,
    min_obs: int = config.MIN_OBS_PER_LINK_HOUR,
) -> list[dict]:
    """Most recent usable hours strictly before ``before`` (newest first).

    ``min_obs`` mirrors the filter used when the model was trained, so lag and
    rolling features are built from exactly the same kind of rows.
    """
    return rows(
        connection.execute(
            """
            SELECT hour_ts, speed_mean, obs_count FROM hourly_traffic
            WHERE link_id = ? AND hour_ts < ? AND obs_count >= ?
            ORDER BY hour_ts DESC LIMIT ?
            """,
            (link_id, before, min_obs, limit),
        )
    )


def speed_at(connection: sqlite3.Connection, link_id: int, hour_ts: str) -> float | None:
    """Observed hourly mean speed at an exact timestamp, if present."""
    row = connection.execute(
        "SELECT speed_mean FROM hourly_traffic WHERE link_id = ? AND hour_ts = ?",
        (link_id, hour_ts),
    ).fetchone()
    return None if row is None else row["speed_mean"]


def boroughs(connection: sqlite3.Connection) -> list[str]:
    return [
        row["borough"]
        for row in connection.execute(
            "SELECT DISTINCT borough FROM link_metadata WHERE borough IS NOT NULL ORDER BY borough"
        )
    ]
