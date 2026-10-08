"""Exact hourly aggregation of per-link cleaned records.

Median, standard deviation and quantile statistics are not additive, so the
aggregation runs one link at a time: each per-link partition (a few tens of
thousands of rows) is small enough to be aggregated exactly in memory, which
keeps the pipeline memory-safe without approximating any statistic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .partition import RECORD_COLUMNS

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

_AGGREGATIONS = {
    "obs_count": ("speed", "size"),
    "speed_mean": ("speed", "mean"),
    "speed_median": ("speed", "median"),
    "speed_std": ("speed", "std"),
    "speed_min": ("speed", "min"),
    "speed_max": ("speed", "max"),
    "travel_time_mean": ("travel_time", "mean"),
    "travel_time_median": ("travel_time", "median"),
    "travel_time_std": ("travel_time", "std"),
    "travel_time_min": ("travel_time", "min"),
    "travel_time_max": ("travel_time", "max"),
}


def read_link_records(path: Path) -> pd.DataFrame:
    """Load one link partition in a compact typed form."""
    records = pd.read_csv(
        path,
        dtype={
            "link_id": "int32",
            "hour_idx": "int32",
            "ts_sec": "int32",
            "speed": "float32",
            "travel_time": "float32",
        },
        usecols=RECORD_COLUMNS,
    )
    return records.sort_values(["hour_idx", "ts_sec"], ignore_index=True)


def aggregate_link(
    records: pd.DataFrame, tolerance: float = config.LENGTH_OUTLIER_TOLERANCE
) -> tuple[pd.DataFrame, dict]:
    """Aggregate one link's records to hourly statistics.

    Returns the hourly frame plus a summary of the link-level cleaning.
    """
    summary = {
        "link_id": int(records["link_id"].iloc[0]) if len(records) else -1,
        "records_in": int(len(records)),
        "duplicate_records": 0,
        "outlier_records": 0,
        "records_used": 0,
        "hours_observed": 0,
        "implied_length_miles": np.nan,
        "first_hour_ts": pd.NaT,
        "last_hour_ts": pd.NaT,
    }
    if records.empty:
        return pd.DataFrame(columns=HOURLY_COLUMNS), summary

    # Global de-duplication on (link, exact timestamp): pagination boundaries
    # can repeat a record across two raw files.
    before = len(records)
    records = records.drop_duplicates(subset="ts_sec", keep="first")
    summary["duplicate_records"] = int(before - len(records))

    implied = records["speed"].to_numpy(dtype="float64") * (
        records["travel_time"].to_numpy(dtype="float64") / 3600.0
    )
    length_median = float(np.median(implied))
    summary["implied_length_miles"] = length_median

    if length_median > 0:
        deviation = np.abs(implied / length_median - 1.0)
        keep = deviation <= tolerance
    else:
        keep = np.ones(len(records), dtype=bool)
    summary["outlier_records"] = int((~keep).sum())
    records = records.loc[keep]
    summary["records_used"] = int(len(records))
    if records.empty:
        return pd.DataFrame(columns=HOURLY_COLUMNS), summary

    hourly = records.groupby("hour_idx", observed=True).agg(**_AGGREGATIONS).reset_index()
    hourly["link_id"] = summary["link_id"]
    hourly["implied_length_miles"] = length_median
    origin = pd.Timestamp(config.HOUR_ORIGIN_ISO)
    hourly["hour_ts"] = origin + pd.to_timedelta(hourly["hour_idx"], unit="h")

    summary["hours_observed"] = int(len(hourly))
    summary["first_hour_ts"] = hourly["hour_ts"].min()
    summary["last_hour_ts"] = hourly["hour_ts"].max()
    return hourly[HOURLY_COLUMNS], summary


def build_hourly_dataset(
    parts_dir: Path, tolerance: float = config.LENGTH_OUTLIER_TOLERANCE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate every per-link partition into one hourly dataset.

    Only one link is materialised at a time; the resulting hourly frames are
    small enough (one row per link-hour) to be collected for the final write.
    """
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for path in sorted(Path(parts_dir).glob("link_*.csv")):
        hourly, summary = aggregate_link(read_link_records(path), tolerance=tolerance)
        summaries.append(summary)
        if not hourly.empty:
            frames.append(hourly)
    hourly_dataset = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HOURLY_COLUMNS)
    )
    if not hourly_dataset.empty:
        hourly_dataset = hourly_dataset.sort_values(
            ["hour_ts", "link_id"], ignore_index=True
        )
    return hourly_dataset, pd.DataFrame(summaries)
