"""Chunk-level cleaning of the raw NYC DOT traffic-speed CSV exports.

The raw files contain every observation the API returned, including stale
records and sentinel values.  Cleaning happens per chunk so that no more than
one chunk of raw rows is ever resident in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import config


@dataclass
class CleaningStats:
    """Running counters describing what happened to the raw records."""

    raw_rows: int = 0
    kept_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    files_read: int = 0
    files_failed: list[str] = field(default_factory=list)

    def add_reason(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    def as_dict(self) -> dict:
        total_dropped = int(sum(self.dropped.values()))
        return {
            "files_read": self.files_read,
            "files_failed": self.files_failed,
            "raw_rows": int(self.raw_rows),
            "kept_rows": int(self.kept_rows),
            "dropped_rows": total_dropped,
            "dropped_by_reason": {
                k: int(v) for k, v in sorted(self.dropped.items(), key=lambda kv: -kv[1])
            },
        }


def iter_raw_chunks(path: Path, chunk_size: int = config.CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Yield the raw CSV in chunks, keeping only the columns we need as text."""
    reader = pd.read_csv(
        path,
        usecols=config.RAW_COLUMNS,
        dtype={col: "string" for col in config.RAW_COLUMNS},
        chunksize=chunk_size,
        engine="c",
    )
    yield from reader


def hour_index(timestamps: pd.Series) -> pd.Series:
    """Whole hours since the configured origin (compact integer time axis)."""
    origin = pd.Timestamp(config.HOUR_ORIGIN_ISO)
    return ((timestamps - origin).dt.total_seconds() // 3600).astype("int32")


def clean_chunk(chunk: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Convert a raw chunk to typed valid records.

    Returns a frame with columns
    ``link_id, hour_idx, ts_sec, speed, travel_time`` holding only records
    that satisfy the documented validity rules.
    """
    stats.raw_rows += len(chunk)

    timestamps = pd.to_datetime(chunk["data_as_of"], format="ISO8601", errors="coerce")
    speed = pd.to_numeric(chunk["speed"], errors="coerce")
    travel_time = pd.to_numeric(chunk["travel_time"], errors="coerce")
    link_id = pd.to_numeric(chunk["link_id"], errors="coerce")
    status = pd.to_numeric(chunk["status"], errors="coerce")

    missing = timestamps.isna() | link_id.isna()
    stats.add_reason("missing_timestamp_or_link_id", missing.sum())
    stats.add_reason("non_numeric_speed", (speed.isna() & ~missing).sum())
    stats.add_reason("non_numeric_travel_time", (travel_time.isna() & ~missing).sum())

    safe = ~missing & speed.notna() & travel_time.notna()
    stale = safe & (status != config.VALID_STATUS)
    stats.add_reason("non_nominal_status", stale.sum())

    usable = safe & ~stale
    bad_speed = usable & (speed <= config.MIN_SPEED_MPH)
    bad_tt = usable & (travel_time <= config.MIN_TRAVEL_TIME_S)
    stats.add_reason("speed_not_positive", bad_speed.sum())
    stats.add_reason("travel_time_not_positive", bad_tt.sum())

    keep = usable & ~bad_speed & ~bad_tt
    records = pd.DataFrame(
        {
            "link_id": link_id[keep].astype("int32"),
            "hour_idx": hour_index(timestamps[keep]),
            "ts_sec": (
                (timestamps[keep] - pd.Timestamp(config.HOUR_ORIGIN_ISO)).dt.total_seconds()
            ).astype("int32"),
            "speed": speed[keep].astype("float32"),
            "travel_time": travel_time[keep].astype("float32"),
        }
    )
    stats.kept_rows += len(records)
    return records


def clean_file(
    path: Path,
    stats: CleaningStats,
    chunk_size: int = config.CHUNK_SIZE,
    on_chunk=None,
) -> pd.DataFrame:
    """Clean one raw file, de-duplicating repeated (link, timestamp) records.

    Only the cleaned records of a single file are held in memory at a time.
    ``on_chunk`` (if given) receives each raw chunk before it is cleaned, so
    callers can collect static metadata in the same pass.
    """
    parts = []
    seen: set[tuple[int, int]] = set()
    for chunk in iter_raw_chunks(path, chunk_size=chunk_size):
        if on_chunk is not None:
            on_chunk(chunk)
        records = clean_chunk(chunk, stats)
        if records.empty:
            continue
        # De-duplicate within the file on (link_id, exact timestamp).
        keys = list(zip(records["link_id"].to_numpy(), records["ts_sec"].to_numpy()))
        fresh = np.fromiter((key not in seen for key in keys), dtype=bool, count=len(keys))
        stats.add_reason("duplicate_record", int((~fresh).sum()))
        seen.update(keys)
        parts.append(records[fresh])
    stats.files_read += 1
    if not parts:
        return pd.DataFrame(
            columns=["link_id", "hour_idx", "ts_sec", "speed", "travel_time"]
        )
    return pd.concat(parts, ignore_index=True)
