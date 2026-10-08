"""Raw-data profiling.

The full raw dataset is 5.4 GB, so the raw-level validation profile is built
from an evenly spaced sample of files, streamed in chunks.  Record-level
counts for the *whole* dataset are produced by the preprocessing pass and are
reported alongside this sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .cleaning import iter_raw_chunks


def sample_files(files: list[Path], count: int) -> list[Path]:
    """Pick ``count`` files spread evenly across the (time-ordered) export."""
    if len(files) <= count:
        return list(files)
    positions = np.linspace(0, len(files) - 1, count).round().astype(int)
    return [files[i] for i in sorted(set(positions.tolist()))]


def profile_raw_files(
    files: list[Path], chunk_size: int = config.CHUNK_SIZE
) -> dict:
    """Stream sampled raw files and summarise their quality."""
    rows = 0
    missing = {col: 0 for col in ("speed", "travel_time", "status", "data_as_of", "link_id")}
    non_numeric = {"speed": 0, "travel_time": 0}
    negative = {"speed": 0, "travel_time": 0}
    zero = {"speed": 0, "travel_time": 0}
    status_counts: dict[str, int] = {}
    unparsable_timestamps = 0
    duplicate_records = 0
    out_of_order_records = 0
    speed_values: list[np.ndarray] = []
    travel_time_values: list[np.ndarray] = []
    link_ids: set[int] = set()
    time_min: pd.Timestamp | None = None
    time_max: pd.Timestamp | None = None

    for path in files:
        previous_ts: pd.Timestamp | None = None
        seen: set[tuple[int, int]] = set()
        for chunk in iter_raw_chunks(path, chunk_size=chunk_size):
            rows += len(chunk)
            for column in missing:
                values = chunk[column]
                missing[column] += int(values.isna().sum() + (values == "").sum())

            speed = pd.to_numeric(chunk["speed"], errors="coerce")
            travel = pd.to_numeric(chunk["travel_time"], errors="coerce")
            non_numeric["speed"] += int(speed.isna().sum())
            non_numeric["travel_time"] += int(travel.isna().sum())
            negative["speed"] += int((speed < 0).sum())
            negative["travel_time"] += int((travel < 0).sum())
            zero["speed"] += int((speed == 0).sum())
            zero["travel_time"] += int((travel == 0).sum())
            for value, counts in chunk["status"].value_counts().items():
                status_counts[str(value)] = status_counts.get(str(value), 0) + int(counts)

            timestamps = pd.to_datetime(chunk["data_as_of"], format="ISO8601", errors="coerce")
            unparsable_timestamps += int(timestamps.isna().sum())
            valid_ts = timestamps.dropna()
            if not valid_ts.empty:
                chunk_min, chunk_max = valid_ts.min(), valid_ts.max()
                time_min = chunk_min if time_min is None else min(time_min, chunk_min)
                time_max = chunk_max if time_max is None else max(time_max, chunk_max)
                if previous_ts is not None and chunk_min < previous_ts:
                    out_of_order_records += 1
                previous_ts = chunk_max

            ids = pd.to_numeric(chunk["link_id"], errors="coerce")
            link_ids.update(ids.dropna().astype("int64").unique().tolist())
            keys = zip(ids.fillna(-1).astype("int64").tolist(), valid_ts.astype("int64").tolist())
            fresh = 0
            for key in keys:
                if key in seen:
                    fresh += 1
                else:
                    seen.add(key)
            duplicate_records += fresh

            speed_values.append(speed.dropna().to_numpy(dtype="float64"))
            travel_time_values.append(travel.dropna().to_numpy(dtype="float64"))

    speed_all = np.concatenate(speed_values) if speed_values else np.array([])
    travel_all = np.concatenate(travel_time_values) if travel_time_values else np.array([])

    def quantiles(values: np.ndarray, points=(0.01, 0.25, 0.5, 0.75, 0.99, 0.999)) -> dict:
        if values.size == 0:
            return {}
        return {str(p): round(float(np.quantile(values, p)), 3) for p in points}

    return {
        "sampled_files": len(files),
        "sampled_file_names": [p.name for p in files],
        "rows_sampled": int(rows),
        "missing_values": missing,
        "non_numeric_values": non_numeric,
        "negative_values": negative,
        "zero_values": zero,
        "status_counts": status_counts,
        "unparsable_timestamps": int(unparsable_timestamps),
        "duplicate_records_in_sample": int(duplicate_records),
        "chunks_out_of_order": int(out_of_order_records),
        "distinct_link_ids": int(len(link_ids)),
        "timestamp_min": None if time_min is None else str(time_min),
        "timestamp_max": None if time_max is None else str(time_max),
        "speed_quantiles": quantiles(speed_all),
        "travel_time_quantiles": quantiles(travel_all),
        "speed_max": None if speed_all.size == 0 else float(speed_all.max()),
        "travel_time_max": None if travel_all.size == 0 else float(travel_all.max()),
    }
