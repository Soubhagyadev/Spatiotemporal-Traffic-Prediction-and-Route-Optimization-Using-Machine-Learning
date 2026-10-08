"""Feature engineering for hourly traffic-speed prediction (plan section 1.4).

Calendar features, lagged speed, rolling averages and the previous-day
same-hour speed are all built from *earlier* observations only.  Lag values
are looked up by exact timestamp (not by row position), so a ``lag_1h`` value
is genuinely the aggregate of the previous hour and never a value from two
hours back.

``link_id`` is a high-cardinality categorical and is encoded with mean-target
encodings (see :mod:`traffic.encoders`); ``borough`` is low-cardinality and is
one-hot encoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .encoders import MeanTargetEncoder

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
LAG_HOURS = (1, 2, 3, 24, 168)
ROLLING_WINDOWS = (3, 6, 24)
LAG_COLUMNS = [f"speed_lag_{hours}h" for hours in LAG_HOURS]
ROLLING_COLUMNS = [f"speed_roll_{window}h" for window in ROLLING_WINDOWS]
PROFILE_SPECS = [
    (["link_id"], "link_speed_profile"),
    (["link_id", "hour"], "link_hour_speed_profile"),
]
ENCODING_COLUMNS = [f"{name}_enc" for _, name in PROFILE_SPECS]


def load_hourly(path: Path = config.HOURLY_DATASET_PATH) -> pd.DataFrame:
    """Read the aggregated dataset with a parsed time axis."""
    frame = pd.read_csv(path)
    frame["hour_ts"] = pd.to_datetime(frame["hour_ts"], format="ISO8601")
    return frame


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Hour of day, day of week, month, weekend flag and cyclical encodings."""
    stamp = frame["hour_ts"]
    frame["hour"] = stamp.dt.hour.astype("int16")
    frame["day_of_week"] = stamp.dt.dayofweek.astype("int16")
    frame["month"] = stamp.dt.month.astype("int16")
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int8")
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24.0)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7.0)
    return frame


def add_lag_features(frame: pd.DataFrame, lag_hours: tuple[int, ...] = LAG_HOURS) -> pd.DataFrame:
    """Exact-timestamp lags of the hourly mean speed.

    A lag is ``NaN`` when that link has no aggregate at the lagged timestamp,
    which keeps the feature honest instead of silently using a stale hour.
    """
    for hours in lag_hours:
        lookup = frame[["link_id", "hour_ts", "speed_mean"]].copy()
        lookup["hour_ts"] = lookup["hour_ts"] + pd.Timedelta(hours=hours)
        lookup = lookup.rename(columns={"speed_mean": f"speed_lag_{hours}h"})
        frame = frame.merge(lookup, on=["link_id", "hour_ts"], how="left")
    return frame


def add_rolling_features(
    frame: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Rolling mean speed over the previous *observed* hours (current excluded)."""
    frame = frame.sort_values(["link_id", "hour_ts"], ignore_index=True)
    previous = frame.groupby("link_id", observed=True)["speed_mean"].shift(1)
    grouped = previous.groupby(frame["link_id"], observed=True)
    for window in windows:
        frame[f"speed_roll_{window}h"] = (
            grouped.rolling(window, min_periods=2).mean().reset_index(level=0, drop=True)
        )
    return frame


def add_borough_dummies(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the low-cardinality ``borough`` attribute.

    One level is dropped so the dummies are not perfectly collinear with the
    intercept; the remaining coefficients are contrasts against the reference
    borough.
    """
    frame = frame.merge(
        metadata[["link_id", "borough", "link_name", "implied_length_miles"]].rename(
            columns={"implied_length_miles": "segment_length_miles"}
        ),
        on="link_id",
        how="left",
    )
    dummies = pd.get_dummies(frame["borough"], prefix="borough", dtype="int8", drop_first=True)
    return pd.concat([frame, dummies], axis=1)


def build_model_dataset(
    hourly: pd.DataFrame, metadata: pd.DataFrame, min_obs: int = config.MIN_OBS_PER_LINK_HOUR
) -> tuple[pd.DataFrame, dict]:
    """Assemble the modelling frame from the hourly aggregates."""
    info = {"hourly_rows": int(len(hourly))}
    frame = hourly[hourly["obs_count"] >= min_obs].copy()
    info["rows_with_enough_observations"] = int(len(frame))
    info["dropped_thin_rows"] = int(len(hourly) - len(frame))

    frame = frame.sort_values(["link_id", "hour_ts"], ignore_index=True)
    frame = add_calendar_features(frame)
    frame = add_lag_features(frame)
    frame = add_rolling_features(frame)
    frame = add_borough_dummies(frame, metadata)

    primary = frame["speed_lag_1h"].notna()
    info["dropped_without_previous_hour"] = int((~primary).sum())
    frame = frame[primary].reset_index(drop=True)
    info["rows_modelled"] = int(len(frame))
    return frame, info


def add_profile_encodings(
    train: pd.DataFrame, others: dict[str, pd.DataFrame], target: str = config.TARGET_COLUMN
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, MeanTargetEncoder]]:
    """Encode ``link_id`` profiles from training data only.

    ``train`` gets out-of-fold (expanding-window) encodings; every other split
    is encoded with statistics from the whole training period.  The fitted
    encoders are returned so they can be persisted for serving.
    """
    train = train.copy()
    others = {name: frame.copy() for name, frame in others.items()}
    encoders: dict[str, MeanTargetEncoder] = {}
    for keys, name in PROFILE_SPECS:
        encoder = MeanTargetEncoder(keys, name).fit(train, target)
        train[encoder.column] = encoder.fit_transform_oof(train, target)
        for frame in others.values():
            frame[encoder.column] = encoder.transform(frame)
        encoders[name] = encoder
    return train, others, encoders


def encoding_provenance(encoders: dict[str, MeanTargetEncoder], train: pd.DataFrame) -> list[dict]:
    """Describe where each encoder's statistics came from (for the reports)."""
    return [
        {
            "encoder": name,
            "keys": encoder.keys,
            "levels_learned": int(len(encoder._mapping)) if encoder._mapping is not None else 0,
            "fit_rows": encoder.fit_rows,
            "fit_end_timestamp": str(encoder.fit_max_timestamp),
            "train_coverage": round(encoder.coverage(train), 4),
        }
        for name, encoder in encoders.items()
    ]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs: calendar, lags, rolling means and encodings."""
    borough = sorted(column for column in frame.columns if column.startswith("borough_"))
    return (
        CALENDAR_FEATURES
        + LAG_COLUMNS
        + ROLLING_COLUMNS
        + ["segment_length_miles"]
        + ENCODING_COLUMNS
        + borough
    )


def impute_features(
    frames: dict[str, pd.DataFrame], columns: list[str], reference: str = "train"
) -> dict[str, float]:
    """Fill missing feature cells using training-period information.

    Lag/rolling gaps are filled with the row's own link-hour profile (the
    expected speed for that segment at that hour); anything still missing is
    filled with the training-period median of the column.
    """
    fallback = ENCODING_COLUMNS[-1]
    medians = {column: float(frames[reference][column].median()) for column in columns}
    for frame in frames.values():
        for column in columns:
            if not frame[column].isna().any():
                continue
            if column in LAG_COLUMNS + ROLLING_COLUMNS and fallback in frame.columns:
                frame[column] = frame[column].fillna(frame[fallback])
            frame[column] = frame[column].fillna(medians[column])
    return medians


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> dict[str, pd.DataFrame]:
    """Split by time into train/validation/test periods (never shuffled).

    The cut points fall between hours, so a link-hour always belongs to
    exactly one split and no timestamp is shared between two splits.
    """
    ordered = frame.sort_values("hour_ts", ignore_index=True)
    hours = np.sort(ordered["hour_ts"].unique())
    train_end = hours[int(len(hours) * train_fraction)]
    val_end = hours[int(len(hours) * (train_fraction + val_fraction))]
    stamps = ordered["hour_ts"]
    return {
        "train": ordered[stamps < train_end].reset_index(drop=True),
        "val": ordered[(stamps >= train_end) & (stamps < val_end)].reset_index(drop=True),
        "test": ordered[stamps >= val_end].reset_index(drop=True),
    }
