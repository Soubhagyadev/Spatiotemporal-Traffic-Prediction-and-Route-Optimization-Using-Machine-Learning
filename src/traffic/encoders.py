"""Mean-target encoding for the high-cardinality ``link_id`` categorical.

The encoder is always fitted on past data only:

* for the validation and test periods it uses the whole training period;
* for the training period itself it uses an expanding-window scheme
  (block *k* is encoded with statistics from blocks ``0..k-1``), which is the
  standard way to avoid a row contributing to its own encoding.

Because traffic has a strong daily cycle, the ``(link_id, hour)`` profile is
encoded as well, which lets a linear model reproduce the per-segment diurnal
pattern with a single coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MeanTargetEncoder:
    """Encode category levels by the mean target observed in past data."""

    def __init__(self, keys: list[str], name: str, min_samples: int = 20) -> None:
        self.keys = list(keys)
        self.name = name
        self.min_samples = min_samples
        self.column = f"{name}_enc"
        self._mapping: pd.Series | None = None
        self._counts: pd.Series | None = None
        self._global_mean: float = np.nan
        self.fit_rows: int = 0
        self.fit_max_timestamp: pd.Timestamp | None = None

    def fit(self, frame: pd.DataFrame, target: str, timestamp: str = "hour_ts") -> "MeanTargetEncoder":
        """Learn level means from the supplied (past) rows."""
        self._global_mean = float(frame[target].mean())
        grouped = frame.groupby(self.keys, observed=True)[target].agg(["mean", "size"])
        keep = grouped["size"] >= self.min_samples
        self._mapping = grouped.loc[keep, "mean"]
        self._counts = grouped["size"]
        self.fit_rows = int(len(frame))
        if timestamp in frame.columns and len(frame):
            self.fit_max_timestamp = pd.to_datetime(frame[timestamp]).max()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        """Map rows to their level mean, falling back to the global mean."""
        if self._mapping is None:
            raise RuntimeError(f"encoder '{self.name}' must be fitted before transform")
        if len(self.keys) == 1:
            index = frame[self.keys[0]]
        else:
            index = pd.MultiIndex.from_frame(frame[self.keys])
        encoded = pd.Series(index.map(self._mapping), index=frame.index, dtype="float64")
        return encoded.fillna(self._global_mean)

    def coverage(self, frame: pd.DataFrame) -> float:
        """Share of rows whose level was seen often enough during fitting."""
        if self._mapping is None or not len(frame):
            return 0.0
        if len(self.keys) == 1:
            index = frame[self.keys[0]]
        else:
            index = pd.MultiIndex.from_frame(frame[self.keys])
        return float(index.isin(self._mapping.index).mean())

    def fit_transform_oof(
        self, frame: pd.DataFrame, target: str, n_blocks: int = 5, timestamp: str = "hour_ts"
    ) -> pd.Series:
        """Expanding-window encoding for rows inside the training period."""
        ordered = frame.sort_values(timestamp)
        small = ordered[self.keys + [target, timestamp]]
        values = np.full(len(ordered), np.nan, dtype="float64")
        starts = np.linspace(0, len(ordered), n_blocks + 1).astype(int)
        for block in range(n_blocks):
            rows = slice(starts[block], starts[block + 1])
            if rows.start == rows.stop:
                continue
            held_out = small.iloc[rows]
            history = small.iloc[: rows.start]
            if len(history) < self.min_samples:
                values[rows] = float(history[target].mean()) if len(history) else np.nan
                continue
            fold = MeanTargetEncoder(self.keys, self.name, self.min_samples).fit(
                history, target, timestamp
            )
            values[rows] = fold.transform(held_out).to_numpy()
        result = pd.Series(values, index=ordered.index, dtype="float64")
        return result.reindex(frame.index)

    @property
    def global_mean(self) -> float:
        return self._global_mean
