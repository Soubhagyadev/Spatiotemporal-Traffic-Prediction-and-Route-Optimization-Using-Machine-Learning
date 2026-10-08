"""Model serving: rebuild the training features for a (segment, hour) request.

This module is the single source of truth for inference.  It reuses the exact
feature definitions from :mod:`traffic.features` (calendar helpers, lag hours,
rolling windows, encoding columns) and the fitted encoders and imputation
values stored in the model bundle, so a served prediction is built from the
same recipe the model was trained on.

Two prediction modes are reported back to the caller:

``observed-lag``
    the previous hours exist in the database, so the lags are real observations;
``forecast``
    some lags are missing (a future hour, or a gap in the feed) and are
    imputed with the segment's hour-of-day profile — the same fallback used
    during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, congestion, features

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def format_hour(stamp) -> str:
    return pd.Timestamp(stamp).strftime(TIMESTAMP_FORMAT)


@dataclass
class SegmentContext:
    """Static attributes of the segment being predicted."""

    link_id: int
    borough: str | None = None
    segment_length_miles: float | None = None


class TrafficPredictor:
    """Loads the trained bundle and turns hourly history into predictions."""

    def __init__(self, bundle_path: Path = config.MODEL_DIR / "model_bundle.joblib") -> None:
        bundle = joblib.load(bundle_path)
        self.models: dict = bundle["models"]
        self.feature_columns: list[str] = bundle["feature_columns"]
        self.target: str = bundle["target"]
        self.thresholds: dict = bundle["congestion_thresholds"]
        self.encoders: dict = bundle["encoders"]["encoders"]
        self.imputation_medians: dict = bundle["encoders"]["imputation_medians"]
        self.trained_on: dict = bundle.get("trained_on", {})
        self.bundle_path = Path(bundle_path)

    # ------------------------------------------------------------------
    # feature construction
    # ------------------------------------------------------------------
    def _calendar(self, hour_ts) -> pd.DataFrame:
        frame = pd.DataFrame({"hour_ts": [pd.Timestamp(hour_ts)]})
        return features.add_calendar_features(frame)

    def _lags(self, frame: pd.DataFrame, history: dict[str, float]) -> dict[str, int]:
        """Exact-timestamp lags; returns how many were observed."""
        observed = 0
        target = pd.Timestamp(frame["hour_ts"].iloc[0])
        for hours in features.LAG_HOURS:
            key = format_hour(target - pd.Timedelta(hours=hours))
            value = history.get(key)
            frame[f"speed_lag_{hours}h"] = np.nan if value is None else float(value)
            observed += int(value is not None)
        return {"lags_observed": observed, "lags_total": len(features.LAG_HOURS)}

    def _rolling(self, frame: pd.DataFrame, ordered_history: list[float]) -> None:
        """Mean of the previous k observations (current hour excluded)."""
        for window in features.ROLLING_WINDOWS:
            values = ordered_history[:window]
            frame[f"speed_roll_{window}h"] = (
                float(np.mean(values)) if len(values) >= 2 else np.nan
            )

    def _encode(self, frame: pd.DataFrame, context: SegmentContext) -> None:
        for keys, name in features.PROFILE_SPECS:
            encoder = self.encoders[name]
            lookup = frame.copy()
            if "link_id" not in lookup.columns:
                lookup["link_id"] = context.link_id
            frame[f"{name}_enc"] = encoder.transform(lookup).to_numpy()

    def _impute(self, frame: pd.DataFrame) -> None:
        fallback = features.ENCODING_COLUMNS[-1]
        for column in features.LAG_COLUMNS + features.ROLLING_COLUMNS:
            if pd.isna(frame[column].iloc[0]):
                frame[column] = frame[column].fillna(frame[fallback])
            frame[column] = frame[column].fillna(self.imputation_medians.get(column, 0.0))

    def _segment(self, frame: pd.DataFrame, context: SegmentContext) -> None:
        frame["segment_length_miles"] = (
            np.nan if context.segment_length_miles is None else float(context.segment_length_miles)
        )
        frame["segment_length_miles"] = frame["segment_length_miles"].fillna(
            self.imputation_medians.get("segment_length_miles", 0.0)
        )
        for column in self.feature_columns:
            if column.startswith("borough_"):
                frame[column] = int(column == f"borough_{context.borough}")

    def build_features(
        self,
        context: SegmentContext,
        hour_ts,
        history: list[dict],
    ) -> tuple[pd.DataFrame, dict]:
        """Assemble one model-ready row plus a small diagnostic dictionary."""
        ordered = sorted(history, key=lambda row: row["hour_ts"], reverse=True)
        speeds = [float(row["speed_mean"]) for row in ordered]
        index = {row["hour_ts"]: float(row["speed_mean"]) for row in ordered}

        frame = self._calendar(hour_ts)
        frame["link_id"] = context.link_id
        lag_info = self._lags(frame, index)
        self._rolling(frame, speeds)
        self._encode(frame, context)
        self._impute(frame)
        self._segment(frame, context)

        diagnostics = {
            **lag_info,
            "history_hours_available": len(ordered),
            "hours_since_last_observation": (
                None
                if not ordered
                else round(
                    (
                        pd.Timestamp(hour_ts)
                        - pd.Timestamp(ordered[0]["hour_ts"])
                    ).total_seconds()
                    / 3600.0,
                    2,
                )
            ),
        }
        diagnostics["mode"] = (
            "observed-lag" if lag_info["lags_observed"] == lag_info["lags_total"] else "forecast"
        )
        return frame[self.feature_columns], diagnostics

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        context: SegmentContext,
        hour_ts,
        history: list[dict],
        model: str = "linear_regression",
    ) -> dict:
        """Predict the hourly mean speed and congestion category."""
        if model not in self.models:
            raise KeyError(f"unknown model '{model}'; available: {sorted(self.models)}")
        frame, diagnostics = self.build_features(context, hour_ts, history)
        values = frame.to_numpy(dtype="float32" if model == "random_forest" else "float64")
        speed = float(np.asarray(self.models[model].predict(values)).ravel()[0])
        return {
            "link_id": context.link_id,
            "borough": context.borough,
            "hour_ts": format_hour(hour_ts),
            "model": model,
            "predicted_speed_mph": round(speed, 3),
            "congestion": str(congestion.categorize([speed], self.thresholds)[0]),
            **diagnostics,
        }

    def predict_horizon(
        self,
        context: SegmentContext,
        start_hour_ts,
        steps: int,
        history: list[dict],
        model: str = "linear_regression",
    ) -> list[dict]:
        """Roll predictions forward, feeding each result into the next step.

        This is the forecasting mode Part 3 needs: the first step uses observed
        lags where available, and later steps use the model's own predictions
        as the most recent history.
        """
        synthetic = sorted(history, key=lambda row: row["hour_ts"], reverse=True)
        start = pd.Timestamp(start_hour_ts)
        results = []
        for step in range(max(1, steps)):
            stamp = start + pd.Timedelta(hours=step)
            prediction = self.predict(context, stamp, synthetic, model=model)
            prediction["step"] = step + 1
            results.append(prediction)
            synthetic.insert(
                0,
                {
                    "hour_ts": format_hour(stamp),
                    "speed_mean": prediction["predicted_speed_mph"],
                    "obs_count": 0,
                },
            )
        return results

    def model_summary(self) -> dict:
        return {
            "models": sorted(self.models),
            "feature_count": len(self.feature_columns),
            "target": self.target,
            "trained_on": self.trained_on,
            "bundle": self.bundle_path.name,
        }
