"""Prediction service: serves the frozen model for arbitrary segment-hours.

A request only supplies a segment and a timestamp; the service pulls the
segment's recent usable hours from the database, rebuilds the training
features through :class:`traffic.serving.TrafficPredictor` and returns the
predicted speed plus its congestion category.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic import config, serving, store

from .data_service import TrafficDataService

HISTORY_HOURS = 200  # enough for the 168-hour lag plus rolling windows


class PredictionService:
    """Model-serving wrapper around the predictor and the data service."""

    def __init__(
        self,
        data_service: TrafficDataService,
        bundle_path: Path = config.MODEL_DIR / "model_bundle.joblib",
    ) -> None:
        self.data = data_service
        self.predictor = serving.TrafficPredictor(bundle_path)

    # ------------------------------------------------------------------
    def thresholds(self) -> dict:
        return self.predictor.thresholds

    def summary(self) -> dict:
        return self.predictor.model_summary()

    def _context(self, link_id: int) -> serving.SegmentContext:
        detail = self.data.link(link_id)
        if detail is None:
            raise LookupError(f"link_id {link_id} is not in the dataset")
        return serving.SegmentContext(
            link_id=link_id,
            borough=detail.get("borough"),
            segment_length_miles=detail.get("implied_length_miles"),
        )

    def _history(self, link_id: int, before: str) -> list[dict]:
        """Recent usable hours for a segment, newest first."""
        connection = self.data.connect()
        try:
            return store.recent_observations(
                connection, link_id, before=before, limit=HISTORY_HOURS
            )
        finally:
            connection.close()

    # ------------------------------------------------------------------
    def predict(
        self, link_id: int, hour_ts: str, model: str = "linear_regression"
    ) -> dict:
        """Single-segment prediction, with the mode and lag coverage reported."""
        stamp = serving.format_hour(pd.Timestamp(hour_ts))
        context = self._context(link_id)
        history = self._history(link_id, stamp)
        return self.predictor.predict(context, stamp, history, model=model)

    def predict_horizon(
        self,
        link_id: int,
        start_hour_ts: str,
        steps: int = 6,
        model: str = "linear_regression",
    ) -> dict:
        """Roll the prediction forward from a start hour."""
        start = serving.format_hour(pd.Timestamp(start_hour_ts))
        context = self._context(link_id)
        history = self._history(link_id, start)
        predictions = self.predictor.predict_horizon(
            context, start, steps, history, model=model
        )
        return {
            "link_id": link_id,
            "start_hour_ts": start,
            "steps": len(predictions),
            "model": model,
            "history_hours_available": len(history),
            "predictions": predictions,
        }

    def predict_batch(self, items: list[dict], model: str = "linear_regression") -> dict:
        """Predict a list of ``{link_id, hour_ts}`` requests."""
        results, errors = [], []
        for item in items[:500]:
            try:
                results.append(
                    self.predict(int(item["link_id"]), str(item["hour_ts"]), model=model)
                )
            except (LookupError, KeyError, ValueError) as exc:
                errors.append({"request": item, "error": str(exc)})
        return {"model": model, "count": len(results), "predictions": results, "errors": errors}

    def latest_prediction(self, link_id: int, model: str = "linear_regression") -> dict:
        """Predict the hour after the segment's most recent observation."""
        detail = self.data.link(link_id)
        if detail is None:
            raise LookupError(f"link_id {link_id} is not in the dataset")
        last = detail.get("last_hour_ts")
        if not last:
            raise LookupError(f"link_id {link_id} has no observations to continue from")
        next_hour = serving.format_hour(pd.Timestamp(last) + pd.Timedelta(hours=1))
        return self.predict(link_id, next_hour, model=model)
