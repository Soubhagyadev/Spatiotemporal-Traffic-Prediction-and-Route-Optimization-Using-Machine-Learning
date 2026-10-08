"""Flask application serving the traffic dataset and the prediction model.

The API is the only component that touches the model and the database, which
keeps the ML pipeline independent of the presentation layers: Django and the
Dash dashboard are HTTP clients of this service.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from .data_service import TrafficDataService
from .prediction_service import PredictionService

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _timestamp(value: str | None, field: str = "hour_ts") -> str | None:
    """Validate and normalise an ISO-like timestamp, or raise ``ValueError``."""
    if value in (None, ""):
        return None
    try:
        return pd.Timestamp(value).strftime(TIMESTAMP_FORMAT)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be a date or timestamp, got {value!r}") from exc


def _int_arg(name: str, default: int, minimum: int = 1, maximum: int = 5000) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _granularity() -> str:
    value = (request.args.get("granularity") or "day").lower()
    if value not in ("hour", "day"):
        raise ValueError("granularity must be 'hour' or 'day'")
    return value


def create_app(db_path=None, bundle_path=None) -> Flask:
    """Application factory: wires the services and registers the routes."""
    app = Flask(__name__)
    CORS(app)
    data = TrafficDataService(db_path) if db_path else TrafficDataService()
    predictions = PredictionService(data, bundle_path) if bundle_path else PredictionService(data)

    @app.errorhandler(ValueError)
    def _bad_request(error):  # pragma: no cover - thin wrapper
        return jsonify({"error": "bad_request", "message": str(error)}), 400

    @app.errorhandler(404)
    def _not_found(error):
        return jsonify({"error": "not_found", "message": "unknown endpoint"}), 404

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        return jsonify({**data.health(), "model": predictions.summary()})

    @app.get("/api/meta")
    def meta():
        return jsonify({**data.meta(), "model": predictions.summary()})

    @app.get("/api/model/metrics")
    def model_metrics():
        return jsonify(data.model_metrics())

    # ------------------------------------------------------------------
    # segments
    # ------------------------------------------------------------------
    @app.get("/api/links")
    def links():
        return jsonify({"links": data.links(borough=request.args.get("borough"))})

    @app.get("/api/links/<int:link_id>")
    def link(link_id: int):
        detail = data.link(link_id)
        if detail is None:
            return jsonify({"error": "not_found", "message": f"link_id {link_id} unknown"}), 404
        return jsonify(detail)

    @app.get("/api/links/<int:link_id>/history")
    def link_history(link_id: int):
        return jsonify(
            {
                "link_id": link_id,
                "granularity": _granularity(),
                "points": data.link_history(
                    link_id,
                    start=_timestamp(request.args.get("start"), "start"),
                    end=_timestamp(request.args.get("end"), "end"),
                    granularity=_granularity(),
                ),
            }
        )

    # ------------------------------------------------------------------
    # analytics
    # ------------------------------------------------------------------
    @app.get("/api/traffic/over_time")
    def over_time():
        raw_link = request.args.get("link_id")
        return jsonify(
            {
                "granularity": _granularity(),
                "borough": request.args.get("borough"),
                "link_id": int(raw_link) if raw_link else None,
                "points": data.over_time(
                    start=_timestamp(request.args.get("start"), "start"),
                    end=_timestamp(request.args.get("end"), "end"),
                    granularity=_granularity(),
                    borough=request.args.get("borough"),
                    link_id=int(raw_link) if raw_link else None,
                ),
            }
        )

    @app.get("/api/patterns/hourly")
    def hourly_pattern():
        raw_link = request.args.get("link_id")
        return jsonify(
            {
                "points": data.hourly_pattern(
                    borough=request.args.get("borough"),
                    link_id=int(raw_link) if raw_link else None,
                )
            }
        )

    @app.get("/api/patterns/weekday_hour")
    def weekday_hour():
        return jsonify({"points": data.weekday_hour(borough=request.args.get("borough"))})

    @app.get("/api/congestion/segments")
    def congestion_segments():
        return jsonify(
            {
                "thresholds": predictions.thresholds(),
                "segments": data.congestion_segments(
                    limit=_int_arg("limit", 20, maximum=200),
                    borough=request.args.get("borough"),
                ),
            }
        )

    @app.get("/api/congestion/boroughs")
    def congestion_boroughs():
        return jsonify(
            {
                "thresholds": predictions.thresholds(),
                "boroughs": data.congestion_boroughs(),
            }
        )

    @app.get("/api/congestion/summary")
    def congestion_summary():
        raw_link = request.args.get("link_id")
        return jsonify(
            {
                "thresholds": predictions.thresholds(),
                "summary": data.congestion_summary(
                    borough=request.args.get("borough"),
                    link_id=int(raw_link) if raw_link else None,
                ),
            }
        )

    # ------------------------------------------------------------------
    # model diagnostics
    # ------------------------------------------------------------------
    @app.get("/api/model/predictions")
    def model_predictions():
        raw_link = request.args.get("link_id")
        return jsonify(
            {
                "points": data.prediction_sample(
                    limit=_int_arg("limit", 4000), link_id=int(raw_link) if raw_link else None
                )
            }
        )

    @app.get("/api/model/errors")
    def model_errors():
        return jsonify({"segments": data.prediction_errors(limit=_int_arg("limit", 15, maximum=100))})

    # ------------------------------------------------------------------
    # predictions
    # ------------------------------------------------------------------
    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        if "link_id" not in payload or "hour_ts" not in payload:
            raise ValueError("body must contain 'link_id' and 'hour_ts'")
        model = payload.get("model", "linear_regression")
        try:
            result = predictions.predict(
                int(payload["link_id"]), _timestamp(payload["hour_ts"]), model=model
            )
        except LookupError as exc:
            return jsonify({"error": "not_found", "message": str(exc)}), 404
        except KeyError as exc:
            return jsonify({"error": "bad_request", "message": str(exc)}), 400
        return jsonify(result)

    @app.post("/api/predict/horizon")
    def predict_horizon():
        payload = request.get_json(silent=True) or {}
        if "link_id" not in payload:
            raise ValueError("body must contain 'link_id'")
        start = payload.get("start_hour_ts") or dt.datetime.now().strftime(TIMESTAMP_FORMAT)
        steps = max(1, min(48, int(payload.get("steps", 6))))
        try:
            return jsonify(
                predictions.predict_horizon(
                    int(payload["link_id"]),
                    _timestamp(start),
                    steps=steps,
                    model=payload.get("model", "linear_regression"),
                )
            )
        except LookupError as exc:
            return jsonify({"error": "not_found", "message": str(exc)}), 404

    @app.post("/api/predict/batch")
    def predict_batch():
        payload = request.get_json(silent=True) or {}
        items = payload.get("requests") or []
        if not isinstance(items, list) or not items:
            raise ValueError("body must contain a non-empty 'requests' list")
        return jsonify(
            predictions.predict_batch(items, model=payload.get("model", "linear_regression"))
        )

    @app.get("/api/predict/latest/<int:link_id>")
    def predict_latest(link_id: int):
        try:
            return jsonify(predictions.latest_prediction(link_id))
        except LookupError as exc:
            return jsonify({"error": "not_found", "message": str(exc)}), 404

    return app
