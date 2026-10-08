"""Score the held-out test period and store the predictions.

Usage
-----
    python scripts/build_predictions.py

Predictions are what the dashboard shows in its "predicted vs actual" and
"model performance" views, so they are computed once here (offline, with the
frozen model bundle) and written both to
``data/processed/test_predictions.csv`` and to the ``test_predictions`` table
of the SQLite database.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traffic import config, congestion, store  # noqa: E402

PREDICTION_COLUMNS = [
    "link_id",
    "hour_ts",
    "observed_speed",
    "predicted_speed",
    "predicted_speed_rf",
    "congestion_actual",
    "congestion_predicted",
]


def main() -> int:
    started = time.perf_counter()
    bundle = joblib.load(config.MODEL_DIR / "model_bundle.joblib")
    features = bundle["feature_columns"]
    target = bundle["target"]
    thresholds = bundle["congestion_thresholds"]

    print(f"Reading test rows from {config.MODEL_DATASET_PATH.name} ...")
    frame = pd.read_csv(
        config.MODEL_DATASET_PATH,
        usecols=["link_id", "hour_ts", "split", target] + features,
    )
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    print(f"  test link-hours: {len(test):,}")
    matrix = test[features].to_numpy(dtype="float32")

    linear = bundle["models"]["linear_regression"]
    forest = bundle["models"].get("random_forest")
    test["observed_speed"] = test[target].astype("float32")
    test["predicted_speed"] = np.asarray(linear.predict(matrix), dtype="float32")
    test["predicted_speed_rf"] = (
        np.asarray(forest.predict(matrix), dtype="float32") if forest else np.nan
    )
    test["congestion_actual"] = congestion.categorize(test["observed_speed"], thresholds)
    test["congestion_predicted"] = congestion.categorize(test["predicted_speed"], thresholds)

    predictions = test[PREDICTION_COLUMNS].copy()
    predictions.to_csv(
        config.TEST_PREDICTIONS_PATH, index=False, float_format="%.4f", lineterminator="\n"
    )

    connection = store.connect(read_only=False)
    try:
        connection.execute("DELETE FROM test_predictions")
        predictions.to_sql("test_predictions", connection, if_exists="append", index=False)
        connection.commit()
        stored = connection.execute("SELECT COUNT(*) FROM test_predictions").fetchone()[0]
    finally:
        connection.close()

    error = (predictions["observed_speed"] - predictions["predicted_speed"]).abs()
    summary = {
        "rows": int(len(predictions)),
        "mae_mph": round(float(error.mean()), 4),
        "stored_in_database": int(stored),
        "csv": str(config.TEST_PREDICTIONS_PATH),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
