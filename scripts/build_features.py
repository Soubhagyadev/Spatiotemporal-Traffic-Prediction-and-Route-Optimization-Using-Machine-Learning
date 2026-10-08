"""Build the model-ready feature dataset (plan section 1.4).

Usage
-----
    python scripts/build_features.py

Reads the hourly aggregates, adds calendar/lag/rolling features, encodes the
``link_id`` categorical from training-period data only, splits the series
chronologically into train/validation/test and writes
``data/features/model_dataset.csv`` together with the fitted encoders and the
feature specification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from traffic import config, features  # noqa: E402


def main() -> int:
    print(f"Reading hourly aggregates: {config.HOURLY_DATASET_PATH}")
    hourly = features.load_hourly(config.HOURLY_DATASET_PATH)
    metadata = pd.read_csv(config.LINK_METADATA_PATH)

    frame, info = features.build_model_dataset(hourly, metadata)
    print(
        f"Modelling frame: {info['rows_modelled']:,} link-hours "
        f"(from {info['hourly_rows']:,} hourly rows, "
        f"{info['dropped_thin_rows']:,} thin rows and "
        f"{info['dropped_without_previous_hour']:,} rows without a previous hour dropped)"
    )

    splits = features.chronological_split(frame)
    train, others, encoders = features.add_profile_encodings(
        splits["train"], {"val": splits["val"], "test": splits["test"]}
    )
    splits = {"train": train, **others}
    columns = features.feature_columns(train)
    medians = features.impute_features(splits, columns)
    provenance = features.encoding_provenance(encoders, train)

    dataset = pd.concat(
        [frame.assign(split=name) for name, frame in splits.items()], ignore_index=True
    ).sort_values(["hour_ts", "link_id"], ignore_index=True)

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.MODEL_DATASET_PATH, index=False, float_format="%.6g")
    joblib.dump(
        {"encoders": encoders, "imputation_medians": medians, "provenance": provenance},
        config.MODEL_DIR / "encoders.joblib",
    )

    specification = {
        "target": config.TARGET_COLUMN,
        "feature_columns": columns,
        "calendar_features": features.CALENDAR_FEATURES,
        "lag_hours": list(features.LAG_HOURS),
        "rolling_windows": list(features.ROLLING_WINDOWS),
        "encoding_columns": features.ENCODING_COLUMNS,
        "imputation_medians": medians,
        "encoding_provenance": provenance,
        "split_fractions": {
            "train": config.TRAIN_FRACTION,
            "val": config.VAL_FRACTION,
            "test": round(1 - config.TRAIN_FRACTION - config.VAL_FRACTION, 4),
        },
        "rows": {name: int(len(split)) for name, split in splits.items()},
        "build_info": info,
    }
    (config.MODEL_DIR / "feature_spec.json").write_text(json.dumps(specification, indent=2) + "\n")

    for name, split in splits.items():
        print(
            f"  {name:5s}: {len(split):>9,} rows  "
            f"{split['hour_ts'].min()} .. {split['hour_ts'].max()}"
        )
    print(f"Features ({len(columns)}): {', '.join(columns)}")
    print(f"Dataset   : {config.MODEL_DATASET_PATH}")
    print(f"Spec      : {config.MODEL_DIR / 'feature_spec.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
