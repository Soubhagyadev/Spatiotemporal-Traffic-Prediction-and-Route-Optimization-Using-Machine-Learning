"""Build the cleaned hourly dataset from the raw CSV exports.

Usage
-----
    python scripts/build_dataset.py [--keep-interim] [--raw-dir DIR]

Pass 1 streams every raw file in chunks, converts the text-typed ``speed``
and ``travel_time`` columns to numeric, drops invalid/stale/duplicate records
and appends the survivors to one small file per road segment.
Pass 2 aggregates each segment exactly to hourly statistics (mean, median,
standard deviation, min, max, observation count) and writes the deliverables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from traffic import build, config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument(
        "--keep-interim",
        action="store_true",
        help="keep the per-link record partitions in data/interim/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    print(f"Raw directory : {args.raw_dir}")
    raw_files = build.find_raw_files(args.raw_dir)
    print(f"Raw files     : {len(raw_files)}")

    print("\nPass 1/2 - streaming, cleaning and partitioning raw chunks ...")
    result = build.run(raw_dir=args.raw_dir)
    stats = result["cleaning_stats"]

    dataset = result["hourly"]
    metadata = result["link_metadata"]
    print(f"\nPass 2/2 - hourly aggregation -> {len(dataset):,} link-hour rows")

    hourly_path = build.write_hourly_dataset(dataset)
    metadata_path = config.LINK_METADATA_PATH
    metadata.to_csv(metadata_path, index=False, float_format="%.6f", lineterminator="\n")

    stats_payload = stats.as_dict()
    stats_payload.update(
        {
            "raw_files": len(raw_files),
            "links": int(metadata["link_id"].nunique()),
            "hourly_rows": int(len(dataset)),
            "hour_ts_min": str(dataset["hour_ts"].min()),
            "hour_ts_max": str(dataset["hour_ts"].max()),
            "outlier_records": int(result["link_summary"]["outlier_records"].sum()),
            "duplicate_records_across_files": int(
                result["link_summary"]["duplicate_records"].sum()
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
    )
    config.PREPROCESS_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PREPROCESS_STATS_PATH.write_text(json.dumps(stats_payload, indent=2) + "\n")

    if not args.keep_interim:
        shutil.rmtree(config.INTERIM_DIR, ignore_errors=True)
        print(f"Removed interim partitions under {config.INTERIM_DIR}")

    print(f"\nHourly dataset : {hourly_path} ({hourly_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Link metadata  : {metadata_path}")
    print(f"Stats          : {config.PREPROCESS_STATS_PATH}")
    print(f"Total time     : {stats_payload['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
