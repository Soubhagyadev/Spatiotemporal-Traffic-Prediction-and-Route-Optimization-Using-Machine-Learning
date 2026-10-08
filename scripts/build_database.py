"""Build the SQLite database used by the Part 2 services.

Usage
-----
    python scripts/build_database.py

Loads the Part 1 artifacts (``data/processed/hourly_traffic.csv`` and
``data/processed/link_metadata.csv``) into ``data/traffic.db`` with the schema
and indexes defined in :mod:`traffic.store`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic import config, store  # noqa: E402


def main() -> int:
    print(f"Hourly dataset : {config.HOURLY_DATASET_PATH}")
    print(f"Link metadata  : {config.LINK_METADATA_PATH}")
    summary = store.build_database()
    for key, value in summary.items():
        print(f"  {key:12s}: {value}")

    connection = store.connect()
    try:
        bounds = store.dataset_bounds(connection)
        print(
            f"  window      : {bounds['start']} .. {bounds['end']} "
            f"({bounds['hours']:,} hours, {bounds['links']} segments)"
        )
        print(f"  boroughs    : {', '.join(store.boroughs(connection))}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
