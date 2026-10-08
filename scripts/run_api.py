"""Run the Flask model-serving API.

Usage
-----
    python scripts/run_api.py [--host 127.0.0.1] [--port 5001] [--debug]

Endpoints are documented in the README ("Part 2 — services"); the interactive
list is at ``GET /api/meta``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from services.api import create_app  # noqa: E402

from traffic import config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not config.DB_PATH.exists():
        print(f"Database {config.DB_PATH} is missing - run scripts/build_database.py first.")
        return 1
    app = create_app()
    print(f"Traffic API on http://{args.host}:{args.port}  (database: {config.DB_PATH.name})")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
