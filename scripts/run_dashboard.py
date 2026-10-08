"""Run the Plotly Dash traffic analytics dashboard.

Usage
-----
    python scripts/run_dashboard.py [--host 127.0.0.1] [--port 8050]

The dashboard reads everything through the Flask API, so start
``scripts/run_api.py`` first (or point ``TRAFFIC_API_URL`` at another host).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from services.dashboard import create_dashboard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--api-url", default=None, help="base URL of the Flask API")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_dashboard(args.api_url)
    if not app.traffic_client.is_available():
        print(
            "Warning: the traffic API is not reachable "
            f"({app.traffic_client.base_url}). Start scripts/run_api.py first."
        )
    print(f"Dashboard on http://{args.host}:{args.port}  (API: {app.traffic_client.base_url})")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
