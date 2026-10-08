"""Run the Django web interface.

Usage
-----
    python scripts/run_web.py [--host 127.0.0.1] [--port 8000]

Serves the browsable interface (overview, segment directory, prediction form,
model performance).  Every page is rendered from the Flask API, so start
``scripts/run_api.py`` first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "services.web.trafficweb.settings")

from django.core.management import execute_from_command_line  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--noreload", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Django web interface on http://{args.host}:{args.port}")
    execute_from_command_line(
        ["manage.py", "runserver", f"{args.host}:{args.port}", "--noreload"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
