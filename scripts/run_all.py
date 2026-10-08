"""Run the complete Part 1 pipeline end to end.

Usage
-----
    python scripts/run_all.py [--skip-forest] [--keep-interim]

Steps: build the hourly dataset -> build features -> validate -> train.
Each step can also be run on its own; see the individual scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PART1_STEPS = (
    ("build_dataset.py", "Build cleaned hourly dataset from the raw CSV exports"),
    ("build_features.py", "Engineer features, encode segments and split chronologically"),
    ("validate_data.py", "Validate raw/processed data and the train-test split"),
    ("train_model.py", "Train and evaluate the prediction models"),
)
PART2_STEPS = (
    ("build_database.py", "Load the processed artifacts into the SQLite database"),
    ("build_predictions.py", "Score the test period for the dashboard"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-forest", action="store_true")
    parser.add_argument("--keep-interim", action="store_true")
    parser.add_argument("--part1-only", action="store_true", help="skip the Part 2 build steps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = PART1_STEPS if args.part1_only else PART1_STEPS + PART2_STEPS
    started = time.perf_counter()
    for index, (script, description) in enumerate(steps, start=1):
        command = [sys.executable, str(SCRIPTS_DIR / script)]
        if script == "build_dataset.py" and args.keep_interim:
            command.append("--keep-interim")
        if script == "train_model.py" and args.skip_forest:
            command.append("--skip-forest")
        print(f"\n{'=' * 78}\n[{index}/{len(steps)}] {description}\n{'=' * 78}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"Step {script} failed with exit code {result.returncode}")
            return result.returncode
    print(f"\nPipeline finished in {time.perf_counter() - started:,.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
