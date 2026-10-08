"""Validate the raw, processed and model-ready data (plan section 1.5).

Usage
-----
    python scripts/validate_data.py [--sample-files 24]

Writes ``reports/validation_report.md`` and ``reports/validation_checks.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from traffic import build, config, leakage, raw_profile, reporting, validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-files", type=int, default=24)
    return parser.parse_args()


def load_splits() -> tuple[dict[str, pd.DataFrame], dict]:
    dataset = pd.read_csv(config.MODEL_DATASET_PATH, parse_dates=["hour_ts"])
    specification = json.loads((config.MODEL_DIR / "feature_spec.json").read_text())
    splits = {
        name: dataset[dataset["split"] == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    return splits, specification


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    hourly = pd.read_csv(config.HOURLY_DATASET_PATH, parse_dates=["hour_ts"])
    metadata = pd.read_csv(config.LINK_METADATA_PATH)
    preprocess = json.loads(config.PREPROCESS_STATS_PATH.read_text())
    raw_files = build.find_raw_files()
    sampled = raw_profile.sample_files(raw_files, args.sample_files)

    print(f"Hourly rows {len(hourly):,} | links {hourly['link_id'].nunique()} "
          f"| raw files {len(raw_files)}")
    print(f"Profiling {len(sampled)} sampled raw files ...")
    profile = raw_profile.profile_raw_files(sampled)

    checks = validation.validate_hourly_dataset(hourly, metadata)
    splits, specification = load_splits()
    target = specification["target"]
    lag_map = {f"speed_lag_{h}h": h for h in specification["lag_hours"]}
    split_checks = leakage.validate_split(
        splits,
        specification["feature_columns"],
        target,
        encoding_provenance=specification.get("encoding_provenance"),
    )
    split_checks += leakage.validate_lag_features(splits, lag_map)

    report = reporting.render_validation_report(
        {"hourly": hourly, "metadata": metadata, "raw_files": raw_files},
        checks,
        split_checks,
        profile,
        preprocess,
    )
    report_path = reporting.write_report(report, config.REPORT_DIR / "validation_report.md")
    checks_path = validation.save_checks(
        checks + split_checks, config.REPORT_DIR / "validation_checks.json"
    )
    (config.REPORT_DIR / "raw_profile.json").write_text(json.dumps(profile, indent=2) + "\n")

    statuses = [c.status for c in checks + split_checks]
    print(
        f"Checks: {statuses.count('PASS')} pass, {statuses.count('WARN')} warn, "
        f"{statuses.count('FAIL')} fail"
    )
    for check in checks + split_checks:
        if check.status != "PASS":
            print(f"  [{check.status}] {check.name}: {check.detail}")
    print(f"Report: {report_path}")
    print(f"Checks: {checks_path}")
    print(f"Elapsed {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
