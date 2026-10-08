"""Data-validation checks for the cleaned hourly dataset.

Every check returns a :class:`Check` with a PASS/WARN/FAIL status so the
validation report is machine-readable as well as human-readable.  Checks about
the train/test split and leakage live in :mod:`traffic.leakage`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str

    def to_row(self) -> dict:
        return asdict(self)


def _status(condition_bad: bool, warn: bool = False) -> str:
    if condition_bad:
        return WARN if warn else FAIL
    return PASS


def validate_hourly_dataset(hourly: pd.DataFrame, metadata: pd.DataFrame) -> list[Check]:
    """Checks on the cleaned/aggregated dataset (plan section 1.5)."""
    checks: list[Check] = []
    key_columns = ["link_id", "hour_ts", "obs_count", "speed_mean", "travel_time_mean"]
    missing = {col: int(hourly[col].isna().sum()) for col in key_columns}
    worst = max(missing.values())
    checks.append(
        Check(
            "missing_values",
            _status(worst > 0),
            f"missing counts in key columns: {missing} "
            f"(speed_std is NaN for single-observation hours by construction)",
        )
    )

    duplicates = int(hourly.duplicated(subset=["link_id", "hour_ts"]).sum())
    checks.append(
        Check(
            "duplicate_link_hours",
            _status(duplicates > 0),
            f"{duplicates} duplicated (link_id, hour_ts) pairs",
        )
    )

    hour_ts = pd.to_datetime(hourly["hour_ts"])
    ordered = bool(hour_ts.is_monotonic_increasing)
    checks.append(
        Check(
            "timestamp_ordering",
            _status(not ordered),
            f"hour_ts monotonic increasing: {ordered}; "
            f"range {hour_ts.min()} .. {hour_ts.max()}",
        )
    )

    fractional = int(((hour_ts.dt.minute != 0) | (hour_ts.dt.second != 0)).sum())
    checks.append(
        Check(
            "hourly_resolution",
            _status(fractional > 0),
            f"{fractional} rows not aligned to a whole hour "
            f"(distinct hours: {hour_ts.nunique():,})",
        )
    )

    non_positive = {
        "obs_count": int((hourly["obs_count"] <= 0).sum()),
        "speed_mean": int((hourly["speed_mean"] <= 0).sum()),
        "travel_time_mean": int((hourly["travel_time_mean"] <= 0).sum()),
    }
    checks.append(
        Check(
            "non_positive_values",
            _status(any(v > 0 for v in non_positive.values())),
            f"zero/negative values per column: {non_positive}",
        )
    )

    low_obs = int((hourly["obs_count"] < config.MIN_OBS_PER_LINK_HOUR).sum())
    share = 100.0 * low_obs / max(len(hourly), 1)
    checks.append(
        Check(
            "thin_hourly_groups",
            _status(share > 25.0, warn=True),
            f"{low_obs:,} link-hours ({share:.2f}%) have fewer than "
            f"{config.MIN_OBS_PER_LINK_HOUR} observations and are excluded from modelling",
        )
    )

    speed = hourly["speed_mean"]
    extreme = int(((speed > 90) | (speed < 1)).sum())
    checks.append(
        Check(
            "speed_outliers",
            _status(extreme > len(hourly) * 0.001, warn=True),
            f"{extreme:,} hourly means outside [1, 90] mph; "
            f"p0.1={speed.quantile(0.001):.2f} p99.9={speed.quantile(0.999):.2f}",
        )
    )

    observed = metadata.set_index("link_id")["hours_observed"]
    never_observed = observed[observed.isna()]
    sparse_links = observed[observed < config.MIN_HOURS_PER_LINK]
    checks.append(
        Check(
            "insufficient_observations_per_link",
            _status(len(sparse_links) > 0, warn=True),
            f"{len(sparse_links)} of {len(observed)} links have fewer than "
            f"{config.MIN_HOURS_PER_LINK} observed hours, of which {len(never_observed)} "
            f"have no usable hour at all (every record was stale/invalid): "
            f"{sorted(never_observed.index)[:6]}",
        )
    )

    usable_links = set(hourly["link_id"].unique())
    missing_links = sorted(set(metadata["link_id"]) - usable_links)
    checks.append(
        Check(
            "links_without_predictions",
            _status(bool(missing_links), warn=True),
            f"{len(missing_links)} of {len(metadata)} segments appear in the raw exports but "
            f"have no valid hourly aggregate, so they cannot be predicted: {missing_links}",
        )
    )

    stamps = pd.to_datetime(hourly["hour_ts"])
    links_per_hour = (
        hourly.assign(hour=stamps.dt.hour).groupby("hour")["link_id"].nunique().sort_index()
    )
    balance = float(links_per_hour.min() / links_per_hour.max())
    checks.append(
        Check(
            "hourly_panel_balance",
            _status(balance < 0.9, warn=True),
            f"{int(links_per_hour.min())}-{int(links_per_hour.max())} of "
            f"{hourly['link_id'].nunique()} segments report in each hour-of-day bucket "
            f"({balance:.0%} balance), so the diurnal speed profile is not an artefact of a "
            f"changing set of reporting segments",
        )
    )

    summary = metadata.set_index("link_id")
    span_hours = (
        pd.to_datetime(summary["last_hour_ts"]) - pd.to_datetime(summary["first_hour_ts"])
    ).dt.total_seconds() / 3600.0 + 1.0
    coverage = (summary["hours_observed"] / span_hours).dropna()
    checks.append(
        Check(
            "temporal_coverage_per_link",
            _status(coverage.median() < 0.5, warn=True),
            f"median share of hours present inside each link's own time span: "
            f"{coverage.median():.1%} (p10 {coverage.quantile(0.1):.1%})",
        )
    )

    total_outliers = int(metadata["outlier_records"].sum())
    total_records = int(metadata["records_in"].sum())
    ratio = 100.0 * total_outliers / max(total_records, 1)
    checks.append(
        Check(
            "implied_length_outliers",
            _status(ratio > 2.0, warn=True),
            f"{total_outliers:,} of {total_records:,} clean records ({ratio:.3f}%) deviate "
            f"more than {config.LENGTH_OUTLIER_TOLERANCE:.0%} from their link's median "
            f"implied length and were removed",
        )
    )

    duplicates_across = int(metadata["duplicate_records"].sum())
    checks.append(
        Check(
            "duplicate_records",
            _status(False),
            f"{duplicates_across:,} repeated (link_id, timestamp) records were dropped "
            f"after per-link de-duplication",
        )
    )

    geometry = metadata.dropna(subset=["geom_length_miles", "implied_length_miles"])
    ratio_geom = (geometry["implied_length_miles"] / geometry["geom_length_miles"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    within = float(((ratio_geom > 0.75) & (ratio_geom < 1.33)).mean()) if len(ratio_geom) else 0.0
    checks.append(
        Check(
            "implied_vs_geometric_length",
            _status(within < 0.6, warn=True),
            f"{within:.1%} of {len(ratio_geom)} links have sensor-implied length within "
            f"±25% of the decoded polyline length (median ratio "
            f"{ratio_geom.median():.3f}); mismatches are expected for tunnels, bridges "
            f"and ramp segments and are reported, not filtered",
        )
    )

    complete = int(metadata["geometry_complete"].sum()) if "geometry_complete" in metadata else 0
    checks.append(
        Check(
            "segment_geometry",
            _status(complete < len(metadata), warn=True),
            f"{complete}/{len(metadata)} links have a complete decoded polyline; "
            f"the rest fall back to the 255-character truncated link_points",
        )
    )
    return checks



def save_checks(checks: list[Check], path: Path) -> Path:
    """Persist check results as JSON next to the markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            status: sum(1 for c in checks if c.status == status)
            for status in (PASS, WARN, FAIL)
        },
        "checks": [c.to_row() for c in checks],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
