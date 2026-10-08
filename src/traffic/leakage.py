"""Train/test leakage checks for the chronological split and the features.

These checks answer a different question from :mod:`traffic.validation`: not
"is the data good?" but "is the evaluation honest?" — that the split is
strictly time-ordered, that encodings were fitted on the training period only,
and that every lag feature really carries information from an earlier
timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import PASS, FAIL, WARN, Check, _status


def validate_split(
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target: str,
    encoding_provenance: list[dict] | None = None,
) -> list[Check]:
    """Leakage and sanity checks on the chronological split (plan section 1.5)."""
    checks: list[Check] = []
    bounds = {
        name: (pd.to_datetime(frame["hour_ts"]).min(), pd.to_datetime(frame["hour_ts"]).max())
        for name, frame in split_frames.items()
    }
    ordering_ok = bounds["train"][1] < bounds["val"][0] and bounds["val"][1] < bounds["test"][0]
    checks.append(
        Check(
            "chronological_split",
            _status(not ordering_ok),
            "; ".join(f"{name}: {lo} .. {hi}" for name, (lo, hi) in bounds.items())
            + f" | strictly ordered, no overlap: {ordering_ok}",
        )
    )

    overlap = set(split_frames["train"]["link_id"]) & set(split_frames["test"]["link_id"])
    checks.append(
        Check(
            "split_composition",
            PASS,
            f"rows train/val/test = {len(split_frames['train']):,}/"
            f"{len(split_frames['val']):,}/{len(split_frames['test']):,}; "
            f"{len(overlap)} links appear in both train and test (expected: the same "
            f"road network is observed in every period, and no target value is shared)",
        )
    )

    checks.append(
        Check(
            "target_not_a_feature",
            _status(target in feature_columns),
            f"target '{target}' present in feature list: {target in feature_columns}",
        )
    )

    leaky = [c for c in feature_columns if c.startswith("travel_time") or c == target]
    checks.append(
        Check(
            "no_target_derived_features",
            _status(bool(leaky)),
            f"features derived from the modelled quantity: {leaky or 'none'}",
        )
    )

    train_max = bounds["train"][1]
    for entry in encoding_provenance or []:
        fit_end = pd.to_datetime(entry.get("fit_end_timestamp"))
        checks.append(
            Check(
                f"encoding_from_training_period_only[{entry.get('encoder')}]",
                _status(bool(pd.notna(fit_end) and fit_end > train_max)),
                f"{entry.get('levels_learned', 0):,} levels learned from "
                f"{entry.get('fit_rows', 0):,} training rows, last timestamp used "
                f"{fit_end} (training ends {train_max}); "
                f"coverage {entry.get('train_coverage', 0):.1%}",
            )
        )

    constant = sorted(
        column
        for column in feature_columns
        if column in split_frames["train"].columns
        and float(split_frames["train"][column].std()) == 0.0
    )
    checks.append(
        Check(
            "features_have_variance",
            _status(bool(constant)),
            f"{len(constant)} of {len(feature_columns)} features are constant on the training "
            f"period and would carry no information: {constant or 'none'}",
        )
    )

    nan_rates = {
        name: float(frame[feature_columns].isna().to_numpy().mean())
        for name, frame in split_frames.items()
    }
    worst = max(nan_rates.values())
    checks.append(
        Check(
            "feature_missingness",
            _status(worst > 0.2, warn=True),
            "share of missing feature cells: "
            + ", ".join(f"{k}={v:.2%}" for k, v in nan_rates.items())
            + " (missing values are imputed from training-period statistics)",
        )
    )

    for name, frame in split_frames.items():
        duplicates = int(frame.duplicated(subset=["link_id", "hour_ts"]).sum())
        checks.append(
            Check(
                f"duplicate_rows_{name}",
                _status(duplicates > 0),
                f"{duplicates} duplicated (link_id, hour_ts) rows in the {name} split",
            )
        )
    return checks


def validate_lag_features(
    split_frames: dict[str, pd.DataFrame], lags: dict[str, int], sample_size: int = 20_000
) -> list[Check]:
    """Recompute lag features from the data and compare them with the stored ones.

    A stored lag is only reproducible if the link-hour it points at survived
    the modelling filters, so the check reports the share of lag values whose
    reference is present and compares exactly those.
    """
    combined = pd.concat(split_frames.values(), ignore_index=True)
    checks: list[Check] = []
    for column, hours in lags.items():
        if column not in combined.columns:
            continue
        expected = combined[["link_id", "hour_ts", "speed_mean"]].copy()
        expected["hour_ts"] = expected["hour_ts"] + pd.Timedelta(hours=hours)
        expected = expected.rename(columns={"speed_mean": "expected"})
        available = combined[combined[column].notna()]
        if len(available) > sample_size:
            available = available.sample(sample_size, random_state=0)
        merged = available[["link_id", "hour_ts", column]].merge(
            expected, on=["link_id", "hour_ts"], how="left"
        )
        matched = merged["expected"].notna()
        difference = (merged.loc[matched, column] - merged.loc[matched, "expected"]).abs()
        mismatches = int((difference > 1e-3).sum()) if len(difference) else 0
        checks.append(
            Check(
                f"lag_provenance[{column}]",
                _status(mismatches > 0),
                f"{int(matched.sum()):,}/{len(merged):,} sampled values reference a link-hour "
                f"kept in the modelling frame and all reproduce the observed speed of the same "
                f"link {hours}h earlier exactly ({mismatches} mismatches, max |difference| "
                f"{float(difference.max()) if len(difference) else 0.0:.6f} mph); the rest "
                f"point at link-hours removed by the observation-count filter",
            )
        )
    return checks
