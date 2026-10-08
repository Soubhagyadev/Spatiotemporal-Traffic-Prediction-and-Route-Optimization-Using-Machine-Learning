"""Train and evaluate the traffic-speed prediction models (plan section 1.6).

Usage
-----
    python scripts/train_model.py [--skip-forest]

Primary model: multiple linear regression.  Comparison model: random forest.
Writes ``models/model_bundle.joblib``, ``models/metrics.json``,
``reports/model_report.md`` and the figures referenced by the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traffic import config, congestion, figures, modeling, reporting  # noqa: E402

SPLIT_ORDER = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-forest", action="store_true", help="train only the linear model")
    return parser.parse_args()


def load_splits() -> tuple[dict[str, pd.DataFrame], dict]:
    dataset = pd.read_csv(config.MODEL_DATASET_PATH, parse_dates=["hour_ts"])
    specification = json.loads((config.MODEL_DIR / "feature_spec.json").read_text())
    splits = {
        name: dataset[dataset["split"] == name].sort_values("hour_ts").reset_index(drop=True)
        for name in SPLIT_ORDER
    }
    return splits, specification


def metrics_table(results: dict) -> str:
    rows = []
    for group in ("baselines", "models"):
        for name, splits in results[group].items():
            rows.append(
                [
                    f"`{name}`",
                    *[
                        f"{splits[split]['mae']:.2f} / {splits[split]['rmse']:.2f} / "
                        f"{splits[split]['r2']:.3f}"
                        for split in SPLIT_ORDER
                        if split in splits
                    ],
                ]
            )
    return reporting.markdown_table(
        ["Predictor", "train MAE/RMSE/R²", "validation MAE/RMSE/R²", "test MAE/RMSE/R²"], rows
    )


def split_table(splits: dict[str, pd.DataFrame]) -> str:
    total = sum(len(frame) for frame in splits.values())
    rows = [
        [
            f"`{name}`",
            f"{len(frame):,}",
            f"{len(frame) / total:.1%}",
            f"{frame['hour_ts'].min():%Y-%m-%d}",
            f"{frame['hour_ts'].max():%Y-%m-%d}",
            f"{frame['link_id'].nunique()}",
        ]
        for name, frame in splits.items()
    ]
    return reporting.markdown_table(
        ["Split", "Link-hours", "Share", "From", "To", "Segments"], rows
    )


def conclusions(results: dict, category_report: dict, coefficients: pd.DataFrame) -> str:
    linear = results["models"]["linear_regression"]["test"]
    persistence = results["baselines"]["persistence_lag_1h"]["test"]
    profile = results["baselines"]["link_hour_profile"]["test"]
    improvement = 100 * (1 - linear["mae"] / persistence["mae"])
    top = ", ".join(
        f"`{row.feature}` ({row.std_coefficient:+.2f})"
        for row in coefficients.head(3).itertuples()
    )
    forest = results["models"].get("random_forest", {}).get("test")
    lines = [
        f"- Multiple linear regression reaches **MAE {linear['mae']:.2f} mph**, "
        f"**RMSE {linear['rmse']:.2f} mph** and **R² {linear['r2']:.3f}** on the held-out "
        f"test period.",
        f"- It improves on the previous-hour persistence baseline by "
        f"**{improvement:.1f}% MAE** ({persistence['mae']:.2f} mph) and on the static "
        f"segment/hour profile by "
        f"{100 * (1 - linear['mae'] / profile['mae']):.1f}%, so the lagged and calendar "
        f"features carry real predictive signal beyond the daily routine.",
        f"- The strongest standardised effects are {top}, i.e. the model is essentially "
        f"extrapolating recent speed while adjusting for the time of day and the segment.",
        f"- Congestion categories agree with the observed category in "
        f"**{category_report['accuracy']:.1%}** of test link-hours.",
    ]
    if forest:
        delta = 100 * (forest["mae"] - linear["mae"]) / linear["mae"]
        lines.append(
            f"- The random forest reference scores MAE {forest['mae']:.2f} mph "
            f"({delta:+.1f}% versus the linear model), which quantifies how much accuracy a "
            f"non-linear model adds on top of the interpretable baseline."
        )
    lines += [
        "",
        "**Next steps (Part 2/3).** The fitted encoders, feature specification and model "
        "bundle are persisted so the Flask service can predict without touching the raw data; "
        "the hourly aggregates and `link_metadata.csv` (segment endpoints, borough, length) "
        "are the inputs the Dash dashboard and the route optimiser need.",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    splits, specification = load_splits()
    columns = specification["feature_columns"]
    target = specification["target"]
    print(f"Training rows {len(splits['train']):,} | features {len(columns)} | target {target}")

    thresholds = congestion.derive_thresholds(splits["train"][target])
    print(
        f"Congestion thresholds from training period: high < "
        f"{thresholds['high_congestion_below_mph']:.2f} mph <= medium < "
        f"{thresholds['low_congestion_above_mph']:.2f} mph <= low"
    )

    print("Fitting multiple linear regression ...")
    linear = modeling.fit_linear_regression(splits["train"][columns], splits["train"][target])
    models = {"linear_regression": linear}
    forest_rows = 0
    if not args.skip_forest:
        print("Fitting random forest (subsampled) ...")
        forest, forest_rows = modeling.fit_random_forest(
            splits["train"][columns], splits["train"][target]
        )
        models["random_forest"] = forest

    results = modeling.evaluate_splits(models, splits, columns, target)
    coefficients = modeling.coefficient_table(linear, splits["train"][columns], splits["train"][target])
    vif = modeling.variance_inflation_factors(splits["train"][columns])
    importances = modeling.feature_importance(models.get("random_forest"), columns)

    test = splits["test"]
    predictions = {name: modeling.predict(model, test[columns]) for name, model in models.items()}
    category_report = congestion.category_report(test[target], predictions["linear_regression"], thresholds)
    diagnostics = modeling.residual_diagnostics(test, predictions["linear_regression"], target)
    distribution = congestion.category_distribution(test[target], thresholds)

    print("Writing figures ...")
    split_boundaries = {
        name: (
            frame["hour_ts"].min(),
            frame["hour_ts"].max(),
            colour,
        )
        for (name, frame), colour in zip(splits.items(), ("#2b6cb0", "#dd6b20", "#38a169"))
    }
    hourly = pd.read_csv(config.HOURLY_DATASET_PATH, parse_dates=["hour_ts"], usecols=[
        "link_id", "hour_ts", "speed_mean"
    ])
    metadata = pd.read_csv(config.LINK_METADATA_PATH)
    figures.plot_speed_over_time(hourly, split_boundaries)
    figures.plot_hourly_pattern(hourly)
    figures.plot_dow_hour_heatmap(hourly)
    figures.plot_congestion_by_segment(hourly, metadata, thresholds)
    figures.plot_predicted_vs_actual(
        test[target], predictions["linear_regression"],
        results["models"]["linear_regression"]["test"],
    )
    figures.plot_model_comparison(results, "mae")
    figures.plot_model_comparison(results, "rmse")
    figures.plot_residuals(test[target], predictions["linear_regression"], test["hour_ts"].dt.hour)
    figures.plot_feature_effects(coefficients, importances)
    figures.plot_congestion_confusion(category_report)

    bundle = {
        "models": models,
        "feature_columns": columns,
        "target": target,
        "congestion_thresholds": thresholds,
        "encoders": joblib.load(config.MODEL_DIR / "encoders.joblib"),
        "trained_on": {
            "rows": int(len(splits["train"])),
            "from": str(splits["train"]["hour_ts"].min()),
            "to": str(splits["train"]["hour_ts"].max()),
        },
    }
    joblib.dump(bundle, config.MODEL_DIR / "model_bundle.joblib", compress=3)

    metrics = {
        "results": results,
        "congestion_thresholds": thresholds,
        "congestion_distribution_test": distribution,
        "congestion_category_report": category_report,
        "coefficients": coefficients.head(20).round(4).to_dict(orient="records"),
        "random_forest_importances": importances,
        "max_vif": max(vif.values()) if vif else 0.0,
        "max_vif_feature": max(vif, key=vif.get) if vif else "",
        "residual_diagnostics": diagnostics,
        "random_forest_training_rows": forest_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (config.MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    worst_vif = sorted(vif.items(), key=lambda item: -item[1])[:5]
    report = reporting.render_model_report(
        {
            "results": results,
            "thresholds": thresholds,
            "category_report": category_report,
            "feature_columns": columns,
            "split_table": split_table(splits),
            "metrics_table": metrics_table(results),
            "coefficient_table": reporting.markdown_table(
                ["Feature", "Coefficient (mph)", "Standardised"],
                [
                    [f"`{row.feature}`", f"{row.coefficient:+.4f}", f"{row.std_coefficient:+.4f}"]
                    for row in coefficients.head(15).itertuples()
                ],
            ),
            "max_vif": max(vif.values()) if vif else 0.0,
            "max_vif_feature": max(vif, key=vif.get) if vif else "",
            "residual_text": (
                f"- Mean residual by hour of day stays within "
                f"{min(diagnostics['mean_residual_by_hour'].values()):+.2f} to "
                f"{max(diagnostics['mean_residual_by_hour'].values()):+.2f} mph, so the model "
                f"is not systematically biased at particular times of day.\n"
                f"- Largest segment-level errors: "
                + ", ".join(
                    f"`{entry['link_id']}` ({entry['mae']:.2f} mph)"
                    for entry in diagnostics["worst_links"][:5]
                )
                + ".\n- Highest VIF: "
                + ", ".join(f"`{name}` {value:.1f}" for name, value in worst_vif)
                + "."
            ),
            "conclusions": conclusions(results, category_report, coefficients),
        }
    )
    report_path = reporting.write_report(report, config.REPORT_DIR / "model_report.md")

    print(
        f"\nTest period -> MAE {results['models']['linear_regression']['test']['mae']:.2f} mph, "
        f"RMSE {results['models']['linear_regression']['test']['rmse']:.2f} mph, "
        f"R² {results['models']['linear_regression']['test']['r2']:.3f}"
    )
    print(f"Category agreement: {category_report['accuracy']:.1%}")
    print(f"Bundle : {config.MODEL_DIR / 'model_bundle.joblib'}")
    print(f"Metrics: {config.MODEL_DIR / 'metrics.json'}")
    print(f"Report : {report_path}")
    print(f"Elapsed {metrics['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
