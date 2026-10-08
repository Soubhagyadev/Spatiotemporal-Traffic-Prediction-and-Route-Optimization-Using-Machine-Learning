# Spatiotemporal Traffic Prediction and Route Optimization

Part 1 of the project plan: a memory-safe data pipeline that turns ~5.4 GB of raw
NYC DOT traffic-speed exports into an hourly, validated modelling dataset, plus a
trained and evaluated **traffic-speed prediction model**.

The dataset is the NYC DOT Traffic Speeds feed (`i4gi-tjb9`) covering
**2025-10-07 → 2026-10-07** for 125 road segments across the five boroughs.

---

## 1. Quickstart

```bash
# Python 3.11+; the ML stack plus the Part 2 web stack
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# run the whole pipeline: Part 1 (data → model) + Part 2 build steps (SQLite, scoring)
.venv/bin/python scripts/run_all.py          # add --part1-only to stop after the model
```

Then start the three Part 2 services, each in its own terminal:

```bash
.venv/bin/python scripts/run_api.py          # Flask API      → http://127.0.0.1:5001
.venv/bin/python scripts/run_dashboard.py    # Dash dashboard → http://127.0.0.1:8050
.venv/bin/python scripts/run_web.py          # Django web UI  → http://127.0.0.1:8000

.venv/bin/python scripts/check_services.py   # acceptance test for all three
```

Individual pipeline stages, in order:

| Command | What it does | Runtime |
|---|---|---|
| `python scripts/build_dataset.py` | streams the raw CSVs, cleans them, aggregates to hourly link statistics | ~4.3 min |
| `python scripts/build_features.py` | calendar/lag/rolling features, segment encodings, chronological split | ~1.2 min |
| `python scripts/validate_data.py` | raw + processed + split validation, writes the validation report | ~25 s |
| `python scripts/train_model.py` | trains MLR + random forest, evaluates, writes the model report | ~1 min |
| `python scripts/build_database.py` | loads the processed artifacts into `data/traffic.db` | ~25 s |
| `python scripts/build_predictions.py` | scores the test period and stores the predictions | ~15 s |

Measured end-to-end on this machine (8 cores, 16 GB RAM): **415 s wall clock,
983 MB peak RSS**, for 242 raw files / 12.06 M records.

Useful flags: `--keep-interim` (keep the per-segment record partitions),
`--skip-forest` (linear model only), `--sample-files N` (how many raw files the
raw-quality profile samples).

---

## 2. Repository layout

```
data_csv/                       raw exports (242 files, 12,064,919 records)
data/
  traffic.db                    deliverable: SQLite database (hourly, metadata, predictions)
  processed/hourly_traffic.csv  deliverable: cleaned hourly dataset (719,046 rows)
  processed/link_metadata.csv   deliverable: per-segment attributes and geometry
  processed/test_predictions.csv  scored test period (94,612 rows)
  features/model_dataset.csv    model-ready features with the train/val/test label
src/traffic/                    reusable pipeline + inference package (each module < 300 lines)
  config.py       paths, cleaning rules, thresholds
  cleaning.py     chunk-level cleaning of raw records
  partition.py    per-segment partitioning of cleaned records
  hourly.py       exact hourly aggregation
  link_meta.py    polyline decoding, segment geometry, borough/name/owner
  build.py        orchestration of the two preprocessing passes
  features.py     calendar, lag, rolling features and the chronological split
  encoders.py     out-of-fold mean-target encoding for link_id
  validation.py   dataset quality checks (missing values, outliers, coverage)
  leakage.py      split ordering, encoding provenance and lag provenance checks
  congestion.py   congestion categories and thresholds
  modeling.py     model fitting, metrics, VIF, residual diagnostics
  figures.py      diagnostic plots
  reporting.py    markdown report rendering
  store.py        SQLite schema, build step and read queries
  analytics.py    aggregate roll-ups (over time, patterns, congestion rankings)
  serving.py      train/serve-consistent feature builder for inference
services/
  common/api_client.py   cached HTTP client shared by the two frontends
  api/                   Flask app: data_service, prediction_service, routes
  dashboard/             Plotly Dash app: api_client-free figures, layouts, callbacks
  web/                   Django project: settings, portal app, templates, static
scripts/          pipeline entry points + four service runners + check_services.py
models/           fitted models, encoders, feature spec, metrics
reports/          validation_report.md, model_report.md, figures, profiles
```

---

## 3. Data

Relevant API fields: `data_as_of`, `link_id`, `speed`, `travel_time`, `status`,
`link_points`, `encoded_poly_line`, `borough`, `link_name`, `owner`.

`speed` and `travel_time` arrive as **text** and are converted explicitly with
`pandas.to_numeric(..., errors="coerce")`; unparsable entries become missing
values instead of breaking the run. Three data quirks drove the design:

1. **`status` is a quality flag.** `status = -101` marks carried-forward
   readings: 81% of those rows repeat the previous value of the same segment
   (versus 6% for `status = 0`), they are ~10× over-represented among
   physically impossible readings, and they hold all of the zero/negative
   speed and travel-time sentinels. Dropping `status != 0` removes 33.6% of
   raw records — the single most important cleaning decision.
2. **`link_points` is truncated at 255 characters** (55% of rows hit the cap)
   and sometimes ends mid-coordinate. The geometry is therefore recovered by
   decoding the untruncated `encoded_poly_line` field; 119/125 segments get a
   complete polyline, and each decoded geometry must agree with the anchor
   point from `link_points` before it is accepted.
3. **`speed × travel_time` is nearly constant per segment** (median relative
   spread 2.4%), which gives a physical consistency test for individual
   readings.

---

## 4. Pipeline

### Stage 1 — memory-safe cleaning and partitioning (`build_dataset.py`)

Raw files are streamed in 250,000-row chunks; only the required columns are
kept and **chunks are never concatenated into one raw DataFrame**. Each chunk is
typed, filtered and appended to one small file per `link_id`:

1. parse `data_as_of`, `speed`, `travel_time`, `status`, `link_id`;
2. drop rows with unparsable timestamp/link, non-numeric speed or travel time;
3. drop `status != 0` (stale/carried-forward);
4. drop non-positive speed or travel time (sentinel zeros);
5. drop repeated `(link_id, timestamp)` records (in-file, then across files);
6. append survivors to `data/interim/link_records/link_<id>.csv`.

### Stage 2 — exact hourly aggregation (`hourly.py`)

Each segment file is then processed **one segment at a time** (~65,000 rows,
a few MB), so every statistic is computed exactly — median and standard
deviation included — without a single large in-memory frame:

- de-duplicate on the exact timestamp across files;
- remove readings deviating more than 30% from the segment's median implied
  length (`speed × travel_time`);
- group by `(link_id, date, hour)` → `obs_count`, mean/median/std/min/max of
  `speed` and `travel_time`.

`12,064,919` raw records → `8,009,907` clean records → **`719,046` link-hours**
(8,381 distinct hours, 110 segments with usable data). The run peaks at a few
hundred MB of RAM and takes ~4.5 minutes.

### Stage 3 — features (`build_features.py`)

- **Calendar**: hour, day of week, month, weekend flag, cyclical sin/cos.
- **Lags by exact timestamp** (`lag_1h`, `2h`, `3h`, `24h`, `168h`): a lag is
  the aggregate of the hour that is exactly *k* hours earlier, and `NaN` when
  that hour is missing — never a silently stale value.
- **Rolling means** over the previous 3/6/24 observations (current hour excluded).
- **Categorical encoding**: `link_id` (125 levels) via mean-target encoding —
  expanding-window out-of-fold inside the training period, training-period
  statistics for validation/test; `borough` via one-hot encoding with a
  reference level.
- **Split**: chronological 70/15/15 by whole hours — train 491,141, validation
  101,264, test 94,612 link-hours; the series is never shuffled.

Missing lag/rolling cells (2% of rows) are imputed with the row's link-hour
profile and finally with training-period medians.

### Stage 4 — validation (`validate_data.py`)

31 automated PASS/WARN/FAIL checks over the raw profile, the hourly dataset and
the split → `reports/validation_report.md`. They cover missing values, invalid
or negative readings, duplicates, timestamp ordering and hourly alignment,
outliers, thin link-hours, insufficient observations per segment, coverage,
panel balance, zero-variance features, encoding provenance and lag provenance
(lag values are recomputed from the data and compared exactly). Current result:
**28 PASS, 3 WARN, 0 FAIL**; the warnings are 15 dead segments with no valid
data, one sparse segment and 6 segments without full geometry.

### Stage 5 — model (`train_model.py`)

**Primary model: multiple linear regression** (23 features, interpretable).
Target: hourly mean speed. Evaluation: **MAE, RMSE, R²** on all three splits,
against two baselines (previous hour's speed, and the segment's average speed
for that hour of day). A random forest on a 200,000-row subsample is trained as
a non-linear reference.

---

## 5. Results

Test period (94,612 link-hours, 2026-08-15 → 2026-10-07):

| Predictor | MAE (mph) | RMSE (mph) | R² |
|---|---|---|---|
| `link_hour_profile` baseline | 6.64 | 9.92 | 0.676 |
| `persistence_lag_1h` baseline | 3.78 | 6.31 | 0.869 |
| **Multiple linear regression** | **3.40** | **5.33** | **0.906** |
| Random forest (reference) | 3.10 | 5.01 | 0.917 |

- The linear model cuts the previous-hour persistence error by **10.0%** and
  the static segment/hour profile error by **48.8%**.
- The random forest adds only 8.9% over the linear model, so most of the
  achievable accuracy is captured by the interpretable model.
- Standardised coefficients: `speed_lag_1h` **+0.72**, `speed_roll_24h` +0.14,
  `speed_lag_24h` +0.13, `speed_lag_168h` +0.13 — the model extrapolates recent
  speed and corrects it with the time-of-day and segment profile.
- **Congestion categories** from training-period percentiles
  (`high < 31.53 mph ≤ medium < 48.05 mph ≤ low`): predicted speeds land in the
  same category as the observation **85.2%** of the time (High recall 89%,
  Low recall 86%, Medium recall 80%).

Charts in `reports/figures/`: speed over time, hourly pattern, weekday×hour
heatmap, congestion by segment, predicted-vs-actual, model comparison,
residuals, feature effects and the congestion confusion matrix.

Sanity check worth noting: 107–110 of 110 segments report in every hour-of-day
bucket (97% panel balance), so the diurnal speed profile is a real traffic
pattern, not an artefact of a changing set of reporting segments.

---

## 6. Part 2 — Flask API, Django web interface, Plotly Dash dashboard

### Architecture

```
        ┌──────────────────────────┐        ┌───────────────────────────┐
        │  Django web interface    │        │  Plotly Dash dashboard    │
        │  (browse, forms, tables) │        │  (interactive analytics)  │
        └────────────┬─────────────┘        └────────────┬──────────────┘
                     │  HTTP (shared cached client: services/common/api_client.py)
                     ▼                                   ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  Flask API  (services/api)                                    │
        │  data_service → traffic.store / traffic.analytics             │
        │  prediction_service → traffic.serving → model_bundle.joblib   │
        └───────────────┬───────────────────────────┬───────────────────┘
                        ▼                           ▼
              data/traffic.db (SQLite)      models/model_bundle.joblib
```

The ML pipeline stays offline: the API only *reads* the SQLite database and the
frozen model bundle, and the two frontends only speak HTTP to the API.

### Build steps

```bash
python scripts/build_database.py      # data/traffic.db: hourly_traffic, link_metadata
python scripts/build_predictions.py   # test_predictions table + CSV (94,612 scored hours)
```

### API endpoints

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/health`, `/api/meta` | dataset window, boroughs, congestion thresholds, model summary |
| GET | `/api/links`, `/api/links/<id>` | segment directory / one segment with its recent hours |
| GET | `/api/links/<id>/history` | hourly or daily series for a segment |
| GET | `/api/traffic/over_time` | mean speed per day/hour for a scope or one segment |
| GET | `/api/patterns/hourly`, `/api/patterns/weekday_hour` | diurnal and weekday×hour profiles |
| GET | `/api/congestion/segments`, `/boroughs`, `/summary` | rankings, per-borough mix, scope KPIs |
| GET | `/api/model/metrics`, `/predictions`, `/errors` | full metrics, scored test rows, worst segments |
| POST | `/api/predict` | predicted speed + congestion category for one segment-hour |
| POST | `/api/predict/horizon` | rolled-forward forecast for the next *n* hours |
| POST | `/api/predict/batch` | up to 500 segment-hours in one call |
| GET | `/api/predict/latest/<id>` | prediction for the hour after the last observation |

`POST /api/predict` rebuilds the training features from the database through
`traffic.serving`: exact-timestamp lags, rolling means over the previous
*usable* hours (`obs_count >= 3`, the same filter used in training), the fitted
segment/hour encodings and the stored imputation values. Each response reports
its `mode` (`observed-lag` when all five lags are real observations, `forecast`
when some were imputed) plus how many lags were observed. Verified
consistency: the live API reproduces the offline test-set predictions exactly
(27.721 mph for segment 4616341 at 2026-09-10 23:00, both models).

### Dashboard views (`scripts/run_dashboard.py`)

Scope controls (borough, segment, daily/hourly resolution, model) drive six
headline numbers and eleven figures: speed over time, hourly pattern,
weekday×hour heatmap, most congested segments, congestion mix per borough,
segment map, selected-segment history with a 6-hour forecast, predicted vs
observed scatter, error comparison against both baselines, feature effects and
the largest segment-level errors. The dashboard is a pure HTTP client — it
never opens the database or the model.

### Django pages (`scripts/run_web.py`)

| Page | Content |
|---|---|
| `/` | dataset summary, KPI cards, borough table, top congested segments, model headline metrics |
| `/segments/` | searchable, sortable segment directory (filter by borough, search by name or id) |
| `/segments/<id>/` | segment metadata, congestion summary, recent-hour table, speed chart, 6-hour forecast |
| `/predict/` | form-driven prediction: segment, timestamp, model, horizon → speed, category, mode, per-step table |
| `/model/` | full metrics by predictor and split, category report and confusion matrix, coefficients, importances, worst segments |
| `/about/` | system description, pipeline summary and architecture |

Every page degrades gracefully with a warning banner when the API is down.

### Verification

`scripts/check_services.py` exercises all three services against their running
HTTP endpoints — API payload shapes and error handling, every Dash callback
executed with realistic control values, and each Django page including a real
CSRF-protected prediction POST — and exits non-zero on any failure. Result on
this machine: **42/42 checks pass**.

---

## 7. Deliverables

**Part 1 (plan items 1, 2, 3, 8)**

| Plan item | Where |
|---|---|
| 1. Cleaned/aggregated dataset | `data/processed/hourly_traffic.csv`, `data/processed/link_metadata.csv` |
| 2. Reproducible preprocessing pipeline | `scripts/build_dataset.py` → `scripts/run_all.py`, `src/traffic/` |
| 3. Trained and evaluated model | `models/model_bundle.joblib`, `models/metrics.json`, `reports/model_report.md` |
| 8. README (setup, processing, model, results) | this file |

Also produced: the model-ready feature table
(`data/features/model_dataset.csv`), the fitted encoders and feature
specification (`models/encoders.joblib`, `models/feature_spec.json`), the
validation report (`reports/validation_report.md`) and the SQLite database
(`data/traffic.db`) with its scored test period.

**Part 2 (plan items 4, 5, 6)**

| Plan item | Where |
|---|---|
| 4. Flask API serving predictions | `services/api/`, run with `scripts/run_api.py` |
| 5. Django web interface | `services/web/`, run with `scripts/run_web.py` |
| 6. Plotly Dash traffic dashboard | `services/dashboard/`, run with `scripts/run_dashboard.py` |
| — | shared client `services/common/api_client.py`, acceptance test `scripts/check_services.py` |

### Linked frontends

The two interfaces link to each other so neither is a dead end: the dashboard
header has a **Web interface →** button pointing at the Django portal, and the
portal navbar has an **Analytics dashboard →** button pointing back. Both URLs
are configurable (`TRAFFIC_WEB_URL` for the dashboard, `TRAFFIC_DASHBOARD_URL`
for Django) and default to `http://127.0.0.1:8000` / `http://127.0.0.1:8050`.
The acceptance test asserts both links are present in the served pages.

### Part 3 (route optimization) — out of scope

Part 3 was deliberately dropped. The infrastructure it would need is in place
(`POST /api/predict/horizon` produces future segment speeds, and `link_metadata`
carries coordinates and lengths), but the data covers 125 limited-access
segments rather than a street grid, so a routing graph would be sparse and many
origin/destination pairs unreachable. Documented here as future work rather
than half-implemented.

---

## 8. Design notes and limitations

- **Why drop `status != 0`?** It removes a third of the records, but those
  records are carried-forward sensor values rather than observations. Keeping
  them would re-introduce ~46,000 zero-speed readings per sample and make the
  hourly means meaningless.
- **Outliers are judged per segment, not globally**, because free-flow speed
  legitimately differs by a factor of three across motorways and local streets.
  0.019% of clean records were removed this way.
- **A small number of extreme but self-consistent readings survive** (67 of
  719,046 hourly means are outside 1–90 mph); they are reported rather than
  clipped, since speed and travel time agree with the segment's own history.
- **`hours_since_prev_obs` was dropped from the feature set**: after requiring
  a previous-hour observation it is constant, so it carried no information.
  The validation suite now checks for zero-variance features automatically.
- **Congestion thresholds are absolute speeds**, so one threshold set is used
  for motorways and local streets alike. A per-segment free-flow ratio is the
  natural refinement.
- **15 segments never report a valid hour** and cannot be predicted until their
  sensors recover.

---

## 9. Reproducibility

Every stage is deterministic: fixed random seeds for the forest and the
sampling-based diagnostics, chronological splits derived from timestamps, and
all thresholds (congestion cut-offs, outlier tolerance, minimum observations)
computed from the training period only. Re-running `python scripts/run_all.py`
from the raw exports reproduces every artifact in this repository.
