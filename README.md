# Spatiotemporal Traffic Prediction and Route Optimization

## Abstract

Urban traffic congestion costs billions in lost productivity and fuel every year. Accurate short-term speed prediction across road segments enables smarter routing, better signal control, and more responsive navigation systems. This project builds an end-to-end machine learning system that ingests a year of NYC DOT traffic-speed sensor data, learns spatiotemporal patterns across 110 road segments spanning all five boroughs, and serves real-time speed predictions and congestion classifications through a production-grade web stack.

---

## Problem

Given a road segment and a target hour, predict the mean traffic speed and classify congestion (High / Medium / Low). The prediction must be:

- **accurate** — better than naive baselines (persistence and historical profile)
- **interpretable** — stakeholders need to understand why a segment is predicted to be congested
- **live** — served through an API that any frontend or system can query

The dataset is the NYC DOT Traffic Speeds feed covering **2025-10-07 → 2026-10-07**, 12 million raw records across 125 road segments, delivered as ~5.4 GB of CSV exports.

---

## Implementation

### Data pipeline

Raw CSVs are streamed in 250,000-row chunks and never loaded into a single DataFrame. Records with carried-forward sensor values (`status != 0`, 33.6% of rows), unparsable fields, and physically impossible readings are dropped. Clean records are partitioned per segment, then aggregated to hourly statistics (mean, median, std, min, max speed and travel time). 12M raw records → 719,046 validated link-hours.

### Feature engineering

23 features per link-hour:

- **Temporal** — hour, day-of-week, month, weekend flag, cyclical sin/cos encodings
- **Lag features** — exact-timestamp lags at 1h, 2h, 3h, 24h, 168h (never silently stale)
- **Rolling means** — previous 3, 6, 24 observations
- **Segment encoding** — mean-target encoding for `link_id` (out-of-fold in training, fixed for val/test); one-hot for borough

### Model

Primary model: **Multiple Linear Regression** (interpretable, 23 features).
Reference model: **Random Forest** on a 200k-row subsample.
Both evaluated against two baselines: previous-hour persistence and segment/hour historical profile.

### Validation

31 automated checks across raw data, processed dataset and train/val/test split covering missing values, duplicates, timestamp ordering, outliers, panel balance, zero-variance features, encoding provenance and lag provenance. Result: **28 PASS, 3 WARN, 0 FAIL**.

---

## Results

Test period — 94,612 link-hours (2026-08-15 → 2026-10-07):

| Model | MAE (mph) | RMSE (mph) | R² |
|---|---|---|---|
| Historical profile baseline | 6.64 | 9.92 | 0.676 |
| Persistence (lag 1h) baseline | 3.78 | 6.31 | 0.869 |
| **Multiple Linear Regression** | **3.40** | **5.33** | **0.906** |
| Random Forest (reference) | 3.10 | 5.01 | 0.917 |

- Linear model cuts persistence error by **10%** and profile error by **49%**
- Congestion category accuracy: **85.2%** (High recall 89%, Low 86%, Medium 80%)
- Random forest adds only 8.9% over the linear model — most signal is captured by the interpretable model

---

## What Makes This Different

Most traffic ML projects load the full dataset into memory, shuffle the time series before splitting, and leak future information into lag features. This project addresses all three:

- **Memory-safe pipeline** — streams 5.4 GB in chunks, peaks at under 1 GB RAM
- **Strict chronological split** — data is never shuffled; train → val → test follows time order
- **Leak-proof features** — lags are computed from exact timestamps and cross-validated against the raw data; the validation suite checks provenance automatically
- **Train/serve consistency** — the live API rebuilds features from the database using the exact same logic as training, so offline and online predictions match exactly (verified to 3 decimal places)
- **Interpretable primary model** — the random forest is a reference point, not the deployed model

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Python, pandas, NumPy |
| ML | scikit-learn (LinearRegression, RandomForestRegressor) |
| Storage | SQLite (`data/traffic.db`) |
| API | Flask + flask-cors |
| Web interface | Django 5 |
| Dashboard | Plotly Dash |
| Containerisation | Docker, Docker Compose |

---

## How to Run

### With Docker (recommended)

Requires Docker Desktop or Docker Engine with Compose.

```bash
# First time — build the image and start
docker compose up --build

# Subsequently
docker compose up
```

Services will be available at:

| Service | URL |
|---|---|
| Django web interface | http://localhost:8000 |
| Plotly Dash dashboard | http://localhost:8050 |
| Flask API | http://localhost:5001 |

To stop:
```bash
docker compose down
```

### Without Docker

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# Start all three services
python start.py
```

> **Note:** The pipeline artifacts (`data/traffic.db`, `models/model_bundle.joblib`) must exist before starting the services. If running from scratch with the raw CSVs, run the pipeline first: `.venv/bin/python scripts/run_all.py`

---

## Conclusion

A clean, memory-efficient pipeline over 12 million sensor records produces a validated hourly dataset that a simple linear model predicts with R² = 0.906 and MAE = 3.40 mph. The full system — from raw CSV to live prediction API with interactive dashboard and web portal — runs end-to-end in under 7 minutes and is fully reproducible from the raw data. The architecture deliberately keeps the ML pipeline offline and the serving layer stateless, making it straightforward to retrain on new data or swap in a better model without touching the frontends.
