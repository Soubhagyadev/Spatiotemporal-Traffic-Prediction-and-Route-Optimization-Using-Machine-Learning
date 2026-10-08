# Spatiotemporal Traffic Prediction and Route Optimization

## Project Goal
Build a data-science system that predicts traffic conditions on road segments and optionally recommends routes with lower predicted congestion.

Dataset: NYC DOT Traffic Speeds API (`i4gi-tjb9`), using approximately one year of data (2025-10-07 to 2026-10-07).

## Part 1 — Data Pipeline, Validation, and ML Model

### 1.1 Raw Data
Relevant fields:
- `data_as_of` — timestamp
- `link_id` — road-segment identifier
- `speed` — observed speed
- `travel_time` — observed travel time
- `status` — data/status indicator
- `link_points` — road-segment coordinates
- `borough`, `link_name`, `owner`

The API returns `speed` and `travel_time` as text, so they must be explicitly converted to numeric types during preprocessing.

### 1.2 Memory-Safe Processing
The raw data is too large to load into a 16 GB RAM machine at once.

Process CSV/API data in chunks:
1. Read one chunk.
2. Keep only required columns.
3. Convert timestamp, speed, and travel time to proper types.
4. Remove invalid/duplicate records.
5. Aggregate the chunk to the required time resolution.
6. Save the aggregated result.
7. Release memory before processing the next chunk.

Do not concatenate all raw chunks into one DataFrame.

### 1.3 Hourly Aggregation
Traffic prediction will use **hourly road-segment observations**.

Group by:
- `link_id`
- date
- hour

Calculate useful statistics such as:
- mean/median speed
- mean travel time
- observation count

This reduces millions of raw observations to a manageable modeling dataset while preserving the major temporal traffic patterns.

### 1.4 Feature Engineering
Create features such as:
- hour of day
- day of week
- month
- weekend/weekday
- road segment (`link_id`)
- lagged speed
- previous-hour speed
- rolling average speed
- previous-day same-hour speed, where available

Handle categorical features using appropriate encoding.

### 1.5 Data Validation
Check:
- missing values
- invalid/negative speed or travel time
- duplicate records
- timestamp ordering
- outliers
- insufficient observations per road segment
- train/test leakage

Use a chronological split; never randomly shuffle time-series data.

### 1.6 Model
Primary model: **Multiple Linear Regression**, because it is interpretable and matches the current course scope.

Target:
- predicted traffic speed (`speed`)

The predicted speed can then be converted into congestion categories (Low/Medium/High) using thresholds defined from the dataset.

Optionally compare MLR with a nonlinear model such as Random Forest if time permits.

Evaluate with:
- MAE
- RMSE
- R²

## Part 2 — Backend and Dashboard

### Stack
- **Flask** — backend/API and model-serving layer
- **Django** — frontend/web application layer
- **Plotly Dash** — interactive traffic analytics dashboard
- **Pandas/NumPy** — data processing
- **Scikit-learn** — ML
- **SQLite/PostgreSQL** — store processed/aggregated data if required

The architecture should keep the ML/data pipeline separate from the web UI so the model can be trained offline and loaded by the backend for prediction.

Dashboard should show:
- traffic speed over time
- congestion by road segment
- hourly/day-of-week patterns
- predicted vs actual traffic
- model performance
- road-segment/location information

## Part 3 — Optional Route Optimization

Use the road-segment coordinates already provided by `link_points`.

Build a road graph:
- nodes = road-segment endpoints/intersections
- edges = road segments
- edge cost = distance/travel time adjusted using predicted congestion

Use a shortest-path algorithm such as **Dijkstra or A\***.

The route recommender should use **future predicted traffic**, not only current traffic, and recommend a route with lower predicted travel cost.

This part is optional and should only be implemented after Parts 1 and 2 are stable.

## Deliverables
1. Cleaned/aggregated dataset.
2. Reproducible preprocessing pipeline.
3. Trained and evaluated traffic prediction model.
4. Flask API serving predictions.
5. Django web interface.
6. Plotly Dash traffic dashboard.
7. Optional predicted-congestion route recommender.
8. README documenting setup, data processing, model, and results.
