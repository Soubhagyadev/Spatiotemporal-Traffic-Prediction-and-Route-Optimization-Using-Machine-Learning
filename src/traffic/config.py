"""Central configuration for the traffic prediction project (Part 1).

All paths are derived from the project root so the pipeline is reproducible
regardless of the working directory it is launched from.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data_csv"
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
LINK_PARTS_DIR = INTERIM_DIR / "link_records"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

HOURLY_DATASET_PATH = PROCESSED_DIR / "hourly_traffic.csv"
LINK_METADATA_PATH = PROCESSED_DIR / "link_metadata.csv"
PREPROCESS_STATS_PATH = REPORT_DIR / "preprocessing_stats.json"
MODEL_DATASET_PATH = FEATURES_DIR / "model_dataset.csv"

# Part 2 artifacts
DB_PATH = DATA_DIR / "traffic.db"
TEST_PREDICTIONS_PATH = PROCESSED_DIR / "test_predictions.csv"

OUTPUT_DIRECTORIES = (
    DATA_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    FEATURES_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
)


def ensure_directories() -> None:
    """Create every output directory the pipeline writes into."""
    for directory in OUTPUT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()

# --------------------------------------------------------------------------
# Raw data layout
# --------------------------------------------------------------------------
RAW_COLUMNS = [
    "link_id",
    "speed",
    "travel_time",
    "status",
    "data_as_of",
    "link_points",
    "encoded_poly_line",
    "borough",
    "link_name",
    "owner",
]

# Columns the API returns as text and that must be converted explicitly.
TEXT_NUMERIC_COLUMNS = ("speed", "travel_time")

# Rows are streamed from disk in chunks of this size and never concatenated
# into a single raw DataFrame.
CHUNK_SIZE = 250_000

# Timestamps are naive local (America/New_York) wall-clock time.  Hour
# buckets are indexed as whole hours since this origin so they can be stored
# as compact integers in the interim files.
HOUR_ORIGIN_ISO = "2025-01-01T00:00:00"

# --------------------------------------------------------------------------
# Cleaning rules (documented in reports/validation_report.md)
# --------------------------------------------------------------------------
# `status` is the dataset's own quality flag: 0 is the nominal state, -101
# marks carried-forward/stale records.
VALID_STATUS = 0
MIN_SPEED_MPH = 0.0  # strictly greater than zero
MIN_TRAVEL_TIME_S = 0.0  # strictly greater than zero

# Outlier rule: implied segment length (speed x travel_time) is nearly
# constant per link, so observations deviating more than this fraction from
# the link's robust median length are treated as measurement outliers.
LENGTH_OUTLIER_TOLERANCE = 0.30

# --------------------------------------------------------------------------
# Aggregation / modelling thresholds
# --------------------------------------------------------------------------
MIN_OBS_PER_LINK_HOUR = 3  # keep hourly rows with at least this many readings
MIN_HOURS_PER_LINK = 24 * 14  # "insufficient observations" reporting threshold

# Chronological split (never shuffled).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # test fraction is the remainder
TARGET_COLUMN = "speed_mean"
