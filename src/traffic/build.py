"""Orchestration for the two memory-safe preprocessing passes."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import config, hourly, link_meta
from .cleaning import CleaningStats, clean_file
from .partition import LinkPartitionWriter


def find_raw_files(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    """Raw CSV exports, deterministically ordered."""
    return sorted(Path(raw_dir).glob("traffic_*.csv"))


def build_link_records(
    raw_files: list[Path],
    parts_dir: Path = config.LINK_PARTS_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    verbose: bool = True,
) -> tuple[dict[int, dict], CleaningStats]:
    """Pass 1: stream every raw file, clean it and partition it by link.

    Each raw chunk is read, cleaned and appended to per-link files before the
    next chunk is read, so peak memory stays at roughly one chunk.
    """
    store: dict[int, dict] = {}
    stats = CleaningStats()
    started = time.perf_counter()
    with LinkPartitionWriter(parts_dir) as writer:
        for index, path in enumerate(raw_files, start=1):
            try:
                records = clean_file(
                    path,
                    stats,
                    chunk_size=chunk_size,
                    on_chunk=lambda chunk: link_meta.collect_link_metadata(chunk, store),
                )
                writer.write(records)
            except Exception as exc:  # pragma: no cover - defensive
                stats.files_failed.append(f"{path.name}: {exc}")
            if verbose and (index % 20 == 0 or index == len(raw_files)):
                elapsed = time.perf_counter() - started
                print(
                    f"  [{index}/{len(raw_files)}] rows kept={stats.kept_rows:,} "
                    f"links={len(store)} elapsed={elapsed:,.0f}s",
                    flush=True,
                )
    return store, stats


def attach_link_summary(metadata: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Combine static link attributes with the data-derived per-link summary."""
    if metadata.empty:
        return summary
    return metadata.merge(summary, on="link_id", how="outer").sort_values("link_id")


def write_hourly_dataset(dataset: pd.DataFrame, path: Path = config.HOURLY_DATASET_PATH) -> Path:
    """Write the aggregated dataset with a stable, readable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")
    return path


def run(raw_dir: Path = config.RAW_DIR, parts_dir: Path = config.LINK_PARTS_DIR) -> dict:
    """Execute both passes and return the produced artifacts."""
    raw_files = find_raw_files(raw_dir)
    if not raw_files:
        raise FileNotFoundError(f"no traffic_*.csv files found in {raw_dir}")

    store, stats = build_link_records(raw_files, parts_dir=parts_dir)
    dataset, summary = hourly.build_hourly_dataset(parts_dir)
    metadata = attach_link_summary(link_meta.metadata_frame(store), summary)
    return {
        "raw_files": raw_files,
        "hourly": dataset,
        "link_metadata": metadata,
        "cleaning_stats": stats,
        "link_summary": summary,
    }
