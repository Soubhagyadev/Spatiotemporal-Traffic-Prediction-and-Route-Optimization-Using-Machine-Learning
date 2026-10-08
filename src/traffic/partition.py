"""Per-link partitioning of cleaned records.

Cleaned records are appended to one small file per ``link_id`` instead of
being held (or concatenated) in memory.  Because the road network only has a
few hundred segments, the number of simultaneously open files stays small and
each per-link file is a few megabytes, so the hourly aggregation can later
process one link at a time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ["link_id", "hour_idx", "ts_sec", "speed", "travel_time"]


class LinkPartitionWriter:
    """Append cleaned records to ``<directory>/link_<id>.csv`` files."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handles: dict[int, object] = {}
        self.rows_written = 0

    def _handle(self, link_id: int):
        handle = self._handles.get(link_id)
        if handle is None:
            path = self.path_for(link_id)
            is_new = not path.exists()
            handle = path.open("w", encoding="utf-8", newline="")
            if is_new or path.stat().st_size == 0:
                handle.write(",".join(RECORD_COLUMNS) + "\n")
            self._handles[link_id] = handle
        return handle

    def path_for(self, link_id: int) -> Path:
        return self.directory / f"link_{int(link_id)}.csv"

    def write(self, records: pd.DataFrame) -> None:
        """Append a cleaned chunk, grouped by link, to the partition files."""
        if records.empty:
            return
        for link_id, group in records.groupby("link_id", sort=False, observed=True):
            group.to_csv(
                self._handle(int(link_id)),
                header=False,
                index=False,
                columns=RECORD_COLUMNS,
                lineterminator="\n",
            )
            self.rows_written += len(group)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "LinkPartitionWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def existing_links(self) -> list[int]:
        """Link ids that already have a partition file on disk."""
        return sorted(
            int(path.stem.split("_")[1]) for path in self.directory.glob("link_*.csv")
        )
