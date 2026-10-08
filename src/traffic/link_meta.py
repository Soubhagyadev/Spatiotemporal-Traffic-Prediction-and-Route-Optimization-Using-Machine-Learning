"""Static road-segment metadata extracted from the raw link geometry fields.

``link_points`` is capped at 255 characters in the API export, so for most
segments it only holds a partial polyline (sometimes ending in a half-written
coordinate).  The ``encoded_poly_line`` field is not truncated, so it is
decoded into the full geometry and used as the primary source, with
``link_points`` only as a fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.7613
# Generous bounding box around New York City, used to reject coordinates that
# were cut in half by the 255-character truncation of ``link_points``.
NYC_BBOX = (40.40, 41.05, -74.35, -73.60)  # lat_min, lat_max, lon_min, lon_max


def decode_polyline(encoded: str | None, precision: int = 5) -> np.ndarray | None:
    """Decode a Google-encoded polyline into an ``(N, 2)`` lat/lon array."""
    if not encoded or not isinstance(encoded, str):
        return None
    factor = float(10**precision)
    coordinates: list[tuple[float, float]] = []
    index = lat = lon = 0
    size = len(encoded)
    while index < size:
        for axis in ("lat", "lon"):
            shift = result = 0
            while True:
                if index >= size:  # truncated input
                    return None
                value = ord(encoded[index]) - 63
                index += 1
                result |= (value & 0x1F) << shift
                shift += 5
                if value < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if axis == "lat":
                lat += delta
            else:
                lon += delta
        coordinates.append((lat / factor, lon / factor))
    if len(coordinates) < 2:
        return None
    return np.asarray(coordinates, dtype="float64")


def parse_link_points(raw: str | None) -> np.ndarray | None:
    """Parse ``"lat,lon lat,lon ..."``, dropping truncated coordinates."""
    if not raw or not isinstance(raw, str):
        return None
    lat_min, lat_max, lon_min, lon_max = NYC_BBOX
    pairs = []
    for token in raw.split():
        lat_text, _, lon_text = token.partition(",")
        try:
            lat, lon = float(lat_text), float(lon_text)
        except ValueError:
            continue
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            pairs.append((lat, lon))
    if len(pairs) < 2:
        return None
    return np.asarray(pairs, dtype="float64")


def polyline_length_miles(coords: np.ndarray) -> float:
    """Great-circle length of the segment polyline in miles."""
    lat1, lon1 = np.radians(coords[:-1, 0]), np.radians(coords[:-1, 1])
    lat2, lon2 = np.radians(coords[1:, 0]), np.radians(coords[1:, 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(np.sum(2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))))


def _inside_bbox(coords: np.ndarray) -> bool:
    lat_min, lat_max, lon_min, lon_max = NYC_BBOX
    return bool(
        ((coords[:, 0] >= lat_min) & (coords[:, 0] <= lat_max)).all()
        and ((coords[:, 1] >= lon_min) & (coords[:, 1] <= lon_max)).all()
    )


def _first_point(raw: str | None) -> np.ndarray | None:
    """First coordinate of ``link_points``; unaffected by its truncation."""
    coords = parse_link_points(raw)
    return None if coords is None else coords[0]


def _accepts(coords: np.ndarray | None, anchor: np.ndarray | None) -> bool:
    """A decoded geometry is accepted if it is in range and starts where
    ``link_points`` says the segment starts."""
    if coords is None or len(coords) < 2 or not _inside_bbox(coords):
        return False
    if anchor is None:
        return True
    return bool(np.abs(coords[0] - anchor).max() < 5e-4)


def _geometry(row) -> tuple[np.ndarray | None, str, bool]:
    """Resolve a segment's full geometry, preferring the encoded polyline.

    The export escapes some characters of the encoded polyline (a backslash
    may be doubled), so the raw string and a de-escaped variant are tried; a
    candidate is only accepted if it lies inside the city and agrees with the
    anchor point from ``link_points``.
    """
    raw = getattr(row, "encoded_poly_line", None)
    anchor = _first_point(getattr(row, "link_points", None))
    candidates = [("encoded_polyline", raw)]
    if isinstance(raw, str) and "\\" in raw:
        candidates.append(("encoded_polyline_unescaped", raw.replace("\\", "")))
    for source, candidate in candidates:
        coords = decode_polyline(candidate)
        if _accepts(coords, anchor):
            return coords, source, True

    points = parse_link_points(getattr(row, "link_points", None))
    if points is not None:
        return points, "link_points_partial", False
    return None, "unavailable", False


def collect_link_metadata(chunk: pd.DataFrame, store: dict[int, dict]) -> None:
    """Record the static attributes of any link seen for the first time.

    Geometry fields are identical for every observation of a link, so a single
    sample per link is parsed once and the rest of the chunk is discarded.
    """
    sample = chunk.drop_duplicates(subset=["link_id"], keep="first")
    sample = sample.loc[~pd.to_numeric(sample["link_id"], errors="coerce").isin(list(store))]
    for row in sample.itertuples(index=False):
        try:
            link_id = int(row.link_id)
        except (TypeError, ValueError):
            continue
        if link_id in store:
            continue
        coords, source, complete = _geometry(row)
        store[link_id] = {
            "link_id": link_id,
            "borough": row.borough,
            "link_name": row.link_name,
            "owner": row.owner,
            "geometry_source": source,
            "geometry_complete": complete,
            "n_polyline_points": 0 if coords is None else int(len(coords)),
            "start_lat": np.nan if coords is None else float(coords[0, 0]),
            "start_lon": np.nan if coords is None else float(coords[0, 1]),
            "end_lat": np.nan if coords is None else float(coords[-1, 0]),
            "end_lon": np.nan if coords is None else float(coords[-1, 1]),
            "geom_length_miles": np.nan if coords is None else polyline_length_miles(coords),
            "link_points_truncated": bool(
                isinstance(row.link_points, str) and len(row.link_points) >= 255
            ),
        }


def metadata_frame(store: dict[int, dict]) -> pd.DataFrame:
    """Convert the collected metadata store into a sorted DataFrame."""
    frame = pd.DataFrame(list(store.values()))
    if frame.empty:
        return frame
    return frame.sort_values("link_id").reset_index(drop=True)
