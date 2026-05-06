"""Data loading helpers.

Loads walks.csv and accommodation.json once at startup so tools can read
from in-memory structures rather than re-parsing on every request.

Walk IDs are normalised to int. Distance/ascent fields are normalised to
numeric types. Accommodation entries get an `osm_id` field constructed
as "{type}/{id}" (e.g. "node/302866833") to match the OSM-style format
used in the trip state plan.
"""

import csv
import json
from functools import lru_cache
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
WALKS_CSV = OUTPUT_DIR / "walks.csv"
ACCOMMODATION_JSON = OUTPUT_DIR / "accommodation.json"


def _parse_int(value):
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=1)
def load_walks() -> list[dict]:
    """Return all walks with normalised types.

    Each walk dict has keys:
        walk_id (int), title, area, wainwrights, county, author,
        grade, distance_miles (float), distance_km (float),
        ascent_feet (int), ascent_metres (int), estimated_time,
        start_lat (float), start_lng (float), start_postcode,
        description, gpx_available (bool)
    """
    walks: list[dict] = []
    with WALKS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            walks.append({
                "walk_id": _parse_int(row["walk_id"]),
                "title": row["title"],
                "area": row["area"],
                "wainwrights": row["wainwrights"],
                "county": row["county"],
                "author": row["author"],
                "grade": row["grade"],
                "distance_miles": _parse_float(row["distance_miles"]),
                "distance_km": _parse_float(row["distance_km"]),
                "ascent_feet": _parse_int(row["ascent_feet"]),
                "ascent_metres": _parse_int(row["ascent_metres"]),
                "estimated_time": row["estimated_time"],
                "start_lat": _parse_float(row["start_lat"]),
                "start_lng": _parse_float(row["start_lng"]),
                "start_postcode": row["start_postcode"],
                "description": row["description"],
                "gpx_available": row["gpx_available"].strip().lower() == "true",
            })
    return walks


@lru_cache(maxsize=1)
def load_accommodation() -> list[dict]:
    """Return all accommodation entries with an osm_id field added.

    osm_id is "{type}/{id}", e.g. "node/302866833" or "way/12345".
    """
    with ACCOMMODATION_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        entry["osm_id"] = f"{entry['type']}/{entry['id']}"
    return data
