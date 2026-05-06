"""LLM-callable tools.

Stage 3 implementations:
  - search_walks
  - get_walk_details
  - search_accommodation

State and map tools (update_trip_state, get_trip_state, show_on_map) are
added in later stages.

Tools raise on invalid input or missing data; the agent loop will catch
exceptions and feed structured error messages back to the LLM.
"""

import math
from typing import Optional

from rapidfuzz import fuzz, process

from server.data import load_accommodation, load_walks


# --------------------------------------------------------------------- helpers


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _resolve_areas(query_areas: list[str], canonical_areas: list[str]) -> set[str]:
    """Match user-typed area names to canonical area names.

    Exact (case-insensitive) matches short-circuit to just that area —
    so "Far Eastern Fells" doesn't bleed into "Eastern Fells" via the
    fuzzy matcher's substring overlap. Anything else falls back to
    rapidfuzz with a permissive cutoff.
    """
    canonical_by_lower = {a.lower(): a for a in canonical_areas}
    matched: set[str] = set()
    for query in query_areas:
        key = query.strip().lower()
        if key in canonical_by_lower:
            matched.add(canonical_by_lower[key])
            continue
        # partial_ratio is more substring-friendly than the default WRatio,
        # which we want for partial area names like "far eastern".
        results = process.extract(
            query,
            canonical_areas,
            scorer=fuzz.partial_ratio,
            score_cutoff=80,
            limit=3,
        )
        for area, _score, _idx in results:
            matched.add(area)
    return matched


def _walk_summary(walk: dict) -> dict:
    """Lightweight summary used by search_walks."""
    desc = walk["description"] or ""
    return {
        "walk_id": walk["walk_id"],
        "title": walk["title"],
        "area": walk["area"],
        "grade": walk["grade"],
        "distance_miles": walk["distance_miles"],
        "ascent_metres": walk["ascent_metres"],
        "estimated_time": walk["estimated_time"],
        "start_lat": walk["start_lat"],
        "start_lng": walk["start_lng"],
        "wainwrights": walk["wainwrights"],
        "short_description": desc[:400],
    }


def _accommodation_summary(entry: dict, distance_km: float) -> dict:
    return {
        "id": entry["osm_id"],
        "name": entry.get("name") or "Unnamed",
        "tourism": entry.get("tourism"),
        "lat": entry["lat"],
        "lon": entry["lon"],
        "distance_km": round(distance_km, 2),
        "addr_street": entry.get("addr_street"),
        "addr_postcode": entry.get("addr_postcode"),
        "phone": entry.get("phone"),
        "website": entry.get("website"),
    }


# ------------------------------------------------------------------- the tools


def search_walks(
    query: Optional[str] = None,
    difficulty: Optional[list[str]] = None,
    distance_miles_min: Optional[float] = None,
    distance_miles_max: Optional[float] = None,
    ascent_metres_max: Optional[int] = None,
    areas: Optional[list[str]] = None,
    near_lat: Optional[float] = None,
    near_lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    limit: int = 10,
) -> list[dict]:
    """Filter walks by structured criteria, return lightweight summaries.

    The LLM should call get_walk_details for the full description of any
    walk it actually wants to consider.

    `query` is a case-insensitive substring match against the walk title
    and the Wainwrights list, useful when the user names a specific peak
    (e.g. "Scafell", "Helvellyn"). Combine with other filters as needed,
    or use alone to see every walk involving that peak regardless of grade.

    Difficulty values must come from the canonical set:
        "easy", "easy/mod", "moderate", "mod/hard",
        "hard", "severe", "very hard"

    Areas are fuzzy-matched (rapidfuzz, cutoff 80) against the canonical
    Lake District fell groupings, so "Far Eastern" matches "Far Eastern Fells".

    Proximity filtering only applies if all three of near_lat, near_lon,
    radius_km are provided. Walks are returned in CSV order, capped to
    `limit`.
    """
    walks = load_walks()

    matched_areas: Optional[set[str]] = None
    if areas:
        canonical = sorted({w["area"] for w in walks})
        matched_areas = _resolve_areas(areas, canonical)

    query_lower = query.strip().lower() if query else None

    use_proximity = (
        near_lat is not None and near_lon is not None and radius_km is not None
    )

    results: list[dict] = []
    for walk in walks:
        if query_lower:
            haystack = f"{walk['title']} {walk['wainwrights'] or ''}".lower()
            if query_lower not in haystack:
                continue
        if difficulty and walk["grade"] not in difficulty:
            continue
        if (
            distance_miles_min is not None
            and walk["distance_miles"] is not None
            and walk["distance_miles"] < distance_miles_min
        ):
            continue
        if (
            distance_miles_max is not None
            and walk["distance_miles"] is not None
            and walk["distance_miles"] > distance_miles_max
        ):
            continue
        if (
            ascent_metres_max is not None
            and walk["ascent_metres"] is not None
            and walk["ascent_metres"] > ascent_metres_max
        ):
            continue
        if matched_areas is not None and walk["area"] not in matched_areas:
            continue
        if use_proximity:
            if walk["start_lat"] is None or walk["start_lng"] is None:
                continue
            d = _haversine_km(near_lat, near_lon, walk["start_lat"], walk["start_lng"])
            if d > radius_km:
                continue
            summary = _walk_summary(walk)
            summary["distance_from_query_km"] = round(d, 1)
            results.append(summary)
        else:
            results.append(_walk_summary(walk))
            if len(results) >= limit:
                break

    if use_proximity:
        results.sort(key=lambda r: r["distance_from_query_km"])

    return results[:limit]


def get_walk_details(walk_id: int) -> dict:
    """Return the full walk row including the long description.

    Raises KeyError if walk_id is not in the dataset.
    """
    for walk in load_walks():
        if walk["walk_id"] == walk_id:
            return walk
    raise KeyError(f"No walk with walk_id={walk_id}")


def search_accommodation(
    near_lat: float,
    near_lon: float,
    radius_km: float = 10.0,
    types: Optional[list[str]] = None,
    limit: int = 15,
) -> list[dict]:
    """Find accommodation within `radius_km` of (near_lat, near_lon).

    Returns up to `limit` entries sorted by distance ascending. `types`
    filters by the OSM `tourism` tag (e.g. ["hotel", "guest_house",
    "camp_site"]); pass None to include all types.
    """
    results: list[dict] = []
    for entry in load_accommodation():
        if entry.get("lat") is None or entry.get("lon") is None:
            continue
        if types and entry.get("tourism") not in types:
            continue
        d = _haversine_km(near_lat, near_lon, entry["lat"], entry["lon"])
        if d > radius_km:
            continue
        results.append(_accommodation_summary(entry, d))

    results.sort(key=lambda x: x["distance_km"])
    return results[:limit]
