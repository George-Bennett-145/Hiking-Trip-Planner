"""Step 3: GPX parsing, circular/linear classification, and trail join point.

Given a walk_id and an accommodation coordinate, find_join_point() returns:
  - whether the route is circular or linear
  - the nearest point on the route to the accommodation (the join coordinate)
  - the display route: GPX points reordered so the map render starts at
    the join point and goes all the way around (circular), or the full
    route unchanged (linear)

Run directly to test:
    python -m server.gpx
"""

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# Anchor to this file's location so the module works no matter what cwd
# the server was started from.
GPX_DIR = Path(__file__).resolve().parent.parent / "output" / "gpx"
CIRCULAR_THRESHOLD_M = 250


# --------------------------------------------------------------------- helpers

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -------------------------------------------------------------------- parsing

def load_gpx_points(walk_id: int) -> list[tuple[float, float]]:
    """Parse a GPX file and return all route points as (lat, lon) tuples.

    Detects the GPX namespace from the root element rather than hardcoding
    it, and falls back to <trkpt> if the file uses tracks instead of
    routes. The Walking Britain dataset uses <rtept> in the topografix
    namespace, but neither assumption is safe for arbitrary GPX input.

    Raises FileNotFoundError if the GPX file does not exist.
    """
    path = GPX_DIR / f"walk_{walk_id}.gpx"
    if not path.exists():
        raise FileNotFoundError(f"No GPX file for walk_id={walk_id} at {path}")

    root = ET.parse(path).getroot()
    ns_match = re.match(r"\{(.+?)\}", root.tag)
    ns = {"gpx": ns_match.group(1)} if ns_match else {}
    prefix = "gpx:" if ns else ""

    elements = (
        root.findall(f".//{prefix}rtept", ns)
        or root.findall(f".//{prefix}trkpt", ns)
    )
    return [(float(e.attrib["lat"]), float(e.attrib["lon"])) for e in elements]


# --------------------------------------------------------------- classification

def is_circular(points: list[tuple[float, float]], threshold_m: float = CIRCULAR_THRESHOLD_M) -> bool:
    """Return True if start and end points are within threshold_m of each other."""
    if len(points) < 2:
        return False
    return _haversine_m(points[0][0], points[0][1], points[-1][0], points[-1][1]) <= threshold_m


# ------------------------------------------------------------------ join point

def find_join_point(
    walk_id: int,
    accom_lat: float,
    accom_lon: float,
) -> dict:
    """Find where accommodation connects to a trail.

    Returns a dict with:
      is_circular    bool    True if start-end gap <= 250m
      join_lat       float   lat of the nearest route point
      join_lon       float   lon of the nearest route point
      join_index     int     index of the join point in the original points list
      join_dist_m    float   crow-flies distance from accommodation to join point
      display_route  list    (lat, lon) tuples for the map, starting at the join
                             point and wrapping around for circular routes, or
                             the full unmodified route for linear routes

    For circular routes the dead section (from canonical start to join point)
    is removed: display_route runs join_point -> loop_end -> loop_start ->
    join_point so the whole loop is preserved but starts where the walker joins.

    For linear routes the join point is always the canonical start (index 0)
    and display_route is the full unmodified GPX sequence.
    """
    points = load_gpx_points(walk_id)
    circular = is_circular(points)

    if not circular:
        return {
            "is_circular": False,
            "join_lat": points[0][0],
            "join_lon": points[0][1],
            "join_index": 0,
            "join_dist_m": _haversine_m(accom_lat, accom_lon, points[0][0], points[0][1]),
            "display_route": list(points),
        }

    # Find the nearest point on the loop to the accommodation.
    best_idx = 0
    best_dist = float("inf")
    for i, (lat, lon) in enumerate(points):
        d = _haversine_m(accom_lat, accom_lon, lat, lon)
        if d < best_dist:
            best_dist = d
            best_idx = i

    join_lat, join_lon = points[best_idx]

    # Reorder the loop so display starts at join_index.
    # points[best_idx:] takes us from join to the canonical end.
    # points[1:best_idx+1] continues from canonical start back to join.
    # The [1:] skips points[0] to avoid duplicating the canonical
    # start/end position (which are nearly identical on a circular route).
    display_route = list(points[best_idx:]) + list(points[1 : best_idx + 1])

    return {
        "is_circular": True,
        "join_lat": join_lat,
        "join_lon": join_lon,
        "join_index": best_idx,
        "join_dist_m": best_dist,
        "display_route": display_route,
    }


# --------------------------------------------------------------------- main

def _run_test(label: str, walk_id: int, accom_lat: float, accom_lon: float) -> None:
    print(f"\n--- {label} ---")
    points = load_gpx_points(walk_id)
    circular = is_circular(points)
    start_end_m = _haversine_m(points[0][0], points[0][1], points[-1][0], points[-1][1])

    print(f"  walk_id={walk_id}, {len(points)} GPX points")
    print(f"  start-end gap: {start_end_m:.1f} m  ->  {'circular' if circular else 'linear'}")

    result = find_join_point(walk_id, accom_lat, accom_lon)
    print(f"  accom at ({accom_lat:.5f}, {accom_lon:.5f})")
    print(f"  join at  ({result['join_lat']:.5f}, {result['join_lon']:.5f})  index={result['join_index']}")
    print(f"  join_dist_m: {result['join_dist_m']:.0f} m")
    print(f"  display_route: {len(result['display_route'])} points")
    print(f"  display_route[0]: {result['display_route'][0]}")
    print(f"  display_route[-1]: {result['display_route'][-1]}")

    if result["is_circular"]:
        # Verify the display route starts and ends at (or near) the join point
        d_start = _haversine_m(
            result["join_lat"], result["join_lon"],
            result["display_route"][0][0], result["display_route"][0][1],
        )
        d_end = _haversine_m(
            result["join_lat"], result["join_lon"],
            result["display_route"][-1][0], result["display_route"][-1][1],
        )
        print(f"  display starts {d_start:.1f}m from join point (expect 0)")
        print(f"  display ends   {d_end:.1f}m from join point (expect ~same as start-end gap)")


if __name__ == "__main__":
    # Test 1: Catbells (walk_2036) - known circular.
    # Accommodation in Keswick town centre, a couple of km from the trail.
    _run_test(
        "Catbells circular, accom in Keswick",
        walk_id=2036,
        accom_lat=54.6005,
        accom_lon=-3.1340,
    )

    # Test 2: Catbells with accommodation right near the canonical start
    # (Gutherscale car park area). Join index should be 0 or close to it.
    _run_test(
        "Catbells circular, accom near canonical start",
        walk_id=2036,
        accom_lat=54.5795,
        accom_lon=-3.1686,
    )

    # Test 3: Fairfield Horseshoe (walk_1002) - circular start from Ambleside.
    _run_test(
        "Fairfield Horseshoe, accom in Ambleside centre",
        walk_id=1002,
        accom_lat=54.4313,
        accom_lon=-2.9622,
    )

    # Test 4: a walk without GPX to confirm the FileNotFoundError is clean.
    print("\n--- Missing GPX (expect FileNotFoundError) ---")
    try:
        load_gpx_points(9999)
    except FileNotFoundError as e:
        print(f"  Caught expected error: {e}")
