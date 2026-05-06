"""Orchestrate gpx + routing into a single accommodation-to-trail connector.

build_connector(walk_id, accom_lat, accom_lon, mode) returns everything
the frontend needs to draw the accommodation, the connector, and the
(possibly trimmed) trail on a single Leaflet map.

Mode determines two things:
  walking - circular routes are rotated to start at the nearest point on
            the loop, and the connector polyline uses the OSM walking
            network (footpaths, bridleways, walkable roads).
  driving - circular routes keep their canonical start, and the connector
            uses the OSM driving network. The walker is assumed to drive
            from the accommodation to the canonical trailhead.

Run directly to test:
    python -m server.connector
"""

import math
import time

from server.gpx import find_join_point, load_gpx_points, is_circular
from server.routing import Mode, route_between


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _polyline_length_m(polyline: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(polyline)):
        total += _haversine_m(
            polyline[i - 1][0], polyline[i - 1][1],
            polyline[i][0], polyline[i][1],
        )
    return total


def build_connector(
    walk_id: int,
    accom_lat: float,
    accom_lon: float,
    mode: Mode = "walking",
) -> dict:
    """Build the full accommodation-to-trail map payload.

    Returns a dict with:
      walk_id              int
      mode                 'walking' | 'driving'
      is_circular          bool
      join_lat/lon         float    coordinates where the connector meets the trail
      join_index           int      index of the join point in the original GPX
      connector            list     (lat, lon) polyline from accommodation to join
      connector_length_m   float    routed distance along the connector
      trail                list     (lat, lon) polyline of the walk
      trail_length_m       float    distance along the displayed trail

    Walking mode rotates circular trails so display starts at the join point.
    Driving mode keeps the canonical start (index 0) and an unrotated trail.
    """
    if mode == "driving":
        # Driver always joins at the canonical trailhead. No rotation, no
        # nearest-point search -- the assumption is they're parking at the
        # official car park and beginning the walk there.
        points = load_gpx_points(walk_id)
        circular = is_circular(points)
        join_lat, join_lon = points[0]
        join_index = 0
        display_route = list(points)
    else:
        join = find_join_point(walk_id, accom_lat, accom_lon)
        circular = join["is_circular"]
        join_lat = join["join_lat"]
        join_lon = join["join_lon"]
        join_index = join["join_index"]
        display_route = join["display_route"]

    routed = route_between(accom_lat, accom_lon, join_lat, join_lon, mode=mode)
    # The routed polyline starts/ends at OSM graph nodes which can be tens of
    # metres from the actual input points, or even degenerate to a single
    # point when both endpoints snap to the same node. Bookend with the real
    # coordinates so the connector is always a visible line from the
    # accommodation pin to where the trail starts.
    connector = [(accom_lat, accom_lon), *routed, (join_lat, join_lon)]

    return {
        "walk_id": walk_id,
        "mode": mode,
        "is_circular": circular,
        "join_lat": join_lat,
        "join_lon": join_lon,
        "join_index": join_index,
        "connector": connector,
        "connector_length_m": _polyline_length_m(connector),
        "trail": display_route,
        "trail_length_m": _polyline_length_m(display_route),
    }


# --------------------------------------------------------------------- main

def _run_test(label: str, walk_id: int, accom_lat: float, accom_lon: float, mode: Mode = "walking") -> None:
    print(f"\n--- {label}  [mode={mode}] ---")
    print(f"  walk_id={walk_id}, accom=({accom_lat:.5f}, {accom_lon:.5f})")
    t0 = time.perf_counter()
    payload = build_connector(walk_id, accom_lat, accom_lon, mode=mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  is_circular: {payload['is_circular']}")
    print(f"  join: ({payload['join_lat']:.5f}, {payload['join_lon']:.5f})  index={payload['join_index']}")
    print(f"  connector: {len(payload['connector'])} points, {payload['connector_length_m']:.0f} m")
    print(f"  trail:     {len(payload['trail'])} points, {payload['trail_length_m']:.0f} m")
    print(f"  total build time: {elapsed_ms:.0f} ms")

    end_lat, end_lon = payload["connector"][-1]
    gap = _haversine_m(end_lat, end_lon, payload["join_lat"], payload["join_lon"])
    print(f"  connector end -> join point gap: {gap:.0f} m (expect small)")

    first = payload["trail"][0]
    gap2 = _haversine_m(first[0], first[1], payload["join_lat"], payload["join_lon"])
    print(f"  trail[0] -> join point gap: {gap2:.0f} m")


if __name__ == "__main__":
    # Same accommodation + walk pair, compared in walking and driving mode.
    _run_test("Catbells, accom in Keswick", walk_id=2036, accom_lat=54.6005, accom_lon=-3.1340, mode="walking")
    _run_test("Catbells, accom in Keswick", walk_id=2036, accom_lat=54.6005, accom_lon=-3.1340, mode="driving")

    _run_test("Fairfield, accom in Ambleside", walk_id=1002, accom_lat=54.4313791, accom_lon=-2.9621712, mode="walking")
    _run_test("Fairfield, accom in Ambleside", walk_id=1002, accom_lat=54.4313791, accom_lon=-2.9621712, mode="driving")
